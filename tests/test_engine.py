from interactive_llm.config import Config
from interactive_llm.engine import Engine, sentences


class FakeLocal:
    model = "fake"
    def __init__(self, script): self.script, self.calls = script, 0
    def stream(self, messages, tools=None, required=False):
        yield from self.script[self.calls]; self.calls += 1


class FakeClaude:
    available = True
    def ask(self, q, image_b64=None, context="", cancel=None): return f"claude:{q}"


def test_sentences_split():
    assert list(sentences(iter(["Hi the", "re. How ", "are you?"]))) == ["Hi there.", "How are you?"]


def test_plain_reply():
    e = Engine(Config(), FakeLocal([[{"text": "Hello."}]]), FakeClaude())
    assert "".join(e.reply("hi")) == "Hello."
    assert e.history[-1]["content"] == "Hello."


def test_tool_escalation():
    call = {"id": "1", "name": "ask_claude", "arguments": '{"question": "2+2"}'}
    e = Engine(Config(), FakeLocal([[{"tool_calls": [call]}], [{"text": "It is 4."}]]), FakeClaude())
    assert "".join(e.reply("hard")) == "Let me think about that. It is 4."
    assert any(m.get("role") == "tool" and m["content"] == "claude:2+2" for m in e.history)


def test_force_claude():
    e = Engine(Config(), FakeLocal([]), FakeClaude())
    assert "".join(e.reply("x", force_claude=True)) == "claude:x"


class SumLocal(FakeLocal):
    def complete(self, messages, max_tokens=300):
        return "User chatted about many things."


def test_compaction_summarizes_old_turns_keeps_recent():
    cfg = Config(); cfg.context_tokens, cfg.context_keep_tokens = 300, 120
    e = Engine(cfg, SumLocal([]), FakeClaude())
    for i in range(12):
        e.history += [{"role": "user", "content": f"question {i} " + "x" * 120}, {"role": "assistant", "content": "answer " + "y" * 120}]
    assert e.compact() is True
    assert e.summary and "summary" in e.history[0]["content"]
    assert e.history[1]["role"] == "user" and e.history[-1]["content"].startswith("answer")
    from interactive_llm.engine import est_tokens
    assert sum(est_tokens(m) for m in e.history[1:]) <= cfg.context_keep_tokens  # kept tail fits the budget
    assert "question 11" in e.history[-2]["content"]


def test_no_compaction_under_budget():
    e = Engine(Config(), SumLocal([]), FakeClaude())
    assert e.compact() is False


def test_old_images_dropped_without_note():
    e = Engine(Config(), FakeLocal([[{"text": "a"}], [{"text": "b"}]]), FakeClaude())
    list(e.reply("one", images=["AAAA"])); list(e.reply("two"))
    assert e.history[1]["content"] == "one"


def test_replace_last_assistant_after_barge_in():
    e = Engine(Config(), FakeLocal([[{"text": "Long reply here."}], [{"text": "Sure."}], [{"text": "Ok."}]]), FakeClaude())
    list(e.reply("hi")); e.replace_last_assistant("Long")
    assert e.history[-1]["content"] == "Long"                      # exactly what was heard, no marker the model could imitate
    list(e.reply("go on"))
    user_msg = e.history[-2]["content"]                              # the model is told once, outside its own text
    assert "interrupted your previous reply" in str(user_msg)
    list(e.reply("and again"))
    assert "interrupted" not in str(e.history[-2]["content"])        # ...and only once


def test_needs_deep_thinking_gate():
    from interactive_llm.engine import needs_deep_thinking as n
    for hard in ["What is seventeen times twenty three?", "solve x squared minus four", "17 * 23", "Here is a riddle for you",
                 "write me a function that sorts a list", "tell me step by step how to do it"]:
        assert n(hard), hard
    for easy in ["Hello there.", "Tell me a fun fact about octopuses.", "Thanks, that's all for now.",
                 "What is the capital of Australia?", "I am person A."]:
        assert not n(easy), easy


def test_interrupted_tool_call_leaves_valid_history():
    import time as _t
    class SlowClaude(FakeClaude):
        def ask(self, q, image_b64=None, context="", cancel=None):
            for _ in range(100):
                if cancel and cancel(): return ""
                _t.sleep(0.05)
            return "late"
    call = {"id": "1", "name": "deep_think", "arguments": '{"question": "x"}'}
    stop = {"v": False}
    e = Engine(Config(), FakeLocal([[{"tool_calls": [call]}]]), SlowClaude())
    g = e.reply("solve this puzzle", cancel=lambda: stop["v"])
    assert next(g).startswith("Let me think")
    stop["v"] = True
    assert list(g) == []                                   # stops promptly instead of waiting for the tool
    roles = [m["role"] for m in e.history]
    assert roles[-2:] == ["assistant", "tool"] and e.history[-1]["tool_call_id"] == "1"  # no dangling tool_calls


def test_tools_only_offered_for_hard_questions():
    seen = []
    class Spy(FakeLocal):
        def stream(self, messages, tools=None):
            seen.append(bool(tools)); yield {"text": "ok."}
    e = Engine(Config(), Spy([]), FakeClaude())
    list(e.reply("Hello there")); list(e.reply("What is 17 times 23"))
    assert seen == [False, True]


def test_note_without_images_does_not_crash():
    e = Engine(Config(), FakeLocal([[{"text": "ok."}]]), FakeClaude())
    assert "".join(e.reply("hi", images=None, note="[Context: x]", speaker="Sam")) == "ok."
    assert e.history[1]["content"][0]["text"] == "Sam: hi" and e.history[1]["content"][1]["text"] == "[Context: x]"


def test_model_can_call_skill_tools_with_conversation_context():
    """A skill tool call runs through the runner, its result enters the history, and the final answer follows."""
    seen = []

    def runner(name, args):
        seen.append((name, args))
        return "VTI last traded at 359.48 USD."

    call = {"id": "1", "name": "stock_quote", "arguments": '{"company": "VTI"}'}
    tools = [{"type": "function", "function": {"name": "stock_quote"}}]
    local = FakeLocal([[{"tool_calls": [call]}], [{"text": "VTI is 359.48."}]])
    e = Engine(Config(), local, FakeClaude())
    out = "".join(e.reply("VTI", skill_tools=tools, skill_runner=runner))
    assert out == "VTI is 359.48."     # an instant lookup needs no spoken filler
    assert seen == [("stock_quote", {"company": "VTI"})]
    assert any(m.get("role") == "tool" and "359.48" in m["content"] for m in e.history)


def test_slow_skill_tool_gets_a_spoken_filler(monkeypatch):
    """A lookup that takes longer than FILLER_AFTER_S says 'One moment' once, before the answer."""
    import time
    from interactive_llm import engine as engine_module
    monkeypatch.setattr(engine_module, "FILLER_AFTER_S", 0.0)

    def runner(name, args):
        time.sleep(0.6)
        return "VTI last traded at 359.48 USD."

    call = {"id": "1", "name": "stock_quote", "arguments": '{"company": "VTI"}'}
    tools = [{"type": "function", "function": {"name": "stock_quote"}}]
    local = FakeLocal([[{"tool_calls": [call]}], [{"text": "VTI is 359.48."}]])
    e = Engine(Config(), local, FakeClaude())
    assert "".join(e.reply("VTI", skill_tools=tools, skill_runner=runner)) == "One moment. VTI is 359.48."


def test_live_data_question_answered_from_memory_is_replaced_by_a_search():
    """If the model answers a live-data question in words, that answer is never spoken; a web search runs instead."""
    seen = []

    def runner(name, args):
        seen.append((name, args))
        return "VOO last traded at 520.10 USD."

    tools = [{"type": "function", "function": {"name": "web_search"}}]
    local = FakeLocal([[{"text": "VOO is at $482.15."}], [{"text": "VOO is 520.10."}]])
    e = Engine(Config(), local, FakeClaude())
    out = "".join(e.reply("what about voo's price", skill_tools=tools, skill_runner=runner, must_use_tool=True))
    assert "482" not in out and "520.10" in out
    assert seen == [("web_search", {"query": "what about voo's price"})]
    assert not any("482" in str(m.get("content")) for m in e.history)


def test_fallback_lookup_uses_the_stock_quote_tool_when_a_ticker_is_named():
    """A capitalised ticker in the sentence selects stock_quote rather than a general search."""
    seen = []
    tools = [{"type": "function", "function": {"name": "web_search"}}, {"type": "function", "function": {"name": "stock_quote"}}]
    local = FakeLocal([[{"text": "VOO is at $1."}], [{"text": "ok"}]])
    e = Engine(Config(), local, FakeClaude())
    "".join(e.reply("and what about the S&P 500 one, VOO?", skill_tools=tools,
                    skill_runner=lambda name, args: seen.append((name, args)) or "quote", must_use_tool=True))
    assert seen == [("stock_quote", {"company": "VOO"})]


def test_fallback_does_not_treat_an_acronym_as_a_ticker():
    """"What's going on with the ASEAN summit?" must fall back to a web search, not a stock quote for "ASEAN"."""
    seen = []
    tools = [{"type": "function", "function": {"name": "web_search"}}, {"type": "function", "function": {"name": "stock_quote"}}]
    local = FakeLocal([[{"text": "Nothing much."}], [{"text": "ok"}]])
    e = Engine(Config(), local, FakeClaude())
    "".join(e.reply("What's going on with the ASEAN summit?", skill_tools=tools,
                    skill_runner=lambda name, args: seen.append(name) or "result", must_use_tool=True))
    assert seen == ["web_search"]
