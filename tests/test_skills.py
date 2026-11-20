"""Tests for the voice skills: parsing, routing, confirmation, honesty and persistence."""
import asyncio
import os
import stat

import pytest

from interactive_llm import settings as user_settings
from interactive_llm.server import SKILL_CLAIM_RE
from interactive_llm.skills.base import (CANCELLED, FAILED, NEEDS_CONFIRMATION, NEEDS_INPUT, OK, SkillContext)
from interactive_llm.skills.mac_control import MacControlSkill
from interactive_llm.skills.manager import SkillManager, event_kind, result_note
from interactive_llm.skills.notes import NotesSkill, NotesStore
from interactive_llm.skills.numbers import parse_duration, spoken_number
from interactive_llm.skills.timers import TimerSkill
from interactive_llm.skills.web_search import WebSearchSkill


def ctx(speaker="Marvin", names=("Marvin", "Hana"), announced=None):
    async def announce(spoken, shown):
        if announced is not None:
            announced.append(spoken)
    return SkillContext(speaker=speaker, known_names=list(names), visible_names=[], announce=announce)


def run(coro):
    return asyncio.run(coro)


def test_parse_duration_phrases():
    cases = {"5 minutes": 300, "half an hour": 1800, "an hour and a half": 5400, "1 hour 30 minutes": 5400,
             "two and a half minutes": 150, "ninety seconds": 90, "twenty five minutes": 1500, "a minute": 60}
    for text, seconds in cases.items():
        assert parse_duration(text) == seconds, text
    assert parse_duration("what time is it") is None and spoken_number("volume to fifty") == 50


def test_timer_set_list_cancel_and_alert():
    async def scenario():
        heard = []
        manager = SkillManager([TimerSkill()], ["timers"])
        context = ctx(announced=heard)
        result = await manager.handle("set a timer for 1 second called pasta", context)
        assert result.kind == OK and "pasta" in result.note
        listing = await manager.handle("what timers are running?", context)
        assert listing.kind == OK and "pasta" in listing.note
        await asyncio.sleep(1.4)
        assert heard == ["Your pasta timer is done."]
        none_left = await manager.handle("cancel my timer", context)
        assert none_left.kind == FAILED                       # nothing to cancel is reported, not faked
        set_again = await manager.handle("start a timer for 10 minutes", context)
        cancelled = await manager.handle("cancel the timer", context)
        assert set_again.kind == OK and cancelled.kind == OK
    run(scenario())


def test_timer_without_duration_asks_then_accepts_followup():
    async def scenario():
        manager = SkillManager([TimerSkill()], ["timers"])
        context = ctx()
        asked = await manager.handle("set a timer", context)
        assert asked.kind == NEEDS_INPUT and asked.followup
        answered = await manager.handle("ten minutes", context)
        assert answered.kind == OK and answered.skill == "timers"
        too_long = await manager.handle("set a timer for 30 hours", context)
        assert too_long.kind == FAILED
        manager.skills["timers"].shutdown()
    run(scenario())


def test_mac_control_always_confirms_and_never_uses_a_shell():
    async def scenario():
        calls = []
        skill = MacControlSkill(apps=["Safari", "Notes"], runner=lambda argv: (calls.append(argv) or (0, "")))
        manager = SkillManager([skill], ["mac_control"])
        context = ctx()
        asked = await manager.handle("open safari", context)
        assert asked.kind == NEEDS_CONFIRMATION and calls == []            # nothing happens before the yes
        done = await manager.handle("yes", context)
        assert done.kind == OK and calls == [["open", "-a", "Safari"]]
        await manager.handle("set the volume to 40", context)
        assert (await manager.handle("go ahead", context)).kind == OK and calls[-1][-1] == "set volume output volume 40"
        await manager.handle("quit notes", context)
        assert (await manager.handle("no", context)).kind == CANCELLED and len(calls) == 2
        await manager.handle("open safari", context)
        await manager.handle("what's the weather", context)                  # any other reply drops the question
        assert (await manager.handle("yes", context)) is None and len(calls) == 2
        blocked = await manager.handle("open spotify app", context)
        assert blocked.kind == FAILED and len(calls) == 2
        assert (await manager.handle("set the volume to 400", context)).kind == FAILED
    run(scenario())


def test_mac_control_reports_real_failure():
    async def scenario():
        skill = MacControlSkill(apps=["Safari"], runner=lambda argv: (1, "not authorised"))
        manager = SkillManager([skill], ["mac_control"])
        await manager.handle("open safari", ctx())
        result = await manager.handle("yes", ctx())
        assert result.kind == FAILED and "not authorised" in result.note
        assert "Do NOT say or imply" in result_note(result) and event_kind(result) == "failed"
    run(scenario())


def test_notes_remember_recall_forget(tmp_path):
    async def scenario():
        store = NotesStore(tmp_path / "notes.json")
        manager = SkillManager([NotesSkill(store)], ["notes"])
        context = ctx()
        saved = await manager.handle("remember that Hana likes tea", context)
        assert saved.kind == OK and store.about("Hana")
        recalled = await manager.handle("what do you know about Hana?", context)
        assert recalled.kind == OK and "likes tea" in recalled.note
        mine = await manager.handle("remember that my dog is called Rex", context)
        assert mine.kind == OK and store.about("Marvin")
        asked = await manager.handle("forget that Hana likes tea", context)
        assert asked.kind == NEEDS_CONFIRMATION and store.about("Hana")  # still there until confirmed
        assert (await manager.handle("yes", context)).kind == OK and not store.about("Hana")
        assert (await manager.handle("forget that", context)) is None     # the idiom is ordinary speech
        unknown = await manager.handle("remember that I like jazz", ctx(speaker=None))
        assert unknown.kind == NEEDS_INPUT                                # never guesses who "I" is
        stranger = await manager.handle("remember that Bob plays the cello", context)   # Bob is not enrolled
        assert stranger.kind == OK and store.about("Bob") and store.about("Bob")[0]["subject"] == "Bob"
        general = await manager.handle("remember that Tomorrow is the deadline", context)
        assert general.kind == OK and store.items[-1]["subject"] == "general"
        recalled_bob = await manager.handle("what do you know about bob", context)
        assert "cello" in recalled_bob.note
        assert stat.S_IMODE(os.stat(tmp_path / "notes.json").st_mode) == 0o600
        assert "Hana" not in store.context(["Hana"]) and "Rex" in store.context(["Marvin"])
    run(scenario())


def test_note_ids_are_unique(tmp_path):
    store = NotesStore(tmp_path / "n.json")
    ids = [store.add("general", f"fact {number}", None)["id"] for number in range(30)]
    assert len(set(ids)) == 30
    store.remove(ids[5])
    assert store.add("general", "later", None)["id"] not in ids[:5] + ids[6:]


def test_notes_save_failure_is_reported(tmp_path):
    async def scenario():
        blocked = tmp_path / "file"
        blocked.write_text("x")
        manager = SkillManager([NotesSkill(NotesStore(blocked / "notes.json"))], ["notes"])   # parent is a file
        result = await manager.handle("remember that Hana likes tea", ctx())
        assert result.kind == FAILED and "nothing was stored" in result.note
    run(scenario())


def test_web_search_success_failure_and_privacy_label():
    async def scenario():
        good = WebSearchSkill(search=lambda query, news: [{"title": "Canberra", "body": "Capital of Australia", "url": "u"}])
        manager = SkillManager([good], ["web_search"])
        result = await manager.handle("search for the capital of australia", ctx())
        assert result.kind == OK and "Capital of Australia" in result.note and "sent online" in result.event
        def boom(query, news):
            raise OSError("offline")
        bad = SkillManager([WebSearchSkill(search=boom)], ["web_search"])
        failed = await bad.handle("look up the latest news on space", ctx())
        assert failed.kind == FAILED and "do not answer from memory" in failed.note
        empty = SkillManager([WebSearchSkill(search=lambda q, n: [])], ["web_search"])
        assert (await empty.handle("google something odd", ctx())).kind == FAILED
        ask = await manager.handle("search", ctx())
        assert ask.kind == NEEDS_INPUT
    run(scenario())


def test_disabled_skills_do_not_run():
    async def scenario():
        manager = SkillManager([TimerSkill()], [])
        assert await manager.handle("set a timer for 5 minutes", ctx()) is None
        manager.set_enabled(["timers"])
        result = await manager.handle("set a timer for 5 minutes", ctx())
        assert result.kind == OK
        manager.skills["timers"].shutdown()
    run(scenario())


def test_missing_arguments_are_asked_for_never_invented():
    async def scenario():
        manager = SkillManager([TimerSkill(), NotesSkill(NotesStore("/nonexistent-dir-for-test/notes.json"))],
                               ["timers", "notes"])
        result = await manager.handle("set a timer", ctx())
        assert result.kind == NEEDS_INPUT and not manager.skills["timers"]._timers   # no timer was created
        assert (await manager.handle("remember that", ctx())) is None
    run(scenario())


def test_claim_guard_and_settings():
    for claim in ["I've set a timer for five minutes.", "I opened Safari for you.", "The timer has been set."]:
        assert SKILL_CLAIM_RE.search(claim), claim
    for fine in ["I couldn't set a timer.", "Should I open Safari?", "I can set a timer if you tell me how long."]:
        assert not SKILL_CLAIM_RE.search(fine), fine
    base = {"context_tokens": 6000, "context_keep_tokens": 2500, "tts_voice": "af_heart", "tts_speed": 1.0,
            "camera_resolution": "640x480", "frame_max_side": 448, "skills_enabled": ["timers"]}
    ok, errors = user_settings.validate({"skills_enabled": ["notes", "web_search"]}, base)
    assert not errors and ok["skills_enabled"] == ["notes", "web_search"]
    assert user_settings.validate({"skills_enabled": ["rm -rf"]}, base)[0] == base


ORDINARY_SENTENCES = """Hello there|How are you today|What do you see right now|I'm Sam, remember me|This is Priya|My name is Marvin|Call me Tim|I am now called Marvin|Forget Lena|Forget about it|Forget that|Remember me|Remember my face|What's my name|Who is on my left|Why are you calling me a man|Can you recognize my voice|Describe the person next to me|What's two plus two|Tell me a joke|Write a function that reverses a string|I need to open up about something|The door is open|Close the window please|How long is the Golden Gate Bridge|What time is it|My dog barks all the time|I set my phone down|Can you start explaining quantum physics|Turn left at the corner|I'd like to search my memory for that|The volume of a sphere is what|She is going to remind me about it|Names and Gertrude|What do you know|I started a new job|Make me a sandwich|Stop talking|Cancel that|No|Yes|I want to look up to my father|He did a google search yesterday|What do you know about my name|What do you know about your face|I read the news about the election|The latest news bores me""".split("|")


def test_ordinary_conversation_triggers_no_skill(tmp_path):
    from interactive_llm.skills import build_skills
    from interactive_llm.skills.catalog import SKILL_IDS

    async def scenario():
        manager = SkillManager(build_skills(NotesStore(tmp_path / "n.json")), SKILL_IDS)
        hits = []
        for sentence in ORDINARY_SENTENCES:
            result = await manager.handle(sentence, ctx())
            if result:
                hits.append((sentence, result.skill))
        manager.skills["timers"].shutdown()
        assert hits == []
        # commands in polite or direct form still work
        for command in ["Search for the capital of France", "Can you look up the weather in Paris", "please google rust", "What's the latest news on space"]:
            assert (await manager.handle(command, ctx())).skill == "web_search", command
        assert (await manager.handle("Remind me to call mum in 10 minutes", ctx())).skill == "timers"
        manager.skills["timers"].shutdown()
    run(scenario())


def test_stock_price_runs_automatically_and_reports_progress():
    """'check amzn stock price' must run a real lookup at once, announce it, and quote only what the lookup returned."""
    quote = {"symbol": "AMZN", "name": "Amazon.com, Inc.", "price": 267.74, "currency": "USD", "exchange": "NasdaqGS",
             "time": 1795035600, "change_pct": 3.8}
    shown: list[str] = []

    async def progress(text):
        shown.append(text)

    skills = SkillManager([WebSearchSkill(stock=lambda company: quote)], ["web_search"])
    context = SkillContext(progress=progress)
    result = asyncio.run(skills.handle("check amzn stock price", context))
    assert result.kind == "ok" and "267.74" in result.note and "AMZN" in result.event
    assert shown and "stock price" in shown[0]


def test_stock_lookup_failure_and_disabled_search_are_reported():
    """No network or search switched off: say so, never let the model invent a price."""
    def offline(company):
        raise OSError("offline")

    failed = asyncio.run(SkillManager([WebSearchSkill(stock=offline)], ["web_search"]).handle("what's the Tesla stock price", SkillContext()))
    assert failed.kind == "failed" and "memory" in failed.note
    off = asyncio.run(SkillManager([WebSearchSkill()], []).handle("check amzn stock price", SkillContext()))
    assert off.kind == "failed" and "Settings" in off.note
    assert asyncio.run(SkillManager([WebSearchSkill()], ["web_search"]).handle("how's the stock market feeling", SkillContext())) is None


def test_plain_live_questions_search_without_saying_search():
    """Weather, results, news and prices need current facts, so they search; ordinary talk about them does not."""
    asked: list[str] = []

    def fake(query, news):
        asked.append(query)
        return [{"title": "Result", "body": "Sunny, 18 C", "url": "x"}]

    skills = SkillManager([WebSearchSkill(search=fake)], ["web_search"])
    for sentence in ["What's the weather in London today", "Who won the F1 race last night", "What's the latest news on AI",
                     "When is the new iPhone release date", "Is it going to rain tomorrow, what's the forecast",
                     "latest business news", "Tech headlines", "weather in Paris tomorrow", "Please give me today's news"]:
        assert asyncio.run(skills.handle(sentence, SkillContext())).kind == "ok", sentence
    assert asked[0].lower().startswith("what's the weather")
    for sentence in ["The weather is nice", "I like the news", "My score was bad", "He won yesterday", "That is the latest one",
                     "The latest news bores me", "Weather is weather", "Good news everyone is here"]:
        assert asyncio.run(skills.handle(sentence, SkillContext())) is None, sentence


def test_listing_a_folder_is_read_only_and_honest():
    """'What files are on my desktop' runs without a yes, reports real names, and reports read failures."""
    skill = MacControlSkill(lister=lambda folder: ["notes.txt", "Photos/"])
    skills = SkillManager([skill], ["mac_control"])
    for sentence in ["What files are in my desktop?", "What files are on my desktop", "list my downloads", "show me what's in my documents"]:
        result = asyncio.run(skills.handle(sentence, SkillContext()))
        assert result.kind == "ok" and "notes.txt" in result.note, sentence

    def denied(folder):
        raise PermissionError("no access")

    failed = asyncio.run(SkillManager([MacControlSkill(lister=denied)], ["mac_control"]).handle("what's on my desktop", SkillContext()))
    assert failed.kind == "failed" and "guess" in failed.note


def test_listing_shows_every_name_and_can_be_repeated():
    """The GUI gets every file name; the model is told to give only the count; "show it here" repeats the list."""
    skill = MacControlSkill(lister=lambda folder: [f"file{n}.txt" for n in range(25)])
    skills = SkillManager([skill], ["mac_control"])
    assert asyncio.run(skills.handle("show it here", SkillContext())) is None      # nothing listed yet
    first = asyncio.run(skills.handle("list the files on my desktop", SkillContext()))
    assert len(first.detail) == 25 and "do NOT describe" in first.note
    again = asyncio.run(skills.handle("show it here", SkillContext()))
    assert again.kind == "ok" and len(again.detail) == 25


def test_failed_stock_lookup_accepts_a_bare_ticker_reply():
    """After 'No stock found', replying just 'VTI' must run the lookup instead of falling to the language model."""
    def stock(company):
        if company.upper() != "VTI":
            raise LookupError(company)
        return {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "price": 300.0, "currency": "USD",
                "exchange": "NYSEArca", "time": None, "change_pct": None}

    skills = SkillManager([WebSearchSkill(stock=stock)], ["web_search"])
    first = asyncio.run(skills.handle("bulatest stock price for vti", SkillContext()))
    assert first.kind == "needs_input" and first.followup
    second = asyncio.run(skills.handle("VTI", SkillContext()))
    assert second.kind == "ok" and "300.00" in second.note


def test_claiming_a_web_lookup_without_one_is_caught():
    """The model saying "according to a live web lookup" when no tool ran must be flagged."""
    assert SKILL_CLAIM_RE.search("VOO is trading at $527.62, according to a live web lookup.")
    assert SKILL_CLAIM_RE.search("From a web search, it is sunny.")
    assert not SKILL_CLAIM_RE.search("The capital of France is Paris.")


def test_stock_follow_ups_use_the_previous_lookup():
    """"and Tesla?" and "check again" refer to the recent stock conversation; unrelated sentences are untouched."""
    asked: list[str] = []

    def stock(company):
        asked.append(company)
        return {"symbol": company.upper(), "name": company, "price": 1.0, "currency": "USD", "exchange": "X", "time": None,
                "change_pct": None}

    skills = SkillManager([WebSearchSkill(stock=stock)], ["web_search"])
    assert asyncio.run(skills.handle("and Tesla?", SkillContext())) is None            # no stock talk yet
    asyncio.run(skills.handle("check vti stock price", SkillContext()))
    assert asyncio.run(skills.handle("and Tesla?", SkillContext())).kind == "ok"
    assert asyncio.run(skills.handle("check again", SkillContext())).kind == "ok"
    assert asked == ["vti", "Tesla", "TESLA"]


def test_llm_tool_calls_run_real_skills_and_reject_bad_arguments():
    """Model tool calls reach the real skill code; unknown tools, bad folders and disabled skills fail honestly."""
    from interactive_llm.skills.llm_tools import call_tool, tool_schemas
    skills = {"web_search": WebSearchSkill(stock=lambda c: {"symbol": "VTI", "name": "VTI", "price": 1.0, "currency": "USD",
                                                          "exchange": "X", "time": None, "change_pct": None},
                                           search=lambda q, n: [{"title": "t", "body": "b", "url": "u"}]),
              "mac_control": MacControlSkill(lister=lambda f: ["a.txt"])}
    enabled = {"web_search", "mac_control"}
    run = lambda name, args, on=enabled: asyncio.run(call_tool(skills, on, name, args, SkillContext()))
    assert run("stock_quote", {"company": "VTI"}).kind == "ok"
    assert run("web_search", {"query": "business news", "news": True}).kind == "ok"
    assert run("list_folder", {"folder": "desktop"}).detail == ["a.txt"]
    assert run("list_folder", {"folder": "/etc"}).kind == "failed"
    assert run("web_search", {"query": ""}).kind == "failed"
    assert run("rm_rf", {}).kind == "failed"
    assert run("web_search", {"query": "x news"}, on={"mac_control"}).kind == "failed"
    assert [t["function"]["name"] for t in tool_schemas({"web_search"})] == ["web_search", "stock_quote"]


def test_model_endpoint_settings_are_validated():
    """Endpoint addresses must be http(s) URLs and model names plain tokens; one bad value applies nothing."""
    base = {"context_tokens": 6000, "context_keep_tokens": 2500, "ollama_url": "http://localhost:11434/v1",
            "ollama_model": "a", "lmstudio_url": "http://localhost:1234/v1", "lmstudio_model": "b"}
    ok, errs = user_settings.validate({"ollama_url": " http://192.0.2.5:11434/v1/ ", "ollama_model": "qwen3:8b"}, base)
    assert not errs and ok["ollama_url"] == "http://192.0.2.5:11434/v1" and ok["ollama_model"] == "qwen3:8b"
    for bad in ({"ollama_url": "ftp://x"}, {"ollama_url": "localhost:11434"}, {"lmstudio_url": "http://a b"},
                {"ollama_model": ""}, {"lmstudio_model": "two words"}):
        accepted, problems = user_settings.validate(bad, base)
        assert problems and accepted == base, bad


def test_vision_mode_setting_is_validated():
    """Camera use must be auto, always or never."""
    base = {"context_tokens": 6000, "context_keep_tokens": 2500, "vision_mode": "auto"}
    assert user_settings.validate({"vision_mode": "always"}, base)[0]["vision_mode"] == "always"
    accepted, problems = user_settings.validate({"vision_mode": "sometimes"}, base)
    assert problems and accepted == base


def test_company_follow_up_ignores_refusals_and_chatter():
    """After a failed stock lookup, "No." or "in your side" must not become a company name; a real ticker still works."""
    def stock(company):
        if company.upper() != "VTI":
            raise LookupError(company)
        return {"symbol": "VTI", "name": "VTI", "price": 1.0, "currency": "USD", "exchange": "X", "time": None, "change_pct": None}

    for chatter in ["No.", "in your side", "Stop, stop, stop.", "Hello.", "What?"]:
        skills = SkillManager([WebSearchSkill(stock=stock)], ["web_search"])
        asyncio.run(skills.handle("check foo stock price", SkillContext()))
        assert asyncio.run(skills.handle(chatter, SkillContext())) is None, chatter
    skills = SkillManager([WebSearchSkill(stock=stock)], ["web_search"])
    asyncio.run(skills.handle("check foo stock price", SkillContext()))
    assert asyncio.run(skills.handle("VTI", SkillContext())).kind == "ok"


def test_context_budget_allows_cloud_sized_windows():
    """A 1M-token cloud model must be configurable; the kept-history budget still obeys the 80% rule."""
    base = {"context_tokens": 6000, "context_keep_tokens": 2500}
    ok, errs = user_settings.validate({"context_tokens": 500_000}, base)
    assert not errs and ok["context_tokens"] == 500_000
    assert user_settings.validate({"context_tokens": 1_000_001}, base)[1]
    assert user_settings.validate({"context_tokens": 500_000, "context_keep_tokens": 450_000}, base)[1]


def test_voice_engine_setting_and_pcm_conversion():
    """Engines are validated, an engine switch falls back to that engine's first voice, and audio is resampled to 24 kHz."""
    import numpy as np
    from interactive_llm.pipeline.tts_engines import to_pcm16
    base = {"context_tokens": 6000, "context_keep_tokens": 2500, "tts_engine": "kokoro", "tts_voice": "af_bella"}
    ok, errs = user_settings.validate({"tts_engine": "chatterbox"}, base)
    assert not errs and ok["tts_engine"] == "chatterbox" and ok["tts_voice"] == "default"
    back, errs = user_settings.validate({"tts_engine": "kokoro", "tts_voice": "am_adam"}, ok)
    assert not errs and back["tts_voice"] == "am_adam"
    assert user_settings.validate({"tts_engine": "nope"}, base)[1]
    assert user_settings.validate({"tts_voice": "default"}, base)[1]            # not a Kokoro voice
    tone = np.sin(np.linspace(0, 440, 12000)).astype(np.float32)               # 0.5 s at 24 kHz
    assert len(to_pcm16(tone, 24000)) == 12000 * 2
    assert abs(len(to_pcm16(tone[:6000], 12000)) - 12000 * 2) <= 8              # 0.5 s at 12 kHz resampled up


def test_each_voice_engine_calls_its_model_with_valid_arguments():
    """Regression: a duplicated keyword argument once made every synthesis (and the start-up warm-up) fail."""
    import types
    import numpy as np
    from interactive_llm.pipeline.tts_engines import ENGINE_CLASSES

    class FakeModel:
        sample_rate = 24000

        def generate(self, text, **kwargs):
            self.kwargs = kwargs
            yield types.SimpleNamespace(audio=np.zeros(2400, np.float32))

    for engine_id, engine_class in ENGINE_CLASSES.items():
        engine = engine_class("unused/repo")
        engine.model = FakeModel()
        audio, rate = engine.synth("Hello there.", "af_heart" if engine_id == "kokoro" else "default", 1.0)
        assert len(audio) == 2400 and rate == 24000, engine_id
    kokoro = ENGINE_CLASSES["kokoro"]("x")
    kokoro.model = FakeModel()
    kokoro.synth("Hi.", "bm_george", 1.1)
    assert kokoro.model.kwargs["lang_code"] == "b" and kokoro.model.kwargs["speed"] == 1.1


def test_search_ads_are_filtered_and_pages_are_read():
    """Sponsored rows are dropped; page excerpts reach the model; the text is labelled untrusted; sources are listed."""
    from interactive_llm.skills.page_reader import best_passages, is_public_http_url, page_blocks
    from interactive_llm.skills.web_search import is_ad
    assert is_ad({"title": "AdLive Weather Updates", "url": "https://www.bing.com/aclick?ld=x"})
    assert not is_ad({"title": "Add a widget to your site", "url": "https://example.com/a"})
    assert not is_ad({"title": "London weather", "url": "https://weather.com/london"})
    html = ("<html><body><nav>Home Menu Login Subscribe now to our newsletter today</nav><script>var x=1;</script>"
            "<article><h1>London weather</h1><p>Today in London it is 14 degrees Celsius with light rain this afternoon.</p>"
            "<p>Cookie settings and privacy policy are shown at the bottom of every page here.</p></article></body></html>")
    picked = best_passages(page_blocks(html), "weather in London today")
    assert "14 degrees" in picked and "Subscribe" not in picked
    assert not is_public_http_url("http://127.0.0.1:8000/x") and not is_public_http_url("http://192.0.2.5/")
    assert not is_public_http_url("file:///etc/passwd") and not is_public_http_url("http://localhost/")

    rows = [{"title": "London weather", "body": "Weather.", "url": "https://weather.example/london"}]
    skill = WebSearchSkill(search=lambda q, n: rows, reader=lambda found, q: ["It is 14 degrees with light rain."])
    result = asyncio.run(skill.run({"query": "weather in London", "news": False}, SkillContext()))
    assert "14 degrees" in result.note and "untrusted" in result.note and result.detail == ["London weather (weather.example)"]


def test_price_question_with_a_ticker_uses_the_quote_service():
    """"What is the price of VTI?" is a quote request, not a web search; acronyms in other questions are not tickers."""
    seen = []
    quote = {"symbol": "VTI", "name": "VTI", "price": 1.0, "currency": "USD", "exchange": "X", "time": None, "change_pct": None}
    skills = SkillManager([WebSearchSkill(stock=lambda company: seen.append(company) or quote)], ["web_search"])
    for sentence in ["What is the price of VTI?", "how much is NVDA trading at", "What's the VOO price"]:
        assert asyncio.run(skills.handle(sentence, SkillContext())).kind == "ok", sentence
    assert seen == ["VTI", "NVDA", "VOO"]
    skills = SkillManager([WebSearchSkill(stock=lambda company: seen.append(company) or quote, search=lambda q, n: [])], ["web_search"])
    assert asyncio.run(skills.handle("What is the latest news on NASA", SkillContext())).kind != "ok" or seen[-1] != "NASA"


def test_everyday_sentences_do_not_count_as_visual_follow_ups():
    """Ordinary talk after a visual question must not attach the camera (and so must not trigger scene narration)."""
    from interactive_llm.server import VISION_FOLLOWUP_RE
    for text in ["You can take it for later.", "Mico, come here and talk to it.", "It's still 10 miles away.", "She was very good.",
                 "Okay, sure.", "That sounds good", "Hello there", "Put it over there"]:
        assert not VISION_FOLLOWUP_RE.search(text), text
    for text in ["What about now?", "And now?", "How about now", "Check again.", "Come closer", "What about the other one?"]:
        assert VISION_FOLLOWUP_RE.search(text), text


def test_tester_found_regressions():
    """Regressions from the QA run: unknown app commands, past-tense timer questions, thin search results, disabled wording."""
    mac = SkillManager([MacControlSkill(apps=["Safari"])], ["mac_control"])
    for sentence, expect in [("Open Spotify", "failed"), ("Please launch Slack", "failed"), ("Open Safari", "needs_confirmation")]:
        assert asyncio.run(mac.handle(sentence, SkillContext())).kind == expect, sentence
    for sentence in ["Open the door", "I need to open up about something", "Close the window please", "She will open Safari later"]:
        assert asyncio.run(mac.handle(sentence, SkillContext())) is None, sentence
    async def timers_scenario():
        timers = SkillManager([TimerSkill()], ["timers"])
        assert await timers.handle("Did you just set a timer for me?", ctx()) is None
        assert (await timers.handle("Set a timer for 5 seconds", ctx())).kind == OK
        timers.skills["timers"].shutdown()
    asyncio.run(timers_scenario())
    off = asyncio.run(SkillManager([WebSearchSkill()], []).handle("check amzn stock price", SkillContext()))
    assert off.kind == "failed" and off.event == "Web search is switched off: turn it on in Settings > Skills"


def test_search_queries_are_cleaned_up():
    """Lead-ins are stripped and topic-less news requests become a useful query instead of the whole sentence."""
    from interactive_llm.skills.web_search import clean_query
    assert clean_query("I said give me the latest news today") == "top news headlines today"
    assert clean_query("Please tell me the latest news") == "top news headlines today"
    assert clean_query("Can you give me the latest news on Tesla") == "the latest news on Tesla"
    assert clean_query("What's the weather in London today") == "What's the weather in London today"
    assert clean_query("I want business news") == "business news"
    asked = []
    skill = WebSearchSkill(search=lambda q, n: asked.append((q, n)) or [{"title": "t", "body": "b", "url": "u"}], reader=None)
    asyncio.run(skill.run({"query": "I said give me the latest news today", "news": False}, SkillContext()))
    assert asked == [("top news headlines today", True)]


def test_news_feed_parsing_sections_and_ages():
    """Headlines are parsed newest first with outlet and age; broad topics use Google's section feeds."""
    from datetime import datetime, timezone
    from interactive_llm.skills.news_feed import age_text, feed_url, parse_feed
    xml = ("<rss><channel>"
           "<item><title>Old story - Wire</title><link>https://n/1</link><pubDate>Mon, 16 Nov 2026 06:00:00 GMT</pubDate></item>"
           "<item><title>Fresh story - BBC</title><link>https://n/2</link><pubDate>Thu, 19 Nov 2026 05:00:00 GMT</pubDate>"
           "<source url='https://bbc'>BBC</source></item></channel></rss>")
    now = datetime(2026, 11, 19, 8, 0, tzinfo=timezone.utc)
    rows = parse_feed(xml, now)
    assert [row["title"] for row in rows] == ["Fresh story", "Old story"] and rows[0]["body"] == "BBC, 3 hours ago"
    assert rows[1]["body"].endswith("3 days ago") and age_text(now, now) == "1 minute ago"
    assert feed_url("top news headlines today").endswith("?hl=en-US&gl=US&ceid=US:en") and "section/topic" not in feed_url("top news")
    assert "section/topic/BUSINESS" in feed_url("business news") and "section/topic/TECHNOLOGY" in feed_url("latest tech news")
    assert "q=Tesla" in feed_url("latest news on Tesla") and "when%3A3d" in feed_url("latest news on Tesla")
    skill = WebSearchSkill(search=lambda q, n: parse_feed(xml, now), reader=lambda rows, q: 1 / 0)
    result = asyncio.run(skill.run({"query": "latest news", "news": True}, SkillContext()))
    assert result.kind == OK and "3 hours ago" in result.note and result.detail[0].startswith("Fresh story")


def test_repeating_looked_up_figures_is_not_a_false_claim():
    """Restating a price an earlier lookup returned (even mid-joke) must not be flagged; an invented price still is."""
    from interactive_llm.server import figures, repeats_backed_data
    backed = figures("VOO last traded at 671.31 USD, -0.3% today (live quote from the web).")
    assert repeats_backed_data("VOO came in at $671.31, down 0.3% today, per a live web lookup. Why did the index fund...", backed)
    assert not repeats_backed_data("VOO is at $482.15, per a live web lookup.", backed)            # invented figure
    assert not repeats_backed_data("Per a live web lookup, it is trending up.", backed)            # no figures to vouch for
    assert not repeats_backed_data("I set a timer for 671.31 seconds.", backed)                    # action claims never pass
    assert figures("Over 1,234.5 shares") == {"1234.5"}


def test_company_text_is_cleaned_before_the_stock_lookup():
    """"Amazon's stock price" must search for Amazon: stray letters (the 's') once matched the ticker S, SentinelOne."""
    from interactive_llm.skills.web_search import clean_company
    assert clean_company("Amazon's stock price") == "Amazon"
    assert clean_company("Amazon’s share price right now") == "Amazon"
    assert clean_company("the S&P 500 index") == "S&P 500 index"
    assert clean_company("VTI") == "VTI"
    assert clean_company("stock price") == "stock price"      # nothing left: keep what the user said


def test_stock_ticker_fallback_never_uses_word_fragments(monkeypatch):
    """When Yahoo finds nothing, only whole 2-5 letter words may be tried as tickers, never pieces of a longer word."""
    import io
    import json
    import pytest
    from interactive_llm.skills import web_search
    asked = []

    def fake_urlopen(request, timeout=0):
        url = request.full_url
        asked.append(url)
        if "/search" in url:
            return io.BytesIO(json.dumps({"quotes": []}).encode())
        raise AssertionError(url)

    monkeypatch.setattr(web_search.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LookupError):
        web_search.lookup_stock("Amazonian's wibble")
    queried = [url.rsplit("q=", 1)[1] for url in asked]
    assert "s" not in [q.lower() for q in queried] and "Amazo" not in queried and "n" not in queried


# ---------------------------------------------------------------- page reader: the second reader must not follow redirects blindly
class _FakeResponse:
    def __init__(self, status, headers=None, text=""):
        self.status_code, self.headers, self.text_markdown = status, headers or {}, text


class _FakeClient:
    """Serves canned answers per URL and records every URL requested and whether redirects were left to us."""

    def __init__(self, pages):
        self.pages, self.requested, self.follow = pages, [], []

    def get(self, url, follow_redirects=None):
        self.requested.append(url)
        self.follow.append(follow_redirects)
        return self.pages[url]


def _public(monkeypatch, *urls):
    from interactive_llm.skills import page_reader
    monkeypatch.setattr(page_reader, "is_public_http_url", lambda u: u in urls)


def test_second_reader_refuses_a_redirect_to_loopback(monkeypatch):
    """Review finding: a public page answering 302 -> http://127.0.0.1:... made the fallback fetch it from this machine."""
    from interactive_llm.skills.page_reader import _fetch_markdown
    _public(monkeypatch, "https://pub.example/x")
    client = _FakeClient({"https://pub.example/x": _FakeResponse(302, {"location": "http://127.0.0.1:8000/api/people"}),
                          "http://127.0.0.1:8000/api/people": _FakeResponse(200, text="SECRET " * 20)})
    assert _fetch_markdown("https://pub.example/x", client) == ""
    assert client.requested == ["https://pub.example/x"]            # the internal URL was never requested
    assert client.follow == [False]                                  # the library was told not to follow redirects itself


def test_second_reader_follows_public_redirects_and_resolves_relative_ones(monkeypatch):
    from interactive_llm.skills.page_reader import _fetch_markdown
    _public(monkeypatch, "https://pub.example/a", "https://pub.example/b")
    client = _FakeClient({"https://pub.example/a": _FakeResponse(301, {"location": "/b"}),
                          "https://pub.example/b": _FakeResponse(200, text="hello page")})
    assert _fetch_markdown("https://pub.example/a", client) == "hello page"
    assert client.requested == ["https://pub.example/a", "https://pub.example/b"]


def test_second_reader_gives_up_on_redirect_loops_and_errors(monkeypatch):
    from interactive_llm.skills.page_reader import MAX_REDIRECTS, _fetch_markdown, read_page_with_ddgs
    _public(monkeypatch, "https://pub.example/loop", "https://pub.example/err", "https://pub.example/nolocation")
    loop = _FakeClient({"https://pub.example/loop": _FakeResponse(302, {"location": "/loop"})})
    assert _fetch_markdown("https://pub.example/loop", loop) == "" and len(loop.requested) == MAX_REDIRECTS + 1
    assert _fetch_markdown("https://pub.example/err", _FakeClient({"https://pub.example/err": _FakeResponse(500, text="x")})) == ""
    assert _fetch_markdown("https://pub.example/nolocation",
                           _FakeClient({"https://pub.example/nolocation": _FakeResponse(302)})) == ""
    assert read_page_with_ddgs("https://pub.example/x", "anything", client=object()) == ""   # a broken client never raises


def test_second_reader_refuses_non_public_start_urls(monkeypatch):
    from interactive_llm.skills.page_reader import _fetch_markdown
    _public(monkeypatch)                                              # nothing is public
    client = _FakeClient({})
    assert _fetch_markdown("http://192.0.2.9/", client) == "" and client.requested == []


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000\\@attacker.example/../api/settings",    # backslash: Python sees attacker.example, fetchers see loopback
    "http://attacker.example\\@127.0.0.1:8000/x",
    "http://user:pass@example.com/", "http://example.com@127.0.0.1/",   # credentials / userinfo
    "http://exa mple.com/", "http://example.com/\n", "http://example.com/\tx", "http://example.com/\x00",
    "http://example.com:notaport/", "ftp://example.com/", "file:///etc/hosts", "//example.com/x", "",
])
def test_ambiguous_or_non_http_urls_are_refused_before_any_lookup(monkeypatch, url):
    """Security review: a parser mismatch let a redirect to a backslash URL pass the check yet connect to loopback."""
    from interactive_llm.skills import page_reader
    monkeypatch.setattr(page_reader.socket, "getaddrinfo", lambda *a, **k: pytest.fail("looked up an ambiguous URL"))
    assert page_reader.is_public_http_url(url) is False


def test_plain_public_urls_still_pass(monkeypatch):
    from interactive_llm.skills import page_reader
    monkeypatch.setattr(page_reader.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    assert page_reader.is_public_http_url("https://example.com/a/b?x=1#frag") is True
    monkeypatch.setattr(page_reader.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 443))])
    assert page_reader.is_public_http_url("https://example.com/") is False


def test_second_reader_refuses_a_backslash_redirect(monkeypatch):
    from interactive_llm.skills import page_reader
    monkeypatch.setattr(page_reader.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    client = _FakeClient({"https://pub.example/x": _FakeResponse(302, {"location": "http://127.0.0.1:8000\\@pub.example/../api/settings"})})
    assert page_reader._fetch_markdown("https://pub.example/x", client) == ""
    assert client.requested == ["https://pub.example/x"]
