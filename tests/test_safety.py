"""Regression tests for the pre-release code review: confirmations, local-server access, stores, auto-learning.

Names are invented (Sam, Hana, Tess); LAN-looking addresses use the documentation range 192.0.2.x.
"""
import asyncio

import pytest

from interactive_llm.pipeline.people import Speaker, is_no, is_yes, same_speaker
from interactive_llm.skills.base import CANCELLED, NEEDS_CONFIRMATION, OK, SkillContext
from interactive_llm.skills.mac_control import MacControlSkill
from interactive_llm.skills.manager import SkillManager
from test_people import mk_db, run


# ---------------------------------------------------------------- yes / no
@pytest.mark.parametrize("text", ["yes", "Yes.", "yeah", "Yes, please", "yeah go ahead", "do it", "Sure.", "okay, do it",
                                  "yes yes", "absolutely"])
def test_plain_yes_forms_confirm(text):
    assert is_yes(text) and not is_no(text)


@pytest.mark.parametrize("text", ["Are you sure?", "I'm not sure", "yeah no, don't", "wait", "hold on a second", "stop",
                                  "no", "cancel that", "I am unsure about that", "yes that sounds good",
                                  "yes I went to the shop", "sure about what?", "Hello there"])
def test_doubt_refusal_or_chatter_never_confirms(text):
    assert not is_yes(text)


@pytest.mark.parametrize("text", ["no", "nope", "cancel", "Are you sure?", "I'm not sure", "wait", "don't", "hold on"])
def test_refusal_and_doubt_count_as_no(text):
    assert is_no(text)


def test_same_speaker_rules():
    assert same_speaker("Sam", "sam") and same_speaker(None, "Hana") and same_speaker("Sam", None)
    assert not same_speaker("Sam", "Hana")


def _mac(calls):
    return SkillManager([MacControlSkill(apps=["Safari", "Notes"], runner=lambda argv: (calls.append(argv) or (0, "")))],
                        ["mac_control"])


@pytest.mark.parametrize("reply", ["Are you sure?", "I'm not sure", "yeah no, don't", "wait", "stop"])
def test_doubtful_reply_never_runs_a_mac_action(reply):
    async def scenario():
        calls = []
        manager, context = _mac(calls), SkillContext(speaker="Sam")
        assert (await manager.handle("quit notes", context)).kind == NEEDS_CONFIRMATION
        result = await manager.handle(reply, context)
        assert result is not None and result.kind == CANCELLED and calls == []
        assert (await manager.handle("yes", context)) is None and calls == []      # the question is gone
    asyncio.run(scenario())


def test_only_the_person_who_was_asked_can_confirm_an_action():
    async def scenario():
        calls = []
        manager = _mac(calls)
        await manager.handle("open safari", SkillContext(speaker="Sam"))
        assert (await manager.handle("yes", SkillContext(speaker="Hana"))) is None and calls == []   # a guest said yes
        done = await manager.handle("yes", SkillContext(speaker="Sam"))                              # still open for Sam
        assert done.kind == OK and calls == [["open", "-a", "Safari"]]
        await manager.handle("open safari", SkillContext(speaker=None))                              # typed: no speaker, allowed
        assert (await manager.handle("yes", SkillContext(speaker="Hana"))).kind == OK
    asyncio.run(scenario())


# ---------------------------------------------------------------- forget everyone
@pytest.mark.parametrize("reply", ["I'm not sure", "Are you sure?", "yeah no, don't", "yes that sounds good", "wait"])
def test_doubtful_reply_never_wipes_everyone(tmp_path, reply):
    m = mk_db(tmp_path)
    sam = Speaker(name="Sam")
    run(m.apply_utterance("Forget everyone.", sam, None))
    run(m.apply_utterance(reply, sam, None))
    assert len(m.db.names()) == 3


def test_wipe_is_cancelled_by_a_no_and_cannot_be_confirmed_afterwards(tmp_path):
    m = mk_db(tmp_path)
    sam = Speaker(name="Sam")
    run(m.apply_utterance("Forget everyone.", sam, None))
    notes = run(m.apply_utterance("no, don't", sam, None))
    assert "cancelled" in notes[0] and len(m.db.names()) == 3
    assert not m.needs_processing("yes")
    run(m.apply_utterance("yes", sam, None))
    assert len(m.db.names()) == 3


def test_wipe_can_only_be_confirmed_by_the_person_who_asked(tmp_path):
    m = mk_db(tmp_path)
    run(m.apply_utterance("Forget everyone.", Speaker(name="Sam"), None))
    run(m.apply_utterance("yes", Speaker(name="Hana"), None))
    assert len(m.db.names()) == 3
    notes = run(m.apply_utterance("yes", Speaker(name="Sam"), None))
    assert "all 3 people" in notes[0] and m.db.names() == []


# ---------------------------------------------------------------- stores: corrupt files, atomic writes, owner-only
import json  # noqa: E402
import os  # noqa: E402
import stat  # noqa: E402

import numpy as np  # noqa: E402

from interactive_llm import settings as user_settings  # noqa: E402
from interactive_llm.pipeline.people import PeopleDB  # noqa: E402


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.mark.parametrize("content", ['{"Sam": {"face": [[0.1', "not json at all", "[1, 2, 3]", '{"Sam": "oops"}'])
def test_unreadable_people_file_is_kept_aside_not_overwritten(tmp_path, content):
    path = tmp_path / "people.json"
    path.write_text(content)
    db = PeopleDB(path)
    assert db.names() == []
    asides = list(tmp_path.glob("people.json.corrupt-*"))
    assert len(asides) == 1 and asides[0].read_text() == content          # the bytes are still there
    db.add("Hana", "face", np.ones(8, np.float32))                         # saving now cannot destroy them
    assert asides[0].read_text() == content and PeopleDB(path).names() == ["Hana"]


def test_people_file_is_written_atomically_with_owner_only_permissions(tmp_path):
    path = tmp_path / "people.json"
    db = PeopleDB(path)
    db.add("Sam", "voice", np.ones(8, np.float32))
    assert _mode(path) == 0o600 and PeopleDB(path).names() == ["Sam"]
    assert [p.name for p in tmp_path.iterdir()] == ["people.json"]          # no temp file left behind


def test_failed_write_leaves_the_old_people_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "people.json"
    db = PeopleDB(path)
    db.add("Sam", "voice", np.ones(8, np.float32))
    before = path.read_text()
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        db.add("Hana", "voice", np.ones(8, np.float32))
    monkeypatch.undo()
    assert path.read_text() == before and [p.name for p in tmp_path.iterdir()] == ["people.json"]


def test_settings_are_saved_atomically_and_privately(tmp_path, monkeypatch):
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "s.json"))
    user_settings.save({"ollama_url": "http://192.0.2.5:11434/v1"})
    assert _mode(tmp_path / "s.json") == 0o600
    assert json.loads((tmp_path / "s.json").read_text())["ollama_url"] == "http://192.0.2.5:11434/v1"
    assert [p.name for p in tmp_path.iterdir()] == ["s.json"]


@pytest.mark.parametrize("content", ['{"tts_speed": 1.', "[]", "nonsense"])
def test_unreadable_settings_file_is_kept_aside(tmp_path, monkeypatch, content):
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "s.json"))
    (tmp_path / "s.json").write_text(content)
    defaults = {"tts_speed": 1.0}
    assert user_settings.load(defaults) == defaults
    asides = list(tmp_path.glob("s.json.corrupt-*"))
    assert len(asides) == 1 and asides[0].read_text() == content


def test_missing_settings_file_just_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "none.json"))
    assert user_settings.load({"a": 1}) == {"a": 1} and list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- local server: only our own page may talk to it
from interactive_llm.server import reject_reason  # noqa: E402


def _req(host="127.0.0.1:8000", origin=None, ctype=None):
    headers = {"host": host}
    if origin is not None:
        headers["origin"] = origin
    if ctype is not None:
        headers["content-type"] = ctype
    return headers


def test_own_page_is_allowed():
    assert reject_reason(_req(origin="http://127.0.0.1:8000"), "GET", "127.0.0.1") is None
    assert reject_reason(_req("localhost:8000", "http://localhost:8000", "application/json"), "POST", "127.0.0.1") is None
    assert reject_reason(_req("[::1]:8000", "http://[::1]:8000"), "GET", "127.0.0.1") is None
    assert reject_reason(_req(ctype="application/json; charset=utf-8"), "POST", "127.0.0.1") is None


def test_scripts_without_an_origin_header_are_allowed():
    assert reject_reason(_req(), "GET", "127.0.0.1") is None


@pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost:9999", "null", "http://127.0.0.1.evil.example:8000",
                                    "file://"])
def test_cross_site_pages_are_refused(origin):
    assert reject_reason(_req(origin=origin), "GET", "127.0.0.1")
    assert reject_reason(_req(origin=origin, ctype="application/json"), "POST", "127.0.0.1")


def test_dns_rebinding_is_refused():
    # the hostile site's own name resolves to 127.0.0.1, so the browser sends Host: evil.example and Origin to match
    assert reject_reason(_req("evil.example:8000", "http://evil.example:8000"), "GET", "127.0.0.1")
    assert reject_reason(_req("evil.example"), "GET", "127.0.0.1")
    assert reject_reason({}, "GET", "127.0.0.1")


@pytest.mark.parametrize("ctype", [None, "text/plain", "application/x-www-form-urlencoded", "multipart/form-data"])
def test_simple_cross_site_style_posts_must_be_json(ctype):
    assert reject_reason(_req(origin="http://127.0.0.1:8000", ctype=ctype), "POST", "127.0.0.1")


def test_wildcard_bind_skips_the_host_allowlist_but_keeps_same_site_rule():
    assert reject_reason(_req("192.0.2.5:8000", "http://192.0.2.5:8000"), "GET", "0.0.0.0") is None
    assert reject_reason(_req("192.0.2.5:8000", "http://evil.example"), "GET", "0.0.0.0")


def test_real_routes_refuse_foreign_pages(monkeypatch):
    """Through the actual app: no lifespan (so no models load); only refused requests, so no settings are written."""
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from interactive_llm import server
    monkeypatch.setattr(server.wcfg, "host", "127.0.0.1")
    client = TestClient(server.app, base_url="http://127.0.0.1:8000")
    assert client.get("/api/settings").status_code == 200                                       # our own page works
    assert client.get("/api/settings", headers={"origin": "https://evil.example"}).status_code == 403
    assert TestClient(server.app, base_url="http://evil.example").get("/api/settings").status_code == 403   # rebinding
    for headers in ({"content-type": "text/plain"}, {"content-type": "application/json", "origin": "https://evil.example"}):
        for route in ("/api/settings", "/api/llm/models", "/api/tts/preview"):
            assert client.post(route, content='{"ollama_url": "http://192.0.2.5/v1"}', headers=headers).status_code == 403
    before = len(server.SESSIONS)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "https://evil.example"}):
            pass
    with pytest.raises(WebSocketDisconnect):
        with TestClient(server.app, base_url="http://evil.example").websocket_connect("/ws"):
            pass
    assert len(server.SESSIONS) == before                                                        # no session was ever created


# ---------------------------------------------------------------- spelling and forget parsing
from interactive_llm.pipeline.people import is_clean_forget, parse_spelling  # noqa: E402
from test_people import known_speaker, mk, vec  # noqa: E402


@pytest.mark.parametrize("text", ["I can never spell it. Today is Tuesday", "I'm bad at spelling. This is Wednesday"])
def test_a_name_in_another_sentence_is_not_a_respelling(text):
    assert parse_spelling(text) is None


def test_real_respelling_phrases_still_parse():
    assert parse_spelling("Spelling should be Marvin") == "Marvin"
    assert parse_spelling("It is spelled M-A-R-V-I-N.") == "Marvin"
    assert parse_spelling("I said it wrong. The spelling is Hana.") == "Hana"


def test_chatter_about_spelling_never_renames_a_stored_person(tmp_path):
    m, spk = known_speaker(tmp_path, "Sam")
    run(m.apply_utterance("I can never spell it. Today is Tuesday.", spk, vec(100)))
    assert m.db.names() == ["Sam"]


def test_a_spelling_that_is_a_different_name_asks_before_renaming(tmp_path):
    m, spk = known_speaker(tmp_path, "Sam")
    notes = run(m.apply_utterance("The spelling should be Tuesday", spk, vec(100)))
    assert m.db.names() == ["Sam"] and "Nothing changed yet" in notes[0] and "needs_input" in [e["kind"] for e in m.events]
    run(m.apply_utterance("yes", spk, vec(100)))                         # the person asked can confirm it
    assert m.db.names() == ["Tuesday"]


def test_a_close_respelling_is_applied_directly(tmp_path):
    m, spk = known_speaker(tmp_path, "Marven")
    run(m.apply_utterance("Spelling should be M-A-R-V-I-N.", spk, vec(100)))
    assert m.db.names() == ["Marvin"]


def _with_people(tmp_path, *names):
    f, m = mk(tmp_path)
    for i, name in enumerate(names):
        m.db.add(name, "voice", vec(300 + i))
    return m


def test_clean_forget_detection():
    known = ["Danielle", "Sam"]
    for clean in ["Forget Sam", "forget Sam and Danielle please", "Delete Daniel", "Forget me", "delete person A",
                  "forget Sam from your memory"]:
        assert is_clean_forget(clean, known), clean
    for messy in ["delete the Daniel Craig playlist", "remove Daniel from the shopping list tomorrow", "Forget Sam right now"]:
        assert not is_clean_forget(messy, known), messy
    assert not is_clean_forget("I forget my keys", known)


def test_a_close_name_inside_ordinary_speech_deletes_nobody(tmp_path):
    m = _with_people(tmp_path, "Danielle", "Sam")
    notes = run(m.apply_utterance("Please delete the Daniel Craig playlist", Speaker(name="Sam"), None))
    assert m.db.names() == ["Danielle", "Sam"]
    assert notes and "nothing was deleted" in notes[0] and "Danielle" in notes[0]


def test_a_clean_forget_with_a_misheard_name_still_works_and_says_what_was_matched(tmp_path):
    m = _with_people(tmp_path, "Danielle", "Sam")
    notes = run(m.apply_utterance("Forget Daniel", Speaker(name="Sam"), None))
    assert m.db.names() == ["Sam"]
    assert "Daniel" in notes[0] and "Danielle" in notes[0] and "matched" in notes[0]       # the assistant must say both names
    assert "heard 'Daniel'" in m.events[-1]["text"]                                          # and the GUI row shows it


def test_exact_names_are_always_forgotten_even_in_a_longer_sentence(tmp_path):
    m = _with_people(tmp_path, "Danielle", "Sam")
    run(m.apply_utterance("Could you forget Danielle right now", Speaker(name="Sam"), None))
    assert m.db.names() == ["Sam"]


# ---------------------------------------------------------------- auto-learning needs lip movement
from test_people import FRAME, face, talking  # noqa: E402


def _known_face_unknown_voice(tmp_path):
    f, m = mk(tmp_path)
    m.db.add("Hana", "face", vec(1))
    f.dets = [face((100, 50, 80, 100), 1)]
    m.update(FRAME)
    return m, m.identify_speaker(vec(55), 0, 2)


def _known_voice_unknown_face(tmp_path):
    f, m = mk(tmp_path)
    m.db.add("Hana", "voice", vec(55))
    f.dets = [face((100, 50, 80, 100), 9)]
    m.update(FRAME)
    return m, m.identify_speaker(vec(55), 0, 2)


def test_a_still_face_never_gets_someone_elses_voice(tmp_path):
    m, spk = _known_face_unknown_voice(tmp_path)         # Hana sits still while a visitor or the TV talks for 2 s
    assert m.co_learn(spk, vec(55), dur=2.0, t0=0, t1=2) is False
    assert m.db.match_voice(vec(55))[0] is None and m.events == []


def test_without_a_time_window_nothing_is_learned(tmp_path):
    m, spk = _known_face_unknown_voice(tmp_path)
    talking(m, 0, 2)
    assert m.co_learn(spk, vec(55), dur=2.0) is False and m.db.match_voice(vec(55))[0] is None


def test_lip_movement_outside_the_utterance_does_not_count(tmp_path):
    m, spk = _known_face_unknown_voice(tmp_path)
    talking(m, 10, 12)                                    # moved their mouth, but long after the utterance
    assert m.co_learn(spk, vec(55), dur=2.0, t0=0, t1=2) is False


def test_a_still_guest_face_never_gets_the_owners_name(tmp_path):
    m, spk = _known_voice_unknown_face(tmp_path)          # the owner's voice is heard while an unknown guest sits still
    assert spk.voice_name == "Hana" and m.co_learn(spk, vec(55), dur=2.0, t0=0, t1=2) is False
    assert m.db.people["Hana"].get("face", []) == []


def test_a_talking_face_is_still_learned(tmp_path):
    m, spk = _known_voice_unknown_face(tmp_path)
    talking(m, 0, 2)
    assert m.co_learn(spk, vec(55), dur=2.0, t0=0, t1=2) is True and len(m.db.people["Hana"]["face"]) == 1
    m2, spk2 = _known_face_unknown_voice(tmp_path / "second") if (tmp_path / "second").mkdir() is None else (None, None)
    talking(m2, 0, 2)
    assert m2.co_learn(spk2, vec(55), dur=2.0, t0=0, t1=2) is True and m2.db.match_voice(vec(55))[0] == "Hana"


# ---------------------------------------------------------------- tools: private files never meet the internet or the cloud
import types  # noqa: E402

from interactive_llm.config import Config  # noqa: E402
from interactive_llm.engine import Engine  # noqa: E402
from interactive_llm.skills.base import SkillResult  # noqa: E402
from test_engine import FakeClaude, FakeLocal  # noqa: E402

SCHEMAS = [{"type": "function", "function": {"name": name}} for name in ("web_search", "stock_quote", "list_folder")]


def call(name, **args):
    return {"id": f"{name}-{len(args)}", "name": name, "arguments": json.dumps(args)}


class SpyLocal(FakeLocal):
    """Records which tools were offered in each round."""
    def __init__(self, script):
        super().__init__(script)
        self.offered = []

    def stream(self, messages, tools=None, required=False):
        self.offered.append(sorted(t["function"]["name"] for t in tools or []))
        yield from self.script[self.calls]
        self.calls += 1


def run_reply(script, text="hello", engine_setup=None, claude=None, images=None):
    ran = []
    local = SpyLocal(script)
    engine = Engine(Config(), local, claude or FakeClaude())
    if engine_setup:
        engine_setup(engine)
    out = "".join(engine.reply(text, skill_tools=SCHEMAS, skill_runner=lambda n, a: ran.append((n, a)) or "result", images=images))
    tool_texts = [m["content"] for m in engine.history if m.get("role") == "tool"]
    return ran, tool_texts, local, out


def test_a_search_and_a_folder_listing_requested_together_are_both_refused():
    """Neither runs: the model must ask for them one at a time, and then the second is blocked by the rules below."""
    ran, tool_texts, _, _ = run_reply([[{"tool_calls": [call("list_folder", folder="documents"), call("web_search", query="x")]}],
                                       [{"text": "ok."}]])
    assert ran == [] and len(tool_texts) == 2 and all(text.startswith("Refused") for text in tool_texts)


def test_after_a_folder_listing_nothing_can_go_to_the_internet_in_later_rounds():
    ran, tool_texts, local, _ = run_reply([[{"tool_calls": [call("list_folder", folder="downloads")]}],
                                           [{"tool_calls": [call("web_search", query="anything")]}], [{"text": "ok."}]])
    assert [name for name, _ in ran] == ["list_folder"] and "Refused" in tool_texts[-1]
    assert local.offered[1] == ["list_folder"]                        # web_search and stock_quote are no longer even offered


def test_a_folder_cannot_be_read_in_a_turn_that_handled_web_text():
    ran, tool_texts, local, _ = run_reply([[{"tool_calls": [call("web_search", query="news")]}],
                                           [{"tool_calls": [call("list_folder", folder="documents")]}], [{"text": "ok."}]])
    assert [name for name, _ in ran] == ["web_search"] and "Refused" in tool_texts[-1]
    assert "list_folder" not in local.offered[1]


def test_a_search_containing_a_listed_file_name_is_refused_in_a_later_turn():
    def setup(engine):
        engine.private_names = {"quarterly_taxes_2024.pdf", "quarterly_taxes_2024"}
    ran, tool_texts, _, _ = run_reply([[{"tool_calls": [call("web_search", query="quarterly_taxes_2024 pdf")]}], [{"text": "ok."}]],
                                      engine_setup=setup)
    assert ran == [] and "Refused" in tool_texts[0]
    ran, _, _, _ = run_reply([[{"tool_calls": [call("web_search", query="weather in Paris")]}], [{"text": "ok."}]], engine_setup=setup)
    assert [name for name, _ in ran] == ["web_search"]                # ordinary searches still work


def test_deep_think_cannot_carry_file_names_to_the_cloud():
    class CountingClaude(FakeClaude):
        asks = 0
        def ask(self, q, image_b64=None, context="", cancel=None):
            type(self).asks += 1
            return "answer"
    ran, tool_texts, _, _ = run_reply([[{"tool_calls": [call("list_folder", folder="documents"), call("deep_think", question="q")]}],
                                       [{"text": "ok."}]], text="solve x squared minus four", claude=CountingClaude())
    assert CountingClaude.asks == 0 and any(t.startswith("Refused") for t in tool_texts)


class CloudClaude(FakeClaude):
    uses_claude = True
    def __init__(self):
        self.asked = []
    def ask(self, q, image_b64=None, context="", cancel=None):
        self.asked.append((q, image_b64, context))
        return "answer"


def test_the_gui_is_told_before_anything_goes_to_the_cloud_and_no_summary_is_sent():
    claude, said = CloudClaude(), []
    def setup(engine):
        engine.summary = "The user mentioned something private about their family."
        engine.on_cloud = lambda text: said.append((text, len(claude.asked)))
    run_reply([[{"tool_calls": [call("deep_think", question="what is 17 times 23", include_camera=True)]}], [{"text": "391."}]],
              text="what is 17 times 23", engine_setup=setup, claude=claude, images=["AAAA"])
    assert said == [("Sent your question and the camera frame to Anthropic's cloud", 0)]      # said first, then sent
    assert claude.asked == [("what is 17 times 23", "AAAA", "")]                              # empty context: no summary


def test_a_local_fallback_reasoner_announces_nothing():
    said = []
    run_reply([[{"tool_calls": [call("deep_think", question="17 times 23")]}], [{"text": "391."}]], text="what is 17 times 23",
              engine_setup=lambda engine: setattr(engine, "on_cloud", said.append))
    assert said == []


def test_listed_file_names_are_remembered_but_common_words_are_not():
    from interactive_llm.server import Session
    stub = types.SimpleNamespace(engine=types.SimpleNamespace(private_names=set()))
    result = SkillResult(OK, "listed", "Listed 4 items", detail=["Quarterly_Taxes_2024.pdf", "Photos/", "notes.txt", "Project Alpha/"])
    result.skill = "mac_control"
    Session._note_private(stub, result)
    assert {"quarterly_taxes_2024.pdf", "quarterly_taxes_2024", "notes.txt", "project alpha"} <= stub.engine.private_names
    assert "photos" not in stub.engine.private_names and "notes" not in stub.engine.private_names
    other = SkillResult(OK, "x", "y", detail=["secret_file.txt"])
    other.skill = "timers"
    Session._note_private(stub, other)
    assert "secret_file.txt" not in stub.engine.private_names                       # only folder listings count


# ---------------------------------------------------------------- web search: personal sentences are never sent to a search engine
from interactive_llm.skills.web_search import WebSearchSkill  # noqa: E402


def _search_skill():
    return WebSearchSkill(search=lambda q, news=False: [], stock=lambda c: {}, reader=None)


@pytest.mark.parametrize("text", ["How should I tell Sam about my test results tomorrow?", "What am I doing tomorrow?",
                                  "Is my flight happening tonight?", "Did we get the results yet?", "How do I tell Hana the news?"])
def test_personal_sentences_do_not_trigger_a_search(text):
    assert not _search_skill().matches(text, SkillContext(speaker="Sam", known_names=["Sam", "Hana"]))


@pytest.mark.parametrize("text", ["What's the weather tomorrow?", "Who won the match last night, any results?", "What's happening in Paris tonight?",
                                  "What's the latest news on the election?", "search for how I should cook rice",
                                  "Look up my favourite band", "Tell me the latest headlines"])
def test_general_live_questions_and_explicit_searches_still_work(text):
    assert _search_skill().matches(text, SkillContext(speaker="Sam", known_names=["Sam", "Hana"]))


# ---------------------------------------------------------------- naming loop, phantom captions, verbs as names
from interactive_llm.pipeline.models import is_junk_transcript, is_sound_caption  # noqa: E402
from interactive_llm.pipeline.people import parse_intros  # noqa: E402


def test_intro_plus_spelling_in_one_sentence_renames_once_and_does_not_ask_again(tmp_path):
    """Regression: "I'm Marven, spelled M-A-R-V-I-N" renamed the person, then asked to confirm the heard spelling."""
    m, spk = known_speaker(tmp_path, "Marven")
    notes = run(m.apply_utterance("Hi, I'm Marven spelled M-A-R-V-I-N.", spk, vec(100), spoken=True))
    assert m.db.names() == ["Marvin"]
    kinds = [e["kind"] for e in m.events]
    assert kinds == ["renamed"], kinds                       # no "Heard ..., kept ..." follow-up, nothing left waiting
    assert not any("kept" in e["text"] for e in m.events) and m._pending is None
    assert all("didn't take" not in n for n in notes)


def test_misheard_third_form_of_the_name_is_also_covered_by_the_spelling(tmp_path):
    m, spk = known_speaker(tmp_path, "Marven")
    run(m.apply_utterance("I'm Marvyn, that's M-A-R-V-I-N.", spk, vec(100), spoken=True))
    assert m.db.names() == ["Marvin"] and [e["kind"] for e in m.events] == ["renamed"]


def test_bare_letters_that_spell_the_stored_name_correct_it(tmp_path):
    """Regression: "No, it's M-A-R-V-I-N" (no word 'spell') was ignored and the model said it can't update names."""
    m, spk = known_speaker(tmp_path, "Marven")
    run(m.apply_utterance("No, it's M-A-R-V-I-N.", spk, vec(100), spoken=True))
    assert m.db.names() == ["Marvin"] and [e["kind"] for e in m.events] == ["renamed"]


@pytest.mark.parametrize("text", ["I work for the U-S-A team", "my favourite letters are A B C D", "Call 1-800 or the F-B-I"])
def test_bare_letters_that_are_not_close_to_the_name_are_ignored(tmp_path, text):
    m, spk = known_speaker(tmp_path, "Sam")
    run(m.apply_utterance(text, spk, vec(100), spoken=True))
    assert m.db.names() == ["Sam"] and not m.events


def test_bare_letters_need_a_current_name():
    assert parse_spelling("No, it's M-A-R-V-I-N.") is None
    assert parse_spelling("No, it's M-A-R-V-I-N.", "Marven") == "Marvin"
    assert parse_spelling("No, it's M-A-R-V-I-N.", "Sam") is None


@pytest.mark.parametrize("text", ["Piano music plays.", "[Applause]", "(soft music)", "*laughs*", "Soft music playing.",
                                  "Thanks for watching!", "Thank you for watching.", "[BLANK_AUDIO]", "♪ la la ♪",
                                  "[Music] [Applause]", "(music) (applause)", "Music playing. Music playing.",
                                  "[Music] Thanks for watching."])
def test_sound_captions_and_signoffs_never_reach_the_assistant(text):
    assert is_sound_caption(text) and is_junk_transcript(text)


@pytest.mark.parametrize("text", ["Play some music.", "Play music", "I like piano music", "Turn the music down", "Okay.",
                                  "Is there noise in the video?", "Silence is golden", "Can you hear the piano?",
                                  "Piano.", "Music.", "Drums", "Typing", "Silence.", "(and then I said", "[Music"])
def test_real_requests_about_sound_are_not_dropped(text):
    assert not is_sound_caption(text) and not is_junk_transcript(text)


@pytest.mark.parametrize("text", ["I'm Cooking dinner", "I'm Listening to music", "This is Working", "Hi, I'm Running."])
def test_loose_verbs_are_not_names(text):
    assert parse_intros(text) == []


def test_explicit_naming_phrases_still_accept_an_unusual_name():
    assert [i.name for i in parse_intros("My name is Sterling")] == ["Sterling"]
    assert [i.name for i in parse_intros("Hi, I'm Sam")] == ["Sam"]


def test_spelled_out_letters_are_never_an_introduction():
    assert parse_intros("I'm Sam, that's S-A-M.") == [i for i in parse_intros("I'm Sam")]
    assert parse_intros("this is M-A-R-V-I-N") == []


def test_server_gate_lets_bare_letters_through_only_when_someone_is_on_record(tmp_path):
    """Regression: the live server only calls the naming code when needs_processing() says so; the unit tests skipped it."""
    m, spk = known_speaker(tmp_path, "Marven")
    assert m.needs_processing("No, it's M-A-R-V-I-N.")
    assert not m.needs_processing("What's the weather today?")
    f, empty = mk(tmp_path / "empty")
    assert not empty.needs_processing("No, it's M-A-R-V-I-N.")


@pytest.mark.parametrize("text", ["Hi, I'm Sterling.", "I'm Harding", "I'm Fleming"])
def test_real_names_ending_in_ing_are_accepted(text):
    assert len(parse_intros(text)) == 1


def test_first_time_intro_with_a_spelling_registers_the_person(tmp_path):
    """Review finding: "Hi, I'm Marvin, spelled M-A-R-V-I-N" from someone not on record learned nobody."""
    f, m = mk(tmp_path)
    run(m.apply_utterance("Hi, I'm Marvin, spelled M-A-R-V-I-N.", Speaker(), vec(1), spoken=True))
    assert m.db.names() == ["Marvin"] and [e["kind"] for e in m.events] == ["learned"]


def test_first_time_intro_with_a_misheard_name_uses_the_spelling(tmp_path):
    f, m = mk(tmp_path)
    run(m.apply_utterance("Hi, I'm Marven, spelled M-A-R-V-I-N.", Speaker(), vec(1), spoken=True))
    assert m.db.names() == ["Marvin"] and [e["kind"] for e in m.events] == ["learned"]


def test_spelling_with_no_intro_and_no_known_speaker_still_fails_honestly(tmp_path):
    f, m = mk(tmp_path)
    run(m.apply_utterance("It is spelled M-A-R-V-I-N.", Speaker(), vec(1), spoken=True))
    assert m.db.names() == [] and [e["kind"] for e in m.events] == ["failed"]


@pytest.mark.parametrize("text", ["I, I, I think so", "A B C D", "I, I, I, I, I", "Spell it out: U-S-A"])
def test_fillers_and_acronyms_do_not_take_the_naming_path(tmp_path, text):
    """Review finding: the gate fired on any three single letters, which also skipped voice/face auto-learning."""
    m, spk = known_speaker(tmp_path, "Sam")
    if "Spell" in text:                 # the word 'spell' is a naming hint, but a far-off acronym is never a respelling
        assert parse_spelling(text, "Sam") is None
        return
    assert not m.needs_processing(text)


def test_a_stutter_does_not_hide_the_spelled_name(tmp_path):
    m, spk = known_speaker(tmp_path, "Marven")
    assert parse_spelling("I, I, I, I, I, I, no it's M-A-R-V-I-N", "Marven") == "Marvin"
    run(m.apply_utterance("I, I, I, I, I, I, no it's M-A-R-V-I-N", spk, vec(100), spoken=True))
    assert m.db.names() == ["Marvin"]


# ---------------------------------------------------------------- spelling dialog: "is that right?" -> "No." -> letters in pieces
def _spoken_intro(tmp_path, heard="Marven"):
    """A spoken introduction, so the assistant is left asking whether the spelling is right."""
    f, m = mk(tmp_path)
    run(m.apply_utterance(f"Hi, I'm {heard}.", Speaker(), vec(1), spoken=True))
    m.drain_events()
    assert m._spell is not None and m._spell.mode == "check"
    return m, Speaker(name=heard, how="voice")


def say_spoken(m, spk, text):
    notes = run(m.apply_utterance(text, spk, vec(1), spoken=True))
    return notes, [(e["kind"], e["text"]) for e in m.drain_events()]


def test_no_then_letters_in_two_pieces_renames_once(tmp_path):
    """The reported conversation: 'is that right?' / 'No.' / 'M-A-R' / 'V-I-N.' must end in a real rename."""
    m, spk = _spoken_intro(tmp_path)
    notes, events = say_spoken(m, spk, "No.")
    assert [k for k, _ in events] == ["needs_input"] and "NOTHING is changed" in notes[0] and m._spell.mode == "collect"
    notes, events = say_spoken(m, spk, "M-A-R")
    assert m.db.names() == ["Marven"] and "M-A-R" in events[0][1] and "NOTHING is changed" in notes[0]
    assert "do not say it was updated" in notes[0]                      # the model may not claim success here
    notes, events = say_spoken(m, spk, "V-I-N.")
    assert m.db.names() == ["Marvin"] and [k for k, _ in events] == ["renamed"] and m._spell is None


def test_yes_closes_the_question_without_changing_anything(tmp_path):
    m, spk = _spoken_intro(tmp_path)
    notes, events = say_spoken(m, spk, "Yes, that's right.")
    assert m.db.names() == ["Marven"] and [k for k, _ in events] == ["info"] and m._spell is None


def test_no_then_a_whole_close_spelling_in_one_go(tmp_path):
    m, spk = _spoken_intro(tmp_path)
    say_spoken(m, spk, "No.")
    notes, events = say_spoken(m, spk, "It's Marvin.")
    assert m.db.names() == ["Marvin"] and [k for k, _ in events] == ["renamed"]


def test_no_then_a_different_name_asks_before_renaming(tmp_path):
    m, spk = _spoken_intro(tmp_path)
    say_spoken(m, spk, "No.")
    notes, events = say_spoken(m, spk, "It's Tuesday.")
    assert m.db.names() == ["Marven"] and "Nothing changed yet" in notes[0]
    say_spoken(m, spk, "yes")
    assert m.db.names() == ["Tuesday"]


def test_the_spelling_can_be_finished_with_that_is_it(tmp_path):
    m, spk = _spoken_intro(tmp_path)
    say_spoken(m, spk, "No.")
    say_spoken(m, spk, "T-E-S")
    say_spoken(m, spk, "S-A")
    assert m.db.names() == ["Marven"]                                   # not close to the old name: keep waiting
    notes, events = say_spoken(m, spk, "That's it.")
    assert m.db.names() == ["Marven"] and "Nothing changed yet" in notes[0]       # a different name always asks first
    say_spoken(m, spk, "yes")
    assert m.db.names() == ["Tessa"]
    assert m._spell is not None and m._spell.name == "Tessa"        # the assistant now spells the new name back


def test_chatter_during_the_question_is_not_taken_as_a_name(tmp_path):
    m, spk = _spoken_intro(tmp_path)
    say_spoken(m, spk, "No.")
    for chatter in ["Hello", "Okay, thanks", "What time is it", "Hmm"]:
        notes, events = say_spoken(m, spk, chatter)
        assert m.db.names() == ["Marven"], chatter
        assert not any(k in ("renamed", "needs_input") for k, _ in events), chatter


def test_cancel_closes_the_question(tmp_path):
    m, spk = _spoken_intro(tmp_path)
    say_spoken(m, spk, "No.")
    notes, events = say_spoken(m, spk, "Never mind.")
    assert m.db.names() == ["Marven"] and m._spell is None and [k for k, _ in events] == ["info"]


def test_only_the_person_asked_can_answer_and_it_expires(tmp_path):
    m, spk = _spoken_intro(tmp_path)
    m.db.add("Sam", "voice", vec(77))
    notes, events = say_spoken(m, Speaker(name="Sam", how="voice"), "No.")   # someone else answering does nothing
    assert events == [] and m._spell.mode == "check"
    m._spell.until = 0.0                                                    # expired
    assert not m.needs_processing("No.") and m._dialog_input("No.") is None


def test_gate_only_opens_while_a_question_is_open(tmp_path):
    f, m = mk(tmp_path)
    assert not m.needs_processing("No.") and not m.needs_processing("M-A-R")
    m, spk = _spoken_intro(tmp_path / "second")
    assert m.needs_processing("No.") and m.needs_processing("Yes") and not m.needs_processing("What's the weather?")


def test_typed_and_third_person_introductions_do_not_open_a_question(tmp_path):
    f, m = mk(tmp_path)
    run(m.apply_utterance("Hi, I'm Marven.", Speaker(), vec(1), spoken=False))
    assert m._spell is None
    m, spk = _spoken_intro(tmp_path / "other")
    run(m.apply_utterance("Say hi to the new name, I am Priya.", Speaker(name="Priya", how="voice"), vec(5), spoken=True))


def test_explicit_commands_still_win_over_the_open_question(tmp_path):
    m, spk = _spoken_intro(tmp_path)
    notes, events = say_spoken(m, spk, "Update name to M-A-R-V-I-N.")
    assert m.db.names() == ["Marvin"] and m._spell is None
    m, spk = _spoken_intro(tmp_path / "forget")
    say_spoken(m, spk, "Forget Marven.")
    assert m.db.names() == [] and m._spell is None


def test_system_prompt_no_longer_tells_the_model_to_explain_rules_or_invent_keywords():
    from dataclasses import fields
    from interactive_llm.config import Config
    prompt = next(f.default for f in fields(Config) if f.name == "system_prompt")
    assert "nothing has changed yet" in prompt and "never explain these rules" in prompt
    assert "ask them to rephrase" not in prompt


# ---------------------------------------------------------------- second review round: dialog edge cases
def test_okay_thanks_after_stray_letters_never_renames(tmp_path):
    """Review finding: 'M-A-R' then 'Okay, thanks' renamed the person to 'Mar' with no confirmation."""
    m, spk = _spoken_intro(tmp_path)
    say_spoken(m, spk, "No.")
    say_spoken(m, spk, "M-A-R")
    for chatter in ["Okay, thanks", "yes", "yeah", "sure"]:
        notes, events = say_spoken(m, spk, chatter)
        assert m.db.names() == ["Marven"], chatter
        assert not any(k == "renamed" for k, _ in events), chatter


def test_name_in_the_same_breath_as_no_is_used(tmp_path):
    """Review finding: "No, it's Marvin." was read as a bare no and the name was thrown away."""
    for phrase in ["No, it's Marvin.", "Wait, it's Marvin.", "Actually it's Marvin"]:
        m, spk = _spoken_intro(tmp_path / phrase.split()[0] / str(len(phrase)))
        notes, events = say_spoken(m, spk, phrase)
        assert m.db.names() == ["Marvin"] and [k for k, _ in events] == ["renamed"], phrase


def test_a_lowercase_word_after_no_is_not_a_name(tmp_path):
    m, spk = _spoken_intro(tmp_path)
    notes, events = say_spoken(m, spk, "No, it's wrong.")
    assert m.db.names() == ["Marven"] and not any(k == "renamed" for k, _ in events)
    assert m._pending is None


def test_stutters_and_lone_letters_do_not_become_a_name(tmp_path):
    """Review finding: 'I, I, I think so' offered to rename the person 'Iii'; a lone 'I.' started collecting junk."""
    m, spk = _spoken_intro(tmp_path)
    notes, events = say_spoken(m, spk, "Um, I, I, I think so")
    assert m._pending is None and not any("Iii" in text for _, text in events)
    notes, events = say_spoken(m, spk, "I.")
    assert events == [] and m._spell.mode == "check"
    say_spoken(m, spk, "No.")
    notes, events = say_spoken(m, spk, "I, I, I")                       # a stutter while collecting adds nothing
    assert m._spell.letters == "" and events == []


def test_sign_offs_that_a_person_may_really_say_are_kept():
    for text in ["Thank you for listening.", "Silence ends.", "The music stops.", "Applause starts."]:
        assert not is_sound_caption(text), text
    for text in ["Thanks for watching!", "Piano music plays.", "Soft music fades."]:
        assert is_sound_caption(text), text
