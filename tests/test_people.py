import asyncio
import time

import cv2
import numpy as np
import pytest

from interactive_llm.pipeline.people import (Detection, FaceEngine, PeopleDB, PeopleManager, Speaker, parse_forget,
                                              parse_intros)

rng = np.random.default_rng(0)


def vec(seed):
    return np.random.default_rng(seed).normal(size=128).astype(np.float32)


# ---------------------------------------------------------------- language
def test_parse_self_and_other():
    i = parse_intros("Hi, I'm Marvin and this is Sam.")
    assert [(x.kind, x.name) for x in i] == [("self", "Marvin"), ("other", "Sam")]
    assert [(x.kind, x.name) for x in parse_intros("Meet my friend Priya")] == [("other", "Priya")]
    assert [(x.kind, x.name) for x in parse_intros("My name is Dana")] == [("self", "Dana")]
    assert [(x.kind, x.name) for x in parse_intros("her name is Joan")] == [("other", "Joan")]


def test_placeholder_and_lowercase_names():
    got = lambda t: [(x.kind, x.name) for x in parse_intros(t)]
    assert got("I am person A.") == [("self", "Person A")]
    assert got("Call me person A") == [("self", "Person A")]
    assert got("i am person 1") == [("self", "Person 1")]
    assert got("this is guest two") == [("other", "Guest Two")]
    assert got("I'm person B and this is Person C") == [("self", "Person B"), ("other", "Person C")]
    assert got("my name is sam") == [("self", "Sam")]       # typed lowercase after an explicit phrase
    assert got("call me marvin") == [("self", "Marvin")]
    assert parse_forget("forget person A") == ["Person A"]


@pytest.mark.parametrize("t", ["i am sam", "i'm going home", "my name is tired", "call me later", "that's fine",
                               "I'm person","I'm tired", "This is amazing", "I'm Good at this", "that's great", "this is a test",
                               "I am going home", "I'm Sorry about that"])
def test_no_false_intros(t):
    assert parse_intros(t) == []


def test_parse_forget():
    assert parse_forget("please forget me") == ["me"]
    assert parse_forget("Forget Sam") == ["Sam"]
    assert parse_forget("I forget it all the time") == []


# ---------------------------------------------------------------- db
def test_db_match_persist_forget(tmp_path):
    db = PeopleDB(tmp_path / "p.json")
    db.add("alice", "face", vec(1)); db.add("Bob", "voice", vec(2))
    assert db.match_face(vec(1))[0] == "alice" and db.match_face(vec(9))[0] is None
    assert db.match_voice(vec(2))[0] == "Bob"
    assert db.add("ALICE", "voice", vec(3)) == "alice"  # case-insensitive identity
    assert PeopleDB(tmp_path / "p.json").match_face(vec(1))[0] == "alice"  # persisted
    assert oct((tmp_path / "p.json").stat().st_mode & 0o777) == "0o600"
    assert db.forget("Alice") and not db.forget("alice")


# ---------------------------------------------------------------- manager with fake faces
class FakeFaces:
    def __init__(self): self.dets = []
    def detect(self, bgr): return self.dets
    def embed(self, bgr, det): return det.raw


def face(box, seed):
    return Detection(box, np.zeros((5, 2), np.float32), 0.9, vec(seed))


FRAME = np.zeros((300, 400, 3), np.uint8)


def mk(tmp_path):
    f = FakeFaces()
    return f, PeopleManager(PeopleDB(tmp_path / "p.json"), f)


def run(c): return asyncio.run(c)


def talking(manager, t0, t1):
    """Record clear mouth movement for every visible face between t0 and t1 (what the camera would have seen)."""
    for track in manager.visible():
        track.motion.extend((t0 + (t1 - t0) * i / 9, 0.6) for i in range(10))


def test_self_intro_single_face_learns_face_and_voice(tmp_path):
    f, m = mk(tmp_path)
    f.dets = [face((100, 50, 80, 100), 1)]; m.update(FRAME)
    spk = m.identify_speaker(vec(100), 0, 1)
    notes = run(m.apply_utterance("Hi, I'm Alice", spk, vec(100)))
    assert "learned Alice's face and voice" in notes[0]
    # later: same voice, no intro -> identified by voice, face track named
    m2 = PeopleManager(m.db, f); m2.update(FRAME)
    assert m2.visible()[0].name == "Alice"
    s2 = m2.identify_speaker(vec(100), 0, 1)
    assert s2.name == "Alice" and s2.how == "voice" and s2.track.name == "Alice"
    assert "named Alice" in m2.context_note(s2)


def test_other_intro_two_unknown_uses_pointing_pick(tmp_path):
    f, m = mk(tmp_path)
    f.dets = [face((20, 50, 80, 100), 1), face((280, 50, 80, 100), 2)]; m.update(FRAME)
    ids = [t.id for t in m.visible()]
    seen = {}

    async def pick(jpeg, text, cands): seen.update(cands=cands, n=len(jpeg)); return ids[1]
    notes = run(m.apply_utterance("This is Bob", Speaker(), None, pick))
    assert "learned Bob's face" in notes[0] and seen["cands"] == ids and seen["n"] > 100
    assert [t.name for t in m.visible()] == [None, "Bob"]


def test_other_intro_ambiguous_learns_nothing(tmp_path):
    f, m = mk(tmp_path)
    f.dets = [face((20, 50, 80, 100), 1), face((280, 50, 80, 100), 2)]; m.update(FRAME)

    async def pick(*a): return None
    notes = run(m.apply_utterance("This is Bob", Speaker(), None, pick))
    assert "unclear which one" in notes[0] and m.db.people == {}


def test_other_intro_excludes_speaker_face(tmp_path):
    f, m = mk(tmp_path)
    m.db.add("Alice", "face", vec(1)); m.db.add("Alice", "voice", vec(100))
    f.dets = [face((20, 50, 80, 100), 1), face((280, 50, 80, 100), 2)]; m.update(FRAME)
    spk = m.identify_speaker(vec(100), 0, 1)
    assert spk.name == "Alice" and spk.track.name == "Alice"
    notes = run(m.apply_utterance("this is Bob", spk, vec(100)))   # only one unknown face left
    assert "learned Bob's face" in notes[0] and sorted(m.db.names()) == ["Alice", "Bob"]


def test_no_unknown_face_in_view(tmp_path):
    f, m = mk(tmp_path); m.update(FRAME)
    notes = run(m.apply_utterance("This is Bob", Speaker(), None))
    assert "no unrecognised face" in notes[0] and not m.db.people


def test_co_learn_voice_from_known_face(tmp_path):
    f, m = mk(tmp_path)
    m.db.add("Alice", "face", vec(1))
    f.dets = [face((100, 50, 80, 100), 1)]; m.update(FRAME)
    spk = m.identify_speaker(vec(55), 0, 2)
    assert spk.name == "Alice" and spk.how == "only face"
    talking(m, 0, 2)                                            # her lips moved while the voice was heard
    assert m.co_learn(spk, vec(55), dur=2.0, t0=0, t1=2) is True    # voice learned silently
    assert m.db.match_voice(vec(55))[0] == "Alice"


def test_co_learn_skipped_with_two_faces(tmp_path):
    f, m = mk(tmp_path)
    m.db.add("Alice", "face", vec(1))
    f.dets = [face((20, 50, 80, 100), 1), face((280, 50, 80, 100), 2)]; m.update(FRAME)
    spk = m.identify_speaker(vec(55), 0, 2)
    assert m.co_learn(spk, vec(55), dur=2.0) is False and m.db.match_voice(vec(55))[0] is None


def test_lip_motion_picks_talker_with_two_faces(tmp_path):
    f, m = mk(tmp_path)
    f.dets = [face((20, 50, 80, 100), 1), face((280, 50, 80, 100), 2)]; m.update(FRAME)
    a, b = m.visible()
    now = time.monotonic()
    for k in range(6):
        a.motion.append((now - 1 + k * 0.1, 0.05)); b.motion.append((now - 1 + k * 0.1, 0.9))
    spk = m.identify_speaker(None, now - 1.2, now)
    assert spk.how == "lips" and spk.track is b
    for k in range(6):  # similar motion -> refuse to guess
        a.motion.append((now + k * 0.1, 0.5)); b.motion.append((now + k * 0.1, 0.55))
    assert m.identify_speaker(None, now - 0.1, now + 1).track is None


def test_voice_conflict_does_not_overwrite(tmp_path):
    f, m = mk(tmp_path)
    m.db.add("Alice", "voice", vec(100))
    f.dets = [face((100, 50, 80, 100), 7)]; m.update(FRAME)
    spk = m.identify_speaker(vec(100), 0, 1)
    notes = run(m.apply_utterance("I'm Bob", spk, vec(100)))
    assert "recognised as Alice but says they are Bob" in notes[0] and m.db.names() == ["Alice"]  # asks, changes nothing yet


def test_forget_me(tmp_path):
    f, m = mk(tmp_path)
    m.db.add("Alice", "face", vec(1)); m.db.add("Alice", "voice", vec(100))
    f.dets = [face((100, 50, 80, 100), 1)]; m.update(FRAME)
    spk = m.identify_speaker(vec(100), 0, 1)
    notes = run(m.apply_utterance("please forget me", spk, vec(100)))
    assert "deleted everything" in notes[0] and not m.db.people and m.visible()[0].name is None


# ---------------------------------------------------------------- real face models
def test_real_face_pipeline_recognises_same_person(tmp_path):
    skimage = pytest.importorskip("skimage.data")
    img = cv2.cvtColor(skimage.astronaut(), cv2.COLOR_RGB2BGR)
    engine = FaceEngine()
    engine.load()          # load the models first: a slow cold load inside update() would age the track out of view
    m = PeopleManager(PeopleDB(tmp_path / "p.json"), engine)
    m.update(img)
    assert len(m.visible()) == 1
    spk = m.identify_speaker(vec(5), 0, 1)
    assert "learned Priya's face" in run(m.apply_utterance("I'm Priya", spk, None))[0]
    darker = cv2.convertScaleAbs(cv2.resize(img, None, fx=0.6, fy=0.6), alpha=0.75, beta=-15)
    m2 = PeopleManager(m.db, engine); m2.update(darker)
    assert m2.visible()[0].name == "Priya" and m2.visible()[0].score > 0.6


# ---------------------------------------------------------------- forgetting (regressions from a real session)
KNOWN = ["Person A", "Lena", "Mena"]


def test_forget_parses_multiple_names_and_verbs():
    assert parse_forget("Forget Lena and Mena.") == ["Lena", "Mena"]
    assert parse_forget("I said earlier to remove a Lena and Mena") == ["Lena", "Mena"]
    assert parse_forget("delete Lena, Mena and person A") == ["Person A", "Lena", "Mena"]
    assert parse_forget("forget lena and mena") == ["Lena"][:0]                    # lowercase ignored without a known list
    assert parse_forget("forget lena and mena", KNOWN) == ["lena", "mena"]         # ...but accepted when they are on record
    assert parse_forget("Please forget Mena from your memory") == ["Mena"]


@pytest.mark.parametrize("t", ["forget it", "Forget about it, never mind", "I forget it all the time", "no, forget that",
                               "I always forget my keys", "forget what I said"])
def test_forget_idioms_are_not_commands(t):
    from interactive_llm.pipeline.people import has_forget_command
    assert not has_forget_command(t) and parse_forget(t, KNOWN) == []


def mk_db(tmp_path):
    f, m = mk(tmp_path)
    for n in KNOWN:
        m.db.add(n, "voice", vec(hash(n) % 1000))
    return m


def test_forget_two_names_deletes_both(tmp_path):
    m = mk_db(tmp_path)
    notes = run(m.apply_utterance("Forget Lena and mena.", Speaker(), None))
    assert "deleted everything you knew about Lena and Mena" in notes[0] and m.db.names() == ["Person A"]


def test_misheard_name_asks_instead_of_guessing(tmp_path):
    m = mk_db(tmp_path)
    notes = run(m.apply_utterance("Forget Vena.", Speaker(), None))       # what Whisper actually heard
    assert "could mean" in notes[0] and "Lena" in notes[0] and "Mena" in notes[0]
    assert m.db.names() == ["Lena", "Mena", "Person A"]                    # nothing deleted on an ambiguous match


def test_close_unique_misspelling_is_resolved(tmp_path):
    m = mk_db(tmp_path)
    notes = run(m.apply_utterance("forget Lina", Speaker(), None))
    assert "Lena" in notes[0] and m.db.names() == ["Mena", "Person A"]


def test_unknown_name_gives_honest_failure_note(tmp_path):
    m = mk_db(tmp_path)
    notes = run(m.apply_utterance("please forget Zoltan", Speaker(), None))
    assert "nothing was deleted" in notes[0] and "Zoltan" in notes[0] and "Lena" in notes[0] and len(m.db.names()) == 3


def test_forget_without_a_name_says_it_could_not_tell(tmp_path):
    m = mk_db(tmp_path)
    notes = run(m.apply_utterance("Oh, I mean forget them.", Speaker(), None))
    assert "could not tell who" in notes[0] and len(m.db.names()) == 3


def test_forget_everyone_needs_confirmation(tmp_path):
    m = mk_db(tmp_path)
    assert m.needs_processing("forget everyone")
    notes = run(m.apply_utterance("Forget everyone.", Speaker(), None))
    assert "Nothing has been deleted yet" in notes[0] and len(m.db.names()) == 3
    assert m.needs_processing("yes please")                                  # armed -> confirmation is processed
    notes = run(m.apply_utterance("Yes, do it", Speaker(), None))
    assert "all 3 people" in notes[0] and m.db.names() == []


def test_confirmation_without_a_pending_wipe_does_nothing(tmp_path):
    m = mk_db(tmp_path)
    assert not m.needs_processing("yes that sounds good")
    assert len(m.db.names()) == 3


# ---------------------------------------------------------------- renaming (regression: "I am now called Marvin")
from interactive_llm.pipeline.people import parse_renames  # noqa: E402


def test_parse_renames():
    assert parse_renames("I am now called Marvin.") == [(None, "Marvin")]
    assert parse_renames("my name is now marvin") == [(None, "Marvin")]
    assert parse_renames("I'm actually Marvin") == [(None, "Marvin")]
    assert parse_renames("Please change my name to Marvin") == [(None, "Marvin")]
    assert parse_renames("I'm not Person A, I'm Marvin") == [(None, "Marvin")]
    assert parse_renames("Rename Person A to Marvin") == [("Person A", "Marvin")]
    assert parse_renames("Person A is actually Marvin") == [("Person A", "Marvin")]
    for no in ["I'm actually tired", "That is actually fine", "I am now going home", "it is now late"]:
        assert parse_renames(no) == [], no


def known_speaker(tmp_path, name="Person A"):
    f, m = mk(tmp_path)
    m.db.add(name, "face", vec(1)); m.db.add(name, "voice", vec(100))
    f.dets = [face((100, 50, 80, 100), 1)]; m.update(FRAME)
    return m, m.identify_speaker(vec(100), 0, 1)


def test_users_exact_sentence_renames_and_keeps_samples(tmp_path):
    m, spk = known_speaker(tmp_path)
    assert m.needs_processing("I am now called Marvin.")
    notes = run(m.apply_utterance("I am now called Marvin.", spk, vec(100)))
    assert "renamed Person A to Marvin" in notes[0]
    assert m.db.names() == ["Marvin"] and len(m.db.people["Marvin"]["face"]) == 1 and len(m.db.people["Marvin"]["voice"]) == 1
    assert spk.name == "Marvin" and m.visible()[0].name == "Marvin"
    s2 = m.identify_speaker(vec(100), 0, 1)                      # recognised under the new name afterwards
    assert s2.name == "Marvin"


def test_placeholder_is_replaced_by_a_plain_real_name(tmp_path):
    m, spk = known_speaker(tmp_path)
    notes = run(m.apply_utterance("I'm Marvin", spk, vec(100)))   # weak phrase, but old name is a placeholder
    assert "renamed Person A to Marvin" in notes[0] and m.db.names() == ["Marvin"]


def test_call_me_renames_a_real_name_immediately(tmp_path):
    m, spk = known_speaker(tmp_path, "Alice")
    notes = run(m.apply_utterance("Call me Bob", spk, vec(100)))
    assert "renamed Alice to Bob" in notes[0] and m.db.names() == ["Bob"]


def test_weak_conflict_asks_then_yes_renames(tmp_path):
    m, spk = known_speaker(tmp_path, "Alice")
    notes = run(m.apply_utterance("I'm Bob", spk, vec(100)))
    assert "Nothing changed yet" in notes[0] and m.db.names() == ["Alice"]
    assert m.needs_processing("Yes.")                              # the follow-up is now handled (it used to be a dead end)
    notes = run(m.apply_utterance("Yes.", spk, vec(100)))
    assert "renamed Alice to Bob" in notes[0] and m.db.names() == ["Bob"]


def test_weak_conflict_no_cancels(tmp_path):
    m, spk = known_speaker(tmp_path, "Alice")
    run(m.apply_utterance("I'm Bob", spk, vec(100)))
    notes = run(m.apply_utterance("No, never mind.", spk, vec(100)))
    assert "cancelled" in notes[0] and m.db.names() == ["Alice"]
    assert not m.needs_processing("yes")                           # nothing pending any more


def test_third_party_rename_with_fuzzy_old_name(tmp_path):
    f, m = mk(tmp_path); m.db.add("Person A", "voice", vec(5))
    notes = run(m.apply_utterance("Rename person A to Marvin", Speaker(), None))
    assert "renamed Person A to Marvin" in notes[0] and m.db.names() == ["Marvin"]
    notes = run(m.apply_utterance("Rename Zoltan to Sam", Speaker(), None))
    assert "nothing was renamed" in notes[0]


def test_rename_into_existing_person_merges(tmp_path):
    f, m = mk(tmp_path); m.db.add("Person A", "voice", vec(5)); m.db.add("Marvin", "face", vec(6))
    run(m.apply_utterance("Rename Person A to Marvin", Speaker(), None))
    assert m.db.names() == ["Marvin"] and m.db.people["Marvin"]["voice"] and m.db.people["Marvin"]["face"]


def test_rename_phrase_with_nobody_recognised_just_introduces(tmp_path):
    f, m = mk(tmp_path)
    f.dets = [face((100, 50, 80, 100), 3)]; m.update(FRAME)
    spk = m.identify_speaker(vec(77), 0, 1)
    assert spk.name is None
    notes = run(m.apply_utterance("I am now called Marvin", spk, vec(77)))
    assert "learned Marvin's face and voice" in notes[0] and m.db.names() == ["Marvin"]


# ---------------------------------------------------------------- events for the GUI + no false confirmations
def kinds(m): return [e["kind"] for e in m.events]


def test_rename_emits_event_and_drain_clears(tmp_path):
    m, spk = known_speaker(tmp_path)
    run(m.apply_utterance("I am now called Marvin.", spk, vec(100)))
    assert m.events == [{"kind": "renamed", "text": "Renamed Person A → Marvin"}]
    assert m.drain_events() and m.events == []


def test_learn_and_forget_emit_events(tmp_path):
    f, m = mk(tmp_path)
    f.dets = [face((100, 50, 80, 100), 1)]; m.update(FRAME)
    spk = m.identify_speaker(vec(100), 0, 1)
    run(m.apply_utterance("Hi, I'm Alice", spk, vec(100)))
    assert kinds(m) == ["learned"] and "Alice" in m.events[0]["text"]
    m.drain_events()
    run(m.apply_utterance("forget Alice", Speaker(), None))
    assert kinds(m) == ["forgot"]


def test_failed_outcomes_tell_the_llm_not_to_confirm(tmp_path):
    m = mk_db(tmp_path)
    for text, ev_kind in [("Forget Zoltan", "failed"), ("Forget Vena", "needs_input"), ("Rename Zoltan to Sam", "failed")]:
        m.drain_events()
        notes = run(m.apply_utterance(text, Speaker(), None))
        assert kinds(m) == [ev_kind], text
        assert "Do NOT say or imply that it was done" in notes[0], text
    assert len(m.db.names()) == 3                                # and indeed nothing changed in the backend


def test_naming_attempt_nobody_understood_is_flagged_as_failure(tmp_path):
    m, spk = known_speaker(tmp_path)
    for text in ["Please call me tired", "Can you change my name please?", "I am now called"]:
        m.drain_events()
        assert m.needs_processing(text), text
        notes = run(m.apply_utterance(text, spk, vec(100)))
        assert kinds(m) == ["failed"] and "NOTHING was changed" in notes[0] and "Do NOT say" in notes[0], text
        assert m.db.names() == ["Person A"]


@pytest.mark.parametrize("t", ["What's my name?", "What is my name now?", "Who am I?", "I am happy today", "That is called a cat",
                               "What is it called?"])
def test_ordinary_sentences_do_not_trigger_naming_logic(tmp_path, t):
    f, m = mk(tmp_path)
    assert not m.needs_processing(t)


def test_successful_turn_does_not_also_report_failure(tmp_path):
    m, spk = known_speaker(tmp_path)
    run(m.apply_utterance("Call me Marvin please", spk, vec(100)))
    assert kinds(m) == ["renamed"]


def test_auto_learn_emits_info_event(tmp_path):
    f, m = mk(tmp_path)
    m.db.add("Alice", "face", vec(1))
    f.dets = [face((100, 50, 80, 100), 1)]; m.update(FRAME)
    spk = m.identify_speaker(vec(55), 0, 2)
    talking(m, 0, 2)
    m.co_learn(spk, vec(55), dur=2.0, t0=0, t1=2)
    assert kinds(m) == ["info"] and "voice automatically" in m.events[0]["text"]


# ---------------------------------------------------------------- spoken names: spelling, near-matches, hints (regressions from a real session)
from interactive_llm.pipeline.people import parse_spelling, similar_name  # noqa: E402
from interactive_llm.pipeline.models import names_prompt  # noqa: E402


@pytest.mark.parametrize("t,expected", [
    ("Spelling should be M-A-R-V-I-N.", "Marvin"),          # exact sentence from the real session (as Whisper transcribed it)
    ("It is spelled M-A-R-V-I-N.", "Marvin"),
    ("it's spelled m a r v i n", "Marvin"),
    ("The spelling is M. A. R. V. I. N.", "Marvin"),
    ("Spelling should be Marvin", "Marvin"),
    ("I am Marvin", None), ("Hello there", None), ("What is my name?", None),
])
def test_parse_spelling(t, expected):
    assert parse_spelling(t) == expected


def test_similar_name():
    assert similar_name("Marvin", "Marven") and similar_name("Marvin", "Marvins")
    assert not similar_name("Marvin", "Marvin")                     # identical is not a "variant"
    assert not similar_name("Lena", "Mena") and not similar_name("Person A", "Marvin")


def test_spelled_out_letters_fix_the_spelling(tmp_path):
    m, spk = known_speaker(tmp_path, "Marven")
    notes = run(m.apply_utterance("Spelling should be M-A-R-V-I-N.", spk, vec(100), spoken=True))
    assert kinds(m) == ["renamed"] and "renamed Marven to Marvin" in notes[0] and m.db.names() == ["Marvin"]
    assert len(m.db.people["Marvin"]["face"]) == 1 and len(m.db.people["Marvin"]["voice"]) == 1   # samples kept


def test_spelling_uses_last_named_when_speaker_unrecognised(tmp_path):
    f, m = mk(tmp_path)
    m.db.add("Marven", "voice", vec(5)); m.last_named = "Marven"
    run(m.apply_utterance("It is spelled M-A-R-V-I-N", Speaker(), None, spoken=True))
    assert m.db.names() == ["Marvin"]


def test_spelling_with_nobody_to_respell_fails_honestly(tmp_path):
    f, m = mk(tmp_path)
    notes = run(m.apply_utterance("Spelling should be M-A-R-V-I-N", Speaker(), None))
    assert kinds(m) == ["failed"] and "Do NOT say" in notes[0]


def test_spelling_already_correct_is_reported(tmp_path):
    m, spk = known_speaker(tmp_path, "Marvin")
    notes = run(m.apply_utterance("Spelling should be M-A-R-V-I-N", spk, vec(100)))
    assert kinds(m) == ["info"] and "already spelled" in notes[0] and m.db.names() == ["Marvin"]


def test_spoken_near_match_does_not_overwrite_correct_spelling(tmp_path):
    m, spk = known_speaker(tmp_path, "Marvin")
    for text in ["Call me Marven.", "I am now called Marven.", "I'm Marven"]:
        m.drain_events()
        notes = run(m.apply_utterance(text, spk, vec(100), spoken=True))
        assert kinds(m) == ["needs_input"] and "nothing was changed" in notes[0] and "Do NOT say" in notes[0], text
        assert m.db.names() == ["Marvin"], text


def test_typed_name_is_always_exact(tmp_path):
    m, spk = known_speaker(tmp_path, "Marvin")
    run(m.apply_utterance("call me Marven", spk, vec(100), spoken=False))      # typed: the user means exactly this
    assert m.db.names() == ["Marven"]


def test_spell_back_only_for_spoken_names(tmp_path):
    m, spk = known_speaker(tmp_path)
    assert "spell it back letter by letter (M-A-R-V-I-N)" in run(m.apply_utterance("Call me Marvin", spk, vec(100), spoken=True))[0]
    m2, spk2 = known_speaker(tmp_path / "typed")
    assert "spell it back" not in run(m2.apply_utterance("Call me Marvin", spk2, vec(100), spoken=False))[0]


def test_possessive_artifacts_are_stripped():
    assert parse_renames("I am now called Marven's") == [(None, "Marven")]
    assert [(i.kind, i.name) for i in parse_intros("Hi, I'm Priya's")] == [("self", "Priya")]


def test_whisper_gets_names_as_context():
    assert names_prompt(["Marvin", "Priya"]) == "Names: Marvin, Priya." and names_prompt([]) is None and names_prompt(None) is None


def test_claim_guard_regex():
    from interactive_llm.server import CLAIM_RE
    for lie in ["I have corrected the spelling to Marven.", "I have updated the spelling to Marven.", "Got it. I have updated your name.",
                "I will call you Marvin from now on.", "Your name has been updated."]:
        assert CLAIM_RE.search(lie), lie
    for fine in ["Your name is Marvin.", "Which spelling would you like?", "I could not change your name.", "Nice to meet you."]:
        assert not CLAIM_RE.search(fine), fine


def test_spell_back_is_limited_to_new_real_names(tmp_path):
    f, m = mk(tmp_path)
    f.dets = [face((100, 50, 80, 100), 1)]; m.update(FRAME)
    spk = m.identify_speaker(vec(100), 0, 1)
    assert "spell it back" not in run(m.apply_utterance("I am person A.", spk, vec(100), spoken=True))[0]   # placeholder
    n = run(m.apply_utterance("Call me Marvin", spk, vec(100), spoken=True))[0]
    assert "(M-A-R-V-I-N)" in n                                                                              # new real name
    assert "spell it back" not in run(m.apply_utterance("Spelling should be Marven", spk, vec(100), spoken=True))[0]  # explicit spelling
    m.db.add("Priya", "face", vec(9))
    s2 = Speaker()
    assert "spell it back" not in run(m.apply_utterance("I'm Priya", s2, vec(55), spoken=True))[0]            # already on record


def test_junk_transcripts_are_dropped():
    from interactive_llm.pipeline.models import is_junk_transcript as junk
    hint = names_prompt(["Marvin", "Nico"])
    assert junk("Names & Nico. " * 20, hint) and junk("Names & Nico.", hint) and junk("Names & Names & Names", hint)
    assert junk("Names " * 40, None) and junk("", None)
    assert not junk("Nico.", hint) and not junk("Her name is Hana.", hint) and not junk("I'm Nico.", hint) and not junk("Can you recognize my voice?", hint)
    assert not junk("No, no, no, no, I said no.", None)


def test_context_note_gives_positions_and_does_not_disown_speaker():
    from interactive_llm.pipeline.people import PeopleManager, Speaker, Track
    m = PeopleManager.__new__(PeopleManager)
    m.frame_size = (1000, 500)
    a = Track.__new__(Track); a.name, a.box = "Marvin", (400, 100, 100, 100)
    b = Track.__new__(Track); b.name, b.box = "Hana", (850, 50, 100, 100)
    m.visible = lambda now=None: [a, b]
    m.db = type("D", (), {"people": {"x": 1}})()
    note = m.context_note(Speaker())
    # Hana is at the image's right edge, which is the user's LEFT when they face the camera.
    assert "Marvin (middle), Hana (left)" in note and "face the camera" in note
    assert "Do not tell them they are unrecognised" in note


def test_hint_continuation_with_invented_name_is_dropped():
    from interactive_llm.pipeline.models import is_junk_transcript as junk
    hint = names_prompt(["Marvin", "Nico"])
    assert junk("Names & Gertrude", hint) and junk("Names: Gertrude.", hint)
    assert not junk("Her name is Gertrude.", hint) and not junk("Names are hard to spell.", hint)


def test_camera_settings_validation():
    from interactive_llm import settings as s
    base = {"context_tokens": 6000, "context_keep_tokens": 2500, "tts_voice": "af_heart", "tts_speed": 1.0,
            "camera_resolution": "640x480", "frame_max_side": 448}
    ok, errs = s.validate({"camera_resolution": "2560x1440", "frame_max_side": 896}, base)
    assert not errs and ok["camera_resolution"] == "2560x1440" and ok["frame_max_side"] == 896
    assert s.validate({"camera_resolution": "9x9"}, base)[1] and s.validate({"camera_resolution": "99999x1"}, base)[1] and s.validate({"frame_max_side": 100}, base)[1]
    assert s.validate({"frame_max_side": 100}, base)[0] == base   # nothing applied when any value is rejected


def test_positions_are_reported_in_the_users_own_left_and_right():
    """The image is not mirrored, so image-left is the user's right; the code does the swap, not the model."""
    from interactive_llm.pipeline.people import side_for_user
    assert side_for_user(0.10) == "right" and side_for_user(0.90) == "left" and side_for_user(0.5) == "middle"


def test_left_right_questions_attach_the_camera():
    """"Is the ball on my left or right?" is about the scene, so a camera frame must be attached."""
    from interactive_llm.pipeline.vision import wants_vision
    for text in ["Is there a ball in view?", "Is anyone in the frame?", "Is the cup visible?", "Is the ball on my left or my right?", "Who is on my left?", "Is it to the right of me?", "Which side is the door on?"]:
        assert wants_vision(text), text
    for text in ["Tell me a joke", "What's two plus two", "Hello there"]:
        assert not wants_vision(text), text


def test_side_questions_get_a_mirrored_frame():
    """Only left/right questions are mirrored, and mirroring flips the image horizontally (and twice restores it)."""
    import base64
    import cv2
    import numpy as np
    from interactive_llm.pipeline.vision import asks_which_side, mirror_jpeg_b64
    assert asks_which_side("Is the ball on my left or my right?") and asks_which_side("Who is on my left?")
    assert not asks_which_side("What is written on the label?") and not asks_which_side("Turn left at the corner")
    image = np.zeros((40, 80, 3), np.uint8)
    image[:, :20] = (0, 0, 255)                                         # red block at the image's left edge
    original = base64.b64encode(cv2.imencode(".jpg", image)[1].tobytes()).decode()
    flipped = cv2.imdecode(np.frombuffer(base64.b64decode(mirror_jpeg_b64(original)), np.uint8), cv2.IMREAD_COLOR)
    assert flipped[20, 70, 2] > 150 and flipped[20, 10, 2] < 80          # now at the right edge
