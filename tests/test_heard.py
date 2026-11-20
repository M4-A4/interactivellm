import time

from interactive_llm.server import Session


def make(spoken, elapsed):
    s = object.__new__(Session)
    s.spoken, s.play_t0 = spoken, time.monotonic() - elapsed
    return s


SPOKEN = [("First sentence here.", 0.0, 2.0), ("Second one follows now.", 2.0, 2.0), ("Third.", 4.0, 1.0)]


def test_heard_nothing_before_playback():
    assert make(SPOKEN, 0.0)._heard() == ""


def test_heard_partial_sentence():
    assert make(SPOKEN, 3.0)._heard() == "First sentence here. Second one"


def test_heard_everything():
    assert make(SPOKEN, 9.0)._heard() == "First sentence here. Second one follows now. Third."
