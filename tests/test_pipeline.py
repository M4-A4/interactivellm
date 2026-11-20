import cv2
import numpy as np

from interactive_llm.engine import sentences
from interactive_llm.pipeline.models import clean_for_speech
from interactive_llm.pipeline.vision import FrameBuffer


def jpeg(v):
    return cv2.imencode(".jpg", np.full((48, 64, 3), v, np.uint8))[1].tobytes()


def test_eager_first_clause():
    out = list(sentences(iter(["Well, that is a really good question, ", "and here is more. Done."])))
    assert out[0].endswith(",") and len(out) >= 2


def test_clean_for_speech():
    assert clean_for_speech("**Hi** <think>x</think> see https://a.b 😊") == "Hi see link"


def test_frame_pick_scene_change():
    fb = FrameBuffer(change_threshold=6.0)
    fb.push(jpeg(20)); fb.mark_start()
    fb.push(jpeg(200))
    assert len(fb.pick()) == 2  # scene changed during speech -> start + end frame
    fb.push(jpeg(200)); fb.mark_start(); fb.push(jpeg(201))
    assert len(fb.pick()) == 1


def test_no_frames_when_stale():
    fb = FrameBuffer(stale_s=0.0)
    fb.push(jpeg(10))
    assert fb.pick() == []


def test_vad_silence_makes_no_events():
    from interactive_llm.pipeline.vad import VADSegmenter
    v = VADSegmenter()
    assert list(v.feed(np.zeros(16000, np.int16).tobytes())) == []


def test_vision_gate():
    from interactive_llm.pipeline.vision import wants_vision as w
    for yes in ["What do you see?", "Look at this", "what's this in my hand", "How do I look?", "what color is my shirt",
                "Can you see the sign behind me", "am I wearing glasses"]:
        assert w(yes), yes
    for no in ["Hello there.", "I am person A.", "Thanks, that's helpful.", "What is the capital of France?",
               "I'm good.", "Tell me a fun fact", "No, not 5 years, maybe 2 and a half years"]:
        assert not w(no), no
    assert w("hello", "always") and not w("what do you see", "never")


def test_speech_cleaning_strips_latex():
    assert clean_for_speech("Let the ball be $x$ dollars, so \\(x\\) is 5") == "Let the ball be x dollars, so (x) is 5"


def test_echoed_notes_are_never_spoken():
    assert clean_for_speech("I do not know your name yet. [interrupted]") == "I do not know your name yet."
    assert clean_for_speech("Hi [System: you renamed X] there") == "Hi there"
