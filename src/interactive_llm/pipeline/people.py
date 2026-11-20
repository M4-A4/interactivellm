"""People recognition learned from conversation.

Faces: YuNet detector + SFace recognizer (OpenCV).  Voices: SpeechBrain ECAPA.
People are enrolled by saying things like "I'm Sam" or "this is Priya" (pointing resolves which
face when several are in view), and refined automatically as they keep talking on camera.
Embeddings are stored locally in a JSON file and never leave the machine.
"""
from __future__ import annotations

import difflib
import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import cv2
import numpy as np

from ..fileutil import quarantine, write_private_atomic

# Timing constants, all in seconds.
TRACK_TTL_S = 2.0      # a face unseen for this long is dropped
VISIBLE_S = 1.0        # "currently in view" window
REID_S = 2.0           # re-run face recognition on a track this often
MIN_VOICE_S = 0.8      # shorter utterances are too short to identify by voice


def _unit(vector) -> np.ndarray:
    """Return `vector` as a flat float32 array scaled to unit length (zero vectors are returned unchanged).

    Unit length makes the dot product of two embeddings equal to their cosine similarity.
    """
    vector = np.asarray(vector, dtype=np.float32).ravel()
    length = float(np.linalg.norm(vector))
    return vector / length if length else vector


# --------------------------------------------------------------------------- storage
class PeopleDB:
    """Thread-safe store of named people, each with face and voice embedding samples, persisted as JSON.

    The file maps ``name -> {"face": [embedding, ...], "voice": [embedding, ...]}``. A match is accepted when
    cosine similarity reaches `face_threshold` / `voice_threshold`. With ``path=None`` the data is memory-only.
    """

    MAX_EMB = 12  # newest embedding samples kept per person and kind

    def __init__(self, path: str | Path | None, face_threshold: float = 0.40, voice_threshold: float = 0.50):
        self.path = Path(path) if path else None
        self.face_threshold, self.voice_threshold = face_threshold, voice_threshold
        self.people: dict[str, dict[str, list[list[float]]]] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if not isinstance(loaded, dict) or not all(isinstance(person, dict) for person in loaded.values()):
                    raise ValueError("not a people database")
                self.people = loaded
            except (OSError, ValueError) as error:
                # Never start empty on top of a file we could not read: the next save would destroy every stored
                # face and voice. Move it aside first (this raises if even that fails, so the server stops loudly).
                quarantine(self.path, error)
                self.people = {}

    def _save(self) -> None:
        """Write the database atomically (temp file + rename). Callers hold the lock."""
        if not self.path:
            return
        write_private_atomic(self.path, json.dumps(self.people))  # biometric templates: owner-only, never half-written

    def canonical(self, name: str) -> str | None:
        """Return the stored spelling of `name` (case-insensitive lookup), or None if unknown."""
        return next((known for known in self.people if known.lower() == name.lower()), None)

    def names(self) -> list[str]:
        """Return all known names, sorted."""
        return sorted(self.people)

    def summary(self) -> list[dict]:
        """Return ``[{"name", "face", "voice"}]`` with sample counts per person, for the UI."""
        return [{"name": name, "face": len(samples.get("face", [])), "voice": len(samples.get("voice", []))}
                for name, samples in sorted(self.people.items())]

    def _match(self, kind: str, embedding, threshold: float) -> tuple[str | None, float]:
        """Find the person whose `kind` sample is most similar to `embedding`.

        Returns ``(name, similarity)``, or ``(None, best similarity floored at 0)`` if below `threshold`.
        """
        query = _unit(embedding)
        best_name, best_similarity = None, -1.0
        with self._lock:
            for name, samples in self.people.items():
                for sample in samples.get(kind, []):
                    similarity = float(query @ np.asarray(sample, dtype=np.float32))  # cosine: both are unit length
                    if similarity > best_similarity:
                        best_name, best_similarity = name, similarity
        return (best_name, best_similarity) if best_similarity >= threshold else (None, max(best_similarity, 0.0))

    def match_face(self, embedding) -> tuple[str | None, float]:
        """Match a face embedding against everyone on record; see `_match`."""
        return self._match("face", embedding, self.face_threshold)

    def match_voice(self, embedding) -> tuple[str | None, float]:
        """Match a voice embedding against everyone on record; see `_match`."""
        return self._match("voice", embedding, self.voice_threshold)

    def add(self, name: str, kind: str, embedding) -> str:
        """Store an embedding sample ("face" or "voice") for `name`, creating the person if needed.

        Returns the stored spelling of the name (an existing person is matched case-insensitively).
        """
        with self._lock:
            name = next((known for known in self.people if known.lower() == name.lower()), name)
            samples = self.people.setdefault(name, {"face": [], "voice": []}).setdefault(kind, [])
            samples.append([round(float(x), 5) for x in _unit(embedding)])
            del samples[: max(0, len(samples) - self.MAX_EMB)]  # keep the newest
            self._save()
        return name

    def rename(self, old: str, new: str) -> str | None:
        """Rename a person, keeping their face/voice samples. If `new` already exists the two are merged."""
        with self._lock:
            key = next((n for n in self.people if n.lower() == old.lower()), None)
            if key is None:
                return None
            entry = self.people.pop(key)
            target = next((n for n in self.people if n.lower() == new.lower()), None)
            if target:  # merging into an existing person
                for kind in ("face", "voice"):
                    merged = self.people[target].setdefault(kind, []) + entry.get(kind, [])
                    self.people[target][kind] = merged[-self.MAX_EMB:]
                final = target
            else:
                self.people[new] = entry
                final = new
            self._save()
            return final

    def forget(self, name: str) -> bool:
        """Delete a person and all their samples. Returns False if the name is not on record."""
        with self._lock:
            key = next((n for n in self.people if n.lower() == name.lower()), None)
            if key:
                del self.people[key]
                self._save()
            return key is not None


# --------------------------------------------------------------------------- engines
@dataclass
class Detection:
    """One detected face: box is (x, y, width, height) in pixels; `raw` is the detector row used for alignment."""

    box: tuple[float, float, float, float]
    landmarks: np.ndarray  # (5,2): right eye, left eye, nose, right mouth corner, left mouth corner
    score: float
    raw: np.ndarray | None = None


def side_for_user(center_fraction: float) -> str:
    """Name the horizontal position of something at `center_fraction` (0 = left edge, 1 = right edge of the camera image).

    The browser sends the true (un-mirrored) camera image, so the left of the image is the RIGHT of a person facing the
    camera. People say "on my left" meaning their own side, so positions are reported that way: the swap is done here in
    code because language models cannot be trusted to do it reliably.
    """
    return "right" if center_fraction < 0.34 else "left" if center_fraction > 0.66 else "middle"


class FaceEngine:
    """OpenCV YuNet face detector plus SFace recognizer; models are downloaded and loaded lazily."""

    def __init__(self):
        self.det = self.rec = None  # detector and recognizer, set by load()

    def load(self) -> None:
        """Download (first run only) and initialise the detector and recognizer models."""
        from huggingface_hub import hf_hub_download
        detector_path = hf_hub_download("opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx")
        recognizer_path = hf_hub_download("opencv/face_recognition_sface", "face_recognition_sface_2021dec.onnx")
        # Positional args: input size (reset per frame in detect), score threshold 0.7, NMS threshold 0.3, top-k 5000.
        self.det = cv2.FaceDetectorYN.create(detector_path, "", (320, 320), 0.7, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(recognizer_path, "")

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        """Return all faces in a BGR frame (loads the models on first use)."""
        if self.det is None:
            self.load()
        height, width = bgr.shape[:2]
        self.det.setInputSize((width, height))
        _, rows = self.det.detect(bgr)
        # YuNet row layout: [x, y, w, h, 5 landmark (x, y) pairs, score].
        return [Detection(tuple(map(float, row[:4])), row[4:14].reshape(5, 2).astype(np.float32), float(row[14]), row)
                for row in (rows if rows is not None else [])]

    def embed(self, bgr: np.ndarray, detection: Detection) -> np.ndarray:
        """Return the unit-length SFace embedding of an aligned face crop."""
        return _unit(self.rec.feature(self.rec.alignCrop(bgr, detection.raw)))


class VoiceEngine:
    """SpeechBrain ECAPA speaker-embedding model (CPU), loaded lazily."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = str(cache_dir or Path.home() / ".cache" / "interactive_llm" / "ecapa")
        self.enc = None  # the speaker encoder, set by load()

    def load(self) -> None:
        """Load (downloading on first run) the speaker encoder."""
        from speechbrain.inference.speaker import EncoderClassifier
        self.enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                  savedir=self.cache_dir, run_opts={"device": "cpu"})

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """Speaker embedding for 16 kHz float32 audio, or None if too short to be reliable."""
        if audio is None or len(audio) < MIN_VOICE_S * 16000:
            return None
        import torch
        if self.enc is None:
            self.load()
        with torch.no_grad():
            return _unit(self.enc.encode_batch(torch.from_numpy(audio.astype(np.float32))[None]).squeeze().numpy())


# --------------------------------------------------------------------------- language
# Capitalised words that can follow "I'm" / "this is" but are not names ("I'm Tired", "This is Great").
STOP = {"The", "This", "That", "Here", "There", "Just", "Not", "Going", "Sorry", "Fine", "Good", "Great", "Okay",
        "Ok", "Well", "Really", "Very", "Back", "Done", "Ready", "Tired", "Happy", "Sure", "Glad", "Afraid", "Trying",
        "Looking", "Thinking", "Wondering", "Curious", "Interested", "Hungry", "Bored", "Busy", "Late", "Early",
        "Right", "Wrong", "So", "Also", "Still", "Now", "Then", "Today", "Yes", "No", "Hi", "Hello", "Hey", "Thanks",
        "Thank", "Wow", "Amazing", "Awesome", "Nice", "Cool", "Lost", "Stuck", "Confused", "Excited", "Worried"}
# Placeholder names such as "Person A" or "Guest 2": accepted as names, so people can be introduced before real names.
_PLACEHOLDER = (r"(?:person|guest|visitor|speaker|user)\s+(?:[a-z]|\d{1,2}|one|two|three|four|five|six|seven|eight"
                r"|nine|ten)\b")
# Regex group capturing a name: a placeholder or a single capitalisable word (2-21 chars, hyphens/apostrophes ok).
_NAME = r"(" + _PLACEHOLDER + r"|[A-Za-z][A-Za-z'\-]{1,20})"
# "I'm X" / "call me X": the speaker introduces themselves.
SELF_RE = re.compile(r"\b(?:i am|i'm|im|my name is|my name's|call me|it's me,?|it is me,?)\s+" + _NAME, re.I)
# "this is X" / "meet X": the speaker introduces someone else.
OTHER_RE = re.compile(r"\b(?:this is|that is|that's|meet|here is|here's|say hi to|say hello to|his name is|her name is|"
                      r"their name is|call him|call her|call them|this guy is|this woman is|this person is)\s+"
                      r"(?:(?:my|our|the)\s+\w+\s+)?" + _NAME, re.I)
_STRONG = re.compile(r"name(?:'s| is)|call (?:me|him|her|them)", re.I)  # explicit naming phrases
_PLACE_FULL = re.compile(_PLACEHOLDER, re.I)
# Lowercase words that must never become a name even after a strong naming phrase.
LOWER_STOP = {w.lower() for w in STOP} | {
    "a", "an", "it", "not", "going", "tired", "fine", "good", "here", "back", "later", "tomorrow", "tonight", "soon",
    "maybe", "sometime", "anytime", "please", "when", "if", "again", "crazy", "stupid", "lazy", "dumb", "silly",
    "anything", "something", "nothing", "whatever", "whenever", "right", "now", "today", "yesterday",
    "first", "last", "next", "once", "twice", "always", "never", "sir", "madam", "boss", "buddy", "friend", "mate",
    # connector words of the naming phrases themselves: "I am now called" (cut off) must not rename someone "Called"
    "called", "named", "known", "as", "by", "to", "into", "now", "actually", "really", "officially", "name", "names"}


@dataclass
class Intro:
    """One introduction found in an utterance."""

    kind: str  # "self" | "other"
    name: str
    pos: int = 0  # character offset in the utterance, used to process introductions in spoken order
    strong: bool = False  # said with an explicit naming phrase ("my name is", "call me")


_COMMON_VERBS = {"watching", "listening", "looking", "working", "trying", "thinking", "talking", "playing", "reading",
                 "eating", "cooking", "running", "sitting", "standing", "waiting", "walking", "driving", "learning",
                 "studying", "sleeping", "leaving", "coming", "doing", "being", "having", "getting", "making", "taking",
                 "feeling", "hoping", "wondering", "testing", "checking", "calling", "typing", "speaking", "texting"}


def _norm_name(raw: str, strong: bool) -> str | None:
    """Normalise a heard name, or None if it is probably not a name. `strong` = said after an explicit naming phrase."""
    raw = re.sub(r"['\u2019]s$", "", raw.strip())   # "Marven's" -> "Marven"
    if not raw:
        return None
    if _PLACE_FULL.fullmatch(raw):
        return " ".join(w.capitalize() for w in raw.split())
    if re.fullmatch(r"(?:[A-Za-z]-){2,}[A-Za-z]", raw):
        return None   # spelled-out letters ("that's M-A-R-V-I-N") are a spelling, never a person
    if not strong and raw.lower() in _COMMON_VERBS:
        return None   # "I'm Watching ..." is a verb, not a name; only an explicit "my name is Watching" may use one
    if raw[0].isupper():
        return raw if raw not in STOP and raw.lower() not in {"called", "named", "known", "as", "by", "to", "into"} else None
    if strong and raw.lower() not in LOWER_STOP:
        return raw.capitalize()
    return None


def _accept(text: str, match: re.Match) -> str | None:
    """Name from group 1 of an introduction match, or None if it is probably not a name ("I'm tired")."""
    return _norm_name(text[match.start(1):match.end(1)], bool(_STRONG.search(match.group(0))))


def parse_intros(text: str) -> list[Intro]:
    """Find self- and third-person introductions ("I'm Sam", "this is Priya") in order of appearance."""
    intros = []
    for kind, pattern in (("self", SELF_RE), ("other", OTHER_RE)):
        for match in pattern.finditer(text):
            name = _accept(text, match)
            if name:
                intros.append(Intro(kind, name, match.start(), bool(_STRONG.search(match.group(0)))))
    return sorted(intros, key=lambda intro: intro.pos)


_CHANGE = r"(?:now|actually|really|officially)"  # adverbs that signal a change rather than a first introduction
# Self-renames: "I'm now called X", "change my name to X", "I'm not Y, I'm X". Group 1 is the new name.
SELF_RENAME_RES = [
    re.compile(r"\b(?:i am|i'm|im|my name is|my name's)\s+" + _CHANGE + r"\s+(?:(?:called|named|known as)\s+)?" + _NAME, re.I),
    re.compile(r"\b(?:change|rename)\s+(?:my name|me|myself)\s+(?:to|as|into)\s+" + _NAME, re.I),
    re.compile(r"\b(?:i am|i'm|im)\s+not\s+[A-Za-z' ]{1,30}?[,.]?\s*(?:but\s+)?(?:i am|i'm|im|it's|it is)\s+" + _NAME, re.I),
]
# Renaming someone else: "rename X to Y", "X is actually Y". Group 1 is the old name, group 2 the new one.
THIRD_RENAME_RES = [
    re.compile(r"\b(?:rename|change)\s+(?:the name of\s+)?" + _NAME + r"\s+(?:to|as|into)\s+" + _NAME, re.I),
    re.compile(r"\b" + _NAME + r"\s+is\s+" + _CHANGE + r"\s+(?:(?:called|named|known as)\s+)?" + _NAME, re.I),
]

SPELL_RE = re.compile(r"\bspell(?:ing|ed|s|t)?\b|\bmisspell\w*", re.I)
# Three or more spelled-out letters ("M-A-R-V-I-N", "M A R V I N"): single letters separated by space/-/./,.
_LETTERS_RE = re.compile(r"(?<![A-Za-z'\u2019])(?:[A-Za-z](?![A-Za-z'\u2019])[\s\-.,]*){3,}")   # not the "s" of "it's"
_AFTER_BE_RE = re.compile(r"\b(?:be|is|as|spelled|spelt|with)\s+([A-Z][a-z]{2,20})\b")


def _letter_words(text: str) -> list[str]:
    """Words spelled out letter by letter in `text` ("M-A-R-V-I-N" -> "Marvin"), 3 to 20 letters each.

    Single letters that follow an apostrophe (the "s" of "it's") never start a run, and fillers such as "I, I, I"
    come out as junk words like "Iii", which callers compare against real names.
    """
    words = []
    for run in _LETTERS_RE.findall(text):
        letters = re.findall(r"[A-Za-z]", run)
        if 3 <= len(letters) <= 20:
            words.append("".join(letters).capitalize())
    return words


def parse_spelling(text: str, current: str | None = None) -> str | None:
    """A corrected spelling of a name: spelled-out letters ("M-A-R-V-I-N") or "spelling should be Marvin".

    Without the word "spell", bare letters count only when they spell a name close to `current` ("No, it's
    M-A-R-V-I-N" for a stored Marven), so acronyms such as "U-S-A" and fillers such as "I, I, I" are never a name.
    """
    words = _letter_words(text)
    close = next((w for w in words if current and (w.lower() == current.lower() or similar_name(current, w))), None)
    if not SPELL_RE.search(text):
        return close
    if words:
        best = close or max(words, key=len)
        # A word far from the stored name counts only when the sentence is about a name ("my name is spelled ...");
        # "Can you spell U-S-A for me" is not a respelling. With nobody on record the spelling is the introduction.
        return best if close or not current or _NAME_SPELL_RE.search(text) else None
    # Only the sentence that mentions spelling counts: "I can never spell it. Today is Tuesday" is not a new name.
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if SPELL_RE.search(sentence):
            written_name = _AFTER_BE_RE.search(sentence)
            if written_name:
                return written_name.group(1)
    return None


def similar_name(first: str, second: str) -> bool:
    """Different spellings that sound alike (Marven/Marvin, Marvins): probably speech-recognition noise."""
    return (first.lower() != second.lower()
            and difflib.SequenceMatcher(None, first.lower(), second.lower()).ratio() >= 0.8)


def parse_renames(text: str) -> list[tuple[str | None, str]]:
    """(old, new) pairs. old=None means "the speaker"."""
    renames: list[tuple[str | None, str]] = []
    for pattern in SELF_RENAME_RES:
        for match in pattern.finditer(text):
            new = _norm_name(text[match.start(1):match.end(1)], True)
            if new:
                renames.append((None, new))
    for pattern in THIRD_RENAME_RES:
        for match in pattern.finditer(text):
            old = text[match.start(1):match.end(1)].strip()
            new = _norm_name(text[match.start(2):match.end(2)], True)
            if new and old.lower() not in LOWER_STOP | {"name", "he", "she", "they", "one", "who", "mine"}:
                renames.append((old, new))
    return list(dict.fromkeys(renames))  # de-duplicate, keeping order


# Forgetting: a delete verb, a "wipe everything" phrase, and yes/no answers to confirmation questions.
_VERBS = r"(?:forget|remove|delete|erase)"
FORGET_VERB_RE = re.compile(r"\b" + _VERBS + r"\b", re.I)
WIPE_RE = re.compile(r"\b(?:forget|remove|delete|erase|wipe|clear)\b[^.!?;]{0,25}\b(?:everyone|everybody|everything|all (?:the |of the )?"
                     r"(?:people|persons|faces|voices)|all of (?:them|us))\b", re.I)
# A confirmation must be the WHOLE utterance ("yes", "yeah go ahead", "yes please"), never a word inside a longer
# sentence: "Are you sure?" and "I'm not sure" contain "sure" but are not a yes. Deleting people or quitting an app
# is too costly to guess at.
_YES_WORD = (r"(?:yes|yeah|yep|yup|yea|sure|ok|okay|confirm(?:ed)?|correct|affirmative|definitely|absolutely|"
             r"do it|go ahead|please do|that'?s right)")
YES_RE = re.compile(r"^\W*" + _YES_WORD + r"(?:\W+(?:" + _YES_WORD + r"|please|thanks|thank you))*\W*$", re.I)
# Phrases that mean "I am trying to name / rename someone", even if the exact name could not be parsed.
NAMING_INTENT_RE = re.compile(
    r"\bcall me\b|\bmy name(?:'s| is)\b|\b(?:change|rename|update)\b[^.!?]{0,20}\bname\b|\brename\b"
    r"|\b(?:i am|i'm|im)\s+(?:now\s+|actually\s+|really\s+)?(?:called|named|known as)\b|\bgo(?:es)? by\b"
    r"|\bnames? (?:is|are) now\b|\bintroduc\w+\b|\bspell(?:ing|ed|s|t)?\b|\bmisspell\w*"
    r"|\bname\b[^.!?]{0,30}\b(?:wrong|incorrect|should be)\b", re.I)
# Anything that sounds like a refusal or doubt wins over a yes: "yeah no, don't", "wait", "I'm not sure".
NO_RE = re.compile(r"\b(?:no|nope|nah|cancel|never ?mind|don'?t|do not|stop|wait|hold on|hang on|not yet|not sure|unsure|"
                   r"are you sure)\b", re.I)


def is_yes(text: str) -> bool:
    """True only if the whole utterance is a plain yes and nothing in it refuses or doubts."""
    return bool(YES_RE.match(text.strip())) and not NO_RE.search(text)


def is_no(text: str) -> bool:
    """True if the utterance refuses, cancels or doubts ("no", "cancel", "wait", "are you sure?")."""
    return bool(NO_RE.search(text))


def same_speaker(asker: str | None, replier: str | None) -> bool:
    """A pending yes/no may only be answered by the person who was asked. Unknown on either side is allowed,
    because voice recognition is not always sure who spoke and typed input has no speaker."""
    return asker is None or replier is None or asker.lower() == replier.lower()
# Words before "forget" that make it ordinary speech, not a command ("I forget", "never forget").
_NOT_A_COMMAND_AFTER = {"i", "we", "they", "he", "she", "often", "always", "never", "sometimes", "dont", "didnt", "wont", "cant"}
# First word after the verb that marks an idiom ("forget it", "forget about everything").
_IDIOM_START = {"it", "that", "this", "what", "everything", "everyone", "everybody", "all"}
# Filler words skipped when collecting the names to forget from a command.
_SKIP_TOKENS = {"a", "an", "the", "my", "about", "from", "your", "memory", "records", "record", "database", "list",
                "please", "it", "them", "those", "these", "and", "or", "also", "then", "as", "of", "on", "in", "to", "for",
                "all", "everything", "everyone", "everybody", "anyone", "that", "this", "now", "too", "again", "both"}


def _command_spans(text: str, trim_from: bool = True) -> list[str]:
    """Text following each real forget/remove/delete command (not "I forget my keys" or "forget it").

    Normally the span stops at "from" ("forget Sam from your memory" -> "Sam"); `trim_from=False` keeps the whole sentence.
    """
    spans = []
    for verb_match in FORGET_VERB_RE.finditer(text):
        before = re.findall(r"[a-z']+", text[: verb_match.start()].lower())
        if before and before[-1].replace("'", "") in _NOT_A_COMMAND_AFTER:
            continue
        span = re.split(r"[.!?;\n]", text[verb_match.end():], maxsplit=1)[0]  # up to the end of the sentence
        if trim_from:
            span = re.split(r"\bfrom\b", span, maxsplit=1, flags=re.I)[0]
        words = re.findall(r"[a-z']+", span.lower())
        if words[:1] == ["about"]:
            words = words[1:]
        first = (words or [""])[0]
        if first in _IDIOM_START:
            continue
        spans.append(span)
    return spans


def has_forget_command(text: str) -> bool:
    """True if `text` contains a real forget/remove/delete command."""
    return bool(_command_spans(text))


def parse_wipe(text: str) -> bool:
    """True if `text` asks to forget everyone ("forget everybody", "wipe all the people")."""
    def preceded_by_non_command_word(match: re.Match) -> bool:
        words_before = re.findall(r"[a-z']+", text[: match.start()].lower()) or [""]
        return words_before[-1].replace("'", "") in _NOT_A_COMMAND_AFTER

    return any(not preceded_by_non_command_word(wipe_match) for wipe_match in WIPE_RE.finditer(text))


def is_clean_forget(text: str, known: list[str] | None = None) -> bool:
    """True if every forget command in `text` consists only of names and filler ("forget Sam and Priya please").

    A close-but-not-exact name is only trusted to mean someone on record in such a sentence. In a longer one
    ("delete the Daniel Craig playlist") it is more likely ordinary speech, so nothing is deleted on a guess.
    """
    known_lower = [name.lower() for name in (known or [])]
    spans = _command_spans(text, trim_from=False)       # the whole sentence, so "... from the shopping list" counts too
    for span in spans:
        for token in re.findall(r"[A-Za-z][A-Za-z'\-]*", re.sub(_PLACEHOLDER, " ", span, flags=re.I)):
            lowered = token.lower()
            if (lowered in _SKIP_TOKENS or lowered == "me" or (token[0].isupper() and token not in STOP)
                    or difflib.get_close_matches(lowered, known_lower, n=1, cutoff=0.7)):
                continue
            return False
    return bool(spans)


def parse_forget(text: str, known: list[str] | None = None) -> list[str]:
    """Names (or "me") the user wants deleted. Capitalised words always count; lowercase words only
    when they look like someone already on record (`known`), so "forget lena and mena" works."""
    known_lower = [name.lower() for name in (known or [])]
    targets: list[str] = []
    for span in _command_spans(text):
        targets += [" ".join(w.capitalize() for w in placeholder.group().split())
                    for placeholder in re.finditer(_PLACEHOLDER, span, re.I)]
        rest = re.sub(_PLACEHOLDER, " ", span, flags=re.I)
        for token in re.findall(r"[A-Za-z][A-Za-z'\-]*", rest):
            lowered = token.lower()
            if lowered == "me":
                targets.append("me")
            elif lowered in _SKIP_TOKENS:
                continue
            elif token[0].isupper() and token not in STOP:
                targets.append(token)
            elif known_lower and difflib.get_close_matches(lowered, known_lower, n=1, cutoff=0.7):
                targets.append(token)
    return list(dict.fromkeys(targets))  # de-duplicate, keeping order


# --------------------------------------------------------------------------- tracking
@dataclass
class Track:
    """A face followed across frames, with its recognised name and recent mouth-movement history."""

    id: int
    box: tuple[float, float, float, float]
    last_seen: float
    name: str | None = None
    score: float = 0.0
    emb: np.ndarray | None = None  # latest face embedding
    emb_t: float = 0.0  # when `emb` was computed (monotonic seconds)
    mouth: np.ndarray | None = None  # normalised mouth-region patch from the previous frame
    # (timestamp, mouth change since the previous frame) samples; used to guess who is talking.
    motion: deque = field(default_factory=lambda: deque(maxlen=80))

    @property
    def cx(self) -> float:
        """Horizontal centre of the face box in pixels."""
        return self.box[0] + self.box[2] / 2


@dataclass
class Speaker:
    """Who is talking right now, and how we know."""

    name: str | None = None
    how: str | None = None          # voice | only face | lips | None
    voice_score: float = 0.0  # similarity of the voice match
    voice_name: str | None = None  # who the voice matched, before any face fallback
    track: Track | None = None  # the on-screen face believed to be speaking


def _mouth_patch(gray: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
    """Crop the mouth region of a face to a 24x16 brightness-normalised patch, or None if it is too small/off-frame.

    The crop is sized from the distance between the mouth corners (landmarks 3 and 4, at least 6 px), so it
    scales with the face. Frame-to-frame differences of these patches measure lip movement.
    """
    corner_a, corner_b = landmarks[3], landmarks[4]
    center, mouth_width = (corner_a + corner_b) / 2, max(float(np.linalg.norm(corner_a - corner_b)), 6.0)
    x0, x1 = int(center[0] - mouth_width * 0.8), int(center[0] + mouth_width * 0.8)
    y0, y1 = int(center[1] - mouth_width * 0.5), int(center[1] + mouth_width * 0.7)  # extra room below for jaw drop
    frame_height, frame_width = gray.shape
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, frame_width), min(y1, frame_height)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    patch = cv2.resize(gray[y0:y1, x0:x1], (24, 16)).astype(np.float32)
    return (patch - patch.mean()) / (patch.std() + 1e-3)  # lighting-invariant


# Async callback ``(annotated JPEG, utterance text, candidate track ids) -> chosen id or None`` that asks a vision
# model which numbered face is being introduced.
VlmPick = Callable[[bytes, str, list[int]], Awaitable[int | None]]


SPELL_CHECK_TTL_S = 120.0     # how long "is that spelling right?" stays open after a spoken name was learned
SPELL_COLLECT_TTL_S = 60.0    # how long we keep waiting for the next letters while someone spells their name
_DONE_RE = re.compile(r"^\W*(?:that'?s (?:it|all|everything|the name)|that is (?:it|all)|done|finished|all done)\W*$", re.I)
_CANCEL_RE = re.compile(r"\b(?:never ?mind|forget it|cancel|leave it|skip it|doesn'?t matter)\b", re.I)
_FRAGMENT_RE = re.compile(r"(?:\W*\b[A-Za-z]\b\W*)+")      # only single letters: "M-E-R", "V I N."
_PREFIXED_NAME_RE = re.compile(r"^\W*(?:(?:no|nope|wait|actually)\W+)?(?:it'?s|it is|that'?s|that is|the spelling is|it should be|"
                               r"spelled)\s+([A-Za-z][A-Za-z'-]{1,19})\W*$", re.I)
_NAME_SPELL_RE = re.compile(r"\b(?:spelled|spelt|misspell\w*|my name|the name|name is|spelling (?:is|should be|of))\b", re.I)
_BARE_NAME_RE = re.compile(r"^\W*([A-Za-z][A-Za-z'-]{1,19})\W*$")


@dataclass
class _SpellDialog:
    """The open question "is the spelling of <name> right?", so later answers are read in that light.

    The language model has the whole chat but may not touch names; this code may, but sees one utterance at a time.
    Remembering what was just asked is how "No." / "M-E-R" / "V-I-N." become one correction.
    """

    name: str
    until: float
    mode: str = "check"        # "check": waiting for yes/no; "collect": gathering letters
    letters: str = ""


class PeopleManager:
    """Tracks faces across frames, works out who is speaking, and applies naming/renaming/forgetting commands.

    Every command outcome is recorded both as a GUI event (`drain_events`) and as a bracketed [System] note that
    is passed to the LLM so it only claims what really happened.
    """

    def __init__(self, db: PeopleDB, faces: FaceEngine, auto_learn: bool = True):
        self.db, self.faces, self.auto_learn = db, faces, auto_learn
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.frame_size = (1, 1)  # (width, height) of the latest camera frame
        self.last_bgr: np.ndarray | None = None
        self.speaker_track_id: int | None = None  # track highlighted as "speaking" in the overlay
        self.speaker_until = 0.0  # monotonic time until which that highlight holds
        self._learn_t = 0.0  # last time a voice sample was refreshed automatically (rate limit)
        self._pending: tuple[str, str, float] | None = None  # (old, new, expires): rename awaiting yes/no
        self._spell: _SpellDialog | None = None              # open "is that spelling right?" conversation
        self.last_named: str | None = None  # most recently learned/renamed person ("the spelling should be ...")
        self._spoken = False                # the current utterance came from speech recognition, not typing
        self.events: list[dict] = []  # outcomes of this turn's naming/forgetting, drained by the server for the GUI
        self._wipe_until = 0.0  # a 'forget everyone' request is armed until this time, awaiting confirmation
        self._wipe_by: str | None = None  # who asked, so only they can confirm
        self.fuzzy: dict[str, str] = {}   # names taken by a close (not exact) match in the last forget/rename: matched -> heard

    # ---- per-frame ---------------------------------------------------------------
    def update(self, bgr: np.ndarray, now: float | None = None) -> None:
        """Detect faces in a frame and match them to existing tracks (or start new ones).

        A detection joins the nearest unclaimed track whose centre is within 0.75 of the face width; otherwise
        it starts a new track. Face recognition is re-run on a track every `REID_S` seconds.
        """
        now = time.monotonic() if now is None else now
        self.last_bgr = bgr
        frame_height, frame_width = bgr.shape[:2]
        self.frame_size = (frame_width, frame_height)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        unclaimed = {i: track for i, track in self.tracks.items() if now - track.last_seen <= TRACK_TTL_S}
        for detection in sorted(self.faces.detect(bgr), key=lambda found: -found.score):  # strongest faces claim first
            center_x = detection.box[0] + detection.box[2] / 2
            center_y = detection.box[1] + detection.box[3] / 2
            nearest, nearest_distance = None, 1e9
            for track in unclaimed.values():
                distance = float(np.hypot(center_x - track.cx, center_y - (track.box[1] + track.box[3] / 2)))
                if distance < 0.75 * max(detection.box[2], track.box[2]) and distance < nearest_distance:
                    nearest, nearest_distance = track, distance
            if nearest is None:
                nearest = Track(self._next_id, detection.box, now)
                self._next_id += 1
                self.tracks[nearest.id] = nearest
            else:
                unclaimed.pop(nearest.id, None)
            nearest.box, nearest.last_seen = detection.box, now
            patch = _mouth_patch(gray, detection.landmarks)
            if patch is not None and nearest.mouth is not None:
                nearest.motion.append((now, float(np.mean(np.abs(patch - nearest.mouth)))))
            nearest.mouth = patch
            if nearest.emb is None or now - nearest.emb_t > REID_S:
                nearest.emb, nearest.emb_t = self.faces.embed(bgr, detection), now
                name, score = self.db.match_face(nearest.emb)
                nearest.name, nearest.score = name, score
        for track_id in [i for i, track in self.tracks.items() if now - track.last_seen > TRACK_TTL_S]:
            del self.tracks[track_id]

    def visible(self, now: float | None = None) -> list[Track]:
        """Return tracks seen within `VISIBLE_S`, ordered left to right."""
        now = time.monotonic() if now is None else now
        return sorted((track for track in self.tracks.values() if now - track.last_seen <= VISIBLE_S),
                      key=lambda track: track.cx)

    # ---- who is speaking ---------------------------------------------------------
    def _lip_pick(self, tracks: list[Track], t0: float, t1: float) -> Track | None:
        """Pick the track whose mouth moved clearly most between times t0 and t1, or None if it is not clear-cut.

        Needs the top mover to exceed an absolute floor (0.2) and be 1.4x the runner-up.
        """
        motion_scores = []
        for track in tracks:
            movements = [amount for (timestamp, amount) in track.motion if t0 <= timestamp <= t1]
            motion_scores.append((float(np.mean(movements)) if len(movements) >= 2 else 0.0, track))
        motion_scores.sort(key=lambda entry: -entry[0])
        if (len(motion_scores) >= 2 and motion_scores[0][0] > 0.2
                and motion_scores[0][0] >= 1.4 * max(motion_scores[1][0], 1e-3)):
            return motion_scores[0][1]
        return None

    def identify_speaker(self, voice_embedding, t0: float, t1: float, now: float | None = None) -> Speaker:
        """Decide who spoke during the window t0..t1 (monotonic seconds).

        Preference order: voice match (then pair it with a matching on-screen face), the only visible face, then
        lip movement among several faces. `how` records which rule fired.
        """
        now = time.monotonic() if now is None else now
        tracks = self.visible(now)
        speaker = Speaker()
        if voice_embedding is not None:
            speaker.voice_name, speaker.voice_score = self.db.match_voice(voice_embedding)
        if speaker.voice_name:
            speaker.how = "voice"
            speaker.track = next((track for track in tracks if track.name == speaker.voice_name), None)
            unnamed = [track for track in tracks if track.name is None]
            if speaker.track is None and len(unnamed) == 1 and len(tracks) == 1:
                speaker.track = unnamed[0]
        elif len(tracks) == 1:
            speaker.track, speaker.how = tracks[0], "only face"
        elif len(tracks) > 1:
            speaker.track = self._lip_pick(tracks, t0, t1)
            speaker.how = "lips" if speaker.track else None
        speaker.name = speaker.voice_name or (speaker.track.name if speaker.track else None)
        if speaker.track:
            self.speaker_track_id, self.speaker_until = speaker.track.id, now + 3.0  # keep the "speaking" highlight for 3 s
        return speaker

    # ---- learning ----------------------------------------------------------------
    @staticmethod
    def _lips_moving(track: Track, t0: float | None, t1: float | None) -> bool:
        """True if this face's mouth clearly moved between t0 and t1 (the same floor `_lip_pick` uses)."""
        if t0 is None or t1 is None:
            return False
        movements = [amount for (timestamp, amount) in track.motion if t0 <= timestamp <= t1]
        return len(movements) >= 2 and float(np.mean(movements)) > 0.2

    def co_learn(self, speaker: Speaker, voice_embedding, dur: float, now: float | None = None,
                 t0: float | None = None, t1: float | None = None) -> bool:
        """Quietly strengthen face<->voice links when exactly one face is in view.

        `dur` is the utterance length in seconds; `t0`..`t1` is the time window of the utterance. Pairing a new voice
        with a face, or a new face with a voice, is only done when that face's lips moved during the utterance:
        otherwise a visitor, a TV or someone off-camera speaking while the owner sits still would be stored as the
        owner. Returns True if a sample was added to the database.
        """
        now = time.monotonic() if now is None else now
        tracks = self.visible(now)
        if not self.auto_learn or voice_embedding is None or len(tracks) != 1:
            return False
        only_track = tracks[0]
        speaking = self._lips_moving(only_track, t0, t1)
        if speaking and only_track.name and only_track.score >= 0.5 and speaker.voice_name is None and dur >= 1.5:
            self.db.add(only_track.name, "voice", voice_embedding)          # known face, unknown voice -> learn voice
            self._event("info", f"Learned {only_track.name}'s voice automatically")
            return True
        if (speaking and speaker.voice_name and speaker.voice_score >= 0.65 and only_track.name is None
                and only_track.emb is not None):
            self.db.add(speaker.voice_name, "face", only_track.emb)       # known voice, unknown face -> learn face
            only_track.name, only_track.score = speaker.voice_name, 1.0
            self._event("info", f"Learned {speaker.voice_name}'s face automatically")
            return True
        # Both already known and agreeing, but only a middling voice score: add a fresh sample, at most every 30 s.
        if (speaker.voice_name and only_track.name == speaker.voice_name and 0.5 <= speaker.voice_score < 0.8
                and now - self._learn_t > 30):
            self.db.add(speaker.voice_name, "voice", voice_embedding)  # refresh with a new sample
            self._learn_t = now
            return True
        return False

    def annotated_jpeg(self, candidates: list[Track]) -> bytes:
        """Return the latest frame as JPEG with each candidate face boxed in yellow and labelled with its track id."""
        image = self.last_bgr.copy()
        for track in candidates:
            x, y, w, h = map(int, track.box)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 3)
            cv2.putText(image, str(track.id), (x + 4, max(y - 8, 24)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    # ---- outcomes: every naming/renaming/forgetting result is an event for the GUI plus a note for the LLM ------
    def _event(self, kind: str, text: str) -> None:
        """Queue a GUI event (kind: learned | renamed | forgot | info | needs_input | failed)."""
        self.events.append({"kind": kind, "text": text})

    def drain_events(self) -> list[dict]:
        """Return and clear the queued GUI events."""
        drained, self.events = self.events, []
        return drained

    def _out(self, kind: str, event: str, note: str) -> str:
        """Record a GUI event and return the LLM note, made explicit that nothing happened for failed/needs_input.

        `kind`: learned | renamed | forgot | info (done or neutral) | needs_input | failed (NOT done).
        """
        self._event(kind, event)
        if kind in ("failed", "needs_input"):
            note = note[:-1].rstrip() + " Do NOT say or imply that it was done.]"
        return note

    # ---- forgetting --------------------------------------------------------------
    def _spells_a_known_name(self, text: str) -> bool:
        """True if letters spelled out in `text` form a name close to someone on record (not "I, I, I" or "U-S-A")."""
        return any(word.lower() == name.lower() or similar_name(name, word)
                   for word in _letter_words(text) for name in self.db.names())

    def needs_processing(self, text: str) -> bool:
        """Cheap check: does `text` possibly contain a naming/rename/forget command or a reply to a pending question?"""
        pending = self._pending is not None and self._pending[2] > time.monotonic()
        return bool(parse_intros(text) or parse_renames(text) or has_forget_command(text) or parse_wipe(text)
                    or NAMING_INTENT_RE.search(text)
                    or self._spells_a_known_name(text)                 # "No, it's M-A-R-V-I-N": maybe a respelling
                    or self._dialog_input(text) is not None            # an answer to the open spelling question
                    or (self._wipe_until > time.monotonic() and (is_yes(text) or is_no(text)))
                    or (pending and (is_yes(text) or is_no(text))))

    def _drop(self, name: str, speaker: Speaker) -> None:
        """Delete a person from the database and clear their name from live tracks and the current speaker."""
        self.db.forget(name)
        if self._spell and self._spell.name.lower() == name.lower():
            self._spell = None
        for track in self.tracks.values():
            if track.name and track.name.lower() == name.lower():
                track.name, track.score = None, 0.0
        if speaker.name and speaker.name.lower() == name.lower():
            speaker.name = None

    def _resolve(self, tokens: list[str], speaker: Speaker, allow_fuzzy: bool = True):
        """Map spoken names (possibly misheard) onto people on record: exact, then a unique close match.

        "me" means the current speaker. Returns ``(found names, unknown tokens, ambiguous [(token, options)])``.
        A close match (similarity >= 0.7) is taken if it is the only one or leads the next by at least 0.1, but only
        when `allow_fuzzy`; otherwise it is reported as ambiguous so nothing is deleted on a guess. Close matches that
        were taken are remembered in ``self.fuzzy`` (matched name -> what was heard).
        """
        self.fuzzy = {}
        names = self.db.names()
        found: list[str] = []
        missing: list[str] = []
        ambiguous: list[tuple[str, list[str]]] = []
        for token in tokens:
            if token.lower() == "me":
                (found.append(speaker.name) if speaker.name else missing.append("me"))
                continue
            exact = self.db.canonical(token)
            if exact:
                found.append(exact)
                continue
            scored = sorted(((difflib.SequenceMatcher(None, token.lower(), known.lower()).ratio(), known)
                             for known in names), reverse=True)
            close = [(similarity, known) for similarity, known in scored if similarity >= 0.7]
            if allow_fuzzy and (len(close) == 1 or (len(close) > 1 and close[0][0] - close[1][0] >= 0.1)):
                found.append(close[0][1])
                self.fuzzy[close[0][1]] = token
            elif close:
                ambiguous.append((token, [known for _, known in close]))
            else:
                missing.append(token)
        return list(dict.fromkeys(found)), missing, ambiguous

    # ---- renaming ----------------------------------------------------------------
    def _spell_back(self, name: str, arm: bool = True) -> str:
        """Names heard by speech recognition may be misspelled, so have the assistant spell it back.

        `arm`: also remember the question, so the answer ("No", then the letters) is handled by code, not left to chat.
        """
        if not self._spoken or _PLACE_FULL.fullmatch(name):   # typed, or a placeholder like "Person A": no need
            return ""
        if arm:
            self._spell = _SpellDialog(name, time.monotonic() + SPELL_CHECK_TTL_S)
        return (f" This name was heard through speech recognition: spell it back letter by letter "
                f"({'-'.join(name.upper())}) and ask if that spelling is right.")

    def _do_rename(self, old: str, new: str, speaker: Speaker, spell: bool = True) -> str:
        """Rename `old` to `new` everywhere and return the LLM note; `spell` asks the LLM to spell the name back."""
        final = self.db.rename(old, new)
        if final is None:
            return self._out("failed", f"Couldn't rename {old}: not on record",
                             f"[System: nothing changed: nobody named '{old}' is on record. Tell the user.]")
        for track in self.tracks.values():
            if track.name and track.name.lower() == old.lower():
                track.name = final
        speaker.name = final if (speaker.name and speaker.name.lower() == old.lower()) else speaker.name
        self._pending = None
        self._spell = None
        self.last_named = final
        return self._out("renamed", f"Renamed {old} → {final}",
                         f"[System: you renamed {old} to {final} and kept everything you knew about them. From now on "
                         f"call them {final}, never {old}. Confirm briefly."
                         + (self._spell_back(final, arm=bool(speaker.name and speaker.name.lower() == final.lower()))
                            if spell else "") + "]")

    async def _handle_renames(self, text: str, speaker: Speaker, voice_embedding, vlm_pick) -> tuple[list[str], set[str]]:
        """Handle explicit respelling and rename phrases.

        Returns ``(LLM notes, lowercase new names already handled)``; the latter stops the same words being
        treated again as a fresh introduction.
        """
        notes: list[str] = []
        handled: set[str] = set()
        target = speaker.name or self.last_named
        spelling = parse_spelling(text, self.db.canonical(target) if target else None)
        if spelling:                                          # "spelling should be M-A-R-V-I-N": explicit re-spelling
            self._spell = None
            handled.add(spelling.lower())
            if target:
                handled.add(target.lower())      # "I'm Marven, spelled M-A-R-V-I-N": the name that was heard is dealt with too
            intros = [intro for intro in parse_intros(text) if intro.kind == "self"]
            if (target is None or self.db.canonical(target) is None) and intros:
                # "Hi, I'm Marven, spelled M-A-R-V-I-N" from someone not on record: introduce them with the spelling
                handled.update(intro.name.lower() for intro in intros)
                notes.append(await self._introduce(Intro("self", spelling, intros[0].pos, strong=True), speaker,
                                                   voice_embedding, text, vlm_pick))
            elif target is None or self.db.canonical(target) is None:
                notes.append(self._out("failed", f"Couldn't change the spelling to {spelling}: don't know whose name",
                                       f"[System: the user gave a spelling ({spelling}) but you do not know whose name it "
                                       f"is, so nothing was changed. Ask whose name to respell.]"))
            elif self.db.canonical(target) == spelling:
                notes.append(self._out("info", f"{target} is already spelled {spelling}",
                                       f"[System: {target} is already spelled exactly {spelling}; nothing to change. Say so.]"))
            else:
                current = self.db.canonical(target)
                if _PLACE_FULL.fullmatch(current) or similar_name(current, spelling):    # a respelling of the same name
                    notes.append(self._do_rename(current, spelling, speaker, spell=False))
                else:                                          # a different name: never rename a stored person unasked
                    self._pending = (current, spelling, time.monotonic() + 60)
                    notes.append(self._out("needs_input", f"Rename {current} → {spelling}? Say yes or no",
                                           f"[System: the user gave a spelling ({spelling}) that is not close to {current}. "
                                           f"Nothing changed yet. Ask whether to rename {current} to {spelling}; they can answer "
                                           f"yes or no.]"))
            return notes, handled
        for old, new in parse_renames(text):
            handled.add(new.lower())
            if old is None:                                   # "I am now called X": rename the speaker
                if speaker.name and self._spoken and similar_name(speaker.name, new):
                    notes.append(self._variant_note(speaker.name, new))
                elif speaker.name:
                    notes.append(self._do_rename(speaker.name, new, speaker))
                else:                                          # nobody known to rename -> it is an introduction
                    notes.append(await self._introduce(Intro("self", new, strong=True), speaker, voice_embedding, text, vlm_pick))
            else:                                              # "rename Person A to X" / "Person A is actually X"
                found, _, ambiguous = self._resolve([old], speaker)
                if found:
                    notes.append(self._do_rename(found[0], new, speaker))
                elif ambiguous:
                    notes.append(self._out("needs_input", f"'{old}' could mean {' or '.join(ambiguous[0][1])}",
                                           f"[System: '{old}' could mean {' or '.join(ambiguous[0][1])}; nothing was "
                                           f"renamed. Ask which one they mean.]"))
                else:
                    notes.append(self._out("failed", f"Couldn't rename '{old}': nobody by that name",
                                           f"[System: nothing was renamed: nobody named '{old}' is on record. People on "
                                           f"record: {', '.join(self.db.names()) or 'nobody'}. Tell the user.]"))
        return notes, handled

    def _variant_note(self, current_name: str, heard: str) -> str:
        """Note for when speech recognition heard a near-identical spelling of an existing name: keep the old one."""
        return self._out("needs_input", f"Heard '{heard}', kept '{current_name}'. Spell it letter by letter or type it to change",
                         f"[System: speech recognition heard '{heard}', which sounds like {current_name} (already on record), so "
                         f"nothing was changed and you keep calling them {current_name}. If they really want the spelling '{heard}' "
                         f"they can spell it out letter by letter or type it.]")

    def _dialog_input(self, text: str) -> tuple[str, str] | None:
        """Classify `text` as an answer to the open spelling question: (kind, value) or None if it is not one.

        Kinds: yes, no, cancel, done, fragment (letters so far), word (a whole spelling or name). A name said with
        the answer is tried before a bare no, so "No, it's Marvin" gives the name instead of asking again.
        """
        dialog = self._spell
        if dialog is None or dialog.until <= time.monotonic() or not text.strip():
            return None
        text = text.strip()
        if _CANCEL_RE.search(text):
            return "cancel", ""
        if dialog.mode == "collect" and _DONE_RE.match(text):
            return ("done", "") if dialog.letters else None
        # Text made only of single letters is a piece of a spelling; a spelled word inside a sentence is a whole name.
        if _FRAGMENT_RE.fullmatch(text):
            letters = "".join(re.findall(r"[A-Za-z]", text)).upper()
            if len(set(letters)) == 1 and len(letters) > 1:
                return None                                         # "I, I, I": a stutter, not letters of a name
            if dialog.mode == "check" and len(letters) < 2:
                return None                                         # a lone "I." while only a yes or no is expected
            return "fragment", letters
        words = [word for word in _letter_words(text) if len(set(word.lower())) >= 2]   # "I, I, I" is not a spelling
        if words:
            return "word", max(words, key=len)
        named = _PREFIXED_NAME_RE.match(text)
        if named and named.group(1).lower() not in LOWER_STOP:
            candidate = named.group(1)
            if candidate[0].isupper() or similar_name(dialog.name, candidate):    # "it's wrong" is not a name
                return "word", candidate.capitalize()
        if is_yes(text):
            return ("yes", "") if dialog.mode == "check" else None   # "okay, thanks" while collecting is plain chat
        if is_no(text):
            return "no", ""
        bare = _BARE_NAME_RE.match(text)
        if bare and (bare.group(1).lower() == dialog.name.lower() or similar_name(dialog.name, bare.group(1))):
            return "word", bare.group(1).capitalize()      # a lone word counts only when it is close to the name
        return None

    def _handle_spell_dialog(self, text: str, speaker: Speaker) -> list[str] | None:
        """Read `text` as an answer to "is that spelling right?"; None if it is not one (normal handling continues)."""
        dialog = self._spell
        answer = self._dialog_input(text)
        if dialog is None or answer is None or not same_speaker(dialog.name, speaker.name):
            return None
        stored = self.db.canonical(dialog.name)
        if stored is None:                                     # the person was deleted meanwhile
            self._spell = None
            return None
        kind, value = answer
        now = time.monotonic()
        if kind == "yes":
            self._spell = None
            return [self._out("info", f"Spelling of {stored} confirmed",
                              f"[System: they confirmed {stored} is spelled right. Acknowledge in a few words.]")]
        if kind == "cancel":
            self._spell = None
            return [self._out("info", f"Spelling change for {stored} cancelled",
                              f"[System: they dropped the spelling change; {stored} is unchanged. Say so briefly.]")]
        if kind == "no":
            dialog.mode, dialog.letters, dialog.until = "collect", "", now + SPELL_COLLECT_TTL_S
            return [self._out("needs_input", f"Waiting for the right spelling of {stored}",
                              f"[System: they said the spelling of {stored} is wrong. NOTHING is changed yet. Ask them to say "
                              f"the right spelling; letter by letter is fine and pauses are fine. Do not explain how "
                              f"names work and do not suggest special phrases.]")]
        if kind == "fragment":
            dialog.mode, dialog.until = "collect", now + SPELL_COLLECT_TTL_S
            dialog.letters += value
            word = dialog.letters.capitalize()
            if len(word) >= 3 and (word.lower() == stored.lower() or similar_name(stored, word)):
                return [self._finish_spelling(stored, word, speaker)]
            return [self._out("needs_input", f"Heard {'-'.join(dialog.letters)} so far, keep going",
                              f"[System: they are spelling their name; the letters so far are {'-'.join(dialog.letters)}. "
                              f"NOTHING is changed yet. Reply with only 'Go on' or 'Mm-hm'. Do not guess or repeat the name, "
                              f"and do not say it was updated.]")]
        if kind == "done":
            return [self._apply_spelling(stored, dialog.letters.capitalize(), speaker)]
        return [self._apply_spelling(stored, value, speaker)]               # a whole spelling or name in one go

    def _finish_spelling(self, stored: str, word: str, speaker: Speaker) -> str:
        """Apply a spelling the user finished giving, and close the question."""
        self._spell = None
        return self._do_rename(stored, word, speaker, spell=False)

    def _apply_spelling(self, stored: str, word: str, speaker: Speaker) -> str:
        """Same rule for every way of finishing: a near-identical respelling applies; a different name asks first."""
        if word.lower() == stored.lower():
            self._spell = None
            return self._out("info", f"{stored} is already spelled {word}",
                             f"[System: {stored} is already spelled exactly {word}; nothing to change. Say so.]")
        if _PLACE_FULL.fullmatch(stored) or similar_name(stored, word):
            return self._finish_spelling(stored, word, speaker)
        self._spell = None
        self._pending = (stored, word, time.monotonic() + 60)
        return self._out("needs_input", f"Rename {stored} → {word}? Say yes or no",
                         f"[System: they gave {word}, which is not close to {stored}. Nothing changed yet. Ask whether to "
                         f"rename {stored} to {word}; they can answer yes or no.]")

    def _handle_pending(self, text: str, speaker: Speaker) -> list[str] | None:
        """Answer a pending "rename X to Y?" question with yes/no from `text`; None if nothing was pending or answered."""
        if self._pending is None:
            return None
        old, new, until = self._pending
        if until <= time.monotonic():
            self._pending = None
            return None
        if not same_speaker(old, speaker.name):      # someone else answering for them does not count
            return None
        if is_yes(text):
            return [self._do_rename(old, new, speaker)]
        if is_no(text):
            self._pending = None
            return [self._out("info", f"Rename {old} → {new} cancelled",
                              f"[System: the rename of {old} to {new} was cancelled; keep calling them {old}. Say so "
                              f"briefly.]")]
        return None

    def _handle_forget(self, text: str, speaker: Speaker) -> list[str]:
        """Handle forget commands, including the two-step "forget everyone" (ask, then confirm within 30 s)."""
        notes: list[str] = []
        now = time.monotonic()
        armed = self._wipe_until > now and not parse_wipe(text) and same_speaker(self._wipe_by, speaker.name)
        if armed and is_no(text):
            self._wipe_until, self._wipe_by = 0.0, None
            return [self._out("info", "Forget everyone cancelled",
                              "[System: the request to forget everyone was cancelled; nobody was deleted. Say so briefly.]")]
        if armed and is_yes(text):
            gone = self.db.names()
            for gone_name in gone:
                self._drop(gone_name, speaker)
            self._wipe_until, self._wipe_by = 0.0, None
            return [self._out("forgot", f"Forgot everyone ({len(gone)} people)",
                              f"[System: as confirmed, you deleted everything you knew about all {len(gone)} people "
                              f"({', '.join(gone) or 'nobody'}). Say so briefly.]")]
        if parse_wipe(text):
            self._wipe_until, self._wipe_by = now + 30, speaker.name
            return [self._out("needs_input", "Forget everyone? Say yes to confirm",
                              f"[System: the user asked to forget EVERYONE ({', '.join(self.db.names()) or 'nobody on record'}). "
                              f"Nothing has been deleted yet. Ask them to say yes to confirm; it cannot be undone.]")]
        if not has_forget_command(text):
            return notes
        tokens = parse_forget(text, self.db.names())
        found, missing, ambiguous = self._resolve(tokens, speaker, allow_fuzzy=is_clean_forget(text, self.db.names()))
        heard = dict(self.fuzzy)
        for name in found:
            self._drop(name, speaker)
        known = ", ".join(self.db.names()) or "nobody"
        if found:
            guessed = "".join(f" (the user said '{said}', which was matched to {name}; say both names so they can object)"
                              for name, said in heard.items())
            notes.append(self._out("forgot", f"Forgot {' and '.join(found)}" + "".join(f" (heard '{said}')" for said in heard.values()),
                                   f"[System: you deleted everything you knew about {' and '.join(found)}.{guessed} Confirm briefly.]"))
        for token, options in ambiguous:
            notes.append(self._out("needs_input", f"'{token}' could mean {' or '.join(options)}",
                                   f"[System: '{token}' could mean {' or '.join(options)}; nothing was deleted for it. Ask "
                                   f"which one they mean.]"))
        for token in missing:
            notes.append(self._out("failed", f"Couldn't forget '{token}': nobody by that name",
                                   f"[System: nothing was deleted: nobody named '{token}' is on record. People on record: "
                                   f"{known}. Tell the user and ask who they meant.]"))
        if not (found or ambiguous or missing):
            notes.append(self._out("failed", "Couldn't tell who to forget",
                                   f"[System: the user asked to forget someone but you could not tell who; nothing was "
                                   f"deleted. People on record: {known}. Ask who they mean.]"))
        return notes

    async def apply_utterance(self, text: str, speaker: Speaker, voice_embedding, vlm_pick: VlmPick | None = None,
                              spoken: bool = False) -> list[str]:
        """Handle "I'm X" / "this is X" / "forget X". Returns notes for the LLM; mutates the DB.
        `spoken`: the text came from speech recognition (names may be misspelled) rather than being typed."""
        self._spoken = spoken
        notes: list[str] = []
        events_before = len(self.events)
        pending = self._handle_pending(text, speaker) if self._wipe_until <= time.monotonic() else None
        if pending is not None:
            return pending
        if (self._spell is not None and self._wipe_until <= time.monotonic() and not parse_intros(text)
                and not parse_renames(text) and not has_forget_command(text) and not parse_wipe(text)
                and not NAMING_INTENT_RE.search(text)):
            answered = self._handle_spell_dialog(text, speaker)    # "No." / "M-E-R" / "V-I-N." after "is that right?"
            if answered is not None:
                return answered
        notes += self._handle_forget(text, speaker)
        rename_notes, renamed = await self._handle_renames(text, speaker, voice_embedding, vlm_pick)
        notes += rename_notes
        for intro in parse_intros(text):
            if any(intro.name.lower() == done or similar_name(intro.name, done) for done in renamed):
                continue   # already handled as an explicit rename or respelling (also the misheard form of it)
            notes.append(await self._introduce(intro, speaker, voice_embedding, text, vlm_pick))
        if len(self.events) == events_before and NAMING_INTENT_RE.search(text):
            # The user sounds like they are naming/renaming someone but nothing was understood or changed.
            notes.append(self._out("failed", "Heard a naming request but couldn't tell what to change",
                                   "[System: the user seems to be naming or renaming someone, but you could not tell the "
                                   "name or who it applies to, so NOTHING was changed. Ask them to say it like 'call me "
                                   "Sam', 'this is Sam' or 'rename Person A to Sam'.]"))
        return [note for note in notes if note]

    async def _introduce(self, intro: Intro, speaker: Speaker, voice_embedding, text: str,
                         vlm_pick: VlmPick | None) -> str:
        """Learn a face and/or voice for the introduced name and return the LLM note.

        "self" introductions attach the name to the speaker's face/voice; "other" introductions attach it to an
        unrecognised face in view (asking the vision model via `vlm_pick` when several are candidates).
        """
        visible_tracks = self.visible()
        name = intro.name
        existed = self.db.canonical(name) is not None       # re-introducing a known name: nothing new to spell back
        if intro.kind == "self":
            current_name = speaker.name
            if current_name and self._spoken and similar_name(current_name, name):
                return self._variant_note(current_name, name)
            if current_name and current_name.lower() != name.lower():
                if _PLACE_FULL.fullmatch(current_name) or intro.strong:   # a placeholder ("Person A") or "call me X": just rename
                    return self._do_rename(current_name, name, speaker)
                self._pending = (current_name, name, time.monotonic() + 60)
                return self._out("needs_input", f"Rename {current_name} → {name}? Say yes or no",
                                 f"[System: the speaker is recognised as {current_name} but says they are {name}. Nothing changed "
                                 f"yet. Ask whether to rename {current_name} to {name}; they can answer yes or no.]")
            face_track = speaker.track
            if face_track is None:
                unnamed = [track for track in visible_tracks if track.name is None]
                face_track = unnamed[0] if len(unnamed) == 1 else None
            learned = []
            if (face_track is not None and face_track.emb is not None
                    and (face_track.name is None or face_track.name.lower() == name.lower())):
                name = self.db.add(name, "face", face_track.emb)
                face_track.name, face_track.score = name, 1.0
                learned.append("face")
            elif len(visible_tracks) > 1 and face_track is None:
                if voice_embedding is not None:
                    self.db.add(name, "voice", voice_embedding)
                    speaker.name, speaker.how = name, "voice"
                    return self._out("needs_input", f"Learned {name}'s voice, but couldn't tell which face is theirs",
                                     f"[System: {name} introduced themselves, and you learned their voice, but several faces "
                                     f"are in view and you can't tell which is theirs. Say you'll learn their face when they "
                                     f"are alone in view or if they say 'this is {name}' while pointing at themselves.]")
                return self._out("failed", f"Couldn't learn {name}: several faces in view and no voice sample",
                                 f"[System: {name} introduced themselves but several faces are in view and the voice sample "
                                 f"was too short, so nothing was learned. Ask them to repeat it.]")
            if voice_embedding is not None:
                name = self.db.add(name, "voice", voice_embedding)
                learned.append("voice")
            speaker.name = name
            if voice_embedding is not None:
                speaker.how = "voice"
            if learned:
                self.last_named = name
                return self._out("learned", f"Learned {name}'s {' and '.join(learned)}",
                                 f"[System: you just learned {name}'s {' and '.join(learned)}; you will recognise them from "
                                 f"now on. Greet them by name briefly." + ("" if existed else self._spell_back(name)) + "]")
            return self._out("failed", f"Couldn't learn {name}: no usable face or voice sample",
                             f"[System: {name} introduced themselves but no face or voice sample was usable (face the camera "
                             f"or speak a little longer), so nothing was saved. Say so and ask them to try again.]")
        # someone else is being introduced
        skip = {speaker.track.id} if speaker.track else set()  # the speaker is not introducing themselves here
        candidates = [track for track in visible_tracks if track.name is None and track.id not in skip]
        if not candidates:
            return self._out("failed", f"Couldn't learn {name}: no unrecognised face in view",
                             f"[System: the speaker introduced '{name}', but there is no unrecognised face in view, so "
                             f"nothing was learned. Ask them to bring {name} into view.]")
        target = candidates[0] if len(candidates) == 1 else None
        if target is None and vlm_pick is not None and self.last_bgr is not None:
            pick = await vlm_pick(self.annotated_jpeg(candidates), text, [track.id for track in candidates])
            target = next((track for track in candidates if track.id == pick), None)
        if target is None or target.emb is None:
            frame_width = self.frame_size[0]  # describe positions as thirds of the frame width
            def third_of_frame(track: Track) -> str:
                return side_for_user(track.cx / frame_width)
            where = ", ".join(f"{i + 1}) {third_of_frame(track)}" for i, track in enumerate(candidates))
            return self._out("needs_input", f"Couldn't tell which of {len(candidates)} people is {name}",
                             f"[System: the speaker introduced '{name}' but {len(candidates)} unrecognised people are in view and "
                             f"it is unclear which one ({where}). Nothing was learned. Ask which person, by position.]")
        name = self.db.add(name, "face", target.emb)
        target.name, target.score = name, 1.0
        self.last_named = name
        return self._out("learned", f"Learned {name}'s face",
                         f"[System: you just learned {name}'s face and will recognise them from now on; their voice will be "
                         f"learned when they speak. Say nice to meet them, briefly." + ("" if existed else self._spell_back(name, arm=False)) + "]")

    # ---- output ------------------------------------------------------------------
    def context_note(self, speaker: Speaker) -> str:
        """Build the bracketed [Context] note for the LLM: who is in view (left to right) and who is speaking."""
        visible_tracks = self.visible()
        parts = []
        if visible_tracks:
            frame_width = self.frame_size[0] or 1

            def where(track):
                """Return left/middle/right as the user would say it (their own sides, not the image's)."""
                return side_for_user((track.box[0] + track.box[2] / 2) / frame_width)
            labels = [track.name or "unrecognised person" for track in visible_tracks]
            if len(visible_tracks) == 1:
                parts.append(f"In view: {labels[0]}. This is for answering questions only: never volunteer who is in view or what they "
                             "look like unless the user asks.")
            else:
                parts.append("In view, with positions given from the user's own point of view as they face the camera (their left "
                             "and right, already corrected, so say them as written): "
                             + ", ".join(f"{label} ({where(track)})" for label, track in zip(labels, visible_tracks))
                             + ". This is for answering questions only: never volunteer who is in view, where they are or "
                             "what they look like unless the user asks.")
        if speaker.name:
            parts.append(f"The person speaking now is named {speaker.name} (recognised by "
                         f"{'voice' if speaker.how == 'voice' else 'face'}). That is their real name for this "
                         f"conversation, even if it looks like a placeholder; use it when asked who they are.")
        elif parts or self.db.people:
            parts.append("It is not known which person is speaking. Do not tell them they are unrecognised, do not ask "
                         "their name, and do not mention this unless they ask to be remembered. You may still say "
                         "who is in view.")
        return f"[Context, never read aloud: {' '.join(parts)}]" if parts else ""

    def overlay(self, now: float | None = None) -> list[dict]:
        """Return visible faces for the browser overlay, with boxes normalised to 0..1 of the frame size."""
        now = time.monotonic() if now is None else now
        frame_width, frame_height = self.frame_size
        return [{"id": track.id,
                 "box": [round(track.box[0] / frame_width, 4), round(track.box[1] / frame_height, 4),
                         round(track.box[2] / frame_width, 4), round(track.box[3] / frame_height, 4)],
                 "name": track.name, "speaking": track.id == self.speaker_track_id and now < self.speaker_until}
                for track in self.visible(now)]
