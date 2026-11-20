"""On-device speech models (MLX). All MLX work runs on ONE dedicated thread, since MLX
GPU streams are bound to the thread that created them."""
from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..config import WebConfig
from ..settings import TTS_ENGINES
from .tts_engines import TTSEngine, build_engine, to_pcm16

# Transcripts Whisper commonly invents from silence or noise; matched case-insensitively against the whole text.
HALLUCINATIONS = {"you", "thank you.", "thanks for watching!", "thanks for watching.", "bye.", ".", ""}


_SOUNDS = (r"music|piano|guitar|drums?|violin|applause|laughter|laughing|clapping|singing|humming|whistling|silence|"
           r"static|noise|beeping|ringing|chiming|typing|coughing")
# Whisper captions non-speech audio ("Piano music plays.", "[Applause]", "(soft music)") instead of returning nothing.
# One or more such captions, and nothing else, make a clip junk. A bare sound word ("Piano.", "Music.") is an ordinary
# answer and is kept: only a bracketed caption or a sound plus a caption verb counts.
_BRACKETED = r"[\[(*♪][^\])*♪]*[\])*♪]"
_SOUND_PLAYS = (r"(?:(?:soft|gentle|upbeat|light|dramatic|background|quiet|loud|instrumental|piano|guitar)\s+){0,2}"
                r"(?:" + _SOUNDS + r")(?:\s+music)?\s+(?:plays?|playing|continues|fades|resumes)")
_SIGN_OFF = r"thank(?:s| you) (?:so much )?for watching(?:\W+see you (?:next time|soon))?"
_CAPTION_UNIT = rf"(?:{_BRACKETED}|{_SOUND_PLAYS}|{_SIGN_OFF})"
_CAPTION_RE = re.compile(rf"\s*{_CAPTION_UNIT}(?:[\s,.;!]*{_CAPTION_UNIT})*\W*", re.I)


def is_sound_caption(text: str) -> bool:
    """True for text that only describes a sound, or is a stock sign-off Whisper invents from noise."""
    return bool(_CAPTION_RE.fullmatch(text))


def names_prompt(names: list[str] | None) -> str | None:
    """Context for Whisper so it spells people already on record the way we stored them (Marvin, not Marven)."""
    names = [n for n in (names or []) if n][:20]
    return f"Names: {', '.join(names)}." if names else None


def is_junk_transcript(text: str, prompt: str | None = None) -> bool:
    """Whisper failure modes on noise or foreign speech: echoing our own name hint ("Names & Nico." x30)
    and degenerate repetition. Such text must never reach the assistant."""
    words = re.findall(r"[\w']+", text.lower())
    if not words or is_sound_caption(text):
        return True
    if prompt and re.match(r"\s*names\s*[&:,]", text, re.I):
        return True  # starts like our own "Names: ..." hint, so it is the model continuing the hint, not speech
    if prompt:
        hint = set(re.findall(r"[\w']+", prompt.lower()))
        if "names" in words and all(w in hint or w in {"and", "the", "a"} for w in words):
            return True  # nothing but our own hint words, e.g. "Names & Nico."
    if len(words) >= 6:
        if len(set(words)) / len(words) < 0.3:
            return True
        grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if max(grams.count(g) for g in set(grams)) >= 4:
            return True
    return False


def clean_for_speech(text: str) -> str:
    """Strip everything that should not be read aloud (think blocks, code, notes, URLs, markdown, emoji)."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    text = re.sub(r"```.*?```", " code omitted ", text, flags=re.S)
    text = re.sub(r"\[[^\]]{0,200}\]", " ", text)            # echoed [System]/[Context]/[interrupted] notes are never spoken
    text = re.sub(r"https?://\S+", " link ", text)
    text = re.sub(r"[*_#`>~|$\\]+", "", text)  # markdown and LaTeX markers are never spoken
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class SpeechModels:
    """Whisper STT and a selectable voice engine (Kokoro by default), with every model call funnelled onto the single MLX worker thread."""

    def __init__(self, cfg: WebConfig):
        self.cfg = cfg
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._engines: dict[str, TTSEngine] = {}   # voice engines loaded so far, by id

    def run(self, fn, *args):
        """Run `fn(*args)` on the MLX thread and return an awaitable for its result."""
        return asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    # ---- run inside the MLX thread -------------------------------------------------
    def _engine(self, engine_id: str) -> TTSEngine:
        """Return the loaded voice engine `engine_id`, loading (and on first use downloading) it if needed."""
        if engine_id not in self._engines:
            engine = build_engine(engine_id, {entry["id"]: os.environ.get(entry["repo_env"], entry["repo"])
                                              for entry in TTS_ENGINES})
            engine.load()
            self._engines[engine_id] = engine
        return self._engines[engine_id]

    def _warm(self) -> None:
        """Load the voice engine and run one dummy synthesis and transcription so the first real turn is fast."""
        try:
            self._engine(self.cfg.tts_engine)
        except Exception as error:   # an engine that cannot load must not stop the app: fall back to Kokoro
            print(f"[warn] voice engine {self.cfg.tts_engine!r} failed to load ({error}); using Kokoro", flush=True)
            self.cfg.tts_engine = "kokoro"
            self.cfg.tts_voice = "af_heart"
            self._engine("kokoro")
        self._synth("Ready.")
        self._stt(np.zeros(16000, dtype=np.float32))

    def _stt(self, audio: np.ndarray, prompt: str | None = None) -> str:
        """Transcribe 16 kHz float32 audio; return "" for silence, hallucinations or junk. `prompt` seeds Whisper."""
        import mlx_whisper
        extra = {"initial_prompt": prompt} if prompt else {}
        # Temperature fallback: a looping/garbled decode (compression ratio > 2.4) is retried instead of returned.
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=self.cfg.whisper_repo, language=self.cfg.whisper_lang,
                                   condition_on_previous_text=False, temperature=(0.0, 0.2, 0.4),
                                   compression_ratio_threshold=2.4, verbose=None, **extra)
        segments = result.get("segments") or []
        # Every segment both likely non-speech and low confidence: treat the clip as silence.
        if segments and all(s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -0.7 for s in segments):
            return ""
        text = result.get("text", "").strip()
        return "" if text.lower() in HALLUCINATIONS or is_junk_transcript(text, prompt) else text

    def _synth(self, text: str, voice: str | None = None, speed: float | None = None, engine: str | None = None) -> bytes:
        """Synthesize `text` to 24 kHz mono int16 PCM bytes; the configured engine, voice and speed are the defaults."""
        engine_id, voice, speed = engine or self.cfg.tts_engine, voice or self.cfg.tts_voice, speed or self.cfg.tts_speed
        audio, sample_rate = self._engine(engine_id).synth(text, voice, speed)
        return to_pcm16(audio, sample_rate) if len(audio) else b""

    # ---- async API -----------------------------------------------------------------
    async def warmup(self) -> None:
        """Load and exercise both models (call once at startup)."""
        await self.run(self._warm)

    async def transcribe(self, audio: np.ndarray, names: list[str] | None = None) -> str:
        """Transcribe `audio`; `names` are known people, passed as a spelling hint to Whisper."""
        return await self.run(self._stt, audio, names_prompt(names))

    async def preview(self, text: str, voice: str, speed: float, engine: str | None = None) -> bytes:
        """Synthesize `text` with an explicit engine, voice and speed (settings preview)."""
        return await self.run(self._synth, text, voice, speed, engine)

    async def ensure_engine(self, engine_id: str) -> None:
        """Load a voice engine now (raises if it cannot be loaded), so switching to it later is instant."""
        await self.run(self._engine, engine_id)

    async def synthesize(self, text: str) -> bytes:
        """Clean `text` for speech and synthesize it with the configured voice; b"" if nothing is speakable."""
        text = clean_for_speech(text)
        return await self.run(self._synth, text) if text else b""
