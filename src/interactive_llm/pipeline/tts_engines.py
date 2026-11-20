"""Text-to-speech engines behind one small interface, so the settings window can switch between them.

Every engine turns text into mono float32 audio at its native sample rate; `to_pcm16` converts that to the 24 kHz
16-bit PCM the browser plays. Models are loaded lazily and must only be called from the single MLX worker thread.
"""
from __future__ import annotations

import math

import numpy as np

from ..settings import TTS_ENGINES

OUTPUT_RATE = 24000   # the browser's playback rate (binary frame tag 0x03)


def to_pcm16(audio: np.ndarray, sample_rate: int) -> bytes:
    """Resample mono float audio to 24 kHz if needed and return little-endian int16 PCM bytes."""
    if sample_rate != OUTPUT_RATE:
        from scipy.signal import resample_poly
        divisor = math.gcd(OUTPUT_RATE, sample_rate)
        audio = resample_poly(audio, OUTPUT_RATE // divisor, sample_rate // divisor)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


class TTSEngine:
    """Base class: `load()` once, then `synth(text, voice, speed)` returns ``(float32 audio, sample_rate)``."""

    id = ""

    def __init__(self, repo: str):
        self.repo = repo
        self.model = None

    def load(self) -> None:
        """Load the model weights (downloads them on first use)."""
        if self.model is None:
            from mlx_audio.tts.utils import load_model
            self.model = load_model(self.repo)

    def _generate(self, text: str, **extra):
        """Run the model and join its chunks into one float32 array."""
        parts = [np.array(result.audio, dtype=np.float32) for result in self.model.generate(
            text=text, verbose=False, **extra)]
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return audio, int(getattr(self.model, "sample_rate", OUTPUT_RATE))

    def synth(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        raise NotImplementedError


class KokoroEngine(TTSEngine):
    """Kokoro-82M: very fast, clear, many voices, adjustable speed."""

    id = "kokoro"

    def synth(self, text, voice, speed):
        # Kokoro voice ids start with the language code (a = American, b = British), which `lang_code` needs.
        return self._generate(text, voice=voice, speed=speed, lang_code=voice[0])


class ChatterboxEngine(TTSEngine):
    """Chatterbox (Resemble AI): more expressive and conversational, one built-in voice, fixed speed."""

    id = "chatterbox"

    def synth(self, text, voice, speed):
        return self._generate(text, exaggeration=0.5, cfg_weight=0.5)


class ChatterboxTurboEngine(TTSEngine):
    """Chatterbox Turbo: a distilled, faster Chatterbox."""

    id = "chatterbox_turbo"

    def synth(self, text, voice, speed):
        return self._generate(text)


ENGINE_CLASSES = {cls.id: cls for cls in (KokoroEngine, ChatterboxEngine, ChatterboxTurboEngine)}


def build_engine(engine_id: str, repos: dict[str, str]) -> TTSEngine:
    """Create (but do not load) the engine `engine_id`; `repos` maps engine ids to Hugging Face repos."""
    return ENGINE_CLASSES[engine_id](repos.get(engine_id) or next(e["repo"] for e in TTS_ENGINES if e["id"] == engine_id))
