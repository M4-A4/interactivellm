"""Microphone listening with energy-based VAD + faster-whisper transcription."""
from __future__ import annotations

import queue

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 480  # samples per audio block: 30 ms at 16 kHz


class Listener:
    """Records one utterance from the mic (RMS-energy gate) and transcribes it with faster-whisper.

    Args:
        threshold: RMS level (float32 audio in [-1, 1]) above which a block counts as speech.
        silence_s: trailing quiet, in seconds, that ends an utterance.
        max_s: hard cap on utterance length in seconds.
    """

    def __init__(self, model_size: str = "small.en", threshold: float = 0.012,
                 silence_s: float = 0.8, max_s: float = 30.0):
        self.model_size = model_size
        self.threshold, self.silence_s, self.max_s = threshold, silence_s, max_s
        self._model = None

    def _load(self):
        """Load the Whisper model on first use and return it."""
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, compute_type="int8")
        return self._model

    def record_utterance(self) -> np.ndarray | None:
        """Block until the user speaks and then pauses. Returns float32 mono audio."""
        import sounddevice as sd

        audio_blocks: queue.Queue[np.ndarray] = queue.Queue()  # filled by the audio callback thread
        frames: list[np.ndarray] = []
        speaking, quiet = False, 0.0
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=BLOCK, callback=lambda data, *_: audio_blocks.put(data.copy())):
            while True:
                block = audio_blocks.get()[:, 0]
                loud = float(np.sqrt(np.mean(block ** 2))) > self.threshold
                if loud:
                    speaking, quiet = True, 0.0
                elif speaking:
                    quiet += BLOCK / SAMPLE_RATE
                if speaking:
                    frames.append(block)
                    if quiet >= self.silence_s or len(frames) * BLOCK / SAMPLE_RATE > self.max_s:
                        break
        return np.concatenate(frames) if frames else None

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 mono 16 kHz audio to text."""
        segments, _ = self._load().transcribe(audio, vad_filter=True, beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()

    def listen(self) -> str:
        """Record one utterance and return its transcript ("" if nothing was captured)."""
        audio = self.record_utterance()
        return self.transcribe(audio) if audio is not None else ""
