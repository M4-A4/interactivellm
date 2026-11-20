"""Streaming voice-activity detection (Silero) with pre-roll and barge-in confirmation.

Feed 16 kHz int16 PCM in any chunk size; yields events:
  ("start", None)          speech began (after a short debounce)
  ("confirm", None)        speech has lasted long enough to be a real interruption
  ("end", float32 audio)   utterance finished (silence), audio includes pre-roll
"""
from __future__ import annotations

from collections import deque
from typing import Iterator

import numpy as np

SR = 16000  # sample rate in Hz
FRAME = 512  # 32 ms, required by Silero at 16 kHz


class VADSegmenter:
    """Turns a stream of PCM into start/confirm/end speech events.

    Args:
        threshold: Silero speech probability at or above which a frame counts as voiced.
        end_silence_ms: trailing silence that ends an utterance.
        confirm_ms: voiced time required before "confirm" fires (barge-in guard).
        preroll_ms: audio kept from before "start" so the first syllable is not clipped.
        start_frames: consecutive voiced frames needed to fire "start" (debounce).
        max_utterance_s: force an "end" after this long.
    """

    def __init__(self, threshold=0.5, end_silence_ms=550, confirm_ms=300, preroll_ms=320,
                 start_frames=2, max_utterance_s=30.0):
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self.model = load_silero_vad()
        self.threshold = threshold
        # A frame is 32 ms, so millisecond settings are converted to frame counts.
        self.end_frames = max(1, end_silence_ms // 32)
        self.confirm_frames = max(1, confirm_ms // 32)
        self.start_frames = start_frames
        self.max_frames = int(max_utterance_s * SR / FRAME)
        self.preroll: deque[np.ndarray] = deque(maxlen=max(1, preroll_ms // 32))
        self._buf = np.zeros(0, dtype=np.int16)
        self.reset()

    def reset(self) -> None:
        """Drop all buffered audio and detector state (also resets the Silero model)."""
        self.model.reset_states()
        self.active = False
        self.confirmed = False
        self.frames: list[np.ndarray] = []
        self.voiced_run = 0
        self.silent_run = 0
        self.preroll.clear()
        self._buf = np.zeros(0, dtype=np.int16)

    def _prob(self, frame_f32: np.ndarray) -> float:
        """Return Silero's speech probability for one float32 frame."""
        with self._torch.no_grad():
            return float(self.model(self._torch.from_numpy(frame_f32), SR).item())

    def feed(self, pcm: bytes) -> Iterator[tuple[str, np.ndarray | None]]:
        """Consume 16 kHz int16 PCM bytes and yield any (event, audio) pairs they complete.

        The "end" audio is None for short unconfirmed blips (<= 0.25 s) so noise is not transcribed.
        """
        self._buf = np.concatenate([self._buf, np.frombuffer(pcm, dtype=np.int16)])
        while len(self._buf) >= FRAME:
            frame = self._buf[:FRAME].astype(np.float32) / 32768.0
            self._buf = self._buf[FRAME:]
            is_speech = self._prob(frame) >= self.threshold
            if not self.active:
                self.preroll.append(frame)
                self.voiced_run = self.voiced_run + 1 if is_speech else 0
                if self.voiced_run >= self.start_frames:
                    self.active, self.confirmed, self.silent_run = True, False, 0
                    self.frames = list(self.preroll)
                    yield "start", None
                continue
            self.frames.append(frame)
            self.silent_run = 0 if is_speech else self.silent_run + 1
            voiced = len(self.frames) - self.silent_run
            if not self.confirmed and voiced >= self.confirm_frames + len(self.preroll):
                self.confirmed = True
                yield "confirm", None
            if self.silent_run >= self.end_frames or len(self.frames) >= self.max_frames:
                # Trim trailing silence, but keep 2 frames (~64 ms) of tail so word endings are not cut.
                audio = np.concatenate(self.frames[: len(self.frames) - self.silent_run + 2])
                was_confirmed = self.confirmed
                self.active, self.confirmed, self.voiced_run, self.frames = False, False, 0, []
                self.preroll.clear()
                self.model.reset_states()
                yield "end", (audio if was_confirmed or len(audio) > SR * 0.25 else None)
