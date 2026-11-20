"""Text-to-speech using the macOS `say` command (no extra dependencies)."""
from __future__ import annotations

import subprocess


class Speaker:
    """Speaks text through `say` as a subprocess so playback can be waited on or cut off."""

    def __init__(self, voice: str = "Samantha", rate: int = 190):  # rate: words per minute
        self.voice, self.rate = voice, rate
        self._proc: subprocess.Popen | None = None

    def speak(self, text: str, block: bool = True) -> None:
        """Speak `text`, first waiting for any current speech; return immediately when `block` is False."""
        text = text.strip()
        if not text:
            return
        self.wait()
        self._proc = subprocess.Popen(["say", "-v", self.voice, "-r", str(self.rate), text])
        if block:
            self.wait()

    def wait(self) -> None:
        """Block until the current utterance (if any) has finished."""
        if self._proc:
            self._proc.wait()
            self._proc = None

    def stop(self) -> None:
        """Cut off the current utterance immediately."""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
