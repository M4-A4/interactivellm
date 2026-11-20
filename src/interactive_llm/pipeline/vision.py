"""Rolling camera-frame buffer with cheap motion scoring."""
from __future__ import annotations

import base64
import re
import time

import cv2
import numpy as np


# Words suggesting the user is asking about something visible (looks, objects, positions, "what is this").
VISUAL_RE = re.compile(
    r"\b(?:look(?:s|ing)?|see(?:ing)?|watch(?:ing)?|show(?:ing|n)?|pictures?|photos?|images?|camera|screen|wear(?:ing)?|"
    r"holding|hold(?:s)?|colou?rs?|shirt|glasses|hair(?:style|cut)?|face|hands?|fingers?|smil(?:e|ing)|behind|beside|"
    r"in view|in frame|in the (?:frame|picture|image|shot|camera)|visible|(?:can|do) you spot|next to|in front|(?:on|to|at) (?:the |my |your |our )?(?:left|right)|(?:left|right) (?:side|of me|of you)|left or (?:my |the )?right|which side|room|background|written|says|sign|label|text|point(?:ing)?|object|"
    r"appears?|read (?:this|that|it)|can you see|do i look|am i (?:wearing|holding|smiling|doing)|"
    r"(?:what|who|which|whose)(?:'s|\s+is|\s+are)?\s+(?:this|that|these|those|in|on|behind|next|here|there)|"
    r"how (?:many|much) .{0,25}(?:here|there|see|picture|frame))\b", re.I)


def wants_vision(text: str, mode: str = "auto") -> bool:
    """Decide whether to attach a camera frame to this turn.

    In "auto" mode a frame is sent only when the words are about what the camera sees (saves ~1s and stops
    unprompted narration); "always" and "never" override the text check.
    """
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(VISUAL_RE.search(text))


class FrameBuffer:
    """Keeps the freshest JPEG frame and a tiny grayscale thumbnail for change detection."""

    def __init__(self, change_threshold: float = 6.0, stale_s: float = 4.0):
        self.change_threshold = change_threshold
        self.stale_s = stale_s
        # Both are (monotonic timestamp, JPEG bytes, 64x48 grayscale thumbnail) or None.
        self.latest: tuple[float, bytes, np.ndarray] | None = None
        self.mark: tuple[float, bytes, np.ndarray] | None = None  # snapshot at speech start

    bgr: np.ndarray | None = None  # most recent decoded colour frame (for face recognition)

    def push(self, jpeg: bytes) -> np.ndarray | None:
        """Store a new JPEG frame; return the decoded BGR image, or None if it could not be decoded."""
        color = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if color is None:
            return None
        self.bgr = color
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        self.latest = (time.monotonic(), jpeg, thumb)
        return color

    def fresh(self) -> bool:
        """True if the latest frame is younger than `stale_s` seconds."""
        return self.latest is not None and time.monotonic() - self.latest[0] < self.stale_s

    def mark_start(self) -> None:
        """Remember the current frame as the speech-start snapshot (None if the feed is stale)."""
        self.mark = self.latest if self.fresh() else None

    def pick(self) -> list[str]:
        """Frames to attach to a turn: the latest, plus the speech-start frame if the scene changed."""
        if not self.fresh():
            return []
        end = self.latest
        chosen = [end]
        if self.mark and self.mark is not end:
            # Mean absolute pixel difference of the thumbnails (0-255 scale) as a cheap scene-change score.
            if float(np.mean(np.abs(self.mark[2] - end[2]))) > self.change_threshold:
                chosen = [self.mark, end]
        self.mark = None
        return [base64.b64encode(frame[1]).decode() for frame in chosen]


# Questions about which side something is on. For these the frame is sent mirrored (see `mirror_jpeg_b64`).
SIDE_RE = re.compile(r"\b(?:left|right)\b|\bwhich side\b", re.I)


def asks_which_side(text: str) -> bool:
    """True if the user asks where something is relative to their own left or right."""
    return bool(SIDE_RE.search(text)) and wants_vision(text)


def mirror_jpeg_b64(jpeg_b64: str) -> str:
    """Flip a base64 JPEG left-to-right (a mirror view) and return it re-encoded as base64.

    The camera image is true, so its left is the user's right. Models answer "on my left/right" far more reliably when the
    image is the mirror view, where left really is the user's left; text in a mirrored image is reversed, so this is
    only used for side questions.
    """
    image = cv2.imdecode(np.frombuffer(base64.b64decode(jpeg_b64), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jpeg_b64
    ok, encoded = cv2.imencode(".jpg", cv2.flip(image, 1), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(encoded.tobytes()).decode() if ok else jpeg_b64
