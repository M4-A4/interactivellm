"""Camera capture via OpenCV. Frames are grabbed on demand (no background stream)."""
from __future__ import annotations

import base64


class Camera:
    """On-demand webcam grabber; `max_side` caps the longer image edge in pixels."""

    def __init__(self, index: int = 0, max_side: int = 768):
        self.index = index
        self.max_side = max_side

    def capture_b64(self) -> str | None:
        """Return a JPEG frame as base64, or None if the camera is unavailable."""
        import cv2

        cap = cv2.VideoCapture(self.index)
        try:
            if not cap.isOpened():
                return None
            for _ in range(5):  # let auto-exposure settle
                cap.read()
            ok, frame = cap.read()
            if not ok:
                return None
            height, width = frame.shape[:2]
            scale = self.max_side / max(height, width)
            if scale < 1:
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(jpeg.tobytes()).decode() if ok else None
        finally:
            cap.release()
