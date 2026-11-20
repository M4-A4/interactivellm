"""Claude client used for complex reasoning (escalation target)."""
from __future__ import annotations

import os

import anthropic


class ClaudeClient:
    """Thin wrapper around the Anthropic Messages API; the SDK client is created lazily on first use."""

    def __init__(self, model: str):
        self.model = model
        self._client: anthropic.Anthropic | None = None

    @property
    def available(self) -> bool:
        """True when ANTHROPIC_API_KEY is set (checked each call so the key can appear later)."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def ask(self, question: str, image_b64: str | None = None, context: str = "", cancel=None) -> str:
        """Ask Claude one question (optionally about a base64 JPEG) and return its text answer.

        `cancel` is accepted for interface parity with DeepReasoner but is unused: the request is not streamed.
        Returns an explanatory message instead of raising when no API key is configured.
        """
        if not self.available:
            return "Claude is unavailable: ANTHROPIC_API_KEY is not set."
        if self._client is None:
            self._client = anthropic.Anthropic()
        content: list[dict] = []
        if image_b64:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}})
        text = f"{context}\n\n{question}".strip()
        content.append({"type": "text", "text": text})
        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system="You are the deep-reasoning backend for a voice assistant. Be accurate and "
                   "reasonably brief; your answer will be relayed aloud. Never identify anyone from their face or appearance; if asked who someone is, say you can't tell from looks.",
            messages=[{"role": "user", "content": content}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
