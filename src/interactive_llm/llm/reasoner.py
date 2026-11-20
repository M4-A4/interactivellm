"""Deep-reasoning tier: Claude when an API key exists, otherwise the local model with thinking on."""
from __future__ import annotations

import os
import time
from typing import Callable

from openai import OpenAI

from .claude import ClaudeClient

# System prompt for the deep-reasoning model.
SYSTEM = ("You are the deep-reasoning backend for a voice assistant. Think carefully, then give an "
          "accurate answer that is brief enough to be read aloud (no markdown, no lists of code)."
          " Never identify anyone from their face or appearance; if asked who someone is, say you can't tell from looks.")


class DeepReasoner:
    """Answers hard questions with Claude if a key is set, else with the local model (reasoning enabled)."""

    available = True  # always: falls back to the local model

    def __init__(self, claude: ClaudeClient, local_url: str, local_model: str):
        self.claude = claude
        self.local_model = local_model
        self._local = OpenAI(base_url=local_url, api_key="local", timeout=120)
        self.time_limit = float(os.environ.get("DEEP_TIMEOUT", "40"))

    @property
    def uses_claude(self) -> bool:
        """True when questions are routed to Claude rather than the local model."""
        return self.claude.available

    @property
    def label(self) -> str:
        """Human-readable name of the active reasoning backend."""
        return "Claude" if self.uses_claude else f"local ({self.local_model})"

    def ask(self, question: str, image_b64: str | None = None, context: str = "",
            cancel: Callable[[], bool] | None = None) -> str:
        """Answer `question` (optionally about a base64 JPEG) using the active backend.

        Local path only: streams the answer so it can stop early when `cancel()` is true (returns "") or after
        `time_limit` seconds (returns what was produced so far).
        """
        if self.uses_claude:
            return self.claude.ask(question, image_b64=image_b64, context=context)
        content: list[dict] = [{"type": "text", "text": f"{context}\n\n{question}".strip()}]
        if image_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        stream = self._local.chat.completions.create(
            model=self.local_model, max_tokens=3000, stream=True,
            extra_body={"reasoning_effort": os.environ.get("DEEP_EFFORT", "low")},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}])
        out, started = "", time.monotonic()
        try:
            for chunk in stream:  # streaming lets us stop promptly on barge-in or timeout
                if cancel and cancel():
                    return ""
                if chunk.choices and chunk.choices[0].delta.content:
                    out += chunk.choices[0].delta.content
                if time.monotonic() - started > self.time_limit:
                    break
        finally:
            stream.close()
        return out.strip() or "I could not finish working that out in time."
