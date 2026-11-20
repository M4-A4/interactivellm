"""Local LLM client for any OpenAI-compatible endpoint (Ollama, LM Studio)."""
from __future__ import annotations

from typing import Iterator

from openai import OpenAI

# OpenAI-style tool schema that lets the local model hand a hard question to the deep reasoner.
DEEP_THINK_TOOL = {
    "type": "function",
    "function": {
        "name": "deep_think",
        "description": "Delegate a hard question (multi-step reasoning, maths, deep analysis, "
                       "careful coding) to a slower, stronger reasoning model. Use sparingly; not for small talk.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Self-contained question for Claude."},
                "include_camera": {"type": "boolean", "description": "Attach the latest camera frame."},
            },
            "required": ["question"],
        },
    },
}


class LocalLLM:
    """Streaming chat client for a local OpenAI-compatible server (Ollama or LM Studio)."""

    def __init__(self, base_url: str, model: str, reasoning: str | None = "none"):
        self.model = model
        # Thinking adds seconds of latency before the first spoken word; keep it off for chat.
        self.reasoning = reasoning
        self.client = OpenAI(base_url=base_url, api_key="local")

    def stream(self, messages: list[dict], tools: list[dict] | None = None, required: bool = False) -> Iterator[dict]:
        """Yield {'text': str} chunks, then a final {'tool_calls': [...]} if any.

        With `required`, the model must call one of `tools` instead of answering in words.

        Each tool call is a dict with `id`, `name` and `arguments` (a JSON string). Tool-call fields arrive
        in fragments across chunks, so they are accumulated per call index before being emitted.
        """
        kwargs = {"model": self.model, "messages": messages, "stream": True}
        if self.reasoning:
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning}
        if tools:
            kwargs["tools"] = tools
            if required:
                kwargs["tool_choice"] = "required"
        calls: dict[int, dict] = {}
        for chunk in self.client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield {"text": delta.content}
            for tool_call_delta in delta.tool_calls or []:
                slot = calls.setdefault(tool_call_delta.index, {"id": "", "name": "", "arguments": ""})
                slot["id"] = tool_call_delta.id or slot["id"]
                if tool_call_delta.function:
                    slot["name"] += tool_call_delta.function.name or ""
                    slot["arguments"] += tool_call_delta.function.arguments or ""
        if calls:
            yield {"tool_calls": [calls[i] for i in sorted(calls)]}

    def complete(self, messages: list[dict], max_tokens: int = 300) -> str:
        """Return one non-streamed completion (used for conversation summaries)."""
        kwargs = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if self.reasoning:
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning}
        return self.client.chat.completions.create(**kwargs).choices[0].message.content or ""
