"""Conversation engine: local LLM for chat, deep_think tool for hard questions, bounded context."""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Callable, Iterator

from .config import Config
from .llm.claude import ClaudeClient
from .llm.local import DEEP_THINK_TOOL, LocalLLM
from .llm.reasoner import DeepReasoner

# Only questions that look hard get the slow deep_think tool at all; everything else stays on the fast path.
HARD_RE = re.compile(
    r"\b(calculate|compute|solve|prove|derive|equations?|integral|derivative|probability|percentages?|step by step|"
    r"riddles?|puzzles?|trick question|algorithms?|debug|regex|sql|pros and cons|multiplied|times|divided|plus|minus|"
    r"square root|factorial|logic|reason(?:ing)?|think (?:hard|carefully|about|through)|write\s+(?:(?:a|an|some|me)\s+)*"
    r"(?:code|function|script|program)|how would you (?:design|build|implement))\b|\d\s*[x×*/^+\-]\s*\d", re.I)


# Capitalised words that are not stock tickers (used when a lookup has to be chosen by code).
FILLER_AFTER_S = 1.2   # a skill lookup that finishes sooner than this gets no spoken "One moment"

# Tool safety. Web page text is untrusted: a page can tell the model to do things. The harm to prevent is private
# data leaving the computer, so within one turn (and for file names heard earlier) these never mix:
LOCAL_DATA_TOOLS = {"list_folder"}                          # read private file names from this computer
UNTRUSTED_TOOLS = {"web_search", "stock_quote"}             # bring in text written by strangers
EGRESS_TOOLS = {"web_search", "stock_quote", "deep_think"}  # send text off this computer (search engine, Anthropic)
REFUSE_EGRESS = ("Refused: private file names from this computer are part of this conversation, so nothing may be sent to "
                 "the internet or a cloud model now. Do not retry. Tell the user briefly that you did not look it up.")
REFUSE_LOCAL = ("Refused: web text is being handled in this same turn, so private folders are not read until the user asks "
                "again separately. Do not retry. Tell the user briefly.")
NOT_TICKERS = {"ETF", "ETFS", "USD", "AI", "US", "UK", "TV", "CEO", "NBA", "NFL", "NHL", "MLB", "EU", "UN", "OK"}


STOCK_CUE_RE = re.compile(r"\b(?:price|prices|stock|stocks|share|shares|etf|etfs|trading|ticker|quote|worth|nasdaq|nyse|dow)\b", re.I)


def needs_deep_thinking(text: str) -> bool:
    """Return True if the request looks hard enough (keywords, arithmetic, or 35+ words) to offer deep_think."""
    return bool(HARD_RE.search(text)) or len(text.split()) >= 35


# Sentence boundary: whitespace following . ! or ? (the punctuation stays with the sentence).
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Softer boundary (comma, semicolon, colon), used only to flush the first chunk early.
CLAUSE_END = re.compile(r"(?<=[,;:])\s+")
IMAGE_TOKENS = 500  # rough cost of one attached 448px frame
CAMERA_NOTE = ("[Live camera view attached. Use it when the user's words are about what is visible, including short "
               "follow-ups to an earlier visual question ('what about now?', 'it's here'); otherwise ignore it and do NOT describe the scene. Answer only "
               "from this image, never from your earlier descriptions. Describe, but never name, the people you see. The image is the "
               "camera's true view, NOT a mirror: the left side of the image is the user's RIGHT and the right side of the image is "
               "the user's LEFT. When you say where something is, always use the user's own left and right (the opposite of the "
               "image side).]")
MIRRORED_CAMERA_NOTE = ("[Live camera view attached, MIRRORED like a mirror: its left side is the user's left and its right side is "
                        "the user's right, so say left and right exactly as they appear in this image. Writing in it looks reversed "
                        "because of the mirroring, so do not try to read text from it. Describe, but never name, the people you see.]")
SUMMARY_PROMPT = ("You are the memory of a voice assistant. Merge the previous summary (if any) and the "
                  "conversation excerpt into ONE updated summary under 150 words: who the user is, facts they "
                  "shared, topics, decisions, open questions. Plain text, no preamble.")


def sentences(chunks: Iterator[str], eager_first: bool = True) -> Iterator[str]:
    """Group streamed text into sentences so TTS can start before the reply ends.

    With eager_first, the very first chunk may be flushed at a clause boundary once it is
    long enough, which cuts time-to-first-audio.
    """
    buf, first = "", True  # `first` stays True until something has been yielded
    for chunk in chunks:
        buf += chunk
        parts = SENTENCE_END.split(buf)
        for sentence in parts[:-1]:  # the last part may be an unfinished sentence: keep buffering it
            first = False
            yield sentence
        buf = parts[-1]
        if eager_first and first and len(buf) >= 28:  # 28 chars: enough text for TTS to sound natural
            clause_ends = list(CLAUSE_END.finditer(buf))
            if clause_ends:
                cut = clause_ends[-1].end()
                yield buf[:cut].strip()
                buf, first = buf[cut:], False
    if buf.strip():
        yield buf


CHARS_PER_TOKEN = 3.5  # rough average for English text


def est_tokens(msg: dict) -> int:
    """Estimate the token cost of one chat message (text, attached images, tool calls) without a tokenizer."""
    content = msg.get("content") or ""
    if isinstance(content, list):
        tokens = sum(len(part.get("text", "")) / CHARS_PER_TOKEN for part in content if part["type"] == "text")
        tokens += IMAGE_TOKENS * sum(1 for part in content if part["type"] == "image_url")
    else:
        tokens = len(content) / CHARS_PER_TOKEN
    for tool_call in msg.get("tool_calls") or []:
        tokens += len(tool_call["function"]["arguments"]) / CHARS_PER_TOKEN + 10  # +10: call framing
    return int(tokens) + 4  # +4: per-message role/format overhead


class Engine:
    """Chat session state (history, rolling summary) plus streaming replies with optional deep_think tool use."""

    def __init__(self, cfg: Config, local: LocalLLM | None = None, claude: ClaudeClient | None = None,
                 camera_fn: Callable[[], str | None] | None = None):
        self.cfg = cfg
        url, model = cfg.local_endpoint()
        self.local = local or LocalLLM(url, model)
        self.claude = claude or DeepReasoner(ClaudeClient(cfg.claude_model), url, model)
        self.camera_fn = camera_fn
        self.summary = ""
        self.history: list[dict] = [{"role": "system", "content": cfg.system_prompt}]
        self.last_frame: str | None = None
        self.lock = threading.RLock()  # guards history edits made outside the reply generator
        self._tool_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deep")
        self.stats: dict = {}  # per-turn diagnostics: {"tool": bool, "tool_ms": int}
        self._cut_off = False  # the user interrupted the previous reply: tell the model once, outside its own text
        self.private_names: set[str] = set()    # lowercase file names listed for the user: never put into a search or cloud call
        self.on_cloud: Callable[[str], None] | None = None   # told, BEFORE anything is sent to Anthropic, what is being sent

    # ---- history helpers -----------------------------------------------------------
    def reset(self) -> None:
        """Forget the conversation and its summary."""
        with self.lock:
            self.summary = ""
            self.private_names = set()
            self.history = [{"role": "system", "content": self.cfg.system_prompt}]

    def adopt(self, other: "Engine") -> None:
        """Carry a conversation over to a new engine (e.g. after switching backend)."""
        with self.lock:
            self.history, self.summary, self.private_names = other.history, other.summary, other.private_names

    def _sync_system(self) -> None:
        """Rewrite the system message so it carries the current rolling summary."""
        extra = f"\n\nEarlier in this conversation (summary): {self.summary}" if self.summary else ""
        self.history[0]["content"] = self.cfg.system_prompt + extra

    @staticmethod
    def _user_message(text: str, images: list[str] | None, note: str | None = None, mirrored: bool = False) -> dict:
        """Build a user chat message; plain string unless images or a context note need a multi-part list."""
        if not images and not note:
            return {"role": "user", "content": text}
        content: list[dict] = [{"type": "text", "text": text}]
        if images:
            content.append({"type": "text", "text": MIRRORED_CAMERA_NOTE if mirrored else CAMERA_NOTE})
        if note:
            content.append({"type": "text", "text": note})
        content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}} for image in images or []]
        return {"role": "user", "content": content}

    def _strip_old_images(self) -> None:
        """Keep images only for the current turn so context (and prefill cost) stays small."""
        for message in self.history:
            if message["role"] == "user" and isinstance(message["content"], list):
                message["content"] = next((part["text"] for part in message["content"] if part["type"] == "text"), "")

    def replace_last_assistant(self, heard: str) -> None:
        """After a barge-in, record only the part of the reply the user actually heard."""
        with self.lock:
            last = self.history[-1]
            if last["role"] == "assistant" and isinstance(last.get("content"), str):
                last["content"] = heard if heard else "…"   # only what was actually heard; no marker the model could imitate
                self._cut_off = True

    def tokens(self) -> int:
        """Return the estimated token size of the whole history."""
        return sum(est_tokens(m) for m in self.history)

    def context_info(self) -> dict:
        """Return token usage, budget and summarised flag for the UI."""
        return {"tokens": self.tokens(), "budget": self.cfg.context_tokens, "summarized": bool(self.summary)}

    def compact(self, force: bool = False) -> bool:
        """If history exceeds the budget (or `force`), fold the oldest turns into a rolling summary.

        Returns True if the history was shortened; False if nothing needed doing or summarising failed.
        """
        with self.lock:
            if not force and self.tokens() <= self.cfg.context_tokens:
                return False
            keep = self.cfg.context_keep_tokens
            user_indices = [i for i, m in enumerate(self.history) if i > 0 and m["role"] == "user"]  # turn starts
            cut = None
            for i in user_indices:  # earliest turn boundary such that the tail fits in `keep`
                if sum(est_tokens(m) for m in self.history[i:]) <= keep:
                    cut = i
                    break
            if cut is None and user_indices:
                cut = user_indices[-1]  # even the last turn is big: keep just that
            if not cut or cut <= 1:
                return False
            old_messages = self.history[1:cut]  # index 0 is the system message, which is always kept
            transcript = "\n".join(
                f"{m['role']}: {m['content'] if isinstance(m.get('content'), str) else ''}"[:1200]
                for m in old_messages if m["role"] in ("user", "assistant") and m.get("content"))
            try:
                self.summary = self.local.complete([
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": f"Previous summary: {self.summary or '(none)'}\n\nExcerpt:\n{transcript}"},
                ], max_tokens=300).strip()
            except Exception:
                return False  # keep full history rather than lose it
            self.history = [self.history[0]] + self.history[cut:]
            self._sync_system()
            return True

    # ---- generation ----------------------------------------------------------------
    def _run_tool(self, call: dict, images: list[str] | None, cancel: Callable[[], bool] | None = None) -> str:
        """Execute one deep_think tool call (runs on the worker thread) and return its text result."""
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError:
            return "Invalid tool arguments."
        frame = images[-1] if (images and args.get("include_camera")) else None
        self._say_cloud(frame is not None)
        # The rolling conversation summary holds personal facts the user shared, so it is never sent to the cloud:
        # the model writes a self-contained question instead.
        return self.claude.ask(args.get("question", ""), image_b64=frame, context="", cancel=cancel)

    def _say_cloud(self, with_frame: bool) -> None:
        """Tell the GUI (before the call is made) that a question is about to go to Anthropic's cloud."""
        if self.on_cloud and getattr(self.claude, "uses_claude", False):
            self.on_cloud("Sent your question" + (" and the camera frame" if with_frame else "") + " to Anthropic's cloud")

    def _refusal(self, call: dict, round_names: set[str], local_seen: bool, untrusted_seen: bool) -> str | None:
        """A reason to refuse one requested tool call, or None. See LOCAL_DATA_TOOLS / UNTRUSTED_TOOLS / EGRESS_TOOLS."""
        name = call["name"]
        if name in EGRESS_TOOLS:
            if local_seen or round_names & LOCAL_DATA_TOOLS:
                return REFUSE_EGRESS
            lowered = str(call.get("arguments", "")).lower()
            if any(private in lowered for private in self.private_names):
                return REFUSE_EGRESS
        if name in LOCAL_DATA_TOOLS and (untrusted_seen or round_names & UNTRUSTED_TOOLS):
            return REFUSE_LOCAL
        return None
    @staticmethod
    def _run_skill_tool(call: dict, runner: Callable[[str, dict], str] | None) -> str:
        """Run one skill tool call (worker thread) and return the text the model will read."""
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError:
            return "Invalid tool arguments; nothing was done. Do not answer from memory."
        if not runner:
            return "That tool is not available."
        return runner(call["name"], args if isinstance(args, dict) else {})

    def reply(self, text: str, use_camera: bool = False, force_claude: bool = False,
              images: list[str] | None = None, note: str | None = None,
              speaker: str | None = None, cancel: Callable[[], bool] | None = None,
              skill_tools: list[dict] | None = None, skill_runner: Callable[[str, dict], str] | None = None,
              must_use_tool: bool = False, mirrored: bool = False) -> Iterator[str]:
        """Yield text chunks of the assistant's answer. Closing the generator saves the partial reply.

        Args:
            use_camera: grab a fresh frame from `camera_fn` for this turn (CLI path).
            force_claude: skip the local model and answer with the deep reasoner directly.
            images: base64 JPEG frames to attach to this turn only.
            note: bracketed context note (e.g. who is speaking) sent to the model but never read aloud.
            speaker: name to prefix the text with so a multi-person history stays readable.
            cancel: polled during slow tool calls; return True to abandon the turn (barge-in).
            skill_tools: OpenAI-style schemas of read-only skills the model may call (web search, stock quote, ...).
            skill_runner: ``(tool name, arguments) -> result text``; blocking, called on the worker thread.
            mirrored: the attached images are mirror views (see `vision.mirror_jpeg_b64`).
            must_use_tool: the question needs live data. The first reply is held back; if the model answers in words
                instead of calling a tool, that answer is discarded and a web search on the user's sentence runs instead.
        """
        self.stats = {"tool": False, "tool_ms": 0}
        hard = needs_deep_thinking(text)
        if speaker:
            text = f"{speaker}: {text}"  # keeps multi-person history readable
        if use_camera and self.camera_fn:
            self.last_frame = self.camera_fn()
            images = [self.last_frame] if self.last_frame else None
        self._strip_old_images()
        if self.tokens() > self.cfg.context_tokens * 1.5:  # safety net; normally compacted after the previous turn
            self.compact()
        if force_claude:
            self._say_cloud(bool(images))
            answer = self.claude.ask(text, image_b64=images[-1] if images else None, context="", cancel=cancel)
            self.history += [self._user_message(text, None), {"role": "assistant", "content": answer}]
            yield answer
            return

        if self._cut_off:
            note = ((note + " ") if note else "") + ("[Context, never read aloud: the user interrupted your previous reply, "
                                                    "so only part of it was heard.]")
            self._cut_off = False
        self.history.append(self._user_message(text, images, note, mirrored))
        skill_names = {tool["function"]["name"] for tool in skill_tools or []}
        local_seen = untrusted_seen = False      # has a private folder / untrusted web text entered this turn yet?
        for round_index in range(3):  # at most a few tool rounds
            tools = ([DEEP_THINK_TOOL] if (round_index == 0 and hard and self.claude.available) else [])
            if skill_tools and skill_runner and round_index < 2:    # the last round must answer in words
                tools = tools + skill_tools
            tools = [tool for tool in tools if not (
                (local_seen and tool["function"]["name"] in EGRESS_TOOLS)
                or (untrusted_seen and tool["function"]["name"] in LOCAL_DATA_TOOLS))]
            tools = tools or None
            collected, calls = "", []
            holding = must_use_tool and round_index == 0 and bool(skill_names)   # buffer this reply until we know
            try:
                for event in (self.local.stream(self.history, tools, required=True) if holding
                              else self.local.stream(self.history, tools)):
                    if "text" in event:
                        collected += event["text"]
                        if not holding:
                            yield event["text"]
                    else:
                        calls = event["tool_calls"]
                if holding:
                    if calls:
                        collected = ""
                    elif "web_search" in skill_names:
                        # The model answered a live-data question from memory: throw that away and look it up instead.
                        spoken = text.split(": ", 1)[-1]
                        # A capitalised word is only a ticker if the sentence is about prices ("ASEAN" is not a stock).
                        about_prices = bool(STOCK_CUE_RE.search(spoken) or re.match(r"^\W*(?:and|what about|how about)\b", spoken, re.I))
                        ticker = next((word for word in re.findall(r"\b[A-Z]{2,5}\b", spoken)
                                       if word not in NOT_TICKERS), None) if about_prices else None
                        if ticker and "stock_quote" in skill_names:
                            fallback = {"name": "stock_quote", "arguments": json.dumps({"company": ticker})}
                        else:
                            fallback = {"name": "web_search", "arguments": json.dumps({"query": spoken[:120]})}
                        collected, calls = "", [{"id": "fallback-lookup", **fallback}]
                    else:
                        yield collected
            except GeneratorExit:  # interrupted (barge-in): remember what was actually said
                if collected.strip():
                    self.history.append({"role": "assistant", "content": collected})
                    self._cut_off = True
                raise
            if not calls:
                self.history.append({"role": "assistant", "content": collected})
                return
            self.history.append({"role": "assistant", "content": collected or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls]})
            round_names = {c["name"] for c in calls}
            refused = {c["id"]: why for c in calls if (why := self._refusal(c, round_names, local_seen, untrusted_seen))}
            local_seen |= any(c["name"] in LOCAL_DATA_TOOLS and c["id"] not in refused for c in calls)
            untrusted_seen |= any(c["name"] in UNTRUSTED_TOOLS and c["id"] not in refused for c in calls)
            said_filler = bool(collected.strip())
            if not said_filler and not all(c["name"] in skill_names or c["id"] in refused for c in calls):
                yield "Let me think about that. "    # deep thinking is always slow, so say so at once
                said_filler = True
            results: dict[str, str] = {}
            tool_started = time.monotonic()
            try:  # always leave a tool result per call, even when interrupted, or the history becomes invalid
                for c in calls:
                    if c["id"] in refused:                  # never run it; the model reads why
                        results[c["id"]] = refused[c["id"]]
                        continue
                    if c["name"] in skill_names:
                        future = self._tool_pool.submit(self._run_skill_tool, c, skill_runner)
                    else:
                        future = self._tool_pool.submit(self._run_tool, c, images, cancel)
                    reassurances_spoken = 0  # "Still working" fillers said so far (max 2)
                    while True:
                        try:
                            results[c["id"]] = future.result(timeout=0.4)  # short poll so `cancel` is checked often
                            break
                        except FutureTimeout:
                            if cancel and cancel():
                                results[c["id"]] = "(interrupted by the user)"
                                return
                            elapsed = time.monotonic() - tool_started
                            if not said_filler and elapsed > FILLER_AFTER_S:   # quick lookups need no "One moment"
                                said_filler = True
                                yield "One moment. "
                            if elapsed > 7 * (reassurances_spoken + 1) and reassurances_spoken < 2:  # at ~7s and ~14s
                                reassurances_spoken += 1
                                yield "Still working on it. "
            finally:
                self.stats = {"tool": True, "tool_ms": int((time.monotonic() - tool_started) * 1000)}
                for c in calls:
                    self.history.append({"role": "tool", "tool_call_id": c["id"],
                                         "content": results.get(c["id"], "(interrupted by the user)")})
            if cancel and cancel():
                return  # user moved on; don't spend another LLM call summarising for nobody
