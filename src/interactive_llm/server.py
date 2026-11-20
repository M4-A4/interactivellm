"""Realtime multimodal web server.

Browser -> server (WebSocket, binary):  0x01 + int16 16 kHz mic PCM | 0x02 + JPEG camera frame
Server -> browser (binary):             0x03 + int16 24 kHz TTS PCM
JSON both ways carries state, transcripts, latency and control messages.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import settings as user_settings
from .config import Config, WebConfig
from .engine import Engine, sentences
from .llm.local import LocalLLM
from .pipeline.models import SpeechModels
from .pipeline.people import FaceEngine, PeopleDB, PeopleManager, VoiceEngine
from .pipeline.vad import VADSegmenter
from .skills import build_skills, default_notes_path
from .skills.base import FAILED as SKILL_FAILED
from .skills.base import NEEDS_INPUT as SKILL_NEEDS_INPUT
from .skills.base import OK as SKILL_OK
from .skills.base import SkillContext
from .skills.catalog import DEFAULT_ENABLED, SKILL_CATALOG
from .skills.llm_tools import LIVE_CUE_RE, call_tool, tool_schemas
from .skills.manager import SkillManager, event_kind, result_note
from .skills.notes import NotesStore
from .pipeline.vision import FrameBuffer, asks_which_side, mirror_jpeg_b64, wants_vision

WEB_DIR = Path(__file__).parent / "web"
SESSIONS: set = set()  # live Session objects, so settings changes can be pushed to every open page
wcfg = WebConfig()
models = SpeechModels(wcfg)
_base = Config()
# Live user settings (persisted to data/settings.json); mutated by POST /api/settings.
current = user_settings.load({"context_tokens": _base.context_tokens, "context_keep_tokens": _base.context_keep_tokens,
                              "tts_engine": wcfg.tts_engine if wcfg.tts_engine in user_settings.TTS_ENGINE_IDS else "kokoro",
                              "tts_voice": wcfg.tts_voice if wcfg.tts_voice in user_settings.VOICE_IDS else "af_heart",
                              "tts_speed": wcfg.tts_speed, "camera_resolution": "640x480", "frame_max_side": 448, "vision_mode": wcfg.vision_mode,
                              "skills_enabled": DEFAULT_ENABLED,
                              "ollama_url": _base.ollama_url, "ollama_model": _base.ollama_model,
                              "lmstudio_url": _base.lmstudio_url, "lmstudio_model": _base.lmstudio_model})
wcfg.tts_engine, wcfg.tts_voice, wcfg.tts_speed = current["tts_engine"], current["tts_voice"], current["tts_speed"]



def apply_settings(cfg: Config) -> None:
    """Copy the live context-window and model-endpoint settings into a session's Config."""
    cfg.context_tokens, cfg.context_keep_tokens = current["context_tokens"], current["context_keep_tokens"]
    for key in (*user_settings.ENDPOINT_KEYS, *user_settings.MODEL_KEYS):
        setattr(cfg, key, current[key])


ready = asyncio.Event()  # set once all models are warm; connections and TTS previews wait on it
# Phrases that claim a name/spelling was changed or saved. Used only to catch the LLM claiming something
# the backend did not do.
CLAIM_RE = re.compile(
    r"\bI(?:'ve| have|'ll| will)?\s+(?:now\s+)?(?:updated|renamed|changed|corrected|fixed|saved|remembered|noted)\b[^.!?]{0,40}"
    r"\b(?:name|spelling|spelled)\b|\b(?:your|the) (?:name|spelling) (?:is|has been|was) (?:now )?(?:updated|changed|corrected)\b"
    r"|\bfrom now on\b[^.!?]{0,30}\b(?:call|address)\b|\b(?:call|address) you\b[^.!?]{0,30}\bfrom now on\b", re.I)
# Short follow-ups that clearly point back at a visual question ("what about now?", "and now?", "check again"). Pronouns and
# place words ("it", "here", "that") are deliberately NOT here: they appear in everyday talk and made the assistant narrate the room.
VISION_FOLLOWUP_RE = re.compile(r"\b(?:what about|how about|what now|and now|closer|further|farther)\b|\b(?:now|again)\s*[?.!]*\s*$", re.I)
VISION_FOLLOWUP_S = 20.0  # after a visual question, a pointed follow-up within this long still gets the camera image
MAX_AHEAD_S = 5.0  # max seconds of synthesized audio allowed to queue ahead of playback (keeps barge-in cheap)
# Past-tense claims that an action was carried out. Used only to catch the LLM claiming something no skill did.
SKILL_CLAIM_RE = re.compile(
    r"\bI(?:'ve| have)?\s+(?:just\s+)?(?:set|started|opened|launched|quit|closed|muted|unmuted|stored|searched|added|cancel+ed|"
    r"deleted|removed)\b[^.!?]{0,40}\b(?:timer|alarm|reminder|volume|app|note|search|safari|notes|calendar|music|mail|finder)\b"
    r"|\bI(?:'ve| have)?\s+(?:just\s+)?(?:searched|looked up|checked)\b[^.!?]{0,40}\b(?:web|online|price|results|stock)\b"
    r"|\bI(?:'ll| will)\s+(?:now\s+)?(?:search|look up|check)\b[^.!?]{0,40}\b(?:web|online|price|stock)\b"
    r"|\b(?:according to|from|based on) (?:a |the )?(?:live |recent )?(?:web|online|internet) (?:search|lookup|look-up|quote)"
    r"|\blive web (?:search|lookup|quote)\b"
    r"|\b(?:timer|reminder|alarm|note)s? (?:is|has been|was) (?:now )?(?:set|started|saved|added|cancel+ed|deleted)\b"
    r"|\bI (?:opened|launched|quit|closed) \w+", re.I)
# The part of SKILL_CLAIM_RE that only cites a source ("per a live web lookup"), as opposed to claiming an action.
CITE_RE = re.compile(r"\b(?:according to|from|based on) (?:a |the )?(?:live |recent )?(?:web|online|internet) (?:search|lookup|look-up|quote)"
                     r"|\blive web (?:search|lookup|quote)\b", re.I)


def figures(text: str) -> set[str]:
    """Numbers in `text` ("1,234.5" -> "1234.5"), used to tell repeated lookup data from invented data."""
    return set(re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")))


def repeats_backed_data(reply: str, backed: set[str]) -> bool:
    """True when the reply's only claim is a source citation and every number in it came from an earlier successful lookup.

    That is the assistant repeating real data it already fetched (for example restating a price while telling a joke), not
    claiming a lookup that never happened. An invented figure, or any action claim, is not covered."""
    if SKILL_CLAIM_RE.search(CITE_RE.sub("", reply)):
        return False
    found = figures(reply)
    return bool(found) and found <= backed


notes_store = NotesStore(default_notes_path())
faces = FaceEngine()
voice = VoiceEngine()
people_db = PeopleDB(wcfg.people_db, wcfg.face_match, wcfg.voice_match)


def _alive(url: str) -> bool:
    """Return True if the OpenAI-compatible server at `url` answers /models within 0.4 s."""
    import urllib.request
    try:
        urllib.request.urlopen(url.rstrip("/") + "/models", timeout=0.4)
        return True
    except Exception:
        return False


def safe_voice_embed(audio):
    """Return the speaker embedding of `audio`, or None if the voice model fails."""
    try:
        return voice.embed(audio)
    except Exception as error:  # a failing speaker model must never break the conversation
        print(f"[warn] voice embedding failed: {error}", flush=True)
        return None


def _warm_people() -> None:
    """Load the face and voice models and push dummy input through them (runs in a worker thread)."""
    faces.load()
    voice.load()
    voice.embed(np.zeros(32000, np.float32))
    faces.detect(np.zeros((240, 320, 3), np.uint8))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start model warm-up in the background at startup; cancel it on shutdown."""
    async def warm():
        """Warm every model, then mark the server ready."""
        warm_started = time.monotonic()
        try:
            await models.warmup()
            await asyncio.get_running_loop().run_in_executor(None, _warm_people)
        except Exception as error:    # without this a failed warm-up leaves the server silently never ready
            print(f"[fatal] model warm-up failed: {type(error).__name__}: {error}", flush=True)
            raise
        cfg = Config()
        apply_settings(cfg)
        try:  # prefill the local LLM so the first real turn is fast
            url, model = cfg.local_endpoint()
            await asyncio.get_running_loop().run_in_executor(None, lambda: list(LocalLLM(url, model).stream(
                [{"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": "hi"}])))
        except Exception as error:
            print(f"[warn] LLM warmup failed: {error}")
        ready.set()
        print(f"[ready] models warm in {time.monotonic() - warm_started:.1f}s — open http://{wcfg.host}:{wcfg.port}", flush=True)
    task = asyncio.create_task(warm())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _split_host(value: str) -> tuple[str, str]:
    """('127.0.0.1', '8000') from '127.0.0.1:8000'; ('::1', '8000') from '[::1]:8000'."""
    value = value.strip().lower()
    if value.startswith("["):
        host, _, rest = value[1:].partition("]")
        return host, rest.lstrip(":")
    host, _, port = value.partition(":")
    return host, port


def reject_reason(headers, method: str, bound_host: str) -> str | None:
    """Why a request to this local server must be refused, or None if it may proceed.

    The server has no login, so it must only talk to the page it served itself. Any other web page open in the
    user's browser can send requests to 127.0.0.1, and DNS rebinding can even make a hostile site look same-origin.
    Three rules: the Host header must be this machine's own name (blocks rebinding); a browser's Origin header, when
    present, must be the same site as Host (blocks cross-site pages, which always send Origin on WebSocket and POST);
    and state-changing requests must be JSON (blocks "simple" cross-site form posts). Scripts that send no Origin,
    such as the test clients, are still allowed: only a browser can be tricked into acting for another site.
    """
    host_header = headers.get("host", "")
    hostname, _ = _split_host(host_header)
    if bound_host not in ("0.0.0.0", "::", "") and hostname not in LOOPBACK_HOSTS | {bound_host.lower()}:
        return f"unexpected Host header {host_header!r}"
    origin = headers.get("origin")
    if origin is not None:
        parsed = urlparse(origin)
        if parsed.scheme not in ("http", "https") or parsed.netloc.lower() != host_header.strip().lower():
            return f"cross-site request from origin {origin!r}"
    if method not in ("GET", "HEAD", "OPTIONS"):
        if headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            return "state-changing requests must be application/json"
    return None


@app.middleware("http")
async def guard_and_revalidate(request: Request, call_next):
    """Refuse requests that did not come from our own page, then make browsers revalidate the GUI files on every
    load so a stale cached app.js never runs against a newer page."""
    reason = reject_reason(request.headers, request.method, wcfg.host)
    if reason:
        print(f"[blocked] {request.method} {request.url.path}: {reason}", flush=True)
        return JSONResponse({"ok": False, "errors": ["Request refused: it did not come from this app's own page."]},
                            status_code=403)
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class Session:
    """State and behaviour for one browser connection: audio/video in, conversation turns, audio out."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.cfg = Config()
        apply_settings(self.cfg)
        self.loop = asyncio.get_running_loop()
        self.engine = Engine(self.cfg)
        self._wire_engine()
        self.vad = VADSegmenter(wcfg.vad_threshold, wcfg.end_silence_ms, wcfg.barge_in_ms)
        self.frames = FrameBuffer(wcfg.scene_change_threshold)
        self.people = PeopleManager(people_db, faces, wcfg.auto_learn)
        self._overlay_t = 0.0  # monotonic time of the last face-overlay message (throttled to 4 per second)
        self._turn_success = False   # did the backend really learn/rename/forget someone this turn?
        self._carry_note = ""        # correction for the assistant after it claimed a change that never happened
        self._skill_success = False  # did a skill really succeed this turn?
        self._backed_numbers: set[str] = set()   # numbers returned by successful lookups earlier in this session
        self._last_vision_at = -1e9   # monotonic time a camera image was last attached (drives the follow-up window)
        self._unfinished = ""        # note for the NEXT turn: a skill request that failed and is still unanswered
        self._skill_handled = False  # did the command layer already handle this turn (so no tools are offered)?
        self.skills = SkillManager(build_skills(notes_store), current["skills_enabled"])
        self.available: list[str] = ["ollama", "lmstudio"]
        self._people_sig = ""  # JSON of the last people list sent, to avoid resending unchanged lists
        self.speak = True  # whether replies are synthesized and sent as audio
        self.state = "idle"  # idle | listening | thinking | speaking (mirrored to the browser)
        self.confirmed = False  # current speech has been confirmed as a real interruption
        self.task: asyncio.Task | None = None  # the running conversational turn
        self.cancel = threading.Event()  # set to abort the current turn (also read by the LLM worker thread)
        self.llm_done: asyncio.Future | None = None  # resolves when the LLM worker thread has finished
        self.play_until = 0.0  # monotonic time at which already-sent audio finishes playing
        self.spoken: list[tuple[str, float, float]] = []  # (sentence, start offset s, duration s) sent this turn
        self.play_t0 = 0.0  # monotonic time at which the first sentence of this turn starts playing
        self.heard_text: str | None = None  # what the user heard before interrupting (applied by the worker)

    # ---- helpers -------------------------------------------------------------------
    async def send(self, **msg) -> None:
        """Send a JSON message to the browser, ignoring a closed socket."""
        try:
            await self.ws.send_text(json.dumps(msg))
        except Exception:
            pass

    async def set_state(self, state: str) -> None:
        """Switch the session state and tell the browser, but only if it actually changed."""
        if state != self.state:
            self.state = state
            await self.send(type="state", value=state)

    def busy(self) -> bool:
        """True while a conversational turn is running."""
        return self.task is not None and not self.task.done()

    def _heard(self) -> str:
        """Text of the reply that had actually been played out loud by now (sentence + partial)."""
        elapsed = time.monotonic() - self.play_t0
        parts: list[str] = []
        for text, offset, duration in self.spoken:
            if elapsed >= offset + duration:
                parts.append(text)
            elif elapsed > offset:  # mid-sentence: estimate how many words were heard, assuming even pacing
                words = text.split()
                parts.append(" ".join(words[: int(len(words) * (elapsed - offset) / duration)]))
                break
            else:
                break
        return " ".join(part for part in parts if part).strip()

    async def interrupt(self) -> None:
        """Stop the current reply (barge-in/user action): cancel the turn, flush client audio, keep what was heard."""
        was_speaking = self.speak and (self.busy() or self.state == "speaking")
        heard = self._heard() if was_speaking else None
        if self.busy():
            self.heard_text = heard  # the LLM worker applies this once it has closed the reply
        elif heard is not None:
            self.engine.replace_last_assistant(heard)
        if self.busy():
            self.cancel.set()
            self.task.cancel()
        if self.busy() or self.state == "speaking":  # client may still hold buffered audio
            await self.send(type="stop_audio")
        self.play_until = 0.0
        await self.set_state("listening" if self.vad.active else "idle")

    # ---- input ---------------------------------------------------------------------
    async def on_audio(self, pcm: bytes) -> None:
        """Feed mic PCM to the VAD and react to its start / confirm / end events."""
        for event, audio in self.vad.feed(pcm):
            if event == "start":
                self.confirmed = False
                self.frames.mark_start()
                if not self.busy():
                    await self.set_state("listening")
            elif event == "confirm":
                self.confirmed = True
                await self.interrupt()  # real speech over the assistant = barge-in
                await self.set_state("listening")
            elif event == "end":
                # Drop blips, and drop speech that ended while a reply was playing but never became a real interruption.
                if audio is None or (self.busy() and not self.confirmed):
                    if not self.busy():
                        await self.set_state("idle")
                    continue
                await self.interrupt()
                self.task = asyncio.create_task(self.run_turn(audio=audio, speech_ended_at=time.monotonic()))

    async def on_frame(self, bgr: np.ndarray) -> None:
        """Run face tracking on a camera frame and send the face overlay to the browser (max 4 per second)."""
        try:
            self.people.update(bgr)
        except Exception as error:
            print(f"[warn] face processing failed: {error}", flush=True)
            return
        now = time.monotonic()
        if now - self._overlay_t >= 0.25:
            self._overlay_t = now
            await self.send(type="faces", faces=self.people.overlay(now))

    async def push_people(self, force: bool = False) -> None:
        """Send the known-people list to the browser when it changed (or always, if `force`)."""
        signature = json.dumps(people_db.summary())
        if force or signature != self._people_sig:
            self._people_sig = signature
            await self.send(type="people", people=people_db.summary())

    async def vlm_pick(self, jpeg: bytes, text: str, ids: list[int]) -> int | None:
        """Ask the vision model which numbered face is being introduced / pointed at."""
        jpeg_b64 = base64.b64encode(jpeg).decode()
        prompt = (f'Numbered yellow boxes mark faces. The speaker just said: "{text}". Which numbered person is being '
                  f"introduced or pointed at (pointing, gesture, or a described position or clothing)? Reply with only "
                  f"the number, or 0 if unclear. Options: {ids}.")
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}}]}]
        try:
            answer = await asyncio.get_running_loop().run_in_executor(
                None, lambda: self.engine.local.complete(messages, max_tokens=8))
        except Exception as error:
            print(f"[warn] pointing disambiguation failed: {error}", flush=True)
            return None
        number = re.search(r"\d+", answer)
        return int(number.group()) if number and int(number.group()) in ids else None

    async def recognise(self, text: str, audio, voice_embedding,
                        speech_ended_at: float) -> tuple[str | None, str | None]:
        """Work out who is speaking, learn from introductions, and build the per-turn note.

        Returns ``(context note for the LLM or None, speaker name or None)``.
        """
        duration_s = len(audio) / 16000 if audio is not None else 0.0
        last_word_at = speech_ended_at - 0.5  # the VAD ends ~0.55 s after the last word
        # Window used to match the voice to faces seen talking: the utterance span plus 0.2 s of slack.
        window = (last_word_at - duration_s - 0.2, last_word_at)
        speaker = self.people.identify_speaker(voice_embedding, *window)
        notes: list[str] = []
        skill_context = SkillContext(
            speaker=speaker.name, known_names=people_db.names(),
            visible_names=[track.name for track in self.people.visible() if track.name], announce=self.announce,
            progress=self.show_progress)
        skill_result = None
        try:
            skill_result = await self.skills.handle(text, skill_context)
        except Exception as error:
            print(f"[warn] skill failed: {error}", flush=True)
        earlier_unfinished, self._unfinished = self._unfinished, ""
        if skill_result and skill_result.kind in (SKILL_FAILED, SKILL_NEEDS_INPUT):
            # The model forgets that a request failed (per-turn notes are not kept in history) and may later treat a bare
            # "ok" or "sure" as agreement and invent the answer, so remind it on the next turn.
            self._unfinished = (f"[System: earlier the user asked \"{text[:90]}\" and that request was NOT completed "
                                f"({skill_result.event}). It is still unanswered: never answer it from memory, and do not treat "
                                "a later 'ok', 'sure' or 'yes' as a new request unless they ask again.]")
        self._skill_success = bool(skill_result and skill_result.kind == SKILL_OK)
        if self._skill_success:
            self._remember_backed(skill_result.note)
        self._skill_handled = skill_result is not None
        if skill_result:
            await self.send(type="skill_event", kind=event_kind(skill_result), text=skill_result.event, detail=skill_result.detail)
            self._note_private(skill_result)
            notes.append(result_note(skill_result))
        try:
            if skill_result:
                self.people.co_learn(speaker, voice_embedding, duration_s, t0=window[0], t1=window[1])   # a skill handled it; skip naming rules
            elif self.people.needs_processing(text):
                notes = await self.people.apply_utterance(
                    text, speaker, voice_embedding, self.vlm_pick, spoken=audio is not None)
            else:
                self.people.co_learn(speaker, voice_embedding, duration_s, t0=window[0], t1=window[1])
        except Exception as error:
            print(f"[warn] people learning failed: {error}", flush=True)
        events = self.people.drain_events()     # live naming/renaming/forgetting outcomes -> GUI
        self._turn_success = any(event["kind"] in ("learned", "renamed", "forgot") for event in events)
        for event in events:
            await self.send(type="people_event", **event)
        await self.send(type="speaker", name=speaker.name, how=speaker.how)
        await self.push_people()
        remembered = notes_store.context([*skill_context.visible_names, *([speaker.name] if speaker.name else [])])
        note = " ".join([self._carry_note, earlier_unfinished, self.people.context_note(speaker), remembered, *notes]).strip()
        self._carry_note = ""
        return (note or None), speaker.name

    def _needs_live_data(self, text: str) -> bool:
        """True if the user's words ask for live data, including a short follow-up right after a stock quote ("and VOO?")."""
        if LIVE_CUE_RE.search(text):
            return True
        search = self.skills.skills.get("web_search")
        return bool(search and search.recent_stock()
                    and re.search(r"^\W*(?:and|what about|how about|then|ok|okay)\b|\b[A-Z]{2,5}\b", text))

    async def skill_tool(self, name: str, args: dict) -> str:
        """Run a model-requested read-only tool, show its outcome in the GUI, and return the text the model reads."""
        result = await call_tool(self.skills.skills, self.skills.enabled, name, args, SkillContext(progress=self.show_progress))
        await self.send(type="skill_event", kind=event_kind(result), text=result.event, detail=result.detail)
        self._note_private(result)
        if result.kind == SKILL_OK:
            self._skill_success = True
            self._remember_backed(result.note)
        return result.note

    def _remember_backed(self, note: str) -> None:
        """Keep the numbers a successful lookup returned so a later repeat of them is not mistaken for an invented claim."""
        if len(self._backed_numbers) > 500:
            self._backed_numbers.clear()
        self._backed_numbers |= figures(note)

    async def show_progress(self, text: str) -> None:
        """Show a "working on it" row in the GUI while a slow skill runs."""
        await self.send(type="skill_event", kind="working", text=text)

    async def announce(self, spoken: str, shown: str) -> None:
        """Speak and display something unprompted (a finished timer), queued behind any audio still playing."""
        await self.send(type="skill_event", kind="done", text=shown)
        await self.send(type="assistant_start")
        await self.send(type="assistant_delta", text=spoken)
        await self.send(type="assistant_done")
        if not self.speak:
            return
        try:
            pcm = await models.synthesize(spoken)
        except Exception as error:
            print(f"[warn] could not speak alert: {error}", flush=True)
            return
        if pcm:
            start, duration = max(self.play_until, time.monotonic()), len(pcm) / 2 / 24000
            self.play_until = start + duration
            try:
                await self.ws.send_bytes(b"\x03" + pcm)
            except Exception:
                pass

    def close(self) -> None:
        """Stop everything this session started (turn worker, timers)."""
        self.cancel.set()
        if self.task:
            self.task.cancel()
        self.skills.skills["timers"].shutdown()

    async def on_text(self, text: str) -> None:
        """Handle typed input: interrupt any reply and start a new turn."""
        await self.interrupt()
        self.task = asyncio.create_task(self.run_turn(text=text, speech_ended_at=time.monotonic()))

    # ---- one conversational turn ---------------------------------------------------
    def _llm_worker(self, loop, queue: asyncio.Queue, text, images, force_claude, cancel,
                    note=None, speaker=None, offer_tools=False, mirrored=False) -> None:
        """Run the (blocking) LLM reply in a worker thread and post results to the event loop.

        Items put on `queue` are ``("delta", text)``, ``("sentence", text)``, ``("error", message)``, and finally
        ``None`` as an end marker. `cancel` is a threading.Event set on barge-in.
        """
        def put(item):
            """Thread-safely hand `item` to the asyncio queue on the event loop."""
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def chunks(reply_stream):
            """Pass text chunks through, also posting each as a raw delta for the live transcript."""
            for chunk in reply_stream:
                if cancel.is_set():
                    return
                put(("delta", chunk))
                yield chunk
        def run_tool(name: str, args: dict) -> str:
            """Blocking bridge from the worker thread to the async skill call."""
            return asyncio.run_coroutine_threadsafe(self.skill_tool(name, args), loop).result(timeout=25)

        tools = tool_schemas(self.skills.enabled) if offer_tools else None
        reply_stream = self.engine.reply(text, images=images, force_claude=force_claude, note=note, speaker=speaker,
                                cancel=cancel.is_set, skill_tools=tools, skill_runner=run_tool if tools else None,
                                         must_use_tool=bool(tools) and self._needs_live_data(text), mirrored=mirrored)
        try:
            for sentence in sentences(chunks(reply_stream)):
                if cancel.is_set():
                    break
                put(("sentence", sentence))
        except Exception as error:
            put(("error", f"{type(error).__name__}: {error}"))
        finally:
            reply_stream.close()  # on interruption this stores the partial reply in history
            if cancel.is_set() and self.heard_text is not None:
                self.engine.replace_last_assistant(self.heard_text)
            put(None)
            try:
                self.engine.compact()  # background: runs after the reply, the next turn waits only if it's still busy
            except Exception as error:
                print(f"[warn] compaction failed: {error}", flush=True)
            asyncio.run_coroutine_threadsafe(self.send(type="context", **self.engine.context_info()), loop)

    async def run_turn(self, audio: np.ndarray | None = None, text: str | None = None,
                       speech_ended_at: float = 0.0) -> None:
        """Run one turn: transcribe, recognise people, stream the LLM reply, synthesize and send audio.

        Pass either `audio` (spoken input) or `text` (typed input). `speech_ended_at` is the monotonic time the
        user finished, the origin for the time-to-first-audio measurement.
        """
        cancel = self.cancel = threading.Event()
        self.spoken, self.heard_text = [], None
        latency_ms: dict[str, int] = {}
        try:
            await self.set_state("thinking")
            voice_embedding = None
            if audio is not None:
                stt_started = time.monotonic()
                text, voice_embedding = await asyncio.gather(
                    models.transcribe(audio, people_db.names()),   # names on record bias Whisper's spelling
                    asyncio.get_running_loop().run_in_executor(None, safe_voice_embed, audio))
                latency_ms["stt_ms"] = int((time.monotonic() - stt_started) * 1000)
            if not text:
                await self.set_state("idle")
                return
            await self.send(type="transcript", role="user", text=text)

            note, speaker = await self.recognise(text, audio, voice_embedding, speech_ended_at)
            force_claude = text.lower().startswith(("/claude ", "/think "))
            if force_claude:
                text = text.split(" ", 1)[1]
            frames = self.frames.pick()
            # "Follow-up" turns ("what about now?", "it's here") carry no visual word but still mean the scene, so in auto
            # mode the image keeps being attached for a minute after the last visual turn.
            in_followup = (time.monotonic() - self._last_vision_at < VISION_FOLLOWUP_S and len(text.split()) <= 8
                           and bool(VISION_FOLLOWUP_RE.search(text)))
            asked_about_vision = wants_vision(text, current["vision_mode"]) or (current["vision_mode"] == "auto" and in_followup)
            images = frames if asked_about_vision else []
            mirrored = bool(images) and asks_which_side(text)
            if mirrored:      # "is it on my left?": show the model the mirror view so its left is the user's left
                images = [mirror_jpeg_b64(image) for image in images]
            if images:
                self._last_vision_at = time.monotonic()
            if not frames and self.frames.latest is not None and not asked_about_vision:
                # The camera was on earlier in this session but has stopped. Earlier scene descriptions in the history
                # are now out of date, and the model tends to repeat them if asked "really?" or "are you sure?".
                note = (note + " " if note else "") + (
                    "[System: the camera is now off, so you cannot see the scene. Do not describe the room, people or "
                    "objects from earlier replies as if you can still see them. Only if the user asks about what is visible, "
                    "say you cannot see right now; otherwise do not mention the camera.]")
            if asked_about_vision and not frames:
                # No camera frame exists (camera off, busy or denied): the model must not invent a description.
                note = (note + " " if note else "") + (
                    "[System: no camera image is available right now. If the user asks what you see, say you "
                    "cannot see anything at the moment; do not describe or guess.]")
            await self.send(type="vision", frames=len(images))
            if self.llm_done:
                await self.llm_done  # previous worker must finish touching history first

            loop = asyncio.get_running_loop()
            worker_queue: asyncio.Queue = asyncio.Queue()
            llm_started = time.monotonic()
            self.llm_done = loop.run_in_executor(None, self._llm_worker, loop, worker_queue, text, images,
                                                 force_claude, cancel, note, speaker,
                                                 not self._skill_handled and not force_claude, mirrored)
            audio_started, first_delta, reply_text = False, True, ""
            await self.send(type="assistant_start")
            while (item := await worker_queue.get()) is not None:
                kind, payload = item
                if kind == "delta":
                    reply_text += payload
                    if first_delta:
                        latency_ms["llm_first_ms"] = int((time.monotonic() - llm_started) * 1000)
                        first_delta = False
                    await self.send(type="assistant_delta", text=payload)
                elif kind == "error":
                    await self.send(type="error", text=payload)
                elif kind == "sentence" and self.speak:
                    while self.play_until - time.monotonic() > MAX_AHEAD_S and not cancel.is_set():
                        await asyncio.sleep(0.1)  # don't run far ahead of playback; keeps barge-in cheap
                    if cancel.is_set():
                        break
                    pcm = await models.synthesize(payload)
                    if cancel.is_set():
                        break
                    if pcm:
                        if not audio_started:
                            latency_ms["first_audio_ms"] = int((time.monotonic() - speech_ended_at) * 1000)
                            await self.send(type="latency", **latency_ms)
                            await self.set_state("speaking")
                        audio_started = True
                        # int16 mono at 24 kHz: 2 bytes per sample. Queue behind audio already scheduled.
                        start, duration = max(self.play_until, time.monotonic()), len(pcm) / 2 / 24000
                        if not self.spoken:
                            self.play_t0 = start
                        self.spoken.append((payload, start - self.play_t0, duration))
                        self.play_until = start + duration
                        await self.ws.send_bytes(b"\x03" + pcm)  # 0x03 = TTS audio frame tag
            if CLAIM_RE.search(reply_text) and not self._turn_success and not cancel.is_set():
                # The assistant said it changed a name but the backend did nothing: never let that stand unflagged.
                await self.send(type="people_event", kind="failed",
                                text="The assistant said it changed a name, but nothing was changed")
                self._carry_note = ("[System: your previous reply claimed a name change that did NOT happen; nothing "
                                    "was changed. Correct yourself in one short sentence.]")
            if (SKILL_CLAIM_RE.search(reply_text) and not self._skill_success and not self._turn_success
                    and not cancel.is_set() and not repeats_backed_data(reply_text, self._backed_numbers)):
                # The assistant said it carried out an action but no skill ran (or the skill failed).
                await self.send(type="skill_event", kind="failed",
                                text="The assistant said it did something, but no skill carried it out")
                self._carry_note = ("[System: your previous reply claimed an action was done, but NO skill ran and "
                                    "nothing happened. Correct yourself in one short sentence.]")
            stats = self.engine.stats
            if stats.get("tool"):
                latency_ms["think_ms"] = stats["tool_ms"]
            print(f"[turn] stt={latency_ms.get('stt_ms', '-')}ms llm1st={latency_ms.get('llm_first_ms', '-')}ms "
                  f"audio1st={latency_ms.get('first_audio_ms', '-')}ms frames={len(images)} "
                  f"deep={stats.get('tool_ms', '-')}ms "
                  f"ctx={self.engine.tokens()}t text={text[:40]!r}", flush=True)
            await self.send(type="assistant_done")
            if not audio_started:
                await self.send(type="latency", **latency_ms)
                await self.set_state("idle")
            # if we spoke, the client reports playback finished -> idle
        except asyncio.CancelledError:
            cancel.set()
            raise

    async def handle_json(self, message: dict) -> None:
        """Dispatch a JSON control message from the browser (text, forget, interrupt, playback, config, reset)."""
        message_type = message.get("type")
        if message_type == "text" and message.get("text", "").strip():
            await self.on_text(message["text"].strip())
        elif message_type == "forget" and message.get("name"):
            gone = people_db.forget(message["name"])
            await self.send(type="people_event", kind="forgot" if gone else "failed",
                            text=f"Forgot {message['name']}" if gone
                            else f"Couldn't forget {message['name']}: not on record")
            for track in self.people.tracks.values():  # un-name live tracks so the overlay updates immediately
                if track.name and track.name.lower() == message["name"].lower():
                    track.name, track.score = None, 0.0
            await self.push_people(force=True)
        elif message_type == "interrupt":
            await self.interrupt()
        elif message_type == "playback" and message.get("state") == "idle" and not self.busy():
            # Client finished playing buffered audio and no turn is running.
            await self.set_state("listening" if self.vad.active else "idle")
        elif message_type == "playback" and message.get("state") == "idle" and self.state == "speaking":
            await self.set_state("idle")
        elif message_type == "config":
            if "speak" in message:
                self.speak = bool(message["speak"])
            if message.get("backend") in ("ollama", "lmstudio") and message["backend"] != self.cfg.backend:
                self.cfg.backend = message["backend"]
                await self.rebuild_engine()
                await self.send(type="info", **self.info())
        elif message_type == "reset":
            await self.interrupt()
            self.engine.reset()
            await self.send(type="context", **self.engine.context_info())

    async def refresh_available(self) -> None:
        """Probe both backends (0.4 s each) and remember which answer, for the header selector."""
        loop = asyncio.get_running_loop()
        up = await asyncio.gather(loop.run_in_executor(None, _alive, self.cfg.ollama_url),
                                  loop.run_in_executor(None, _alive, self.cfg.lmstudio_url))
        self.available = [name for name, is_up in zip(("ollama", "lmstudio"), up) if is_up] or [self.cfg.backend]

    async def rebuild_engine(self) -> None:
        """Switch to the endpoint now in `self.cfg`, keeping the conversation so far."""
        await self.interrupt()
        new_engine = Engine(self.cfg)
        new_engine.adopt(self.engine)
        self.engine = new_engine
        self._wire_engine()

    def _wire_engine(self) -> None:
        """Let the engine tell the GUI, from its worker thread and before the call, when something goes to the cloud."""
        def tell(text: str) -> None:
            asyncio.run_coroutine_threadsafe(self.send(type="skill_event", kind="info", text=text), self.loop)
        self.engine.on_cloud = tell

    def _note_private(self, result) -> None:
        """Remember file names the assistant just listed so they are never put into a web search or a cloud call."""
        names = result.detail if (result.skill == "mac_control" and isinstance(result.detail, list)) else []
        for name in names:
            plain = str(name).rstrip("/").lower()
            stem = plain.rsplit(".", 1)[0]
            for candidate in {plain, stem}:
                if len(candidate) >= 8 or ("." in candidate) or any(ch.isdigit() for ch in candidate) and len(candidate) >= 5:
                    self.engine.private_names.add(candidate)
        if len(self.engine.private_names) > 400:           # bounded: keep the newest listing's names only
            self.engine.private_names = {n.rstrip("/").lower() for n in names}

    def info(self) -> dict:
        """Return the backend/model summary sent to the browser as the ``info`` message."""
        return {"available": self.available, "backend": self.cfg.backend, "model": self.engine.local.model,
                "deep": self.engine.claude.label, "tts": f"{wcfg.tts_engine}/{wcfg.tts_voice}", "stt": wcfg.whisper_repo.split("/")[-1]}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """Serve one browser connection: binary frames carry mic/camera data, text frames carry JSON messages."""
    reason = reject_reason(ws.headers, "GET", wcfg.host)
    if reason:                                   # refused before the handshake completes, so no data is ever sent
        print(f"[blocked] websocket: {reason}", flush=True)
        await ws.close(code=1008)
        return
    await ws.accept()
    await ready.wait()
    session = Session(ws)
    SESSIONS.add(session)
    await session.refresh_available()
    await session.send(type="info", **session.info())
    await session.send(type="state", value="idle")
    await session.send(type="context", **session.engine.context_info())
    await session.push_people(force=True)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                tag, payload = msg["bytes"][0], msg["bytes"][1:]
                if tag == 1:  # 0x01: int16 16 kHz mic PCM
                    await session.on_audio(payload)
                elif tag == 2:  # 0x02: JPEG camera frame
                    bgr = session.frames.push(payload)
                    if bgr is not None:
                        await session.on_frame(bgr)
            elif msg.get("text"):
                await session.handle_json(json.loads(msg["text"]))
    finally:
        SESSIONS.discard(session)
        session.close()


def _settings_payload() -> dict:
    """Current settings plus the voice list and numeric limits the settings dialog needs."""
    return {**current, "voices": [{"id": voice_id, "label": label} for voice_id, label in user_settings.VOICES],
            "tts_engines": user_settings.TTS_ENGINES,
            "frame_sides": user_settings.FRAME_SIDES,
            "vision_modes": [{"id": mode, "label": label} for mode, label in user_settings.VISION_MODES], "skills": SKILL_CATALOG,
            "limits": {**user_settings.LIMITS, "keep_min": user_settings.KEEP_MIN}}


@app.get("/api/settings")
async def get_settings():
    """Return the current settings and their limits."""
    return _settings_payload()


@app.post("/api/settings")
async def post_settings(request: Request):
    """Validate and apply. Nothing changes unless every submitted value is acceptable."""
    try:
        raw = await request.json()
    except ValueError:
        raw = None
    if not isinstance(raw, dict):
        return JSONResponse({"ok": False, "errors": ["Expected a JSON object"]}, status_code=400)
    new, errs = user_settings.validate(raw, current)
    if errs:
        return JSONResponse({"ok": False, "errors": errs}, status_code=422)
    if new["tts_engine"] != current["tts_engine"]:
        try:      # load (and on first use download) the new voice engine before committing to it
            await models.ensure_engine(new["tts_engine"])
        except Exception as error:
            return JSONResponse({"ok": False, "errors": [f"Could not load that voice engine ({type(error).__name__}: {error}). "
                                                         "Nothing was changed."]}, status_code=422)
    current.update(new)
    wcfg.tts_engine, wcfg.tts_voice, wcfg.tts_speed = current["tts_engine"], current["tts_voice"], current["tts_speed"]
    for session in list(SESSIONS):  # push the new settings to every open page
        endpoint_before = session.cfg.local_endpoint()
        apply_settings(session.cfg)
        session.skills.set_enabled(current["skills_enabled"])
        try:
            if session.cfg.local_endpoint() != endpoint_before:
                await session.rebuild_engine()      # a different server or model: continue the conversation on it
            await session.refresh_available()
            await session.send(type="info", **session.info())
            await session.send(type="context", **session.engine.context_info())
        except Exception:
            pass
    try:
        user_settings.save(current)
    except OSError as error:
        return JSONResponse({"ok": True, "saved": False, "warning": f"Applied, but could not save: {error}",
                             **_settings_payload()})
    return {"ok": True, "saved": True, **_settings_payload()}


@app.post("/api/llm/models")
async def llm_models(request: Request):
    """List the models offered by an OpenAI-compatible endpoint (``GET <url>/models``) so the settings page can verify it."""
    try:
        body = await request.json()
    except ValueError:
        body = None
    url = user_settings.clean_url(body.get("url", "")) if isinstance(body, dict) else None
    if not url:
        return JSONResponse({"ok": False, "error": "Enter a web address such as http://localhost:11434/v1"}, status_code=422)

    def fetch() -> list[str]:
        import urllib.request
        with urllib.request.urlopen(url + "/models", timeout=3) as page:
            return sorted({row["id"] for row in json.load(page)["data"]})
    try:
        names = await asyncio.get_running_loop().run_in_executor(None, fetch)
    except Exception as error:
        return {"ok": False, "error": f"Could not reach it ({type(error).__name__})"}
    return {"ok": True, "models": names}


@app.post("/api/tts/preview")
async def tts_preview(request: Request):
    """Speak a short sample with the (unsaved) voice/speed from the form; returns a WAV."""
    await ready.wait()
    raw = await request.json()
    voice_id, speed = raw.get("tts_voice"), raw.get("tts_speed", current["tts_speed"])
    probe, errs = user_settings.validate({"tts_engine": raw.get("tts_engine", current["tts_engine"]), "tts_voice": voice_id,
                                          "tts_speed": speed}, current)
    if errs:
        return JSONResponse({"ok": False, "errors": errs}, status_code=422)
    try:
        pcm = await models.preview("Hi, this is how I sound. How can I help?", probe["tts_voice"], probe["tts_speed"],
                                   probe["tts_engine"])
    except Exception as error:
        return JSONResponse({"ok": False, "errors": [f"Voice failed to load: {error}"]}, status_code=500)
    import io, wave
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:  # mono, 16-bit, 24 kHz: the Kokoro output format
        wav_file.setnchannels(1); wav_file.setsampwidth(2); wav_file.setframerate(24000); wav_file.writeframes(pcm)
    return Response(wav_buffer.getvalue(), media_type="audio/wav")


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main() -> None:
    """Start the web server (blocking)."""
    uvicorn.run(app, host=wcfg.host, port=wcfg.port, log_level="warning")


if __name__ == "__main__":
    main()
