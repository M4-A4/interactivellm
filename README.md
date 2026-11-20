# Interactive_LLM

Multimodal interactive assistant in Python. Local models (Ollama / LM Studio) handle live
conversation; Claude handles hard reasoning.

## Architecture
- `llm/local.py` – OpenAI-compatible streaming client (Ollama `:11434/v1`, LM Studio `:1234/v1`)
- `llm/claude.py` – Anthropic client (escalation target, accepts a camera frame)
- `engine.py` – history, streaming, sentence chunking for TTS, **`ask_claude` tool**: the local
  model decides when to escalate; `/claude` forces it
- `audio/stt.py` – mic + energy VAD + faster-whisper; `audio/tts.py` – macOS `say`
- `vision/camera.py` – OpenCV frame grab → JPEG base64 (sent to the local vision model)
- `app.py` – CLI

## Run
    cp .env.example .env        # add ANTHROPIC_API_KEY for Claude escalation
    uv sync --extra dev
    uv run interactive-llm                  # text chat
    uv run interactive-llm --voice --speak  # talk and listen
    uv run pytest

Commands: `/cam <msg>`, `/claude <msg>`, `/voice`, `/speak`, `/backend`, `/reset`, `/quit`.
macOS will prompt for microphone and camera permission for your terminal on first use.

## Realtime web app (continuous audio + vision)
    uv run interactive-llm-web      # then open http://127.0.0.1:8000 and press Start

![Realtime web app architecture. The browser streams microphone audio (16 kHz PCM, echo cancelled) and camera frames (about 2 fps) over one WebSocket to the server, which runs Silero VAD, then MLX Whisper, then a local LLM with the camera frames, then Kokoro TTS. The spoken reply streams back sentence by sentence. Talking over the assistant cancels the reply (barge-in).](docs/architecture.svg)

Tunables (environment variables): `END_SILENCE_MS`, `BARGE_IN_MS`, `VAD_THRESHOLD`, `MLX_WHISPER_REPO`, `KOKORO_VOICE`.

### Running fully local (no API key)
The "deep think" tier uses Claude only if `ANTHROPIC_API_KEY` is set; otherwise the same local
model is called again with thinking enabled (`DEEP_EFFORT=low|medium|high`, default `low`:
~5 s and correct on a bat-and-ball test vs 30-50 s at `high`). The assistant says "Let me think
about that" while it works. `/think <msg>` forces it. Fast chat has thinking off.

## Knowing who is talking (learned from conversation)
Say **"I'm Sam"**, **"my name is Sam"**, or **"this is Priya"** / **"meet my friend Dana"** (typing works too).
- One unrecognised face in view -> that face is learned. Several -> numbered boxes are drawn on the frame and the
  vision model picks the one you point at / describe; if it can't tell, nothing is learned and the assistant asks.
- "I'm X" also learns your voice. Afterwards a known face talking alone teaches the system their voice (and vice
  versa); disable with `AUTO_LEARN=0`.
- Who is speaking: voice match -> only face in view -> lip motion (experimental, only when several faces are visible).
- Each turn the LLM gets a hidden note ("In view, left to right: Sam, unrecognised person. Speaker: Sam"), and history
  is labelled per speaker. The assistant is told never to identify anyone from appearance.
- **"Forget me" / "forget Sam"**, or the x on a chip in the UI, deletes a person.
- Stack: OpenCV YuNet + SFace (faces), SpeechBrain ECAPA (voices). Embeddings (not photos or audio) are stored in
  `data/people.json` (owner-only, git-ignored). Tunables: `FACE_MATCH` (0.40), `VOICE_MATCH` (0.50), `PEOPLE_DB`.

### Names heard by voice
Speech recognition can misspell names ("Marven" for "Marvin"). What the app does about it:
- Names already on record are given to Whisper as context, so they come back spelled the way they were stored.
- A newly heard name is spelled back letter by letter ("M-A-R-V-E-N, is that right?"). Correct it by spelling it out
  ("spelling should be M-A-R-V-I-N") or by typing it.
- A spoken near-match of someone already on record (Marven vs Marvin) never overwrites the stored spelling; typed names are exact.
- Every naming/renaming/forgetting attempt shows as a live event row in the GUI. If nothing was changed it says so, and if the
  assistant claims a change that did not happen it is flagged in the GUI and corrected on the next turn.

## Settings

The gear icon in the top bar opens a settings window: context budget and recent-history size, voice engine, voice and speech speed (with a preview button), camera resolution, skills, and the language-model endpoints. Under *Language model* you set the address and model name for Ollama and for LM Studio (any OpenAI-compatible server works), and **Check** lists the models an address offers (Ollama cloud models such as `name:cloud` appear there too). Saving switches the active backend over without losing the conversation; the environment variables `OLLAMA_URL`, `OLLAMA_MODEL`, `LMSTUDIO_URL` and `LMSTUDIO_MODEL` remain the defaults. Values are validated, applied live to open sessions, and saved to `data/settings.json` (override the path with `SETTINGS_FILE`).

## Skills

Voice-triggered skills run as code, not prompts, so they work reliably with small local models: the server decides
when one applies, parses the arguments itself, runs it, and the model only phrases the real outcome. Every outcome
appears as a row in the conversation, and a failure is never reported as success.

| Skill | Example | Notes |
|-------|---------|-------|
| Timers & reminders | "set a timer for 5 minutes called pasta", "what timers are running", "cancel the timer" | In memory only; lost on restart. A spoken alert plays when one ends. |
| Notes | "remember that Hana likes tea", "what do you know about Hana", "forget that Hana likes tea" | Stored in `data/notes.json` (owner-only). Deleting asks for a yes. |
| Mac control | "open Safari", "set the volume to 30", "mute" | Always asks "yes or no" first. Apps come from an allow-list (`MAC_APPS`, comma-separated). |
| Web search | "search for the capital of Australia" | Off by default; the query leaves this machine. Uses `ddgs` if installed, otherwise Wikipedia search. |

Turn skills on or off in Settings. Set `NOTES_DB` to move the notes file.

### Voice engines

Settings > Voice has a **Voice engine** dropdown: **Kokoro** (default: very fast, 27 voices, adjustable speed),
**Chatterbox** and **Chatterbox Turbo** (more expressive and conversational; one built-in voice; speed is not adjustable).
On this Mac a reply starts speaking about 0.3 s after you stop talking with Kokoro and about 0.8 s with Chatterbox.
Choosing a new engine downloads its model on first use (about 1 GB for Chatterbox, 0.7 GB for Turbo). Override the
Hugging Face repos with `KOKORO_REPO`, `CHATTERBOX_REPO` and `CHATTERBOX_TURBO_REPO`, or the default engine with `TTS_ENGINE`.

### How web search reads the web

Search snippets alone were too thin, so the web search skill now (1) searches with `ddgs` (retrying its occasional
"no results" and dropping sponsored results), (2) reads the top three result pages over plain HTTP and keeps the passages
that best match the question, using `ddgs`'s extractor for pages that block plain requests, and (3) falls back to the bare
snippets (and Wikipedia if search itself is down). Only public pages are fetched; local and private addresses are refused.
Page text is passed to the model labelled as untrusted. Price questions with a ticker use Yahoo Finance's quote endpoint.
A benchmark of 10 queries found plain HTTP reading as accurate as a headless Chrome and Jina Reader at a quarter of the time,
so no headless browser is used.

## License

MIT License, see [LICENSE](LICENSE). The models the app downloads (speech, voice and face models) have their own licenses.
