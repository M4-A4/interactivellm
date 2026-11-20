# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Realtime multimodal voice assistant: Python backend (FastAPI + WebSocket) with a plain-JS web front end, running on local models. Apple silicon only (MLX).

## Commands

- Setup: `cp .env.example .env`, then `uv sync --extra dev`. Use `uv run` / `uv add`, not pip.
- Web app: `uv run interactive-llm-web` (http://127.0.0.1:8000, WebSocket `/ws`). Wait for the `[ready] models warm` log line (about 5 s; check it with `grep -a`, the log has binary progress bars so plain `grep` never matches). Needs Ollama or LM Studio running.
- `uv run interactive-llm` is the terminal chat, a different entry point. It does not start the server.
- Lint: `uv run ruff check src tests` (report only; rules are pyflakes + syntax errors, deliberately no style rules).
- Tests: `uv run pytest`; one test: `uv run pytest tests/test_skills.py::<name> -q`.

## Scratch servers (use these for experiments)

`data/` holds the owner's real biometric and settings data and a real server usually runs on port 8000. For any throwaway run set all four, never just some:

`PORT=8001 PEOPLE_DB=$TMP/p.json SETTINGS_FILE=$TMP/s.json NOTES_DB=$TMP/n.json uv run interactive-llm-web`

- Copy `data/settings.json` to the scratch path if the run needs the same model endpoints.
- Stop a server by its port (`kill $(lsof -tiTCP:<port> -sTCP:LISTEN)`). Never `pkill -f interactive-llm-web`, it also kills the real one.
- After first run `data/settings.json` overrides `.env` defaults, which can confuse debugging.

## Architecture rules

- **One MLX thread.** All MLX work (Whisper, Kokoro, Chatterbox) runs on a single-worker pool in `pipeline/models.py`; MLX streams are bound to the creating thread. Never call MLX/TTS from other threads or `run_in_executor(None, ...)`.
- **Skills are code-routed, not prompt-routed.** Each skill in `skills/` implements `matches`/`extract`/`run`; `SkillManager` routes by regex and parses arguments in code, so the LLM never invents values. The LLM only phrases the outcome from a `[System: ...]` note. Read-only skills (`web_search`, `stock_quote`, `list_folder`) are also exposed as LLM tool calls in `skills/llm_tools.py`.
- **Adding a skill** touches three places: `SKILL_CATALOG` in `skills/catalog.py` (also drives settings validation, keep it import-light), `build_skills()` in `skills/__init__.py`, and a `default_on` flag. Side-effect skills (`side_effect=True`) need a spoken yes first.
- **Never claim success the backend did not produce.** Failures must show in the GUI as a `skill_event` and the assistant must say what actually happened. `server.py` backs the prompt with regex guards (`CLAIM_RE`, `SKILL_CLAIM_RE`) and a correction note on the next turn. Keep new skills and prompt edits compatible with this. A reply that quotes live data (prices, news) with no lookup event in the GUI is a bug.
- **People are named by code only** (`parse_intros`, `parse_forget` in `pipeline/people.py`). The model can never name, rename or delete anyone. Only embeddings are stored.
- **Settings** go through `settings.validate()` (all-or-nothing, bounds in `settings.py`); `POST /api/settings` applies live to all sessions, then persists.
- **Stores** (`people.json`, `notes.json`) are written atomically (temp file + rename) with mode 0600. Keep that for any new store.
- **Local LLM thinking stays off** (`reasoning_effort=none`) on the fast chat path, for latency.
- Web page text is untrusted and labelled as such; only public http(s) URLs are fetched (`skills/page_reader.py`).
- Vision: an image is attached only on visual turns. Left/right questions get a mirrored frame (`mirror_jpeg_b64`) so the user's left is the image's left.

## Style and tests

- `from __future__ import annotations` and a module docstring in every module; long lines (120+) are normal; no formatter is configured, so match surrounding code.
- Tests are plain pytest functions with hand-written fakes and `asyncio.run(...)` helpers (no pytest-asyncio); use `tmp_path` for files. `tests/test_skills.py` imports `interactive_llm.server`.
- Frontend is plain JS with no build step. Bump the `?v=` cache query in `index.html` when changing `app.js`/`style.css`.

## Public repository: no personal data

This repo is public on GitHub, and anything pushed stays in history even after a later delete. This is a voice and face app, so personal data is the main risk.

- **Never commit:** anything from `data/` (biometric embeddings, settings, notes), `.env` or any key/token, audio, images or camera frames, model weights, embeddings (`.npy`, `.npz`, `.pkl`, `.pt`, `.wav`, `.jpg`, `.png`), logs, or transcripts of real sessions. Add a new data or log path to `.gitignore` and confirm it with `git check-ignore -v <path>` before the first write.
- **Use invented names** in code, tests, docs, comments, skills and evals (Sam, Priya, Marvin). Never use the owner's real name or the names of people or pets stored in `data/`. Do not paste real transcripts, camera descriptions or settings values; write a made-up equivalent.
- **No personal identifiers anywhere in tracked files:** email addresses, phone numbers, home paths (`/Users/<name>/`), real LAN IPs, hostnames, account IDs. Use `~`, `$TMP`, `localhost` or the documentation range `192.0.2.x`.
- **Before every commit:** stage files by name (never `git add -A` or `git add .`), then review `git status --short` and `git diff --cached --stat`. Scan the staged diff: `git diff --cached -U0 -- . ':!uv.lock' | grep -nIE "^\+.*([A-Za-z0-9._-]+@[a-z0-9-]+\.[a-z]{2,}|/Users/|sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|(api[_-]?key|secret|token|password)[\"']?\s*[:=]\s*[\"'][^\"' ]{8,})"`. Resolve every hit, and also grep the staged files for the owner's real name and any name stored in `data/`. LLM "tokens" (`max_tokens`) are not secrets.
- **Commit identity** must be the GitHub noreply address already set in this repo's local git config. Check `git config user.email` first; never put a personal email in a commit. Before any push, run `git fetch` and `git log --all --format='%h %an <%ae> / %ce'`: every author and committer address in the history must be a `@users.noreply.github.com` address. Commits made on github.com (such as the one it creates with a new repo) use the account's real email unless "Keep my email addresses private" is on.
- **Do not push, force-push or rewrite published history** without being asked in that turn. If something sensitive was already pushed, tell the user straight away: deleting the file is not enough, because the history still has it.
