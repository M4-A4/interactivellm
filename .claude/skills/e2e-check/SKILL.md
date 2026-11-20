---
name: e2e-check
description: Run the test suite, then drive a scripted text conversation over the WebSocket against a scratch server and compare the GUI skill events with the bot's replies. Use to check that a change did not make the assistant claim something the backend did not do.
argument-hint: "[user turns separated by | , optional]"
---

Turns to send: `$ARGUMENTS`. If empty, use a default set that covers a lookup, a follow-up that depends on the previous turn, a plain-chat turn and a local action, for example: `What is the Vanguard Total Stock Market ETF?` | `What's it trading at?` | `And what about the S&P 500 one, VOO?` | `Tell me a joke.` | `What have I got in my downloads folder?`

1. Run `uv run pytest -q`. Report failures with their output and stop if any fail.
2. Start a scratch server with the `scratch-server` skill (port 8001 unless the user names one, never 8000). If the requested port is taken, the scratch-server skill picks the next free one; say so at the top of the report so the user knows which port ran.
3. Write a script in the scratchpad directory that connects to `ws://127.0.0.1:<port>/ws` with `websockets` and, for each turn, sends `{"type": "text", "text": <turn>}`, then collects JSON messages until the reply goes quiet. Handle `assistant_start` (reset text), `assistant_delta` (append `text`) and `skill_event` (record `kind` and `text`); ignore binary frames. Wait long enough for lookups: web search with page reading can take over 12 s, so wait for the reply to be quiet for several seconds rather than a fixed delay.
4. Print per turn: the user text, the GUI events, and the bot reply. Run with `uv run python <script>`.
5. Flag every turn where the reply states live data (prices, news, file counts, "done", "set", "opened") but no matching `done` skill event appeared, or where a `failed` event is followed by a reply that sounds successful. Those are bugs, not nits.
6. Stop the scratch server by its port with the `scratch-server` skill. Report plainly what was and was not covered (real camera and microphone are not covered).
