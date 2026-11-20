---
name: scratch-server
description: Start or stop a throwaway Interactive_LLM web server on its own port with scratch data files, so experiments never touch the real server on port 8000 or the real data/ files. Use before any live test of server behavior.
argument-hint: "[start|stop] [port, default 8001]"
---

Arguments: `$ARGUMENTS` (action `start` or `stop`, then an optional port; default action `start`, default port `8001`). Never use port 8000.

## start

1. Check the port is free with both `lsof -iTCP:<port> -sTCP:LISTEN` and a bind test, `python3 -c "import socket; socket.socket().bind(('127.0.0.1', <port>))"`. lsof cannot see every listener (a port it reported free was in use in testing), so a free lsof result alone is not enough. If either check says the port is taken, pick the next free port instead of killing anything, and tell the user the port changed and why.
2. Use the session scratchpad directory (or `mktemp -d`) as `$TMP`. Copy the real settings so model endpoints match: `cp data/settings.json $TMP/s.json`. Do not copy `data/people.json` (biometric data); let the scratch people file start empty.
3. Start from the project root in the background, logging to `$TMP/server.log`:
   `PORT=<port> PEOPLE_DB=$TMP/p.json SETTINGS_FILE=$TMP/s.json NOTES_DB=$TMP/n.json uv run interactive-llm-web`
   All four variables are required. The entry point is `interactive-llm-web`, not `interactive-llm` (that is the terminal chat).
4. Wait for the `[ready] models warm` line in the log (about 5 s when the models are cached, up to about 30 s on a cold start). If a `[fatal]` line appears, report it. The log contains binary progress-bar characters, so a plain `grep` never matches and the wait runs to its limit: always use `grep -a`, and stop the wait early on the ready line, a fatal line or a Python traceback, with a 60 s cap:
   `for i in $(seq 1 30); do grep -aqE "\[ready\]|\[fatal\]|Traceback" $TMP/server.log && break; sleep 2; done; grep -aE "\[ready\]|\[fatal\]|Traceback" $TMP/server.log`
   If neither line shows after the cap, say so and show the last log lines instead of waiting longer.
5. Tell the user the URL, the port, the `$TMP` path and the log path. Say this even when the user also asked you to stop the server straight away, because the paths are how they check what ran.

## stop

Stop only the scratch server, by its port: `kill $(lsof -tiTCP:<port> -sTCP:LISTEN)`. Never `pkill -f interactive-llm-web`; that also kills the real server. Confirm the port is free, confirm port 8000 is still listening if it was before, then delete the scratch files in `$TMP`.
