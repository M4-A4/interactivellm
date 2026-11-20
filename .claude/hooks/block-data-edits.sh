#!/bin/bash
# PreToolUse (Write|Edit): never edit the real data/ directory (people.json holds biometric data).
#
# Works wherever the repo is cloned: it compares against "$CLAUDE_PROJECT_DIR/data", not a folder name.
# The path is resolved first, so "src/../data/x", symlinks into data/, and a different letter case all count.
# It fails closed: if jq or python3 is missing, or the path cannot be read or resolved, the edit is blocked.
# Limit: this only watches Claude's Write and Edit tools. It cannot see shell commands or other programs.
block() {
  echo "Blocked: $1" >&2
  exit 2
}

command -v jq >/dev/null 2>&1 || block "jq is not installed, so the data/ guard cannot check this edit (brew install jq)."
command -v python3 >/dev/null 2>&1 || block "python3 is not installed, so the data/ guard cannot check this edit."

input=$(cat)
f=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty') || block "could not read the tool input, so the edit is refused."
[ -n "$f" ] || exit 0                       # nothing to check: not a file edit

root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
resolve() { python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]).lower())' "$1"; }
target=$(resolve "$f") && data=$(resolve "$root/data") || block "could not resolve the path $f, so the edit is refused."

case "$target" in
  "$data" | "$data"/*)
    block "$f is in the real data/ directory (real people.json / settings.json). Use a scratch copy via PEOPLE_DB / SETTINGS_FILE / NOTES_DB, or ask the user." ;;
esac
exit 0
