#!/bin/bash
# PostToolUse (Write|Edit): run the test suite after a Python edit under src/ or tests/.
# On failure, print the tail of the output to stderr and exit 2 so Claude sees it.
f=$(jq -r '.tool_response.filePath // .tool_input.file_path // empty')
root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
case "$f" in
  "$root"/src/*.py|"$root"/tests/*.py) ;;
  *) exit 0 ;;
esac
cd "$root" || exit 0
# Skip the one test that does live web searches (about 16 s, needs network); run the full suite by hand.
out=$(uv run pytest -q -x --deselect tests/test_skills.py::test_ordinary_conversation_triggers_no_skill 2>&1)
status=$?
if [ $status -ne 0 ]; then
  echo "pytest failed after editing $f:" >&2
  echo "$out" | tail -25 >&2
  exit 2
fi
exit 0
