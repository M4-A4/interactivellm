"""Mac control: open or quit allow-listed apps, set the volume (each asks for a spoken yes first) and list folder contents.

There is no shell passthrough: commands are fixed argument lists and the only variable parts are an app name taken
from the allow-list and an integer.
"""
from __future__ import annotations

import asyncio
import difflib
import os
import re
import subprocess
from typing import Callable

from .base import FAILED, NEEDS_INPUT, OK, Skill, SkillContext, SkillResult
from .numbers import spoken_number

DEFAULT_APPS = ["Safari", "Notes", "Calendar", "Music", "Finder", "Mail", "Reminders", "Calculator", "Terminal",
                "System Settings", "Photos", "Maps"]
_APP_RE = re.compile(r"\b(open|launch|start up|quit|close|exit)\s+(?:the\s+)?(.+?)(?:\s+(?:app|application))?\s*[.!?]*$", re.I)
FOLDERS = {"desktop": "Desktop", "documents": "Documents", "downloads": "Downloads"}   # the only folders that can be listed
MAX_LISTED = 200
_LIST_RE = re.compile(r"\b(?:what|which|list|show|tell|see)\b[^.!?]{0,40}\b(?:files?|folders?|stuff|items?|documents|contents?|in|on)\b"
                      r"[^.!?]{0,30}\b(desktop|documents|downloads)\b|\b(?:list|show)\b[^.!?]{0,20}\b(desktop|documents|downloads)\b", re.I)
_SHOW_AGAIN_RE = re.compile(r"\b(?:show|display|put|print|give)\b[^.!?]{0,25}\b(?:it|them|that|those|list|files)\b|\b(?:on|to) (?:the )?screen\b|"
                            r"\bshow (?:it|them|that) (?:here|again)\b", re.I)
_VOLUME_RE = re.compile(r"\b(?:set|turn|put|change|make)\b[^.!?]{0,20}\bvolume\b|\bvolume\s+(?:to|at)\b|\b(?:mute|unmute)\b", re.I)


# A command to open/quit something with a capitalised name that is not on the allow-list ("Open Spotify"): the skill must answer
# "not allowed" itself, otherwise the model improvises a confirmation question for an action nothing would perform.
_COMMAND_RE = re.compile(r"^\s*(?i:(?:(?:please|can you|could you|will you|go ahead and)\s+)*(?:open|launch|quit|close))\s+"
                         r"(?P<name>[A-Z][\w.&-]*(?:\s+[A-Z][\w.&-]*)?)\s*(?i:app|application)?\s*[.!?]*$")


def allowed_apps() -> list[str]:
    """The allow-list: ``MAC_APPS`` (comma-separated) or the built-in default."""
    configured = [name.strip() for name in os.environ.get("MAC_APPS", "").split(",") if name.strip()]
    return configured or DEFAULT_APPS


def list_folder(name: str) -> list[str]:
    """Visible entry names in one of the allow-listed home folders (folders get a trailing slash).

    Raises:
        OSError: the folder cannot be read (for example macOS has not granted file access).
    """
    folder = os.path.join(os.path.expanduser("~"), FOLDERS[name])
    return sorted((entry + "/" if os.path.isdir(os.path.join(folder, entry)) else entry
                   for entry in os.listdir(folder) if not entry.startswith(".")), key=str.lower)


def run_command(argv: list[str]) -> tuple[int, str]:
    """Run a fixed argument list (no shell); return (exit code, first line of stderr)."""
    done = subprocess.run(argv, capture_output=True, text=True, timeout=8)
    return done.returncode, (done.stderr or "").strip().splitlines()[0] if done.stderr.strip() else ""


class MacControlSkill(Skill):
    """Open/quit allow-listed apps and set or mute the volume."""

    id = "mac_control"
    side_effect = True

    def __init__(self, apps: list[str] | None = None, runner: Callable[[list[str]], tuple[int, str]] = run_command,
                 lister: Callable[[str], list[str]] = list_folder):
        self.apps = apps or allowed_apps()
        self.runner = runner
        self.lister = lister
        self.last_folder: str | None = None   # lets "show it here" repeat the last list on screen

    def needs_confirmation(self, args: dict) -> bool:
        """Listing a folder only reads names, so it runs at once; everything else asks first."""
        return args["action"] != "list"

    def _closest_app(self, spoken: str) -> str | None:
        """Allow-listed app that `spoken` names (exact, or a close spelling such as "safari" / "system setting")."""
        lowered = {name.lower(): name for name in self.apps}
        spoken = spoken.lower().strip()
        if spoken in lowered:
            return lowered[spoken]
        close = difflib.get_close_matches(spoken, list(lowered), n=1, cutoff=0.78)
        return lowered[close[0]] if close else None

    def matches(self, text: str, ctx: SkillContext) -> bool:
        if _LIST_RE.search(text) or (self.last_folder and _SHOW_AGAIN_RE.search(text)):
            return True
        if _VOLUME_RE.search(text):
            return True
        found = _APP_RE.search(text)
        return bool((found and (self._closest_app(found.group(2)) or re.search(r"\b(?:app|application)\b", text, re.I)))
                    or _COMMAND_RE.match(text))

    def extract(self, text: str, ctx: SkillContext) -> dict | SkillResult | None:
        lowered = text.lower()
        listing = _LIST_RE.search(text)
        if not listing and self.last_folder and _SHOW_AGAIN_RE.search(text):
            return {"action": "list", "folder": self.last_folder}
        if listing:
            return {"action": "list", "folder": (listing.group(1) or listing.group(2)).lower()}
        if re.search(r"\bunmute\b", lowered):
            return {"action": "mute", "muted": False}
        if re.search(r"\bmute\b", lowered):
            return {"action": "mute", "muted": True}
        if _VOLUME_RE.search(text):
            level = spoken_number(text)
            if level is None:
                return SkillResult(NEEDS_INPUT, "The user wants to change the volume but gave no level from 0 to 100. Ask for it.",
                                   "What volume level (0-100)?")
            if not 0 <= level <= 100:
                return SkillResult(FAILED, "Volume must be between 0 and 100; nothing was changed.",
                                   "Volume not changed: must be 0-100")
            return {"action": "volume", "level": int(level)}
        found = _APP_RE.search(text)
        if not found:
            return None
        app = self._closest_app(found.group(2))
        verb = "open" if found.group(1).lower() in ("open", "launch", "start up") else "quit"
        if app is None:
            return SkillResult(FAILED, f"'{found.group(2)}' is not on the allowed app list ({', '.join(self.apps)}); "
                               "nothing was opened or closed.", f"Not allowed: {found.group(2)}")
        return {"action": verb, "app": app}

    def confirm_prompt(self, args: dict) -> str:
        return {"open": f"Open {args.get('app')}?", "quit": f"Quit {args.get('app')}?",
                "volume": f"Set the volume to {args.get('level')} percent?",
                "mute": "Mute the volume?" if args.get("muted") else "Unmute the volume?"}[args["action"]]

    async def run(self, args: dict, ctx: SkillContext) -> SkillResult:
        action = args["action"]
        if action == "list":
            return await self._list(args["folder"])
        if action == "open":
            argv, done = ["open", "-a", args["app"]], f"Opened {args['app']}"
        elif action == "quit":
            argv, done = ["osascript", "-e", f'tell application "{args["app"]}" to quit'], f"Quit {args['app']}"
        elif action == "volume":
            argv, done = ["osascript", "-e", f"set volume output volume {int(args['level'])}"], f"Volume set to {int(args['level'])}%"
        else:
            argv = ["osascript", "-e", f"set volume {'with' if args['muted'] else 'without'} output muted"]
            done = "Muted" if args["muted"] else "Unmuted"
        try:
            code, error = await asyncio.get_running_loop().run_in_executor(None, self.runner, argv)
        except Exception as exc:
            return SkillResult(FAILED, f"The command could not run ({exc}); nothing changed.", f"Failed: {exc}")
        if code != 0:
            return SkillResult(FAILED, f"The command failed ({error or 'exit code ' + str(code)}); nothing changed. "
                               "macOS may need permission under Privacy & Security > Automation.",
                               f"Failed: {error or 'exit code ' + str(code)}")
        return SkillResult(OK, f"{done}.", done)

    async def _list(self, folder: str) -> SkillResult:
        """Report the real folder contents, or the real reason they could not be read."""
        try:
            names = await asyncio.get_running_loop().run_in_executor(None, self.lister, folder)
        except Exception as exc:
            return SkillResult(FAILED, f"The {folder} folder could not be read ({type(exc).__name__}); do not guess its "
                               "contents. macOS may need permission under Privacy & Security > Files and Folders.",
                               f"Could not read {folder}: {type(exc).__name__}")
        if not names:
            return SkillResult(OK, f"The {folder} folder is empty.", f"{folder.title()} is empty")
        self.last_folder = folder
        shown = names[:MAX_LISTED]
        extra = f" (+{len(names) - MAX_LISTED} more)" if len(names) > MAX_LISTED else ""
        return SkillResult(OK, f"The {folder} folder has {len(names)} items and the full list is now shown on screen. Say only "
                           "how many items there are and that the list is on screen; do NOT describe, group or guess what the "
                           f"files are. If asked about a specific file, answer only from this list: {', '.join(shown)}.",
                           f"Listed {len(names)} items in {folder.title()}{extra}", detail=shown)
