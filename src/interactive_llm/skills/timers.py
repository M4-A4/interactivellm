"""Timers and reminders: set, list and cancel; an alert is spoken when one finishes."""
from __future__ import annotations

import asyncio
import itertools
import re
import time
from dataclasses import dataclass

from .base import FAILED, NEEDS_INPUT, OK, Skill, SkillContext, SkillResult
from .numbers import parse_duration

MIN_SECONDS, MAX_SECONDS = 1, 24 * 3600

_SET_RE = re.compile(r"\b(?:set|start|create|make|begin|give me|need)\b[^.!?]{0,25}\b(?:timer|alarm|countdown)\b"
                     r"|\b(?:timer|countdown)\s+(?:for|of)\b|^\s*(?:(?:please|can you|could you|would you|hey|ok|okay|and|also)\s+)*remind me\b", re.I)
_CANCEL_RE = re.compile(r"\b(?:cancel|stop|clear|delete|remove|kill)\b[^.!?]{0,20}\b(?:timers?|alarms?|reminders?)\b", re.I)
_LIST_RE = re.compile(r"\b(?:what|which|list|any|how many|show)\b[^.!?]{0,25}\b(?:timers?|reminders?|alarms?)\b"
                      r"|\bhow (?:much time|long)\b[^.!?]{0,20}\b(?:left|remaining)\b", re.I)
_REMIND_RE = re.compile(r"\bremind me (?:to|about|that)\s+(.+?)(?:\s+(?:in|after|for)\s+(?:\d|an?\b|half|quarter|"
                        r"one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|ninety)|[.!?]*$)", re.I)
_LABEL_RE = re.compile(r"\b(?:called|named|labelled|labeled|for the)\s+([a-z0-9' ]{2,30}?)(?:\s+(?:for|in|after)\b|[.!?]*$)", re.I)


@dataclass
class Timer:
    """One running timer."""
    number: int
    label: str
    reminder: bool
    due: float                 # time.monotonic() when it fires
    task: asyncio.Task


def describe_seconds(seconds: float) -> str:
    """Human wording such as "1 hour 30 minutes" or "45 seconds"."""
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{hours} hour{'s' if hours != 1 else ''}" if hours else "",
             f"{minutes} minute{'s' if minutes != 1 else ''}" if minutes else "",
             f"{secs} second{'s' if secs != 1 else ''}" if secs and not hours else ""]
    return " ".join(part for part in parts if part) or "0 seconds"


_QUESTION_ABOUT_PAST_RE = re.compile(r"^\s*(?:and\s+|so\s+)?(?:did|have|has|had|was|were|why)\b", re.I)


class TimerSkill(Skill):
    """Per-session timers. They live in memory only, so they do not survive a server restart."""

    id = "timers"

    def __init__(self) -> None:
        self._timers: dict[int, Timer] = {}
        self._numbers = itertools.count(1)

    def matches(self, text: str, ctx: SkillContext) -> bool:
        if _QUESTION_ABOUT_PAST_RE.match(text):
            return False          # "Did you just set a timer for me?" is a question, not a command
        return bool(_SET_RE.search(text) or _CANCEL_RE.search(text) or _LIST_RE.search(text))

    def accepts_followup(self, text: str, ctx: SkillContext) -> bool:
        return parse_duration(text) is not None

    def extract(self, text: str, ctx: SkillContext) -> dict | SkillResult | None:
        if _CANCEL_RE.search(text):
            label = _LABEL_RE.search(text)
            return {"action": "cancel", "label": label.group(1).strip().lower() if label else "",
                    "all": bool(re.search(r"\b(?:all|every|everything)\b", text, re.I))}
        if _LIST_RE.search(text) and not _SET_RE.search(text):
            return {"action": "list"}
        seconds = parse_duration(text)
        remind = _REMIND_RE.search(text)
        label = (remind.group(1) if remind else (_LABEL_RE.search(text).group(1) if _LABEL_RE.search(text) else "")).strip()
        if seconds is None:
            return SkillResult(NEEDS_INPUT, "The user wants a timer but did not say for how long. Ask how long.",
                               "For how long should the timer run?", followup=True)
        return {"action": "set", "seconds": seconds, "label": label, "reminder": bool(remind)}

    async def run(self, args: dict, ctx: SkillContext) -> SkillResult:
        action = args["action"]
        if action == "list":
            return self._list()
        if action == "cancel":
            return self._cancel(args["label"], args.get("all", False))
        seconds = args["seconds"]
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            return SkillResult(FAILED, "Timers must be between 1 second and 24 hours; none was set.",
                               "Timer not set: must be between 1 second and 24 hours")
        if ctx.announce is None:
            return SkillResult(FAILED, "No way to alert the user is available, so no timer was set.",
                               "Timer not set: no alert channel")
        number = next(self._numbers)
        timer = Timer(number, args["label"], args["reminder"], time.monotonic() + seconds,
                      asyncio.create_task(self._fire(number, seconds, ctx.announce)))
        self._timers[number] = timer
        what = f"reminder to {timer.label}" if timer.reminder and timer.label else (f"{timer.label} timer" if timer.label else "timer")
        return SkillResult(OK, f"A {what} is running for {describe_seconds(seconds)}; an alert will be spoken when it ends.",
                           f"Set {what}: {describe_seconds(seconds)}")

    async def _fire(self, number: int, seconds: float, announce) -> None:
        """Wait, then speak and show the alert."""
        await asyncio.sleep(seconds)
        timer = self._timers.pop(number, None)
        if timer is None:
            return
        if timer.reminder and timer.label:
            spoken = f"Reminder: {timer.label}."
        elif timer.label:
            spoken = f"Your {timer.label} timer is done."
        else:
            spoken = "Your timer is done."
        await announce(spoken, spoken)

    def _list(self) -> SkillResult:
        if not self._timers:
            return SkillResult(OK, "No timers are running.", "No timers running")
        now = time.monotonic()
        lines = [f"{timer.label or 'unnamed timer'}: {describe_seconds(max(0, timer.due - now))} left"
                 for timer in self._timers.values()]
        return SkillResult(OK, "Running timers - " + "; ".join(lines) + ".", f"{len(lines)} timer(s) running")

    def _cancel(self, label: str, everything: bool) -> SkillResult:
        if not self._timers:
            return SkillResult(FAILED, "There are no timers to cancel.", "No timers to cancel")
        chosen = [timer for timer in self._timers.values() if everything or not label or label in timer.label.lower()]
        if not chosen:
            return SkillResult(FAILED, f"No timer matches '{label}'; none was cancelled.", f"No timer called {label}")
        if not everything and not label and len(self._timers) > 1:
            return SkillResult(NEEDS_INPUT, "Several timers are running; ask which to cancel, or whether to cancel all.",
                               "Which timer should I cancel?")
        for timer in chosen:
            timer.task.cancel()
            self._timers.pop(timer.number, None)
        return SkillResult(OK, f"Cancelled {len(chosen)} timer(s).", f"Cancelled {len(chosen)} timer(s)")

    def shutdown(self) -> None:
        """Cancel every pending timer (session ended)."""
        for timer in self._timers.values():
            timer.task.cancel()
        self._timers.clear()
