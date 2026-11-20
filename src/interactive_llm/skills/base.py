"""Skill framework: a skill decides deterministically whether it applies, extracts its arguments, and runs.

Skills are code-routed rather than prompt-routed because small local models cannot be trusted to follow long
instruction files at voice latency. The language model only phrases the outcome; it can never make a skill
"succeed" by saying so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

OK = "ok"
FAILED = "failed"
NEEDS_INPUT = "needs_input"
NEEDS_CONFIRMATION = "needs_confirmation"
CANCELLED = "cancelled"


@dataclass
class SkillResult:
    """Outcome of one skill invocation.

    Attributes:
        kind: One of OK, FAILED, NEEDS_INPUT, NEEDS_CONFIRMATION, CANCELLED.
        note: Plain facts for the language model to phrase (never shown verbatim).
        event: Short text for the GUI event row.
        skill: Id of the skill that produced it (filled in by the manager).
        detail: Full lines for the GUI row (for example every file name); never needs to fit the model's context.
        followup: True if the next utterance should be treated as the answer to this result's question.
    """
    kind: str
    note: str
    event: str
    skill: str = ""
    followup: bool = False
    detail: list[str] = field(default_factory=list)


@dataclass
class SkillContext:
    """What a skill may know about the current turn."""
    speaker: str | None = None
    known_names: list[str] = field(default_factory=list)
    visible_names: list[str] = field(default_factory=list)
    # announce(spoken_text, gui_text): speak and show something unprompted, e.g. a finished timer.
    announce: Callable[[str, str], Awaitable[None]] | None = None
    # progress(gui_text): show a "working on it" row while a slow skill (a web lookup) runs.
    progress: Callable[[str], Awaitable[None]] | None = None


class Skill:
    """Base class. Subclasses set `id` and `side_effect` and implement `matches`, `extract` and `run`."""

    id = ""
    disabled_event: str | None = None    # short GUI wording for the same situation (the notice above is written for the model)
    disabled_notice: str | None = None   # if set, tell the user this when they ask for the skill while it is switched off
    side_effect = False        # True: every run needs a spoken yes first (override `needs_confirmation` for finer control)

    def matches(self, text: str, ctx: SkillContext) -> bool:
        """Return True if this utterance is addressed to the skill (cheap and deterministic)."""
        raise NotImplementedError

    def extract(self, text: str, ctx: SkillContext) -> dict | SkillResult | None:
        """Return arguments, a ready-made SkillResult (e.g. NEEDS_INPUT), or None if they cannot be parsed."""
        raise NotImplementedError

    def accepts_followup(self, text: str, ctx: SkillContext) -> bool:
        """Return True if `text` answers a question this skill asked on the previous turn."""
        return False

    def abandon_followup(self) -> None:
        """Called when the user answered something else, so the skill can stop waiting for its follow-up."""

    def needs_confirmation(self, args: dict) -> bool:
        """Return True if these arguments must be confirmed by voice before running (default: `side_effect`)."""
        return self.side_effect

    def confirm_prompt(self, args: dict) -> str:
        """Question to ask before running a side-effect skill."""
        return "Should I go ahead?"

    async def run(self, args: dict, ctx: SkillContext) -> SkillResult:
        """Do the work and report the real outcome."""
        raise NotImplementedError
