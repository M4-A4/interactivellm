"""Routes utterances to skills, handles spoken confirmation, and turns results into LLM notes."""
from __future__ import annotations

import time
from typing import Iterable

from ..pipeline.people import is_no, is_yes, same_speaker
from .base import (CANCELLED, FAILED, NEEDS_CONFIRMATION, NEEDS_INPUT, OK, Skill, SkillContext, SkillResult)

CONFIRM_TTL_S = 30.0   # how long a pending yes/no or follow-up answer stays valid


class SkillManager:
    """Holds the enabled skills for one session and the single pending question (confirmation or follow-up)."""

    def __init__(self, skills: Iterable[Skill], enabled: Iterable[str]):
        """Arguments are always parsed by code, never guessed by the language model, so a skill can never be run with
        invented values (for example a made-up timer length)."""
        self.skills = {skill.id: skill for skill in skills}
        self.enabled: set[str] = set(enabled)
        self._pending_confirmation: tuple[Skill, dict, float, str | None] | None = None   # skill, args, expiry, who was asked
        self._pending_followup: tuple[Skill, float] | None = None

    def set_enabled(self, ids: Iterable[str]) -> None:
        """Replace the set of enabled skills (applies immediately)."""
        self.enabled = set(ids) & set(self.skills)

    def _active(self) -> list[Skill]:
        return [skill for skill_id, skill in self.skills.items() if skill_id in self.enabled]

    async def handle(self, text: str, ctx: SkillContext) -> SkillResult | None:
        """Return the outcome if a skill handled this utterance, else None (ordinary conversation)."""
        now = time.monotonic()
        pending = self._pending_confirmation
        if pending and pending[2] > now and pending[0].id in self.enabled:
            skill, args, _, asked = pending
            if not same_speaker(asked, ctx.speaker):
                return None      # someone else said it: the question stays open for the person who was asked
            if is_yes(text):
                self._pending_confirmation = None
                return self._tag(skill, await self._run_safely(skill, args, ctx))
            if is_no(text):
                self._pending_confirmation = None
                return self._tag(skill, SkillResult(CANCELLED, "The user said no, so nothing was done.",
                                                    "Cancelled"))
        self._pending_confirmation = None     # any other reply drops the question
        follow = self._pending_followup
        self._pending_followup = None
        if follow:
            answers = follow[1] > now and follow[0].id in self.enabled and follow[0].accepts_followup(text, ctx)
            if answers:
                return await self._dispatch(follow[0], text, ctx)
            follow[0].abandon_followup()     # the user moved on: stop waiting for what the skill asked
        for skill in self._active():
            if skill.matches(text, ctx):
                return await self._dispatch(skill, text, ctx)
        for skill in self.skills.values():
            # A switched-off skill that was clearly asked for must say so, or the model would invent an answer.
            if skill.id not in self.enabled and skill.disabled_notice and skill.matches(text, ctx):
                return self._tag(skill, SkillResult(FAILED, skill.disabled_notice, skill.disabled_event or "Skill is switched off in Settings"))
        return None

    async def _dispatch(self, skill: Skill, text: str, ctx: SkillContext) -> SkillResult:
        """Extract arguments, then ask for confirmation or run."""
        extracted = skill.extract(text, ctx)
        if isinstance(extracted, SkillResult):
            result = self._tag(skill, extracted)
            if result.followup:
                self._pending_followup = (skill, time.monotonic() + CONFIRM_TTL_S)
            return result
        if extracted is None:
            return self._tag(skill, SkillResult(NEEDS_INPUT, f"The user seems to want the {skill.id} skill but the "
                                                "request was unclear.", "Couldn't tell what to do; please rephrase"))
        if skill.needs_confirmation(extracted):
            self._pending_confirmation = (skill, extracted, time.monotonic() + CONFIRM_TTL_S, ctx.speaker)
            question = skill.confirm_prompt(extracted)
            return self._tag(skill, SkillResult(NEEDS_CONFIRMATION, question, f"{question} Say yes or no"))
        result = self._tag(skill, await self._run_safely(skill, extracted, ctx))
        if result.followup:          # e.g. a failed lookup that asks the user to repeat the name
            self._pending_followup = (skill, time.monotonic() + CONFIRM_TTL_S)
        return result

    async def _run_safely(self, skill: Skill, args: dict, ctx: SkillContext) -> SkillResult:
        """Run a skill; any exception becomes an honest FAILED result."""
        try:
            return await skill.run(args, ctx)
        except Exception as error:
            return SkillResult(FAILED, f"The {skill.id} skill crashed: {error}", f"{skill.id} failed: {error}")

    @staticmethod
    def _tag(skill: Skill, result: SkillResult) -> SkillResult:
        result.skill = skill.id
        return result


def result_note(result: SkillResult) -> str:
    """Render a result as the bracketed [System] note given to the language model."""
    if result.kind == OK:
        return f"[System: skill '{result.skill}' succeeded. {result.note} Tell the user in one or two short sentences.]"
    if result.kind == NEEDS_CONFIRMATION:
        return (f"[System: skill '{result.skill}' is waiting for a yes or no and has NOT done anything yet. Ask the "
                f"user exactly this, in your own words: {result.note} Do NOT say or imply it was done.]")
    if result.kind == CANCELLED:
        return (f"[System: skill '{result.skill}' was cancelled. {result.note} Say only that nothing was done; do not say "
                "anything about the current state of any app, file or setting, because it was not checked.]")
    return (f"[System: skill '{result.skill}' FAILED or needs more input. {result.note} Do NOT say or imply it was "
            "done; tell the user briefly what went wrong or what you need.]")


def event_kind(result: SkillResult) -> str:
    """Map a result to the GUI event style."""
    return {OK: "done", FAILED: "failed", NEEDS_INPUT: "needs_input", NEEDS_CONFIRMATION: "needs_input",
            CANCELLED: "info"}[result.kind]
