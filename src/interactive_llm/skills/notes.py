"""Notes about people and things: "remember that Hana likes tea", "what do you know about Hana"."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path

from .base import FAILED, NEEDS_INPUT, OK, Skill, SkillContext, SkillResult

MAX_NOTES = 500
MAX_NOTE_CHARS = 300
CONTEXT_CHARS = 350    # cap on remembered facts added to a turn's context
_REMEMBER_RE = re.compile(r"\b(?:remember|note|keep in mind|don't forget|do not forget) that\s+(.+)|"
                          r"\b(?:make|take|add) a note\b[^:,.]{0,20}[:,]?\s*(.+)", re.I)
_RECALL_RE = re.compile(r"\bwhat (?:do you know|did i (?:tell|say)|have you noted|notes? do you have)\b[^.!?]{0,15}\babout\s+(.+?)[?.!]*$"
                        r"|\bwhat (?:notes|facts) (?:do you have|are on record)(?: about\s+(.+?))?[?.!]*$", re.I)
_FORGET_RE = re.compile(r"\b(?:forget|delete|remove)\s+(?:the\s+)?(?:note|fact)s?\s+(?:that|about)\s+(.+?)[?.!]*$"
                        r"|\bforget that\s+(.+?)[?.!]*$", re.I)
# Questions about someone's identity belong to the people/naming system, not to notes.
_IDENTITY_RE = re.compile(r"^(?:my|your|our|the|his|her|their)\s+(?:name|face|voice|identity|spelling)\b", re.I)
_SELF_WORDS = {"me", "myself", "i", "my"}
_STOP = {"the", "a", "an", "that", "about", "is", "are", "was", "of", "to", "and", "my", "i", "likes", "like"}
# Capitalised first words that are not names, so "Tomorrow is the deadline" is a general note, not a note about "Tomorrow".
_NOT_NAMES = {"the", "a", "an", "it", "its", "this", "that", "these", "those", "there", "here", "we", "they", "he", "she",
              "you", "your", "our", "his", "her", "their", "today", "tomorrow", "yesterday", "tonight", "monday",
              "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "if", "when", "while", "because",
              "some", "all", "every", "no", "not", "what", "who", "how", "why", "where", "please", "also", "then"}


class NotesStore:
    """A small JSON file of facts: ``[{"id", "subject", "text", "by", "created"}]``. Owner-only permissions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.items: list[dict] = []
        try:
            data = json.loads(self.path.read_text())
            self.items = [item for item in data if isinstance(item, dict) and "text" in item][-MAX_NOTES:]
        except (OSError, ValueError):
            self.items = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(dir=self.path.parent, prefix=".notes")
        with os.fdopen(descriptor, "w") as handle:
            json.dump(self.items, handle, indent=1)
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, self.path)

    def add(self, subject: str, text: str, by: str | None) -> dict:
        """Store a fact and return it; raises OSError if the file cannot be written (state is rolled back)."""
        item = {"id": max((existing["id"] for existing in self.items), default=0) + 1, "subject": subject, "text": text[:MAX_NOTE_CHARS], "by": by,
                "created": time.strftime("%Y-%m-%d %H:%M")}
        self.items.append(item)
        del self.items[:-MAX_NOTES]
        try:
            self._save()
        except OSError:
            self.items.remove(item)
            raise
        return item

    def about(self, subject: str) -> list[dict]:
        """Facts whose subject matches `subject` case-insensitively."""
        wanted = subject.lower()
        return [item for item in self.items
                if item["subject"].lower() == wanted
                or (item["subject"] == "general" and re.search(r"\b" + re.escape(wanted) + r"\b", item["text"].lower()))]

    def remove(self, item_id: int) -> bool:
        """Delete one fact by id; True if it existed."""
        before = len(self.items)
        self.items = [item for item in self.items if item["id"] != item_id]
        if len(self.items) == before:
            return False
        self._save()
        return True

    def context(self, names: list[str]) -> str:
        """A short, capped summary of facts about the given people, for the model's per-turn context."""
        parts, used = [], 0
        for name in dict.fromkeys(names):
            facts = [item["text"] for item in self.about(name)][-3:]
            if facts:
                line = f"{name}: " + "; ".join(facts)
                if used + len(line) > CONTEXT_CHARS:
                    break
                parts.append(line)
                used += len(line)
        return f"Facts on record (use only if relevant, never read aloud): {' | '.join(parts)}." if parts else ""


def _content_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9']+", text.lower()) if word not in _STOP}


class NotesSkill(Skill):
    """Remember, recall and (with confirmation) delete notes."""

    id = "notes"

    def __init__(self, store: NotesStore):
        self.store = store

    def _subject_for(self, sentence: str, ctx: SkillContext) -> tuple[str | None, str]:
        """Return (subject, reason). Subject is a known person, the speaker (for "my"/"I"), or "general"."""
        words = re.findall(r"[A-Za-z']+", sentence)
        first = re.sub(r"'s$", "", words[0].lower()) if words else ""
        first_raw = words[0].lower() if words else ""
        if first_raw in _SELF_WORDS or first_raw in {"i'm", "i've", "i'll", "i'd"}:
            return (ctx.speaker, "speaker") if ctx.speaker else (None, "unknown_speaker")
        for name in ctx.known_names:
            if first == name.lower() or first_raw == name.lower():
                return name, "named"
        if words and words[0][0].isupper() and first not in _NOT_NAMES and len(first) >= 2:
            return words[0].title() if words[0].isupper() else words[0], "new_name"   # a person not enrolled yet
        return "general", "general"

    def matches(self, text: str, ctx: SkillContext) -> bool:
        recall = _RECALL_RE.search(text)
        if recall and _IDENTITY_RE.match((recall.group(1) or recall.group(2) or "").strip()):
            return False
        if _REMEMBER_RE.search(text) or recall:
            return True
        forget = _FORGET_RE.search(text)
        return bool(forget and self._find_note(forget.group(1) or forget.group(2) or "", ctx))

    def _find_note(self, description: str, ctx: SkillContext) -> dict | None:
        """The stored fact that best overlaps `description` (at least half of its content words), or None."""
        wanted = _content_words(description)
        if not wanted:
            return None
        best, best_score = None, 0.0
        for item in self.store.items:
            overlap = len(wanted & (_content_words(item["text"]) | _content_words(item["subject"]))) / len(wanted)
            if overlap > best_score:
                best, best_score = item, overlap
        return best if best_score >= 0.5 else None

    def extract(self, text: str, ctx: SkillContext) -> dict | SkillResult | None:
        forget = _FORGET_RE.search(text)
        if forget:
            item = self._find_note(forget.group(1) or forget.group(2) or "", ctx)
            return {"action": "forget", "id": item["id"], "text": item["text"], "subject": item["subject"]} if item else None
        recall = _RECALL_RE.search(text)
        if recall:
            raw = (recall.group(1) or recall.group(2) or "").strip().lower()
            if raw in _SELF_WORDS or raw in {"us"}:
                subject = ctx.speaker
                if not subject:
                    return SkillResult(NEEDS_INPUT, "The speaker is not recognised, so it is unclear whose notes to read. "
                                       "Ask who they mean.", "Whose notes? I don't know who is speaking")
            else:
                subject = next((name for name in ctx.known_names if name.lower() == raw), raw.title() if raw else None)
            return {"action": "recall", "subject": subject}
        remember = _REMEMBER_RE.search(text)
        sentence = (remember.group(1) or remember.group(2) or "").strip() if remember else ""
        if len(sentence) < 3:
            return None
        subject, why = self._subject_for(sentence, ctx)
        if subject is None:
            return SkillResult(NEEDS_INPUT, "The note is about 'me' but the speaker is not recognised. Ask who it is about.",
                               "Who is this note about?")
        return {"action": "remember", "subject": subject, "text": sentence}

    def confirm_prompt(self, args: dict) -> str:
        return f"Delete the note \"{args['text']}\"?"

    def needs_confirmation(self, args: dict) -> bool:
        """Only deleting a note asks first; remembering and recalling do not."""
        return args.get("action") == "forget"

    async def run(self, args: dict, ctx: SkillContext) -> SkillResult:
        action = args["action"]
        if action == "remember":
            try:
                self.store.add(args["subject"], args["text"], ctx.speaker)
            except OSError as error:
                return SkillResult(FAILED, f"The note could not be saved ({error}); nothing was stored.",
                                   "Note not saved: could not write the file")
            who = "general notes" if args["subject"] == "general" else args["subject"]
            return SkillResult(OK, f"Saved a note under {who}: {args['text']}", f"Noted for {who}")
        if action == "recall":
            facts = self.store.about(args["subject"]) if args["subject"] else []
            if not facts:
                return SkillResult(OK, f"There are no notes about {args['subject']}. Say so plainly.",
                                   f"No notes about {args['subject']}")
            return SkillResult(OK, f"Notes about {args['subject']}: " + " | ".join(item["text"] for item in facts[-8:]),
                               f"{len(facts)} note(s) about {args['subject']}")
        try:
            removed = self.store.remove(args["id"])
        except OSError as error:
            return SkillResult(FAILED, f"The note could not be deleted ({error}).", "Note not deleted")
        return (SkillResult(OK, "The note was deleted.", "Note deleted") if removed
                else SkillResult(FAILED, "That note no longer exists.", "Note not found"))
