"""Voice-triggered skills (timers, notes, Mac control, web search); see `base.py` for the design."""
from __future__ import annotations

from pathlib import Path

from .base import Skill
from .mac_control import MacControlSkill
from .notes import NotesSkill, NotesStore
from .timers import TimerSkill
from .web_search import WebSearchSkill


def build_skills(notes_store: NotesStore) -> list[Skill]:
    """Create one session's skills (timers are per session; the notes store is shared)."""
    return [TimerSkill(), NotesSkill(notes_store), MacControlSkill(), WebSearchSkill()]


def default_notes_path() -> Path:
    """Where notes are kept unless ``NOTES_DB`` is set: data/notes.json beside people.json."""
    import os
    return Path(os.environ.get("NOTES_DB", Path(__file__).resolve().parents[3] / "data" / "notes.json"))
