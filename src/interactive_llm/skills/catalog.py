"""Static list of the available skills (kept import-light so settings validation can use it)."""
from __future__ import annotations

# id, label shown in the settings window, one-line description, enabled by default?
SKILL_CATALOG = [
    {"id": "timers", "label": "Timers & reminders",
     "description": "Set, list and cancel timers by voice; a spoken alert plays when one finishes.",
     "default_on": True},
    {"id": "notes", "label": "Notes about people & things",
     "description": "Remember facts (\"remember that Hana likes tea\") and recall them. Stored on this machine only.",
     "default_on": True},
    {"id": "mac_control", "label": "Mac control",
     "description": "Open or quit allow-listed apps and set the volume (asks for a spoken yes first); list the files on your Desktop, Documents or Downloads.",
     "default_on": True},
    {"id": "web_search", "label": "Web search",
     "description": "Look things up online (\"search for ...\", \"check the AMZN stock price\"). Your query leaves this machine.",
     "default_on": True},
]
SKILL_IDS = [entry["id"] for entry in SKILL_CATALOG]
DEFAULT_ENABLED = [entry["id"] for entry in SKILL_CATALOG if entry["default_on"]]
