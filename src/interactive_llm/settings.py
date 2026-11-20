"""User-tweakable settings (context window and TTS), validated and persisted to data/settings.json."""
from __future__ import annotations

import json
import re
import os
from pathlib import Path
from urllib.parse import urlparse

from .fileutil import quarantine, write_private_atomic
from .skills.catalog import SKILL_IDS

# English Kokoro voices (prefix: a = American, b = British; f/m = female/male).
VOICES = [
    ("af_heart", "Heart (US, female)"), ("af_bella", "Bella (US, female)"), ("af_nicole", "Nicole (US, female)"),
    ("af_sarah", "Sarah (US, female)"), ("af_sky", "Sky (US, female)"), ("af_nova", "Nova (US, female)"),
    ("af_alloy", "Alloy (US, female)"), ("af_aoede", "Aoede (US, female)"), ("af_jessica", "Jessica (US, female)"),
    ("af_kore", "Kore (US, female)"), ("af_river", "River (US, female)"),
    ("am_adam", "Adam (US, male)"), ("am_michael", "Michael (US, male)"), ("am_eric", "Eric (US, male)"),
    ("am_liam", "Liam (US, male)"), ("am_echo", "Echo (US, male)"), ("am_onyx", "Onyx (US, male)"),
    ("am_fenrir", "Fenrir (US, male)"), ("am_puck", "Puck (US, male)"),
    ("bf_emma", "Emma (UK, female)"), ("bf_isabella", "Isabella (UK, female)"), ("bf_alice", "Alice (UK, female)"),
    ("bf_lily", "Lily (UK, female)"),
    ("bm_george", "George (UK, male)"), ("bm_daniel", "Daniel (UK, male)"), ("bm_lewis", "Lewis (UK, male)"),
    ("bm_fable", "Fable (UK, male)"),
]
VOICE_IDS = {v for v, _ in VOICES}
# Voice engines. `repo` is the default Hugging Face model (override with the env var in `repo_env`); `speed` says whether the
# engine honours the speed slider; `voices` are the choices shown for that engine.
TTS_ENGINES = [
    {"id": "kokoro", "label": "Kokoro (fast, clear)", "repo": "mlx-community/Kokoro-82M-bf16", "repo_env": "KOKORO_REPO",
     "speed": True, "voices": [{"id": v, "label": label} for v, label in VOICES],
     "note": "82M parameters; a reply starts speaking in about 0.1 s."},
    {"id": "chatterbox", "label": "Chatterbox (more expressive)", "repo": "mlx-community/chatterbox-8bit",
     "repo_env": "CHATTERBOX_REPO", "speed": False, "voices": [{"id": "default", "label": "Default voice"}],
     "note": "More natural intonation; about 0.4 s per sentence. First use downloads about 1 GB."},
    {"id": "chatterbox_turbo", "label": "Chatterbox Turbo (expressive, faster)",
     "repo": "mlx-community/chatterbox-turbo-8bit", "repo_env": "CHATTERBOX_TURBO_REPO", "speed": False,
     "voices": [{"id": "default", "label": "Default voice"}],
     "note": "A faster Chatterbox; about 0.3 s per sentence. First use downloads about 0.7 GB."},
]
TTS_ENGINE_IDS = [engine["id"] for engine in TTS_ENGINES]

# Inclusive (min, max) bounds for numeric settings. The "keep" budget reuses the context maximum as its ceiling.
# Camera capture sizes are discovered by the browser from the actual hardware, so the server only checks the format
# ("WIDTHxHEIGHT") and sane bounds instead of keeping a fixed list.
CAMERA_RESOLUTION_RE = re.compile(r"^(\d{3,4})x(\d{3,4})$")
CAMERA_SIDE_BOUNDS = (160, 4320)
# When the camera image is attached to a turn: "auto" = when you ask about the scene and for a minute afterwards,
# "always" = every turn (slower, and the most accurate for follow-ups), "never" = never.
VISION_MODES = [("auto", "When I ask about what it sees (and follow-ups)"), ("always", "Every turn"), ("never", "Never")]
VISION_MODE_IDS = {mode for mode, _ in VISION_MODES}
# Longest side, in pixels, of the frame sent to the model. Larger = more legible detail but slower turns.
FRAME_SIDES = [320, 448, 640, 896, 1280]

LIMITS = {"context_tokens": (1500, 1_000_000), "tts_speed": (0.6, 1.6)}   # context up to 1M tokens for cloud models
KEEP_MIN = 500  # smallest allowed "recent history kept" budget, in tokens


# Language-model endpoints: any OpenAI-compatible server (Ollama and LM Studio by default).
ENDPOINT_KEYS = {"ollama_url": "Ollama URL", "lmstudio_url": "LM Studio URL"}
MODEL_KEYS = {"ollama_model": "Ollama model", "lmstudio_model": "LM Studio model"}
MODEL_NAME_RE = re.compile(r"^[^\s\x00-\x1f]{1,120}$")


def clean_url(value: object) -> str | None:
    """Return `value` as a normalised http(s) base URL without a trailing slash, or None if it is not one."""
    text = str(value).strip().rstrip("/")
    parts = urlparse(text)
    if len(text) > 200 or parts.scheme not in ("http", "https") or not parts.netloc or re.search(r"\s", text):
        return None
    return text


def path() -> Path:
    """Return the settings file location (``SETTINGS_FILE`` env var, else ``data/settings.json``)."""
    default = Path(__file__).resolve().parents[2] / "data" / "settings.json"
    return Path(os.environ.get("SETTINGS_FILE", default))


def validate(raw: dict, current: dict) -> tuple[dict, list[str]]:
    """Merge `raw` over `current`; return (accepted settings, human-readable problems).

    If any problem is found nothing is applied and `current` is returned unchanged.
    """
    out, errs = dict(current), []
    for key, label in (("context_tokens", "Context budget"), ("context_keep_tokens", "Recent history kept")):
        if key in raw:
            try:
                number = int(raw[key])
            except (TypeError, ValueError):
                errs.append(f"{label} must be a whole number")
                continue
            lo, hi = LIMITS["context_tokens"] if key == "context_tokens" else (KEEP_MIN, LIMITS["context_tokens"][1])
            if not lo <= number <= hi:
                errs.append(f"{label} must be between {lo} and {hi}")
            else:
                out[key] = number
    if "tts_speed" in raw:
        try:
            speed = float(raw["tts_speed"])
            lo, hi = LIMITS["tts_speed"]
            if lo <= speed <= hi:
                out["tts_speed"] = round(speed, 2)
            else:
                errs.append(f"Speech speed must be between {lo} and {hi}")
        except (TypeError, ValueError):
            errs.append("Speech speed must be a number")
    if "tts_engine" in raw:
        if raw["tts_engine"] in TTS_ENGINE_IDS:
            out["tts_engine"] = raw["tts_engine"]
        else:
            errs.append(f"Unknown voice engine: {raw['tts_engine']!r}")
    engine = next((e for e in TTS_ENGINES if e["id"] == out.get("tts_engine", "kokoro")), TTS_ENGINES[0])
    voice_ids = [voice["id"] for voice in engine["voices"]]
    if "tts_voice" in raw and raw["tts_voice"] in voice_ids:
        out["tts_voice"] = raw["tts_voice"]
    elif "tts_voice" in raw and ("tts_engine" not in raw or raw["tts_engine"] == current.get("tts_engine", "kokoro")):
        errs.append(f"Unknown voice: {raw['tts_voice']!r}")
    if out.get("tts_voice") not in voice_ids:      # switched engine: fall back to that engine's first voice
        out["tts_voice"] = voice_ids[0]
    if "skills_enabled" in raw:
        ids = raw["skills_enabled"]
        if isinstance(ids, list) and all(skill_id in SKILL_IDS for skill_id in ids):
            out["skills_enabled"] = list(dict.fromkeys(ids))
        else:
            errs.append("skills_enabled must be a list of known skill ids")
    if "camera_resolution" in raw:
        match = CAMERA_RESOLUTION_RE.match(str(raw["camera_resolution"]))
        lo, hi = CAMERA_SIDE_BOUNDS
        if match and all(lo <= int(side) <= hi for side in match.groups()):
            out["camera_resolution"] = raw["camera_resolution"]
        else:
            errs.append(f"Camera resolution must look like 1280x720 (sides {lo}-{hi})")
    if "vision_mode" in raw:
        if raw["vision_mode"] in VISION_MODE_IDS:
            out["vision_mode"] = raw["vision_mode"]
        else:
            errs.append(f"Camera use must be one of {sorted(VISION_MODE_IDS)}")
    if "frame_max_side" in raw:
        if raw["frame_max_side"] in FRAME_SIDES:
            out["frame_max_side"] = raw["frame_max_side"]
        else:
            errs.append(f"Frame size must be one of {FRAME_SIDES}")
    for key, label in ENDPOINT_KEYS.items():
        if key in raw:
            url = clean_url(raw[key])
            if url:
                out[key] = url
            else:
                errs.append(f"{label} must be a web address such as http://localhost:11434/v1")
    for key, label in MODEL_KEYS.items():
        if key in raw:
            if isinstance(raw[key], str) and MODEL_NAME_RE.match(raw[key].strip()):
                out[key] = raw[key].strip()
            else:
                errs.append(f"{label} must be a model name without spaces (for example qwen3.5:35b)")
    if out["context_keep_tokens"] > out["context_tokens"] * 0.8:
        errs.append("Recent history kept must be at most 80% of the context budget")
    return (current if errs else out), errs


def load(defaults: dict) -> dict:
    """Return the saved settings merged over `defaults`; fall back to `defaults` if the file is missing or invalid."""
    settings_file = path()
    try:
        saved = json.loads(settings_file.read_text())
        if not isinstance(saved, dict):
            raise ValueError("not a settings object")
    except FileNotFoundError:
        return defaults
    except (OSError, ValueError) as error:
        quarantine(settings_file, error)         # keep the unreadable file instead of silently losing the settings
        return defaults
    merged, errs = validate(saved, defaults)
    return defaults if errs else merged


def save(values: dict) -> None:
    """Write `values` to the settings file as JSON, creating the directory if needed."""
    write_private_atomic(path(), json.dumps(values, indent=2))   # endpoint URLs may hold credentials: owner-only
