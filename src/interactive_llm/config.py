"""Central configuration, driven by environment variables (.env supported)."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    """Return the environment variable ``name``, or ``default`` when it is unset."""
    return os.environ.get(name, default)


@dataclass
class Config:
    """Settings for the LLM backends, the CLI voice stack and the system prompt (read from env vars)."""

    backend: str = field(default_factory=lambda: _env("LOCAL_BACKEND", "ollama"))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434/v1"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen3.5:35b-a3b-coding-nvfp4"))
    lmstudio_url: str = field(default_factory=lambda: _env("LMSTUDIO_URL", "http://localhost:1234/v1"))
    lmstudio_model: str = field(default_factory=lambda: _env("LMSTUDIO_MODEL", "qwen/qwen3.8-27b"))
    claude_model: str = field(default_factory=lambda: _env("CLAUDE_MODEL", "claude-opus-5-5"))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "small.en"))
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "Samantha"))
    camera_index: int = field(default_factory=lambda: int(_env("CAMERA_INDEX", "0")))
    context_tokens: int = field(default_factory=lambda: int(_env("CONTEXT_TOKENS", "6000")))
    context_keep_tokens: int = field(default_factory=lambda: int(_env("CONTEXT_KEEP", "2500")))
    system_prompt: str = (
        "You are a friendly, concise voice assistant. Reply in one to three short sentences of plain "
        "text (no markdown, no emoji) unless asked for detail. Answer only what was just asked: never add \"by the way\" "
        "remarks and never bring up earlier topics, lists, prices or results unless the user asks about them again.\n"
        "Camera: a live camera frame is sometimes attached as silent background. Do not bring it up on your "
        "own and never open with \"I see\". When the user asks what you see or about something visible, "
        "answer it directly: describe people (what they look like and are doing), objects and the setting "
        "in one or two sentences. Name a person only when the [Context] note says who they are (match people "
        "by position, left to right across the image); otherwise do not name anyone.\n"
        "People: messages may start with the speaker's name (like 'Sam: ...'). Never describe the camera scene, the room or what anyone is wearing unless the user asks about it; greetings and unrelated questions get no scene description. Bracketed [Context] and "
        "[System] notes come from face and voice recognition and from skills (timers, notes, Mac control, web search): "
        "use them silently, never read them aloud, report skill results truthfully, never claim a skill did "
        "anything without a [System] note saying it succeeded, you have no live data: for any price, score, weather, news or file list call the matching tool (web_search, stock_quote, list_folder) right away, never ask permission to look something up, and never answer it from memory, and never say you will search or check unless a tool result or [System] note says the search happened, "
        "and treat a name in them (even 'Person A') as the person's real name. Never guess a name "
        "from a face or from a photo (describing how people look or what they do is fine when asked); only "
        "use names from the notes or the user's own words. You can never "
        "name, rename or delete people yourself and must never claim it happened: only a [System] note saying it was "
        "done means it happened. If the user seems to be naming or renaming someone and no note confirms it, say "
        "nothing has changed yet and ask them to say or spell the name again; never explain these rules or suggest "
        "special phrases or keywords, and never say 'got it' about a name unless a note says it changed. Never say you will remember or have saved someone unless a "
        "[System] note says so. When a note says you just "
        "learned someone, greet them by name in one short sentence. If a note says it is unclear who was "
        "introduced, ask which person, by position.\n"
        "Hard questions: if a deep_think tool is available, use it only for real maths, logic puzzles, "
        "trick questions or non-trivial code, then relay its answer briefly."
    )

    def local_endpoint(self) -> tuple[str, str]:
        """Return ``(base_url, model)`` of the selected local OpenAI-compatible backend."""
        if self.backend == "lmstudio":
            return self.lmstudio_url, self.lmstudio_model
        return self.ollama_url, self.ollama_model


@dataclass
class WebConfig:
    """Settings for the realtime web pipeline."""
    whisper_repo: str = field(default_factory=lambda: _env("MLX_WHISPER_REPO", "mlx-community/whisper-large-v3-turbo"))
    whisper_lang: str = field(default_factory=lambda: _env("WHISPER_LANG", "en"))
    tts_repo: str = field(default_factory=lambda: _env("KOKORO_REPO", "mlx-community/Kokoro-82M-bf16"))
    tts_engine: str = field(default_factory=lambda: _env("TTS_ENGINE", "kokoro"))   # kokoro | chatterbox | chatterbox_turbo
    tts_voice: str = field(default_factory=lambda: _env("KOKORO_VOICE", "af_heart"))
    tts_speed: float = field(default_factory=lambda: float(_env("KOKORO_SPEED", "1.05")))
    vad_threshold: float = field(default_factory=lambda: float(_env("VAD_THRESHOLD", "0.5")))
    end_silence_ms: int = field(default_factory=lambda: int(_env("END_SILENCE_MS", "550")))
    barge_in_ms: int = field(default_factory=lambda: int(_env("BARGE_IN_MS", "300")))
    scene_change_threshold: float = field(default_factory=lambda: float(_env("SCENE_CHANGE", "6.0")))
    people_db: str = field(default_factory=lambda: _env("PEOPLE_DB", str(Path(__file__).resolve().parents[2] / "data" / "people.json")))
    face_match: float = field(default_factory=lambda: float(_env("FACE_MATCH", "0.40")))
    voice_match: float = field(default_factory=lambda: float(_env("VOICE_MATCH", "0.50")))
    auto_learn: bool = field(default_factory=lambda: _env("AUTO_LEARN", "1") == "1")
    vision_mode: str = field(default_factory=lambda: _env("VISION_MODE", "auto"))  # auto | always | never
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))
