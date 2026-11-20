"""Interactive CLI. Text or voice in; text and/or speech out; optional camera."""
from __future__ import annotations

import argparse

from .audio.tts import Speaker
from .config import Config
from .engine import Engine, sentences
from .vision.camera import Camera

HELP = """Commands:
  /cam <msg>    send msg with a fresh camera frame
  /think <msg>  force deep reasoning (Claude if key set, else local)
  /voice        toggle microphone input
  /speak        toggle spoken replies
  /backend      switch ollama <-> lmstudio
  /reset        clear conversation
  /quit         exit"""


def main() -> None:
    """Run the interactive read-eval-reply loop until the user quits or input ends."""
    parser = argparse.ArgumentParser(description="Interactive multimodal LLM")
    parser.add_argument("--voice", action="store_true", help="start with microphone input")
    parser.add_argument("--speak", action="store_true", help="start with spoken replies")
    parser.add_argument("--backend", choices=["ollama", "lmstudio"])
    args = parser.parse_args()

    cfg = Config()
    if args.backend:
        cfg.backend = args.backend
    camera = Camera(cfg.camera_index)
    engine = Engine(cfg, camera_fn=camera.capture_b64)
    speaker, listener = Speaker(cfg.tts_voice), None
    voice, speak = args.voice, args.speak

    print(f"Local: {cfg.backend} ({engine.local.model}) | deep think: "
          f"{engine.claude.label}")
    print(HELP)
    while True:
        try:
            if voice:
                from .audio.stt import Listener
                listener = listener or Listener(cfg.whisper_model)
                print("🎤 listening...")
                text = listener.listen()
                print(f"you> {text}")
            else:
                text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        use_camera = force_claude = False  # per-message flags set by slash commands
        if text.startswith("/"):
            cmd, _, rest = text.partition(" ")
            if cmd == "/quit":
                break
            elif cmd == "/voice":
                voice = not voice; print(f"voice={voice}"); continue
            elif cmd == "/speak":
                speak = not speak; print(f"speak={speak}"); continue
            elif cmd == "/reset":
                engine.reset(); print("cleared"); continue
            elif cmd == "/backend":
                cfg.backend = "lmstudio" if cfg.backend == "ollama" else "ollama"
                new_engine = Engine(cfg, camera_fn=camera.capture_b64); new_engine.adopt(engine); engine = new_engine; print(f"backend={cfg.backend}"); continue
            elif cmd == "/cam":
                use_camera, text = True, rest or "What do you see?"
            elif cmd in ("/claude", "/think"):
                force_claude, text = True, rest
            else:
                print(HELP); continue
        try:
            print("bot> ", end="", flush=True)
            chunks = engine.reply(text, use_camera=use_camera, force_claude=force_claude)
            for sent in sentences(chunks):
                print(sent + " ", end="", flush=True)
                if speak:
                    speaker.speak(sent, block=False)
            print()
            engine.compact()
            speaker.wait()  # don't let the mic hear the assistant
        except Exception as error:  # connection errors etc.
            print(f"\n[error] {type(error).__name__}: {error}")
    speaker.stop()


if __name__ == "__main__":
    main()
