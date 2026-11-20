"""Safe file helpers shared by the stores that hold private data (people.json, settings.json, notes.json)."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def write_private_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` so a crash never leaves a half-written file and no one else can read it.

    The temp file is created by ``mkstemp`` (mode 0600 from the first byte), in the same folder so the final
    ``os.replace`` is atomic. On any failure the temp file is removed and the original file is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-")
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def quarantine(path: Path, reason: object) -> Path:
    """Move an unreadable data file aside as ``<name>.corrupt-<time>`` so the next save cannot overwrite it.

    Prints a warning with the new location. Raises OSError if it cannot be moved: starting with empty data on top of
    a file we could neither read nor move would destroy it on the next save, so the caller must stop instead.
    """
    aside = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
    os.replace(path, aside)
    print(f"[warn] {path.name} could not be read ({reason}); kept a copy as {aside.name} and started empty", flush=True)
    return aside
