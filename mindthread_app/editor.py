"""Helper utilities for launching a text editor."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional


def launch_editor(initial_text: str) -> Optional[str]:
    """Open the user's preferred editor and return the edited text."""

    editor = os.environ.get("EDITOR")
    fallback = shutil.which("nano") or shutil.which("vi") or shutil.which("vim")
    command = editor or fallback
    if not command:
        return None

    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
        tmp.write(initial_text)
        tmp_path = tmp.name

    try:
        subprocess.run([command, tmp_path], check=False)
        with open(tmp_path, "r", encoding="utf-8") as handle:
            return handle.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


__all__ = ["launch_editor"]
