"""Storage utilities for persisting notes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .config import PROJECT_ROOT, get_settings


LEGACY_NOTES_FILE = PROJECT_ROOT / "notes.json"


class StorageError(RuntimeError):
    """Raised when note storage operations fail."""


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_notes() -> List[Dict[str, Any]]:
    """Load notes from the configured storage backend."""

    settings = get_settings()
    path = settings.notes_file

    if path.exists():
        source = path
    elif LEGACY_NOTES_FILE.exists():
        source = LEGACY_NOTES_FILE
    else:
        return []

    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise StorageError(f"Failed to parse notes file at {source}") from exc


def save_notes(notes: List[Dict[str, Any]]) -> None:
    """Persist notes to the configured storage backend."""

    settings = get_settings()
    path = settings.notes_file
    _ensure_parent(path)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(notes, handle, indent=2)


def next_note_id(notes: List[Dict[str, Any]]) -> str:
    """Compute the next note identifier for JSON storage."""

    if not notes:
        return "1"

    max_id = max(int(note["id"]) for note in notes if note.get("id"))
    return str(max_id + 1)


__all__ = ["StorageError", "load_notes", "save_notes", "next_note_id"]
