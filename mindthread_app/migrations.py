"""Utilities for migrating legacy JSON storage into SQLite."""

from __future__ import annotations

import json
from array import array
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from .config import get_settings
from .db import ensure_schema, get_connection, get_database_path
from .storage import load_notes


@dataclass(slots=True)
class MigrationReport:
    """Summary of the JSON → SQLite migration run."""

    migrated: int
    skipped: int
    errors: List[str]
    dry_run: bool
    database_path: Path

    @property
    def success(self) -> bool:
        return not self.errors


def _encode_embedding(values: Sequence[float] | None) -> bytes | None:
    if not values:
        return None
    arr = array("f", [float(value) for value in values])
    return arr.tobytes()


def _to_json(value: object) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


def _coerce_iso(value: str | None) -> str:
    if value:
        try:
            datetime.fromisoformat(value)
            return value
        except ValueError:
            pass
    return datetime.now().isoformat()


def migrate_to_sqlite(*, dry_run: bool = False, force: bool = False) -> MigrationReport:
    """Migrate legacy JSON notes into SQLite storage."""

    settings = get_settings()
    if settings.storage_type not in {"sqlite", "json"}:
        raise RuntimeError(f"Unsupported STORAGE_TYPE '{settings.storage_type}' for migration")

    notes = load_notes()
    database_path = get_database_path()

    ensure_schema()

    migrated = 0
    skipped = 0
    errors: List[str] = []

    with get_connection() as connection:
        # Guard against double migration unless explicitly forced.
        existing_count = connection.execute("SELECT COUNT(1) FROM notes").fetchone()[0]
        if existing_count and not force:
            errors.append(
                "Notes table already contains data. Re-run with force=True to overwrite."  # noqa: E501
            )
            return MigrationReport(migrated, skipped, errors, dry_run, database_path)

        if force and not dry_run:
            connection.execute("DELETE FROM notes")

        for record in notes:
            try:
                raw_id = record.get("id")
                note_id = int(raw_id) if raw_id is not None else None
            except (TypeError, ValueError):
                skipped += 1
                errors.append(f"Skipping note with invalid id: {record!r}")
                continue

            if note_id is None:
                skipped += 1
                errors.append(f"Skipping note missing id: {record!r}")
                continue

            payload = {
                "id": note_id,
                "type": record.get("type", "note"),
                "title": record.get("title", "Untitled"),
                "body": record.get("text", ""),
                "category": record.get("category"),
                "tags": _to_json(record.get("tags")),
                "threads": _to_json(record.get("threads")),
                "related_ids": _to_json(record.get("related_ids")),
                "embedding": _encode_embedding(record.get("embedding")),
                "metadata": _to_json(record.get("metadata")),
                "created_at": _coerce_iso(record.get("created_at")),
                "updated_at": record.get("updated_at"),
            }

            migrated += 1
            if dry_run:
                continue

            connection.execute(
                """
                INSERT OR REPLACE INTO notes (
                    id, type, title, body, category, tags, threads,
                    related_ids, embedding, metadata, created_at, updated_at
                ) VALUES (
                    :id, :type, :title, :body, :category, :tags, :threads,
                    :related_ids, :embedding, :metadata, :created_at, :updated_at
                )
                """,
                payload,
            )

        if not dry_run:
            connection.commit()

    return MigrationReport(migrated, skipped, errors, dry_run, database_path)


__all__ = ["MigrationReport", "migrate_to_sqlite"]
