"""Thread registry utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import get_settings

THREADS_FILENAME = "threads.json"
SLUG_SANITIZER = re.compile(r"[^a-z0-9]+")


def normalize_slug(raw: str) -> str:
    """Normalize raw labels to filesystem-friendly slugs."""

    slug = SLUG_SANITIZER.sub("-", raw.strip().lower())
    return slug.strip("-")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class ThreadRecord:
    slug: str
    title: str
    intent: str
    note_id: Optional[str]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ThreadRecord":
        return cls(**data)


def _threads_path() -> Path:
    settings = get_settings()
    return settings.data_dir / THREADS_FILENAME


def load_threads() -> Dict[str, ThreadRecord]:
    path = _threads_path()
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {slug: ThreadRecord.from_dict(record) for slug, record in data.items()}


def save_threads(threads: Dict[str, ThreadRecord]) -> None:
    path = _threads_path()
    _ensure_parent(path)
    serializable = {slug: record.to_dict() for slug, record in sorted(threads.items())}
    path.write_text(json.dumps(serializable, indent=2))


def get_thread(slug: str) -> Optional[ThreadRecord]:
    threads = load_threads()
    return threads.get(slug)


def upsert_thread(record: ThreadRecord) -> None:
    threads = load_threads()
    threads[record.slug] = record
    save_threads(threads)


def delete_thread(slug: str) -> None:
    threads = load_threads()
    if slug in threads:
        del threads[slug]
        save_threads(threads)


def list_threads() -> List[ThreadRecord]:
    return sorted(load_threads().values(), key=lambda r: r.created_at)


def _dedupe_slug(base_slug: str, existing: Iterable[str]) -> str:
    taken = set(existing)
    if base_slug not in taken:
        return base_slug
    index = 2
    while True:
        candidate = f"{base_slug}-{index}"
        if candidate not in taken:
            return candidate
        index += 1


def create_thread(slug: str, title: str, intent: str, note_id: Optional[str]) -> ThreadRecord:
    now = datetime.now().isoformat()
    normalized = normalize_slug(slug or title)
    if not normalized:
        raise ValueError("Thread slug cannot be empty")
    threads = load_threads()
    unique_slug = _dedupe_slug(normalized, threads.keys())
    record = ThreadRecord(
        slug=unique_slug,
        title=title,
        intent=intent,
        note_id=note_id,
        created_at=now,
        updated_at=now,
    )
    threads[unique_slug] = record
    save_threads(threads)
    return record


def update_thread(slug: str, *, title: Optional[str] = None, intent: Optional[str] = None, note_id: Optional[str] = None) -> Optional[ThreadRecord]:
    threads = load_threads()
    record = threads.get(slug)
    if not record:
        return None
    if title is not None:
        record.title = title
    if intent is not None:
        record.intent = intent
    if note_id is not None:
        record.note_id = note_id
    record.updated_at = datetime.now().isoformat()
    threads[slug] = record
    save_threads(threads)
    return record
