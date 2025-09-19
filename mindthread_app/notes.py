"""Core note operations for mindthread backed by SQLite storage."""

from __future__ import annotations

import json
from array import array
from collections import Counter
from collections.abc import Iterable as IterableABC
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .db import ensure_schema, get_connection
from .services.openai_service import AIServiceError, generate_embedding, generate_metadata


Note = Dict[str, Any]

_SCHEMA_READY = False


def _ensure_db_ready() -> None:
    global _SCHEMA_READY
    if not _SCHEMA_READY:
        ensure_schema()
        _SCHEMA_READY = True


def normalize_threads(value: Any) -> List[str]:
    """Return a clean list of thread slugs from arbitrary input."""

    if value is None:
        return []

    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, IterableABC):
        candidates = list(value)
    else:
        return []

    seen: set[str] = set()
    cleaned: List[str] = []
    for item in candidates:
        if not isinstance(item, str):
            continue
        slug = item.strip()
        if not slug or slug in seen:
            continue
        cleaned.append(slug)
        seen.add(slug)
    return cleaned


def _now_iso() -> str:
    return datetime.now().isoformat()


def _decode_json_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [str(item) for item in data if isinstance(item, (str, int))]
    return []


def _to_json_or_none(values: Iterable[Any] | None) -> str | None:
    if not values:
        return None
    cleaned = list(values)
    if not cleaned:
        return None
    return json.dumps(cleaned)


def _decode_embedding(blob: bytes | None) -> List[float]:
    if not blob:
        return []
    arr = array("f")
    arr.frombytes(blob)
    return [float(value) for value in arr]


def _encode_embedding(values: Sequence[float] | None) -> bytes | None:
    if not values:
        return None
    arr = array("f", [float(v) for v in values])
    return arr.tobytes()


def _coerce_note_id(raw: Any) -> int | None:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _ensure_defaults(note: Note) -> Note:
    note.setdefault("type", "note")
    note["threads"] = normalize_threads(note.get("threads"))
    note.setdefault("tags", [])
    note.setdefault("related_ids", [])
    note.setdefault("embedding", [])
    note.setdefault("created_at", _now_iso())
    return note


def _row_to_note(row) -> Note:
    note: Note = {
        "id": str(row["id"]),
        "type": row["type"] or "note",
        "title": row["title"] or "Untitled",
        "text": row["body"] or "",
        "category": row["category"] or "",
        "tags": _decode_json_list(row["tags"]),
        "threads": normalize_threads(_decode_json_list(row["threads"])),
        "related_ids": [str(item) for item in _decode_json_list(row["related_ids"])],
        "embedding": _decode_embedding(row["embedding"]),
        "created_at": row["created_at"] or _now_iso(),
        "updated_at": row["updated_at"],
    }
    return _ensure_defaults(note)


def auto_enrich_note(
    text: str,
    existing_categories: Sequence[str] | None = None,
    existing_tags: Sequence[str] | None = None,
) -> Tuple[Dict[str, Any], List[float]]:
    """Return (metadata, embedding) for the provided note text."""

    metadata = generate_metadata(text, existing_categories, existing_tags)
    metadata.setdefault("type", "note")
    embedding = generate_embedding(text)
    return metadata, embedding


def build_note(
    text: str,
    metadata: Dict[str, Any],
    embedding: Sequence[float],
    related_ids: Sequence[str] | None = None,
) -> Note:
    """Create a note dictionary ready for persistence."""

    related = sorted({str(rid) for rid in (related_ids or []) if str(rid)})
    return {
        "id": None,
        "type": metadata.get("type", "note"),
        "text": text,
        "title": metadata.get("title", "Untitled"),
        "category": metadata.get("category", "General"),
        "tags": list(metadata.get("tags", ["untagged"])),
        "threads": normalize_threads(metadata.get("threads")),
        "embedding": list(embedding),
        "created_at": _now_iso(),
        "updated_at": None,
        "related_ids": related,
    }


def _next_note_id(connection) -> int:
    row = connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM notes").fetchone()
    return int(row[0])


def persist_note(note: Note, linked_note_ids: Sequence[str] | None = None) -> Note:
    """Persist a note into SQLite and return the stored record."""

    _ensure_db_ready()

    threads = normalize_threads(note.get("threads"))
    tags = [tag for tag in (note.get("tags") or []) if tag]
    related = sorted({str(rid) for rid in (linked_note_ids or note.get("related_ids") or []) if str(rid)})
    now = _now_iso()

    with get_connection() as connection:
        note_id = _coerce_note_id(note.get("id"))
        if note_id is None:
            note_id = _next_note_id(connection)

        created_at = note.get("created_at") or now
        connection.execute(
            """
            INSERT OR REPLACE INTO notes (
                id, type, title, body, category, tags, threads,
                related_ids, embedding, metadata, created_at, updated_at
            ) VALUES (
                :id, :type, :title, :body, :category, :tags, :threads,
                :related_ids, :embedding, NULL, :created_at, :updated_at
            )
            """,
            {
                "id": note_id,
                "type": note.get("type", "note"),
                "title": note.get("title", "Untitled"),
                "body": note.get("text", ""),
                "category": note.get("category"),
                "tags": _to_json_or_none(tags),
                "threads": _to_json_or_none(threads),
                "related_ids": _to_json_or_none(related),
                "embedding": _encode_embedding(note.get("embedding")),
                "created_at": created_at,
                "updated_at": now,
            },
        )

        if related:
            for target in related:
                target_id = _coerce_note_id(target)
                if target_id is None or target_id == note_id:
                    continue
                existing = connection.execute(
                    "SELECT related_ids FROM notes WHERE id = ?",
                    (target_id,),
                ).fetchone()
                if not existing:
                    continue
                current = set(_decode_json_list(existing["related_ids"]))
                current.add(str(note_id))
                connection.execute(
                    "UPDATE notes SET related_ids = ?, updated_at = ? WHERE id = ?",
                    (_to_json_or_none(sorted(current)), now, target_id),
                )

        connection.commit()

    stored = get_note(str(note_id))
    return stored if stored else _ensure_defaults(dict(note, id=str(note_id)))


def list_all_notes() -> List[Note]:
    _ensure_db_ready()
    with get_connection(readonly=True) as connection:
        rows = connection.execute(
            "SELECT * FROM notes ORDER BY datetime(created_at) ASC, id ASC"
        ).fetchall()
    return [_row_to_note(row) for row in rows]


def note_counts_by_day(limit: int | None = None) -> List[Tuple[str, int]]:
    _ensure_db_ready()
    with get_connection(readonly=True) as connection:
        rows = connection.execute(
            """
            SELECT substr(created_at, 1, 10) AS day, COUNT(1) AS total
            FROM notes
            GROUP BY day
            ORDER BY day
            """
        ).fetchall()
    items = [(row["day"], int(row["total"])) for row in rows if row["day"]]
    if limit:
        items = items[-limit:]
    return items


def tag_frequency() -> List[Tuple[str, int]]:
    counter: Counter[str] = Counter()
    for note in list_all_notes():
        for tag in note.get("tags", []):
            if tag:
                counter[tag] += 1
    return counter.most_common()


def search_notes(query: str) -> List[Note]:
    query_lower = query.lower()

    def matches(note: Note) -> bool:
        return (
            query_lower in note.get("text", "").lower()
            or query_lower in note.get("title", "").lower()
            or any(query_lower in tag.lower() for tag in note.get("tags", []))
        )

    return [note for note in list_all_notes() if matches(note)]


def get_note(note_id: str) -> Note | None:
    _ensure_db_ready()
    target_id = _coerce_note_id(note_id)
    if target_id is None:
        return None
    with get_connection(readonly=True) as connection:
        row = connection.execute("SELECT * FROM notes WHERE id = ?", (target_id,)).fetchone()
    if not row:
        return None
    return _row_to_note(row)


def _notes_with_embeddings(exclude_ids: Iterable[str] | None = None) -> List[Note]:
    excluded = {str(nid) for nid in (exclude_ids or [])}
    return [note for note in list_all_notes() if note.get("embedding") and note.get("id") not in excluded]


def suggest_related_by_embedding(
    embedding: Sequence[float],
    exclude_ids: Iterable[str] | None = None,
    top_k: int = 5,
) -> List[Tuple[Note, float]]:
    candidates = _notes_with_embeddings(exclude_ids)
    if not candidates:
        return []

    target_vector = np.array(embedding).reshape(1, -1)
    other_vectors = np.array([note["embedding"] for note in candidates])

    similarities = cosine_similarity(target_vector, other_vectors)[0]
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    return [(candidates[idx], float(similarities[idx])) for idx in ranked_indices]


def find_related_notes(note_id: str, top_k: int = 5) -> Tuple[Note, List[Tuple[Note, float]]]:
    target = get_note(note_id)
    if target is None:
        raise ValueError(f"Note {note_id} not found")

    embedding = target.get("embedding")
    if not embedding:
        raise ValueError(f"Note {note_id} does not have an embedding")

    others = _notes_with_embeddings(exclude_ids=[note_id])
    if not others:
        return target, []

    target_vector = np.array(embedding).reshape(1, -1)
    other_vectors = np.array([note["embedding"] for note in others])
    similarities = cosine_similarity(target_vector, other_vectors)[0]
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    related = [(others[idx], float(similarities[idx])) for idx in ranked_indices]
    return target, related


def remove_note(note_id: str) -> bool:
    _ensure_db_ready()
    target_id = _coerce_note_id(note_id)
    if target_id is None:
        return False

    with get_connection() as connection:
        exists = connection.execute("SELECT 1 FROM notes WHERE id = ?", (target_id,)).fetchone()
        if not exists:
            return False

        now = _now_iso()
        rows = connection.execute(
            "SELECT id, related_ids FROM notes WHERE related_ids IS NOT NULL"
        ).fetchall()
        for row in rows:
            related = set(_decode_json_list(row["related_ids"]))
            if str(target_id) not in related:
                continue
            related.discard(str(target_id))
            connection.execute(
                "UPDATE notes SET related_ids = ?, updated_at = ? WHERE id = ?",
                (_to_json_or_none(sorted(related)), now, row["id"]),
            )

        connection.execute("DELETE FROM notes WHERE id = ?", (target_id,))
        connection.commit()

    return True


def update_note_text(note_id: str, new_text: str, regenerate_embedding: bool = True) -> bool:
    _ensure_db_ready()
    target_id = _coerce_note_id(note_id)
    if target_id is None:
        return False

    with get_connection() as connection:
        exists = connection.execute("SELECT 1 FROM notes WHERE id = ?", (target_id,)).fetchone()
        if not exists:
            return False

        embedding_blob = None
        if regenerate_embedding:
            embedding = generate_embedding(new_text)
            embedding_blob = _encode_embedding(embedding)

        now = _now_iso()
        if embedding_blob is not None:
            connection.execute(
                "UPDATE notes SET body = ?, embedding = ?, updated_at = ? WHERE id = ?",
                (new_text, embedding_blob, now, target_id),
            )
        else:
            connection.execute(
                "UPDATE notes SET body = ?, updated_at = ? WHERE id = ?",
                (new_text, now, target_id),
            )
        connection.commit()

    return True


def rename_category(old: str, new: str) -> bool:
    if old == new:
        return False
    _ensure_db_ready()

    now = _now_iso()
    with get_connection() as connection:
        result = connection.execute(
            "UPDATE notes SET category = ?, updated_at = ? WHERE category = ?",
            (new, now, old),
        )
        connection.commit()
        return result.rowcount > 0


def rename_tag(old: str, new: str) -> bool:
    if old == new:
        return False

    _ensure_db_ready()
    now = _now_iso()
    changed = False

    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, tags FROM notes WHERE tags IS NOT NULL"
        ).fetchall()
        for row in rows:
            tags = _decode_json_list(row["tags"])
            if old not in tags:
                continue
            updated: List[str] = []
            seen: set[str] = set()
            for tag in tags:
                replacement = new if tag == old else tag
                if not replacement:
                    continue
                if replacement not in seen:
                    seen.add(replacement)
                    updated.append(replacement)
            connection.execute(
                "UPDATE notes SET tags = ?, updated_at = ? WHERE id = ?",
                (_to_json_or_none(updated), now, row["id"]),
            )
            changed = True

        if changed:
            connection.commit()

    return changed


def get_notes_between(start: datetime, end: datetime | None = None) -> List[Note]:
    _ensure_db_ready()
    start_iso = start.isoformat()
    if end is not None:
        end_iso = end.isoformat()
        query = (
            "SELECT * FROM notes WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?)"
        )
        params = (start_iso, end_iso)
    else:
        query = "SELECT * FROM notes WHERE datetime(created_at) >= datetime(?)"
        params = (start_iso,)

    with get_connection(readonly=True) as connection:
        rows = connection.execute(query, params).fetchall()
    return [_row_to_note(row) for row in rows]


def notes_since(days: int) -> List[Note]:
    start = datetime.now() - timedelta(days=days)
    return get_notes_between(start)


__all__ = [
    "AIServiceError",
    "auto_enrich_note",
    "build_note",
    "persist_note",
    "list_all_notes",
    "search_notes",
    "get_note",
    "find_related_notes",
    "remove_note",
    "update_note_text",
    "rename_category",
    "rename_tag",
    "note_counts_by_day",
    "tag_frequency",
    "notes_since",
    "get_notes_between",
    "normalize_threads",
    "suggest_related_by_embedding",
]
