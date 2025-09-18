"""Core note operations for mindthread."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .services.openai_service import AIServiceError, generate_embedding, generate_metadata
from .storage import load_notes, next_note_id, save_notes


Note = Dict[str, Any]


def _ensure_defaults(note: Note) -> Note:
    note.setdefault("type", "note")
    return note


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

    notes = load_notes()
    note_id = next_note_id(notes)

    return {
        "id": note_id,
        "type": metadata.get("type", "note"),
        "text": text,
        "title": metadata.get("title", "Untitled"),
        "category": metadata.get("category", "General"),
        "tags": metadata.get("tags", ["untagged"]),
        "embedding": list(embedding),
        "created_at": datetime.now().isoformat(),
        "related_ids": list({str(rid) for rid in (related_ids or [])}),
    }


def persist_note(note: Note, linked_note_ids: Sequence[str] | None = None) -> None:
    """Append a note to storage, updating bidirectional links."""

    notes = load_notes()
    note.setdefault("type", "note")
    notes.append(note)

    if linked_note_ids:
        unique_related = sorted({str(rid) for rid in linked_note_ids})
        note["related_ids"] = unique_related

        by_id = {existing.get("id"): existing for existing in notes}
        for target_id in unique_related:
            target = by_id.get(target_id)
            if not target:
                continue
            current = set(target.get("related_ids", []))
            current.add(note["id"])
            target["related_ids"] = sorted(current)

    else:
        note.setdefault("related_ids", [])

    save_notes(notes)


def list_all_notes() -> List[Note]:
    return [_ensure_defaults(note) for note in load_notes()]


def note_counts_by_day(limit: int | None = None) -> List[Tuple[str, int]]:
    notes = [_ensure_defaults(note) for note in load_notes()]
    counter: Counter[str] = Counter()
    for note in notes:
        created = note.get("created_at")
        if not created:
            continue
        date = created[:10]
        counter[date] += 1
    items = sorted(counter.items())
    if limit:
        items = items[-limit:]
    return items


def tag_frequency() -> List[Tuple[str, int]]:
    notes = [_ensure_defaults(note) for note in load_notes()]
    counter: Counter[str] = Counter()
    for note in notes:
        for tag in note.get("tags", []):
            if tag:
                counter[tag] += 1
    return counter.most_common()


def search_notes(query: str) -> List[Note]:
    notes = [_ensure_defaults(note) for note in load_notes()]
    if not notes:
        return []

    query_lower = query.lower()

    def matches(note: Note) -> bool:
        return (
            query_lower in note.get("text", "").lower()
            or query_lower in note.get("title", "").lower()
            or any(query_lower in tag.lower() for tag in note.get("tags", []))
        )

    return [note for note in notes if matches(note)]


def get_note(note_id: str) -> Note | None:
    for note in load_notes():
        note = _ensure_defaults(note)
        if note.get("id") == note_id:
            return note
    return None


def suggest_related_by_embedding(
    embedding: Sequence[float],
    exclude_ids: Iterable[str] | None = None,
    top_k: int = 5,
) -> List[Tuple[Note, float]]:
    notes = [_ensure_defaults(note) for note in load_notes()]
    if not notes:
        return []

    excluded = set(exclude_ids or [])

    candidates = [n for n in notes if n.get("embedding") and n.get("id") not in excluded]
    if not candidates:
        return []

    target_vector = np.array(embedding).reshape(1, -1)
    other_vectors = np.array([n["embedding"] for n in candidates])

    similarities = cosine_similarity(target_vector, other_vectors)[0]
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    return [(candidates[idx], float(similarities[idx])) for idx in ranked_indices]


def find_related_notes(note_id: str, top_k: int = 5) -> Tuple[Note, List[Tuple[Note, float]]]:
    notes = [_ensure_defaults(note) for note in load_notes()]
    if not notes:
        raise ValueError("No notes available")

    target_note = next((n for n in notes if n.get("id") == note_id), None)
    if target_note is None:
        raise ValueError(f"Note {note_id} not found")

    target_embedding = target_note.get("embedding")
    if not target_embedding:
        raise ValueError(f"Note {note_id} does not have an embedding")

    others = [n for n in notes if n is not target_note and n.get("embedding")]
    if not others:
        return target_note, []

    target_vector = np.array(target_embedding).reshape(1, -1)
    other_vectors = np.array([n["embedding"] for n in others])

    similarities = cosine_similarity(target_vector, other_vectors)[0]
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    related = [(others[idx], float(similarities[idx])) for idx in ranked_indices]
    return target_note, related


def remove_note(note_id: str) -> bool:
    notes = [_ensure_defaults(note) for note in load_notes()]
    remaining = [note for note in notes if note.get("id") != note_id]
    if len(remaining) == len(notes):
        return False
    save_notes(remaining)
    return True


def update_note_text(note_id: str, new_text: str, regenerate_embedding: bool = True) -> bool:
    notes = [_ensure_defaults(note) for note in load_notes()]
    updated = False

    for note in notes:
        if note.get("id") != note_id:
            continue
        note["text"] = new_text
        note["updated_at"] = datetime.now().isoformat()
        if regenerate_embedding:
            note["embedding"] = generate_embedding(new_text)
        updated = True
        break

    if updated:
        save_notes(notes)

    return updated


def rename_category(old: str, new: str) -> bool:
    if old == new:
        return False

    notes = [_ensure_defaults(note) for note in load_notes()]
    changed = False
    for note in notes:
        if note.get("category") == old:
            note["category"] = new
            changed = True
    if changed:
        save_notes(notes)
    return changed


def rename_tag(old: str, new: str) -> bool:
    if old == new:
        return False

    notes = [_ensure_defaults(note) for note in load_notes()]
    changed = False
    for note in notes:
        tags = note.get("tags", [])
        if not tags:
            continue
        updated = []
        tag_changed = False
        for tag in tags:
            if tag == old:
                if new:
                    updated.append(new)
                tag_changed = True
                changed = True
            else:
                updated.append(tag)
        if tag_changed:
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for tag in updated:
                if tag not in seen:
                    seen.add(tag)
                    deduped.append(tag)
            note["tags"] = deduped
    if changed:
        save_notes(notes)
    return changed


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
]


def get_notes_between(start: datetime, end: datetime | None = None) -> List[Note]:
    notes = [_ensure_defaults(note) for note in load_notes()]
    results: List[Note] = []
    for note in notes:
        created = note.get("created_at")
        if not created:
            continue
        try:
            created_dt = datetime.fromisoformat(created)
        except ValueError:
            continue
        if created_dt >= start and (end is None or created_dt <= end):
            results.append(note)
    return results


def notes_since(days: int) -> List[Note]:
    start = datetime.now() - timedelta(days=days)
    return get_notes_between(start)
