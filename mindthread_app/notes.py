"""Core note operations for mindthread."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .services.openai_service import AIServiceError, generate_embedding, generate_metadata
from .storage import load_notes, next_note_id, save_notes


Note = Dict[str, Any]


def auto_enrich_note(text: str) -> Tuple[Dict[str, Any], List[float]]:
    """Return (metadata, embedding) for the provided note text."""

    metadata = generate_metadata(text)
    embedding = generate_embedding(text)
    return metadata, embedding


def build_note(text: str, metadata: Dict[str, Any], embedding: Sequence[float]) -> Note:
    """Create a note dictionary ready for persistence."""

    notes = load_notes()
    note_id = next_note_id(notes)

    return {
        "id": note_id,
        "text": text,
        "title": metadata.get("title", "Untitled"),
        "category": metadata.get("category", "General"),
        "tags": metadata.get("tags", ["untagged"]),
        "embedding": list(embedding),
        "created_at": datetime.now().isoformat(),
    }


def persist_note(note: Note) -> None:
    """Append a note to storage."""

    notes = load_notes()
    notes.append(note)
    save_notes(notes)


def list_all_notes() -> List[Note]:
    return load_notes()


def search_notes(query: str) -> List[Note]:
    notes = load_notes()
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
    notes = load_notes()
    return next((note for note in notes if note.get("id") == note_id), None)


def find_related_notes(note_id: str, top_k: int = 5) -> Tuple[Note, List[Tuple[Note, float]]]:
    notes = load_notes()
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
    notes = load_notes()
    remaining = [note for note in notes if note.get("id") != note_id]
    if len(remaining) == len(notes):
        return False
    save_notes(remaining)
    return True


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
]
