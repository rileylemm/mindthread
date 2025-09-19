"""AI-assisted thread orchestration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .notes import (
    Note,
    build_note,
    get_note,
    list_all_notes,
    normalize_threads,
    persist_note,
)
from .services.openai_service import (
    generate_thread_discovery,
    generate_thread_note_update,
    generate_thread_review,
)
from .thread_store import (
    ThreadRecord,
    create_thread,
    list_threads,
    update_thread,
)


DISCOVERY_NOTE_LIMIT = 40
REVIEW_CANDIDATE_LIMIT = 10


@dataclass(frozen=True)
class ThreadCandidate:
    slug: str
    title: str
    intent: str
    summary: str
    journal_entry: str
    note_ids: List[str]


@dataclass(frozen=True)
class ThreadReviewSuggestion:
    note_id: str
    reason: str
    confidence: str


@dataclass(frozen=True)
class ThreadCreationResult:
    record: ThreadRecord
    thread_note: Note
    added_notes: List[Note]
    summary: str
    journal_entry: str


@dataclass(frozen=True)
class ThreadUpdateResult:
    record: ThreadRecord
    thread_note: Note
    added_notes: List[Note]
    summary: str
    journal_entry: str


def _sorted_recent_notes(notes: Sequence[Note], limit: int) -> List[Note]:
    def created_at(note: Note) -> datetime:
        raw = note.get("created_at", "")
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return datetime.min

    filtered = [note for note in notes if note.get("type", "note") != "thread"]
    return sorted(filtered, key=created_at, reverse=True)[:limit]


def discover_thread_candidates(
    notes: Sequence[Note] | None = None,
    max_threads: int = 5,
) -> List[ThreadCandidate]:
    """Return AI-proposed thread clusters for review."""

    all_notes = list(notes) if notes is not None else list_all_notes()
    recent_notes = _sorted_recent_notes(all_notes, DISCOVERY_NOTE_LIMIT)
    existing = [record.to_dict() for record in list_threads()]

    suggestions = generate_thread_discovery(recent_notes, existing, max_threads=max_threads)

    notes_by_id = {str(note.get("id")): note for note in all_notes}
    candidates: List[ThreadCandidate] = []
    for item in suggestions:
        note_ids = [nid for nid in item.get("note_ids", []) if nid in notes_by_id]
        if len(note_ids) < 2:
            continue
        slug = str(item.get("slug", "")).strip()
        if not slug:
            continue
        candidate = ThreadCandidate(
            slug=slug,
            title=item.get("title", slug).strip(),
            intent=item.get("intent", "").strip(),
            summary=item.get("summary", "").strip(),
            journal_entry=item.get("journal_entry", "").strip(),
            note_ids=note_ids,
        )
        candidates.append(candidate)

    return candidates


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _select_review_candidates(slug: str, notes: Sequence[Note], limit: int) -> List[Note]:
    members: List[Note] = []
    candidates: List[Note] = []
    for note in notes:
        if note.get("type") == "thread":
            continue
        threads = normalize_threads(note.get("threads"))
        if slug in threads:
            members.append(note)
        else:
            candidates.append(note)

    member_vectors = [np.array(note.get("embedding", []), dtype=float) for note in members if note.get("embedding")]
    if member_vectors:
        centroid = np.mean(member_vectors, axis=0)
        scored: List[Tuple[Note, float]] = []
        for note in candidates:
            embedding = note.get("embedding")
            if not embedding:
                continue
            vector = np.array(embedding, dtype=float)
            similarity = _cosine_similarity(centroid, vector)
            scored.append((note, similarity))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        ranked = [note for note, _ in scored[:limit]]
    else:
        ranked = []

    if len(ranked) < limit:
        remaining_slots = limit - len(ranked)
        existing_ids = {note.get("id") for note in ranked}
        recency_pool = sorted(
            [note for note in candidates if note.get("id") not in existing_ids],
            key=lambda n: n.get("created_at", ""),
            reverse=True,
        )
        ranked.extend(recency_pool[:remaining_slots])

    return ranked[:limit]


def review_thread_suggestions(
    record: ThreadRecord,
    notes: Sequence[Note] | None = None,
    max_candidates: int = REVIEW_CANDIDATE_LIMIT,
    max_suggestions: int = 6,
) -> List[ThreadReviewSuggestion]:
    """Return AI-ranked candidate notes that may fit an existing thread."""

    all_notes = list(notes) if notes is not None else list_all_notes()
    thread_note = get_thread_note(record.slug, all_notes)
    thread_text = thread_note.get("text", "") if thread_note else ""
    current_ids = [str(note.get("id")) for note in all_notes if record.slug in normalize_threads(note.get("threads"))]

    candidate_notes = _select_review_candidates(record.slug, all_notes, max_candidates)
    suggestions = generate_thread_review(
        {
            "slug": record.slug,
            "title": record.title,
            "intent": record.intent,
        },
        thread_text,
        current_ids,
        candidate_notes,
        max_suggestions=max_suggestions,
    )

    formatted: List[ThreadReviewSuggestion] = []
    for item in suggestions:
        formatted.append(
            ThreadReviewSuggestion(
                note_id=item["note_id"],
                reason=item.get("reason", "").strip(),
                confidence=item.get("confidence", "medium"),
            )
        )
    return formatted


def get_thread_note(slug: str, notes: Sequence[Note] | None = None) -> Note | None:
    """Return the dedicated thread note for a slug, if any."""

    for note in notes or list_all_notes():
        if note.get("type") != "thread":
            continue
        if slug in normalize_threads(note.get("threads")):
            return note
    return None


def _build_thread_note_body(title: str, summary: str, journal_entries: Sequence[str]) -> str:
    lines: List[str] = [f"# Thread: {title}", "", "## Overview", summary.strip() or "(pending summary)", "", "## Journal"]
    entries = [entry for entry in journal_entries if entry.strip()]
    if not entries:
        lines.append("- (journal entries will appear here)")
    else:
        lines.extend(entries)
    return "\n".join(lines)


def _parse_thread_note(note_text: str) -> Tuple[str, List[str]]:
    if not note_text:
        return "", []
    summary = ""
    journal_lines: List[str] = []
    sections = note_text.split("## Journal", 1)
    if len(sections) == 2:
        before, journal = sections
        summary_section = before.split("## Overview", 1)
        if len(summary_section) == 2:
            summary = summary_section[1].strip()
        journal_lines = [line.rstrip() for line in journal.strip().splitlines() if line.strip()]
    else:
        parts = note_text.split("## Overview", 1)
        if len(parts) == 2:
            summary = parts[1].strip()
    cleaned_entries = [entry for entry in journal_lines if not entry.startswith("- (journal entries")]
    return summary, cleaned_entries


def _attach_thread_to_notes(slug: str, note_ids: Iterable[str]) -> List[Note]:
    target_ids = [str(nid) for nid in note_ids if str(nid)]
    attached: List[Note] = []
    now = datetime.now().isoformat()

    for note_id in target_ids:
        note = get_note(note_id)
        if not note:
            continue
        threads = normalize_threads(note.get("threads"))
        if slug not in threads:
            threads.append(slug)
            note["threads"] = threads
        note["updated_at"] = now
        note["related_ids"] = note.get("related_ids", [])
        updated = persist_note(note, note.get("related_ids"))
        if updated:
            attached.append(updated)
    if not attached:
        attached = [note for note in (get_note(nid) for nid in target_ids) if note]
    return attached


def _update_thread_note_storage(
    note_id: str,
    new_text: str,
    additional_related: Iterable[str],
    new_title: str | None = None,
) -> None:
    note = get_note(note_id)
    if not note:
        return

    combined = set(note.get("related_ids", [])) | {str(rid) for rid in additional_related if str(rid)}
    note["text"] = new_text
    if new_title is not None:
        note["title"] = new_title
    note["related_ids"] = sorted(combined)
    note["updated_at"] = datetime.now().isoformat()
    persist_note(note, note.get("related_ids"))


def create_thread_from_candidate(candidate: ThreadCandidate) -> ThreadCreationResult:
    """Persist a new thread, its note, and attach member notes."""

    record = create_thread(candidate.slug, candidate.title, candidate.intent, note_id=None)
    slug = record.slug
    summary = candidate.summary or candidate.intent
    entry_text = candidate.journal_entry or f"Connected notes {' '.join(candidate.note_ids)} into the thread."
    date_str = datetime.now().date().isoformat()
    journal_line = f"- {date_str}: {entry_text.strip()}"
    note_body = _build_thread_note_body(record.title, summary, [journal_line])

    metadata = {
        "type": "thread",
        "title": record.title,
        "category": "Threads",
        "tags": ["thread", f"thread:{slug}"],
        "threads": [slug],
    }
    note = build_note(note_body, metadata, embedding=[], related_ids=candidate.note_ids)
    persist_note(note, linked_note_ids=candidate.note_ids)

    updated_record = update_thread(slug, note_id=note["id"], intent=candidate.intent, title=record.title) or record

    attached = _attach_thread_to_notes(slug, candidate.note_ids)

    refreshed_note = get_thread_note(slug)
    return ThreadCreationResult(
        record=updated_record,
        thread_note=refreshed_note or note,
        added_notes=attached,
        summary=summary,
        journal_entry=journal_line,
    )


def update_thread_with_notes(
    record: ThreadRecord,
    additions: Sequence[ThreadReviewSuggestion],
    reason_lookup: Dict[str, str],
) -> ThreadUpdateResult:
    """Attach accepted notes to a thread and refresh its note."""

    if not additions:
        raise ValueError("No additions provided for thread update")

    slug = record.slug
    all_notes = list_all_notes()
    thread_note = get_thread_note(slug, all_notes)
    if not thread_note:
        raise ValueError(f"Thread note for '{slug}' was not found")

    additions_map = {suggestion.note_id: suggestion for suggestion in additions}
    added_notes = [note for note in all_notes if str(note.get("id")) in additions_map]
    attached = _attach_thread_to_notes(slug, [note.get("id") for note in added_notes])

    addition_payload = []
    for note in added_notes:
        note_id = str(note.get("id"))
        addition_payload.append(
            {
                "note_id": note_id,
                "title": note.get("title", "Untitled"),
                "text": note.get("text", ""),
                "reason": reason_lookup.get(note_id, additions_map[note_id].reason),
            }
        )

    update = generate_thread_note_update(
        record.title,
        record.intent,
        thread_note.get("text", ""),
        addition_payload,
    )

    summary, existing_entries = _parse_thread_note(thread_note.get("text", ""))
    summary = update["summary"]
    intent = update.get("intent", record.intent)
    date_str = datetime.now().date().isoformat()
    journal_line = f"- {date_str}: {update['journal_entry'].strip()}"
    entries = existing_entries + [journal_line]
    new_body = _build_thread_note_body(record.title, summary, entries)

    existing_related = thread_note.get("related_ids", [])
    _update_thread_note_storage(
        thread_note["id"],
        new_body,
        list(existing_related) + [note.get("id") for note in added_notes],
    )

    updated_record = update_thread(slug, intent=intent, note_id=thread_note.get("id")) or record
    refreshed_note = get_thread_note(slug)

    combined_added = attached or added_notes
    return ThreadUpdateResult(
        record=updated_record,
        thread_note=refreshed_note or thread_note,
        added_notes=combined_added,
        summary=summary,
        journal_entry=journal_line,
    )


def summarize_thread_note(note: Note) -> Tuple[str, List[str]]:
    """Return the current summary and journal entries for a thread note."""

    summary, entries = _parse_thread_note(note.get("text", ""))
    return summary, entries


def apply_thread_metadata_changes(
    record: ThreadRecord,
    new_title: str | None = None,
    new_intent: str | None = None,
) -> ThreadRecord:
    """Update stored thread metadata and keep the thread note in sync."""

    pending_title = (new_title or record.title).strip()
    pending_intent = (new_intent or record.intent).strip()

    changed_title = pending_title != record.title
    changed_intent = pending_intent != record.intent
    if not changed_title and not changed_intent:
        return record

    updated_record = update_thread(
        record.slug,
        title=pending_title if changed_title else None,
        intent=pending_intent if changed_intent else None,
        note_id=record.note_id,
    ) or record

    if record.note_id:
        note = get_thread_note(record.slug)
        if note:
            summary, entries = summarize_thread_note(note)
            body = _build_thread_note_body(pending_title, summary, entries)
            _update_thread_note_storage(
                record.note_id,
                body,
                note.get("related_ids", []),
                new_title=pending_title,
            )

    return updated_record


__all__ = [
    "ThreadCandidate",
    "ThreadReviewSuggestion",
    "ThreadCreationResult",
    "ThreadUpdateResult",
    "discover_thread_candidates",
    "review_thread_suggestions",
    "create_thread_from_candidate",
    "update_thread_with_notes",
    "get_thread_note",
    "summarize_thread_note",
    "apply_thread_metadata_changes",
]
