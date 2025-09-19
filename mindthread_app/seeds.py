"""Seed management utilities backed by SQLite."""

from __future__ import annotations

import json
from array import array
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .db import ensure_schema, get_connection
from .seed_templates import get_template, suggest_template_from_format
from .services.openai_service import AIServiceError, generate_embedding


SeedState = str


@dataclass(slots=True)
class Seed:
    id: str
    name: str
    bloom: str
    spark: str
    intention: str
    format_profile: dict
    care_cadence: dict
    care_window: Optional[dict]
    first_action: str
    constraints: List[str]
    planting_story: dict
    template_slug: Optional[str]
    state: SeedState
    momentum_score: Optional[float]
    embedding: List[float]
    created_at: str
    updated_at: Optional[str]
    next_check_at: Optional[str]
    origin_note_id: Optional[str]
    last_note_at: Optional[str]
    note_count: int
    cadence_score: Optional[float]
    last_quick_note_at: Optional[str]
    last_ritual_at: Optional[str]


@dataclass(slots=True)
class SeedNote:
    id: str
    seed_id: str
    created_at: str
    note_type: str
    text: str
    next_action: Optional[str]
    metadata: dict
    embedding: List[float]


@dataclass(slots=True)
class SeedTend:
    id: str
    seed_id: str
    performed_at: str
    prompt_type: str
    reflection: str
    actions: List[str]
    ai_assist: Optional[str]


STATE_ORDER = ["Dormant", "Sprouting", "Budding", "Bloomed", "Compost"]


def _ensure_schema_ready() -> None:
    ensure_schema()


def _encode_embedding(values: Sequence[float] | None) -> bytes | None:
    if not values:
        return None
    arr = array("f", [float(v) for v in values])
    return arr.tobytes()


def _decode_embedding(blob: bytes | None) -> List[float]:
    if not blob:
        return []
    arr = array("f")
    arr.frombytes(blob)
    return [float(v) for v in arr]


def _pack_json(value: object | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


def _unpack_json(value: str | None) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _unpack_json_list(value: str | None) -> List[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _now_iso() -> str:
    return datetime.now().isoformat()


def _cadence_interval_days(cadence: dict) -> int:
    interval = cadence.get("interval_days")
    if isinstance(interval, (int, float)) and interval > 0:
        return int(interval)
    kind = cadence.get("kind")
    if kind == "daily":
        return 1
    if kind == "weekly":
        return 7
    return 3


def _compute_next_check(cadence: dict, last: datetime | None = None) -> str:
    anchor = last or datetime.now()
    delta = timedelta(days=_cadence_interval_days(cadence))
    return (anchor + delta).isoformat()


def _row_to_seed(row) -> Seed:
    cadence = _unpack_json(row["care_cadence"])
    window = _unpack_json(row["care_window"])
    constraints = _unpack_json_list(row["constraints"])
    format_profile = _unpack_json(row["format_profile"])
    planting_story = _unpack_json(row["planting_story"])
    return Seed(
        id=str(row["id"]),
        name=row["name"],
        bloom=row["bloom"] or "",
        spark=row["spark"] or "",
        intention=row["intention"] or "",
        format_profile=format_profile,
        care_cadence=cadence,
        care_window=window or None,
        first_action=row["first_action"] or "",
        constraints=constraints,
        planting_story=planting_story,
        template_slug=row["template_slug"],
        state=row["state"] or "Dormant",
        momentum_score=row["momentum_score"],
        embedding=_decode_embedding(row["embedding"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        next_check_at=row["next_check_at"],
        origin_note_id=str(row["origin_note_id"]) if row["origin_note_id"] is not None else None,
        last_note_at=row["last_note_at"],
        note_count=int(row["note_count"] or 0),
        cadence_score=row["cadence_score"],
        last_quick_note_at=row["last_quick_note_at"],
        last_ritual_at=row["last_ritual_at"],
    )


def _row_to_seed_note(row) -> SeedNote:
    return SeedNote(
        id=str(row["id"]),
        seed_id=str(row["seed_id"]),
        created_at=row["created_at"],
        note_type=row["note_type"],
        text=row["text"],
        next_action=row["next_action"],
        metadata=_unpack_json(row["metadata"]),
        embedding=_decode_embedding(row["embedding"]),
    )


def _row_to_tend(row) -> SeedTend:
    return SeedTend(
        id=str(row["id"]),
        seed_id=str(row["seed_id"]),
        performed_at=row["performed_at"],
        prompt_type=row["prompt_type"] or "germinate",
        reflection=row["reflection"] or "",
        actions=_unpack_json_list(row["actions"]),
        ai_assist=row["ai_assist"],
    )


def list_seeds(*, order_by: str = "state") -> List[Seed]:
    _ensure_schema_ready()
    with get_connection(readonly=True) as connection:
        if order_by == "created_at":
            rows = connection.execute(
                "SELECT * FROM seeds ORDER BY datetime(created_at) ASC, id ASC"
            ).fetchall()
        else:
            state_case = "CASE state " + " ".join(
                f"WHEN '{state}' THEN {index}" for index, state in enumerate(STATE_ORDER)
            ) + " ELSE 99 END"
            rows = connection.execute(
                f"SELECT * FROM seeds ORDER BY {state_case}, datetime(created_at) ASC"
            ).fetchall()
    return [_row_to_seed(row) for row in rows]


def get_seed(seed_id: str) -> Seed | None:
    _ensure_schema_ready()
    with get_connection(readonly=True) as connection:
        row = connection.execute("SELECT * FROM seeds WHERE id = ?", (seed_id,)).fetchone()
    if not row:
        return None
    return _row_to_seed(row)


def list_due_seeds(reference: datetime | None = None) -> List[Seed]:
    _ensure_schema_ready()
    ref = (reference or datetime.now()).isoformat()
    with get_connection(readonly=True) as connection:
        rows = connection.execute(
            """
            SELECT * FROM seeds
            WHERE next_check_at IS NOT NULL
              AND datetime(next_check_at) <= datetime(?)
            ORDER BY datetime(next_check_at) ASC
            """,
            (ref,),
        ).fetchall()
    return [_row_to_seed(row) for row in rows]


def list_seed_notes(seed_id: str, *, limit: int | None = None) -> List[SeedNote]:
    _ensure_schema_ready()
    query = "SELECT * FROM seed_notes WHERE seed_id = ? ORDER BY datetime(created_at) DESC"
    params: list[object] = [seed_id]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    with get_connection(readonly=True) as connection:
        rows = connection.execute(query, params).fetchall()
    return [_row_to_seed_note(row) for row in rows]


def get_latest_note(seed_id: str) -> SeedNote | None:
    notes = list_seed_notes(seed_id, limit=1)
    return notes[0] if notes else None


def similar_seeds(seed_id: str, *, top_k: int = 3, min_similarity: float = 0.2) -> List[tuple[Seed, float]]:
    """Return Seeds similar to the target based on embeddings."""

    target = get_seed(seed_id)
    if target is None or not target.embedding:
        return []

    target_vec = np.array(target.embedding, dtype=float)
    if np.linalg.norm(target_vec) == 0:
        return []

    candidates = [seed for seed in list_seeds(order_by="state") if seed.id != seed_id and seed.embedding]
    if not candidates:
        return []

    vectors = np.array([seed.embedding for seed in candidates], dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    valid_mask = norms > 0
    if not valid_mask.any():
        return []

    vectors = vectors[valid_mask]
    candidates = [seed for seed, valid in zip(candidates, valid_mask) if valid]
    similarities = vectors @ target_vec / (np.linalg.norm(target_vec) * np.linalg.norm(vectors, axis=1))
    ranking = np.argsort(similarities)[::-1]

    results: list[tuple[Seed, float]] = []
    for idx in ranking:
        score = float(similarities[idx])
        if score < min_similarity:
            continue
        results.append((candidates[idx], round(score, 3)))
        if len(results) >= top_k:
            break
    return results


def list_seed_tends(seed_id: str, *, limit: int | None = None) -> List[SeedTend]:
    _ensure_schema_ready()
    query = "SELECT * FROM seed_tends WHERE seed_id = ? ORDER BY datetime(performed_at) DESC"
    params = [seed_id]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    with get_connection(readonly=True) as connection:
        rows = connection.execute(query, params).fetchall()
    return [_row_to_tend(row) for row in rows]


def get_latest_tend(seed_id: str) -> SeedTend | None:
    tends = list_seed_tends(seed_id, limit=1)
    return tends[0] if tends else None


def _prepare_constraints(constraints: Sequence[str] | str | None) -> List[str]:
    if constraints is None:
        return []
    if isinstance(constraints, str):
        raw_items = [part.strip() for part in constraints.replace("\n", ",").split(",")]
    else:
        raw_items = [str(item).strip() for item in constraints]
    return [item for item in raw_items if item]


def create_seed(
    name: str,
    spark: str,
    intention: str,
    care_cadence: dict,
    *,
    care_window: dict | None = None,
    constraints: Sequence[str] | str | None = None,
    origin_note_id: str | None = None,
    bloom: str | None = None,
    format_profile: dict | None = None,
    first_action: str | None = None,
    planting_story: dict | None = None,
    template_slug: str | None = None,
) -> Seed:
    _ensure_schema_ready()

    try:
        embedding = generate_embedding("\n\n".join(filter(None, [name, spark, intention])))
    except AIServiceError:
        embedding = []

    now = datetime.now()
    next_check = _compute_next_check(care_cadence, now)
    constraint_list = _prepare_constraints(constraints)
    format_profile = format_profile or {}
    template_slug = template_slug or suggest_template_from_format(format_profile)
    bloom = (bloom or intention or name).strip()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO seeds (
                name, bloom, spark, intention, format_profile, care_cadence, care_window,
                first_action, constraints, planting_story, template_slug, state,
                momentum_score, embedding, created_at, updated_at, next_check_at, origin_note_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name.strip(),
                bloom,
                spark.strip(),
                intention.strip(),
                _pack_json(format_profile),
                _pack_json(care_cadence),
                _pack_json(care_window),
                (first_action or "").strip(),
                _pack_json(constraint_list),
                _pack_json(planting_story or {}),
                template_slug,
                "Dormant",
                None,
                _encode_embedding(embedding),
                now.isoformat(),
                now.isoformat(),
                next_check,
                _safe_int(origin_note_id),
            ),
        )
        seed_id = cursor.lastrowid
        connection.commit()

    return get_seed(str(seed_id))


def create_seed_note(
    seed_id: str,
    text: str,
    *,
    note_type: str = "quick",
    next_action: str | None = None,
    metadata: dict | None = None,
    embed: bool = False,
) -> SeedNote:
    """Insert a note for a seed and update cadence/state metadata."""

    _ensure_schema_ready()
    metadata = metadata or {}
    created_at = datetime.now()
    embedding_vector: List[float] = []
    if embed:
        try:
            embedding_vector = generate_embedding(text)
        except AIServiceError:
            embedding_vector = []

    with get_connection() as connection:
        seed_row = connection.execute(
            "SELECT care_cadence, state, next_check_at, note_count FROM seeds WHERE id = ?",
            (int(seed_id),),
        ).fetchone()
        if not seed_row:
            raise ValueError(f"Seed {seed_id} not found")

        care_cadence = _unpack_json(seed_row["care_cadence"])
        next_check = None
        if care_cadence:
            next_check = _compute_next_check(care_cadence, created_at)

        cursor = connection.execute(
            """
            INSERT INTO seed_notes (
                seed_id, created_at, note_type, text, next_action, metadata, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(seed_id),
                created_at.isoformat(),
                note_type,
                text.strip(),
                next_action,
                _pack_json(metadata),
                _encode_embedding(embedding_vector),
            ),
        )

        note_count = int(seed_row["note_count"] or 0) + 1
        momentum = _calculate_momentum(note_count, created_at, care_cadence)
        state = seed_row["state"]
        if state == "Dormant":
            state = "Sprouting"

        connection.execute(
            """
            UPDATE seeds
               SET state = ?,
                   updated_at = ?,
                   next_check_at = ?,
                   last_note_at = ?,
                   note_count = ?,
                   momentum_score = ?,
                   cadence_score = ?,
                   last_quick_note_at = CASE WHEN ? = 'quick' THEN ? ELSE last_quick_note_at END,
                   last_ritual_at = CASE WHEN ? = 'ritual' THEN ? ELSE last_ritual_at END
             WHERE id = ?
            """,
            (
                state,
                created_at.isoformat(),
                next_check,
                created_at.isoformat(),
                note_count,
                momentum,
                momentum,
                note_type,
                created_at.isoformat(),
                note_type,
                created_at.isoformat(),
                int(seed_id),
            ),
        )

        connection.commit()
        note_id = cursor.lastrowid

    inserted = list_seed_notes(str(seed_id), limit=1)
    return inserted[0] if inserted else SeedNote(
        id=str(note_id),
        seed_id=str(seed_id),
        created_at=created_at.isoformat(),
        note_type=note_type,
        text=text.strip(),
        next_action=next_action,
        metadata=metadata,
        embedding=embedding_vector,
    )


def log_seed_tend(
    seed_id: str,
    prompt_type: str,
    reflection: str,
    actions: Sequence[str] | None = None,
    *,
    ai_assist: str | None = None,
) -> SeedTend:
    _ensure_schema_ready()
    actions_payload = [item.strip() for item in (actions or []) if item and item.strip()]
    metadata = {
        "prompt_type": prompt_type,
        "actions": actions_payload,
        "ai_assist": ai_assist,
    }

    note = create_seed_note(
        seed_id,
        reflection,
        note_type="ritual",
        metadata=metadata,
    )

    performed_at = note.created_at

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO seed_tends (
                seed_id, performed_at, prompt_type, reflection, actions, ai_assist
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(seed_id),
                performed_at,
                prompt_type,
                reflection.strip(),
                _pack_json(actions_payload) if actions_payload else None,
                ai_assist,
            ),
        )
        connection.commit()
        tend_id = cursor.lastrowid

    return SeedTend(
        id=str(tend_id),
        seed_id=str(seed_id),
        performed_at=performed_at,
        prompt_type=prompt_type,
        reflection=reflection.strip(),
        actions=actions_payload,
        ai_assist=ai_assist,
    )


def update_seed_next_check(seed_id: str, *, reference: datetime | None = None) -> None:
    _ensure_schema_ready()
    reference = reference or datetime.now()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT care_cadence, note_count FROM seeds WHERE id = ?",
            (int(seed_id),),
        ).fetchone()
        if not row:
            raise ValueError(f"Seed {seed_id} not found")
        care_cadence = _unpack_json(row["care_cadence"])
        next_check = None
        if care_cadence:
            next_check = _compute_next_check(care_cadence, reference)
        momentum = _calculate_momentum(int(row["note_count"] or 0), reference, care_cadence)
        connection.execute(
            "UPDATE seeds SET updated_at = ?, next_check_at = ?, momentum_score = ?, cadence_score = ? WHERE id = ?",
            (reference.isoformat(), next_check, momentum, momentum, int(seed_id)),
        )
        connection.commit()


def snooze_seed(seed_id: str, days: int) -> None:
    _ensure_schema_ready()
    if days <= 0:
        return
    with get_connection() as connection:
        row = connection.execute(
            "SELECT next_check_at FROM seeds WHERE id = ?",
            (int(seed_id),),
        ).fetchone()
        if not row:
            raise ValueError(f"Seed {seed_id} not found")
        current = row["next_check_at"]
        if current:
            try:
                base = datetime.fromisoformat(current)
            except ValueError:
                base = datetime.now()
        else:
            base = datetime.now()
        new_time = (base + timedelta(days=days)).isoformat()
        connection.execute(
            "UPDATE seeds SET next_check_at = ?, updated_at = ? WHERE id = ?",
            (new_time, datetime.now().isoformat(), int(seed_id)),
        )
        connection.commit()


def update_seed_state(seed_id: str, state: SeedState) -> None:
    _ensure_schema_ready()
    now = datetime.now().isoformat()
    with get_connection() as connection:
        connection.execute(
            "UPDATE seeds SET state = ?, updated_at = ? WHERE id = ?",
            (state, now, int(seed_id)),
        )
        connection.commit()


def schedule_next_check(seed_id: str, cadence: dict, from_time: datetime | None = None) -> None:
    _ensure_schema_ready()
    next_check = _compute_next_check(cadence, from_time)
    with get_connection() as connection:
        row = connection.execute(
            "SELECT note_count FROM seeds WHERE id = ?",
            (int(seed_id),),
        ).fetchone()
        count = int(row["note_count"] or 0) if row else 0
        momentum = _calculate_momentum(count, datetime.now(), cadence)
        connection.execute(
            "UPDATE seeds SET next_check_at = ?, updated_at = ?, momentum_score = ? WHERE id = ?",
            (next_check, datetime.now().isoformat(), momentum, int(seed_id)),
        )
        connection.commit()


def remove_seed(seed_id: str) -> None:
    _ensure_schema_ready()
    with get_connection() as connection:
        connection.execute("DELETE FROM seeds WHERE id = ?", (int(seed_id),))
        connection.commit()


def all_states() -> List[str]:
    return list(STATE_ORDER)


__all__ = [
    "Seed",
    "SeedNote",
    "SeedTend",
    "create_seed",
    "create_seed_note",
    "list_seeds",
    "get_seed",
    "list_due_seeds",
    "list_seed_notes",
    "list_seed_tends",
    "get_latest_note",
    "log_seed_tend",
    "set_seed_template",
    "record_prompt_feedback",
    "update_seed_state",
    "update_seed_next_check",
    "snooze_seed",
    "remove_seed",
    "all_states",
]


def set_seed_template(seed_id: str, template_slug: str, format_profile: dict | None = None) -> None:
    _ensure_schema_ready()
    now = datetime.now().isoformat()
    with get_connection() as connection:
        if format_profile is not None:
            connection.execute(
                "UPDATE seeds SET template_slug = ?, format_profile = ?, updated_at = ? WHERE id = ?",
                (template_slug, _pack_json(format_profile), now, int(seed_id)),
            )
        else:
            connection.execute(
                "UPDATE seeds SET template_slug = ?, updated_at = ? WHERE id = ?",
                (template_slug, now, int(seed_id)),
            )
        connection.commit()


def record_prompt_feedback(seed_id: str, prompt_key: str, delta: int) -> None:
    _ensure_schema_ready()
    now = datetime.now().isoformat()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT score FROM seed_prompt_preferences WHERE seed_id = ? AND prompt_key = ?",
            (int(seed_id), prompt_key),
        ).fetchone()
        if row:
            new_score = int(row["score"]) + delta
            connection.execute(
                "UPDATE seed_prompt_preferences SET score = ?, updated_at = ? WHERE seed_id = ? AND prompt_key = ?",
                (new_score, now, int(seed_id), prompt_key),
            )
        else:
            connection.execute(
                "INSERT INTO seed_prompt_preferences (seed_id, prompt_key, score, updated_at) VALUES (?, ?, ?, ?)",
                (int(seed_id), prompt_key, delta, now),
            )
        connection.commit()
def _safe_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _calculate_momentum(note_count: int, reference: datetime, care_cadence: dict | None) -> float:
    if note_count <= 0:
        return 0.0

    interval_days = _cadence_interval_days(care_cadence or {})
    if interval_days <= 0:
        interval_days = 7

    days_since = max(0.0, (datetime.now() - reference).total_seconds() / 86400.0)
    freshness = max(0.0, 1.0 - (days_since / interval_days))
    streak = min(1.0, note_count / 12.0)
    return round(freshness * 0.7 + streak * 0.3, 3)


def link_seeds(
    seed_a: str,
    seed_b: str,
    *,
    link_type: str = "related",
    metadata: dict | None = None,
) -> None:
    _ensure_schema_ready()
    if seed_a == seed_b:
        raise ValueError("Cannot link a Seed to itself")

    a = min(int(seed_a), int(seed_b))
    b = max(int(seed_a), int(seed_b))

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO seed_links (a_seed_id, b_seed_id, link_type, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(a_seed_id, b_seed_id) DO UPDATE SET
                link_type = excluded.link_type,
                metadata = excluded.metadata,
                created_at = excluded.created_at
            """,
            (
                a,
                b,
                link_type,
                _pack_json(metadata or {}),
                datetime.now().isoformat(),
            ),
        )
        connection.commit()


def unlink_seeds(seed_a: str, seed_b: str) -> bool:
    _ensure_schema_ready()
    if seed_a == seed_b:
        return False

    a = min(int(seed_a), int(seed_b))
    b = max(int(seed_a), int(seed_b))

    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM seed_links WHERE a_seed_id = ? AND b_seed_id = ?",
            (a, b),
        )
        connection.commit()
        return cursor.rowcount > 0


def list_seed_links(seed_id: str) -> List[tuple[Seed, str, dict]]:
    _ensure_schema_ready()
    seed_int = int(seed_id)
    with get_connection(readonly=True) as connection:
        rows = connection.execute(
            """
            SELECT a_seed_id, b_seed_id, link_type, metadata
              FROM seed_links
             WHERE a_seed_id = ? OR b_seed_id = ?
            ORDER BY datetime(created_at) DESC
            """,
            (seed_int, seed_int),
        ).fetchall()

    results: list[tuple[Seed, str, dict]] = []
    for row in rows:
        other_id = row["b_seed_id"] if row["a_seed_id"] == seed_int else row["a_seed_id"]
        other_seed = get_seed(str(other_id))
        if not other_seed:
            continue
        metadata = _unpack_json(row["metadata"])
        results.append((other_seed, row["link_type"], metadata))
    return results


def serialize_seed(seed: Seed, include_notes: bool = True, *, include_links: bool = True) -> dict:
    payload = {
        "id": seed.id,
        "name": seed.name,
        "bloom": seed.bloom,
        "spark": seed.spark,
        "intention": seed.intention,
        "format_profile": seed.format_profile,
        "care_cadence": seed.care_cadence,
        "care_window": seed.care_window,
        "first_action": seed.first_action,
        "constraints": seed.constraints,
        "planting_story": seed.planting_story,
        "template": seed.template_slug,
        "state": seed.state,
        "momentum_score": seed.momentum_score,
        "cadence_score": seed.cadence_score,
        "next_check_at": seed.next_check_at,
        "created_at": seed.created_at,
        "updated_at": seed.updated_at,
        "note_count": seed.note_count,
        "last_note_at": seed.last_note_at,
        "last_quick_note_at": seed.last_quick_note_at,
        "last_ritual_at": seed.last_ritual_at,
    }

    if include_notes:
        notes = [
            {
                "id": note.id,
                "created_at": note.created_at,
                "type": note.note_type,
                "text": note.text,
                "next_action": note.next_action,
                "metadata": note.metadata,
            }
            for note in list_seed_notes(seed.id)
        ]
        payload["notes"] = notes

    if include_links:
        payload["links"] = [
            {
                "seed": serialize_seed(link_seed, include_notes=False, include_links=False),
                "link_type": link_type,
                "metadata": metadata,
            }
            for link_seed, link_type, metadata in list_seed_links(seed.id)
        ]

    return payload
