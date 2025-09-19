"""Conversational planting ceremony for Mindthread Seeds."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import get_settings
from .seed_templates import suggest_template_from_format


@dataclass(frozen=True)
class CeremonyResult:
    name: str
    bloom: str
    spark: str
    format_profile: dict
    care_cadence: dict
    care_window: Optional[dict]
    first_action: str
    constraints: list[str]
    planting_story: list[dict[str, str]]
    template_slug: Optional[str]


_QUESTIONS = [
    {
        "key": "name",
        "prompt": "Name this Seed:",
        "required": True,
    },
    {
        "key": "bloom",
        "prompt": "What do you want this Seed to bloom into?",
        "required": True,
    },
    {
        "key": "spark",
        "prompt": "What sparked it (moment, fragment, scene)?",
        "required": True,
    },
    {
        "key": "format",
        "prompt": "How should it grow (e.g., playlist + alignment notes + arc narrative)?",
        "required": False,
    },
    {
        "key": "cadence",
        "prompt": "How often should we tend it (e.g., weekly Sunday evening)?",
        "required": False,
    },
    {
        "key": "constraints",
        "prompt": "Any ingredients, boundaries, or collaborators to respect?",
        "required": False,
    },
    {
        "key": "first_action",
        "prompt": "What’s the first tiny action that would make it feel alive?",
        "required": False,
    },
]

_DRAFT_FILE = "seed_drafts.json"


def run_seed_ceremony(
    *,
    name: str | None = None,
    initial_answers: dict[str, Any] | None = None,
    origin_note_id: str | None = None,
) -> Optional[CeremonyResult]:
    """Run the interactive planting ceremony. Returns None if cancelled."""

    initial_answers = dict(initial_answers or {})
    if name:
        initial_answers.setdefault("name", name)

    draft = _maybe_resume_draft(origin_note_id)
    if draft:
        answers = draft.get("answers", {})
        answers.update(initial_answers)
        step = draft.get("step", 0)
        planting_story = draft.get("planting_story", [])
    else:
        answers = dict(initial_answers)
        step = 0
        planting_story: list[dict[str, str]] = []

    draft_id = draft.get("id") if draft else str(uuid.uuid4())
    draft_meta = {
        "id": draft_id,
        "created_at": (draft.get("created_at") if draft else datetime.now().isoformat()),
        "origin_note_id": origin_note_id,
    }

    try:
        while step < len(_QUESTIONS):
            question = _QUESTIONS[step]
            key = question["key"]
            if answers.get(key):
                step += 1
                continue

            _print_preview(answers)
            prompt = question["prompt"] + " "
            try:
                response = input(prompt).strip()
            except KeyboardInterrupt:
                print("\n🌱 Ceremony paused. Resume it later with 'mindthread seed plant'.")
                _save_draft(draft_meta, answers, step, planting_story)
                return None

            if response.lower() in {":quit", ":abort"}:
                print("🌱 Ceremony saved for later.")
                _save_draft(draft_meta, answers, step, planting_story)
                return None

            if response.lower() == ":back":
                step = max(0, step - 1)
                continue

            if not response and answers.get(key):
                step += 1
                continue

            if not response and question.get("required"):
                print("This answer is required. Use :quit to save for later.")
                continue

            answers[key] = response
            planting_story.append({"question": question["prompt"], "answer": response})
            step += 1
            _save_draft(draft_meta, answers, step, planting_story)

        while True:
            _print_preview(answers)
            confirm = input("Create this Seed? (Y/n/edit): ").strip().lower()
            if confirm in {"", "y", "yes"}:
                break
            if confirm in {"n", "no"}:
                print("🌱 Ceremony saved. Resume later with 'mindthread seed plant'.")
                _save_draft(draft_meta, answers, step, planting_story)
                return None
            if confirm == "edit":
                key = input("Which field to edit (name/bloom/spark/format/cadence/constraints/first_action)? ").strip()
                indices = {q["key"]: idx for idx, q in enumerate(_QUESTIONS)}
                if key in indices:
                    step = indices[key]
                    answers.pop(key, None)
                    continue
                print("Unknown field.")
                continue

        _clear_draft()

        format_profile = _parse_format_profile(answers.get("format", ""))
        care_cadence, care_window = _parse_cadence_answer(answers.get("cadence", ""))
        constraints = _parse_constraints_answer(answers.get("constraints", ""))
        story = [item for item in planting_story if item.get("answer")]

        template_slug = suggest_template_from_format(format_profile)

        return CeremonyResult(
            name=answers.get("name", "Untitled").strip() or "Untitled",
            bloom=answers.get("bloom", "").strip(),
            spark=answers.get("spark", "").strip(),
            format_profile=format_profile,
            care_cadence=care_cadence,
            care_window=care_window,
            first_action=answers.get("first_action", "").strip(),
            constraints=constraints,
            planting_story=story,
            template_slug=template_slug,
        )

    except KeyboardInterrupt:
        print("\n🌱 Ceremony paused. Resume it later with 'mindthread seed plant'.")
        _save_draft(draft_meta, answers, step, planting_story)
        return None


def _print_preview(answers: dict[str, Any]) -> None:
    if not answers:
        return
    print("\nCurrent Seed preview:")
    name = answers.get("name", "(unnamed)")
    bloom = answers.get("bloom", "")
    spark = answers.get("spark", "")
    cadence = answers.get("cadence", "")
    first_action = answers.get("first_action", "")
    print(f"  • Name: {name}")
    if bloom:
        print(f"  • Bloom: {bloom}")
    if spark:
        print(f"  • Spark: {spark[:80]}" + ("…" if len(spark) > 80 else ""))
    if cadence:
        print(f"  • Cadence: {cadence}")
    if first_action:
        print(f"  • First action: {first_action}")
    print("")


def _parse_format_profile(answer: str) -> dict:
    if not answer:
        return {"raw": ""}
    normalized = answer.replace("+", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    return {
        "raw": answer.strip(),
        "components": parts,
    }


def _parse_cadence_answer(answer: str) -> tuple[dict, Optional[dict]]:
    if not answer:
        return ({"kind": "custom", "interval_days": 7, "label": "Weekly"}, None)

    text = answer.strip()
    lower = text.lower()
    kind = "custom"
    interval = None

    if "daily" in lower or "every day" in lower:
        kind = "daily"
        interval = 1
    elif "weekly" in lower or "every week" in lower:
        kind = "weekly"
        interval = 7
    elif "month" in lower:
        kind = "monthly"
        interval = 30

    for token in lower.replace("/", " ").split():
        if token.isdigit():
            value = int(token)
            if "day" in lower:
                interval = value
                kind = "daily"
            elif "week" in lower:
                interval = value * 7
                kind = "weekly"
            elif "month" in lower:
                interval = value * 30
                kind = "monthly"
            break

    return (
        {
            "kind": kind,
            "interval_days": interval or 7,
            "label": text,
        },
        None,
    )


def _parse_constraints_answer(answer: str) -> list[str]:
    if not answer:
        return []
    fragments = [part.strip() for part in answer.replace("+", ",").split(",")]
    return [fragment for fragment in fragments if fragment]


def _draft_path() -> Path:
    settings = get_settings()
    return settings.data_dir / _DRAFT_FILE


def _load_draft_file() -> dict:
    path = _draft_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def _maybe_resume_draft(origin_note_id: str | None) -> Optional[dict]:
    data = _load_draft_file()
    draft = data.get("draft")
    if not draft:
        return None

    resume_prompt = draft.get("name") or "Seed draft"
    choice = input(f"Resume saved ceremony for '{resume_prompt}'? (Y/n): ").strip().lower()
    if choice in {"", "y", "yes"}:
        if origin_note_id and draft.get("origin_note_id") not in {None, origin_note_id}:
            print("Stored draft is linked to another note; starting fresh instead.")
            _clear_draft()
            return None
        return draft

    _clear_draft()
    return None


def _save_draft(
    meta: dict,
    answers: dict[str, Any],
    step: int,
    planting_story: list[dict[str, str]],
) -> None:
    payload = {
        "draft": {
            **meta,
            "name": answers.get("name"),
            "answers": answers,
            "step": step,
            "planting_story": planting_story,
        }
    }
    path = _draft_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _clear_draft() -> None:
    path = _draft_path()
    if path.exists():
        path.unlink()


__all__ = ["run_seed_ceremony", "CeremonyResult"]
