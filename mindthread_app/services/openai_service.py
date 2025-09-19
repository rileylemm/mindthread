"""OpenAI helpers with light error handling."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence, Callable

import openai
from openai import OpenAI, OpenAIError

from ..config import get_settings
from ..thread_store import normalize_slug


class AIServiceError(RuntimeError):
    """Raised when the OpenAI service cannot fulfill a request."""


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise AIServiceError("OPENAI_API_KEY is not configured.")
        _client = openai.OpenAI(api_key=settings.openai_api_key)
    return _client


def _trim_text(text: str, limit: int = 400) -> str:
    """Collapse whitespace and trim long text for prompt payloads."""

    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _parse_json_response(content: str, error_message: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise AIServiceError(error_message) from exc


def generate_embedding(text: str) -> List[float]:
    """Generate an embedding vector for the provided text."""

    settings = get_settings()
    client = _get_client()
    try:
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
    except OpenAIError as exc:
        raise AIServiceError("Failed to generate embedding from OpenAI") from exc

    if not response.data:
        raise AIServiceError("OpenAI embedding response did not include data")

    return response.data[0].embedding


def _build_metadata_prompt(text: str, categories: Sequence[str], tags: Sequence[str]) -> str:
    sections = [
        "You help organize personal notes.",
        "Return JSON with keys: title, category, tags.",
        "- title: Short descriptive title (max 5 words)",
        "- category: Single category name",
        "- tags: Array of 3-5 relevant tags",
    ]

    if categories:
        sections.append(
            "Existing categories (prefer one of these if it fits): "
            + ", ".join(sorted(categories))
        )
    else:
        sections.append("No existing categories yet. Introduce one that fits the note.")

    if tags:
        sections.append(
            "Existing tags (prefer reuse when appropriate): " + ", ".join(sorted(tags))
        )
    else:
        sections.append("No existing tags yet. Introduce relevant tags.")

    sections.append(f"Note: \"{text}\"")
    sections.append("Return only JSON.")
    return "\n".join(sections)


def generate_metadata(
    text: str,
    existing_categories: Sequence[str] | None = None,
    existing_tags: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Generate title, category, and tags using GPT."""

    settings = get_settings()
    client = _get_client()
    prompt = _build_metadata_prompt(
        text,
        existing_categories or [],
        existing_tags or [],
    )
    try:
        response = client.chat.completions.create(
            model=settings.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
    except OpenAIError as exc:
        raise AIServiceError("Failed to generate metadata from OpenAI") from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise AIServiceError("OpenAI metadata response did not contain choices")

    content = choices[0].message.content.strip()
    try:
        metadata = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AIServiceError("OpenAI returned malformed metadata JSON") from exc

    # Ensure mandatory fields are present
    metadata.setdefault("title", "Untitled")
    metadata.setdefault("category", "General")
    metadata.setdefault("tags", ["untagged"])

    return metadata


def generate_recap_summary(
    notes: Sequence[Dict[str, Any]],
    instructions: str | None = None,
) -> str:
    """Generate an analytical recap across multiple notes."""

    if not notes:
        raise AIServiceError("No notes supplied for recap generation")

    settings = get_settings()
    client = _get_client()

    note_blocks = []
    for note in notes:
        block = (
            f"ID: {note.get('id')}\n"
            f"Title: {note.get('title')}\n"
            f"Category: {note.get('category', '')}\n"
            f"Tags: {', '.join(note.get('tags', []))}\n"
            f"Text: {note.get('text', '')}"
        )
        note_blocks.append(block)

    default_instructions = (
        "Produce a concise recap that surfaces deeper insights from these notes. "
        "Highlight emerging themes, unexpected connections, and concrete next steps. "
        "Reference note IDs when calling out specifics. Structure the response using clear headings."
    )

    user_content = (
        "Notes to analyze:\n\n"
        + "\n\n".join(note_blocks)
        + "\n\n"
        + (instructions or default_instructions)
    )

    try:
        response = client.chat.completions.create(
            model=settings.gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an insightful synthesizer who draws connections across personal notes.",
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            max_tokens=600,
        )
    except OpenAIError as exc:
        raise AIServiceError("Failed to generate recap summary") from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise AIServiceError("OpenAI recap response did not contain choices")

    return choices[0].message.content.strip()


def generate_seed_suggestion(note: Dict[str, Any]) -> Dict[str, Any]:
    """Draft a Mindthread Seed outline from an existing note."""

    if not note:
        raise AIServiceError("Cannot promote an empty note into a seed")

    settings = get_settings()
    client = _get_client()

    title = note.get("title", "Untitled")
    text = note.get("text", "")
    body_preview = _trim_text(text, limit=800)
    category = note.get("category", "")
    tags = note.get("tags", [])

    tag_line = ", ".join(tags) if tags else "(no tags)"

    user_prompt = (
        "You are a poetic product strategist who turns personal notes into living Seeds.\n"
        "Take the provided note and propose a Seed in JSON with the keys:\n"
        "name (short title), spark (evocative paragraph), intention (desired bloom),\n"
        "care_cadence (object with kind: daily/weekly/custom, interval_days integer, label string),\n"
        "care_window (object with label string, optional), constraints (array of short phrases).\n"
        "Prefer lyrical but actionable language.\n"
        "Return only JSON.\n\n"
        f"Note Title: {title}\n"
        f"Category: {category}\n"
        f"Tags: {tag_line}\n"
        f"Body:\n{body_preview}\n"
    )

    try:
        response = client.chat.completions.create(
            model=settings.gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": "You translate notes into Seed blueprints that balance ritual and action.",
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=350,
        )
    except OpenAIError as exc:
        raise AIServiceError("Failed to generate seed suggestion") from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise AIServiceError("Seed suggestion response did not contain choices")

    content = choices[0].message.content.strip()
    suggestion = _parse_json_response(content, "OpenAI returned malformed seed suggestion JSON")

    cadence = suggestion.get("care_cadence", {})
    if isinstance(cadence, dict):
        cadence.setdefault("kind", "weekly")
        cadence.setdefault("interval_days", 7 if cadence.get("kind") == "weekly" else 1)
        cadence.setdefault("label", cadence.get("kind", "Weekly"))
        suggestion["care_cadence"] = cadence
    else:
        suggestion["care_cadence"] = {"kind": "weekly", "interval_days": 7, "label": "Weekly"}

    if "constraints" in suggestion and not isinstance(suggestion["constraints"], list):
        suggestion["constraints"] = [str(suggestion["constraints"])]

    return suggestion


def generate_eli5_explanation(
    subject: str,
    level_label: str,
    level_instructions: str,
    context_summary: str,
    previous_answer: str | None = None,
) -> str:
    """Generate an explanation tailored to a specific audience level."""

    settings = get_settings()
    client = _get_client()

    user_content = (
        f"Explain the following request: {subject}\n\n"
        f"Audience: {level_label}\n"
        f"Guidelines: {level_instructions}\n"
        "Important: Do NOT reuse sentences or phrases from prior notes. Invent new metaphors and wording."
        " Highlight differences from earlier explanations when appropriate."
        " Reference note IDs only when it adds value.\n\n"
    )

    if previous_answer:
        user_content += (
            "Here is the original explanation you already provided. Build upon it when relevant, "
            "clarify further, and answer the follow-up specifically.\n"
            f"Original explanation:\n{previous_answer}\n\n"
        )

    user_content += f"Context overview:\n{context_summary}"

    try:
        response = client.chat.completions.create(
            model=settings.gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain ideas using the requester's own notes when possible. "
                        "Be accurate, friendly, and keep the tone aligned with the specified audience."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.8,
            max_tokens=600,
        )
    except OpenAIError as exc:
        raise AIServiceError("Failed to generate eli5 explanation") from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise AIServiceError("OpenAI eli5 response did not contain choices")

    return choices[0].message.content.strip()


def generate_thread_discovery(
    notes: Sequence[Dict[str, Any]],
    existing_threads: Sequence[Dict[str, Any]],
    max_threads: int = 5,
) -> List[Dict[str, Any]]:
    """Ask GPT to suggest new thread groupings across notes."""

    if not notes:
        return []

    settings = get_settings()
    client = _get_client()

    note_blocks: List[str] = []
    for note in notes:
        snippet = _trim_text(str(note.get("text", "")), 420)
        tags = ", ".join(note.get("tags", [])) or "none"
        existing = ", ".join(note.get("threads", [])) or "none"
        block = (
            f"ID: {note.get('id')}\n"
            f"Title: {note.get('title', 'Untitled')}\n"
            f"Category: {note.get('category', 'General')}\n"
            f"Tags: {tags}\n"
            f"Threads: {existing}\n"
            f"Summary: {snippet}\n"
        )
        note_blocks.append(block)

    existing_section = "None"
    if existing_threads:
        formatted = []
        for record in existing_threads:
            formatted.append(
                f"- {record.get('slug')}: {record.get('title', '')} — {record.get('intent', '')}"
            )
        existing_section = "\n".join(formatted)

    user_content = (
        "You connect related notes into thematic threads."
        " Review the notes and propose up to {max_threads} new thread ideas.".format(max_threads=max_threads)
        + "\nExisting threads (do not duplicate these):\n"
        + existing_section
        + "\n\nNotes to analyze:\n"
        + "\n---\n".join(note_blocks)
        + "\n\nRespond with JSON of the form {\"threads\": [...]} where each item has:"
          " slug (lowercase hyphenated), title, intent, note_ids (array of strings),"
          " summary (2-3 sentences capturing the theme), and journal_entry (1 sentence referencing the note IDs)."
        + " Only include suggestions that feel cohesive and have at least two note_ids."
        + " If no threads are apparent, return {\"threads\": []}."
    )

    try:
        response = client.chat.completions.create(
            model=settings.gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert curator of interconnected ideas."
                        " You name threads succinctly and explain the linkage clearly."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            max_tokens=900,
        )
    except OpenAIError as exc:
        raise AIServiceError("Failed to generate thread discovery suggestions") from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise AIServiceError("OpenAI thread discovery response did not contain choices")

    payload = _parse_json_response(
        choices[0].message.content.strip(),
        "OpenAI returned malformed thread discovery JSON",
    )

    threads_payload = payload.get("threads", [])
    if not isinstance(threads_payload, list):
        raise AIServiceError("OpenAI thread discovery response missing 'threads' list")

    cleaned: List[Dict[str, Any]] = []
    seen_slugs: set[str] = set()
    for item in threads_payload:
        if not isinstance(item, dict):
            continue
        slug_raw = str(item.get("slug", "")).strip()
        slug = normalize_slug(slug_raw) if slug_raw else ""
        if not slug or slug in seen_slugs:
            continue
        title = str(item.get("title", "")).strip() or slug.replace("-", " ").title()
        intent = str(item.get("intent", "")).strip()
        summary = str(item.get("summary", "")).strip()
        journal_entry = str(item.get("journal_entry", "")).strip()
        note_ids_raw = item.get("note_ids", [])
        if not isinstance(note_ids_raw, list):
            continue
        note_ids: List[str] = []
        for ident in note_ids_raw:
            ident_str = str(ident).strip()
            if ident_str:
                note_ids.append(ident_str)
        if len(note_ids) < 2:
            continue
        cleaned.append(
            {
                "slug": slug,
                "title": title,
                "intent": intent,
                "summary": summary,
                "journal_entry": journal_entry,
                "note_ids": note_ids,
            }
        )
        seen_slugs.add(slug)

    return cleaned


def generate_thread_review(
    thread_context: Dict[str, Any],
    thread_note_text: str,
    current_note_ids: Sequence[str],
    candidate_notes: Sequence[Dict[str, Any]],
    max_suggestions: int = 6,
) -> List[Dict[str, Any]]:
    """Ask GPT which candidate notes should join an existing thread."""

    if not candidate_notes:
        return []

    settings = get_settings()
    client = _get_client()

    candidate_blocks: List[str] = []
    for note in candidate_notes:
        snippet = _trim_text(str(note.get("text", "")), 360)
        tags = ", ".join(note.get("tags", [])) or "none"
        block = (
            f"ID: {note.get('id')}\n"
            f"Title: {note.get('title', 'Untitled')}\n"
            f"Category: {note.get('category', 'General')}\n"
            f"Tags: {tags}\n"
            f"Summary: {snippet}\n"
        )
        candidate_blocks.append(block)

    current_ids = ", ".join(sorted(str(x) for x in current_note_ids)) or "none"
    thread_snapshot = _trim_text(thread_note_text or "", 1500)

    user_content = (
        f"Thread slug: {thread_context.get('slug')}\n"
        f"Title: {thread_context.get('title', '')}\n"
        f"Intent: {thread_context.get('intent', '')}\n"
        f"Current members: {current_ids}\n"
        f"Thread note excerpt:\n{thread_snapshot}\n\n"
        "Evaluate the candidate notes below and choose up to {max} additions that truly belong".format(max=max_suggestions)
        + ". Explain why each addition strengthens the thread."
        + "\n\nCandidates:\n"
        + "\n---\n".join(candidate_blocks)
        + "\n\nRespond with JSON {\"suggestions\": [...]} where each item has note_id, reason (2 sentences),"
          " and confidence (low/medium/high). Return an empty list if none fit."
    )

    try:
        response = client.chat.completions.create(
            model=settings.gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a meticulous curator ensuring notes fit the thread's thesis."
                        " Only propose additions that clearly align."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=700,
        )
    except OpenAIError as exc:
        raise AIServiceError("Failed to generate thread review suggestions") from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise AIServiceError("OpenAI thread review response did not contain choices")

    payload = _parse_json_response(
        choices[0].message.content.strip(),
        "OpenAI returned malformed thread review JSON",
    )

    suggestions_payload = payload.get("suggestions", [])
    if not isinstance(suggestions_payload, list):
        raise AIServiceError("OpenAI thread review response missing 'suggestions' list")

    valid_candidates = {str(note.get("id")) for note in candidate_notes}
    cleaned: List[Dict[str, Any]] = []
    for item in suggestions_payload:
        if not isinstance(item, dict):
            continue
        note_id = str(item.get("note_id", "")).strip()
        if not note_id or note_id not in valid_candidates:
            continue
        reason = str(item.get("reason", "")).strip()
        confidence = str(item.get("confidence", "")).strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium" if confidence else "medium"
        cleaned.append({"note_id": note_id, "reason": reason, "confidence": confidence})

    return cleaned[:max_suggestions]


def generate_thread_note_update(
    thread_title: str,
    intent: str,
    existing_note_text: str,
    additions: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    """Ask GPT to refresh the thread note summary and craft a new journal entry."""

    if not additions:
        return {"summary": existing_note_text, "journal_entry": ""}

    settings = get_settings()
    client = _get_client()

    excerpt = _trim_text(existing_note_text or "", 1600)
    addition_blocks: List[str] = []
    for addition in additions:
        snippet = _trim_text(str(addition.get("text", "")), 360)
        block = (
            f"Note ID: {addition.get('note_id')}\n"
            f"Title: {addition.get('title', 'Untitled')}\n"
            f"Summary: {snippet}\n"
            f"Reason: {addition.get('reason', '')}\n"
        )
        addition_blocks.append(block)

    user_content = (
        f"Thread Title: {thread_title}\n"
        f"Intent: {intent}\n"
        f"Existing thread note (may be empty):\n{excerpt or '<<no existing note>>'}\n\n"
        "The thread note has two sections: an 'Overview' that stays current, and an append-only 'Journal'."
        " Review the new additions and update both sections along with the thread's intent if it shifted."
        "\n\nNew additions:\n"
        + "\n---\n".join(addition_blocks)
        + "\n\nRespond with JSON containing:"
        " summary (updated Overview content, 1-2 paragraphs),"
        " intent (a single sentence capturing the refined aim of the thread),"
        " and journal_entry (a concise paragraph referencing the new notes by ID and how they extend the thread)."
        " Do not rewrite previous journal entries; just supply the new entry text without any date prefix."
    )

    try:
        response = client.chat.completions.create(
            model=settings.gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You maintain living documentation of thematic threads."
                        " Keep the tone analytical and connective."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            max_tokens=600,
        )
    except OpenAIError as exc:
        raise AIServiceError("Failed to refresh thread note") from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise AIServiceError("OpenAI thread note update response did not contain choices")

    payload = _parse_json_response(
        choices[0].message.content.strip(),
        "OpenAI returned malformed thread note update JSON",
    )

    summary = str(payload.get("summary", "")).strip()
    journal_entry = str(payload.get("journal_entry", "")).strip()
    updated_intent = str(payload.get("intent", intent)).strip() or intent
    if not summary:
        raise AIServiceError("GPT did not provide an updated thread summary")
    if not journal_entry:
        raise AIServiceError("GPT did not provide a journal entry for the thread update")

    return {"summary": summary, "journal_entry": journal_entry, "intent": updated_intent}


__all__ = [
    "AIServiceError",
    "generate_embedding",
    "generate_metadata",
    "generate_recap_summary",
    "generate_eli5_explanation",
    "generate_thread_discovery",
    "generate_thread_review",
    "generate_thread_note_update",
    "generate_chat_reply",
]


def generate_chat_reply(
    messages: Sequence[Dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
    stream: bool = False,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """Generate a conversational reply using chat completions."""

    settings = get_settings()
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=model or settings.gpt_model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=600,
            stream=stream,
        )
    except OpenAIError as exc:
        raise AIServiceError("Failed to generate chat reply") from exc

    if stream:
        chunks: List[str] = []
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                chunks.append(delta)
                if on_chunk:
                    on_chunk(delta)
        return "".join(chunks).strip()

    choices = getattr(response, "choices", None)
    if not choices:
        raise AIServiceError("OpenAI chat response did not contain choices")

    return choices[0].message.content.strip()
