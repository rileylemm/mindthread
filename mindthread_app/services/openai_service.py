"""OpenAI helpers with light error handling."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence

import openai
from openai import OpenAI, OpenAIError

from ..config import get_settings


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


def generate_eli5_explanation(
    subject: str,
    level_label: str,
    level_instructions: str,
    notes: Sequence[Dict[str, Any]],
    previous_answer: str | None = None,
) -> str:
    """Generate an explanation tailored to a specific audience level."""

    settings = get_settings()
    client = _get_client()

    note_blocks: List[str] = []
    for note in notes:
        block = (
            f"ID: {note.get('id')}\n"
            f"Title: {note.get('title')}\n"
            f"Category: {note.get('category', '')}\n"
            f"Tags: {', '.join(note.get('tags', []))}\n"
            f"Text: {note.get('text', '')}"
        )
        note_blocks.append(block)

    context = "\n\n".join(note_blocks) if note_blocks else "(No related notes found.)"

    user_content = (
        f"Explain the following request: {subject}\n\n"
        f"Audience: {level_label}\n"
        f"Guidelines: {level_instructions}\n"
        "Important: Do NOT reuse sentences or phrases from the context. Invent new metaphors and wording."
        " Reference note IDs when relevant.\n\n"
    )

    if previous_answer:
        user_content += (
            "Here is the original explanation you already provided. Build upon it when relevant, "
            "clarify further, and answer the follow-up specifically.\n"
            f"Original explanation:\n{previous_answer}\n\n"
        )

    user_content += f"Context notes:\n{context}"

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
            temperature=0.7,
            max_tokens=600,
        )
    except OpenAIError as exc:
        raise AIServiceError("Failed to generate eli5 explanation") from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise AIServiceError("OpenAI eli5 response did not contain choices")

    return choices[0].message.content.strip()


__all__ = [
    "AIServiceError",
    "generate_embedding",
    "generate_metadata",
    "generate_recap_summary",
    "generate_eli5_explanation",
]
