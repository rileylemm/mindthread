"""OpenAI helpers with light error handling."""

from __future__ import annotations

import json
from typing import Any, Dict, List

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


_METADATA_PROMPT = """
Given this note, return JSON with:
- title: Short descriptive title (max 5 words)
- category: Single category name
- tags: Array of 3-5 relevant tags

Note: "{text}"

Return only JSON.
"""


def generate_metadata(text: str) -> Dict[str, Any]:
    """Generate title, category, and tags using GPT."""

    settings = get_settings()
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=settings.gpt_model,
            messages=[{"role": "user", "content": _METADATA_PROMPT.format(text=text)}],
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


__all__ = ["AIServiceError", "generate_embedding", "generate_metadata"]
