"""Agent-facing project brief output."""

from __future__ import annotations

from textwrap import dedent

from .config import get_settings


def get_agent_brief() -> str:
    settings = get_settings()
    notes_location = settings.notes_file

    return dedent(
        f"""
        mindthread – terminal-first augmented note taking
        -------------------------------------------------
        Purpose: Capture short-form ideas, auto-tag them with GPT, embed for semantic search, and explore relationships via cosine similarity.

        Runtime layout:
        - mindthread_app/cli.py – command dispatch and user prompts
        - mindthread_app/notes.py – core note CRUD, search, embedding utilities, link suggestions
        - mindthread_app/services/openai_service.py – GPT metadata + embedding helpers
        - mindthread_app/catalog.py – tracks known categories/tags for reuse
        - mindthread_app/storage.py – JSON persistence utilities (notes live at {notes_location})
        - mindthread_app/config.py – dotenv-backed settings loader

        Data model: JSON list of dicts with fields (id, text, title, category, tags, embedding, created_at). Embeddings are OpenAI vectors stored inline.

        CLI commands:
        - add ["note"] – interactive or direct capture with GPT enrichment
        - list – render all notes
        - search "query" – substring match across text/title/tags
        - show <id> – pager view with inline help/edit actions
        - related <id> – cosine-similarity suggestions using stored embeddings
        - add (interactive) now offers AI-suggested links + category/tag selection from catalog
        - stats – sparkline capture history and high-level counts
        - tags [limit] – tag heatmap with ASCII bars
        - catalog – review categories/tags, rename or remove tags inline
        - clip – capture the current clipboard as a note (pbpaste required on macOS)
        - list/search/stats/tags use a pager by default (pass --no-pager to disable)
        - remove <id> – delete note with confirmation prompt
        - agent-brief – print this orientation block
        - help – command recap

        Config: .env mirrors env.example. Key vars OPENAI_API_KEY, EMBEDDING_MODEL, GPT_MODEL, STORAGE_TYPE (currently json), DATA_DIR (defaults to project root).

        Extensibility hooks: swap storage backend by extending storage module, reuse auto_enrich_note() in notes.py for new ingestion surfaces, and call find_related_notes() for graph-style features.
        """
    ).strip()


__all__ = ["get_agent_brief"]
