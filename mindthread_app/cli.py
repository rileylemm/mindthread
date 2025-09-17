"""Command-line interface entry point for mindthread."""

from __future__ import annotations

import sys
from typing import Sequence

from .briefing import get_agent_brief
from .config import get_settings
from .notes import (
    AIServiceError,
    auto_enrich_note,
    build_note,
    find_related_notes,
    get_note,
    list_all_notes,
    persist_note,
    remove_note,
    search_notes,
)


def _require_api_key() -> bool:
    if get_settings().openai_api_key:
        return True
    print("❌ Error: OPENAI_API_KEY not found in environment")
    print("Create a .env file with: OPENAI_API_KEY=sk-...")
    return False


def _print_note_summary(note: dict) -> None:
    print(f"[{note['id']}] {note['title']}")
    print(f"Category: {note['category']}")
    print(f"Tags: {', '.join(note['tags'])}")
    print(f"Created: {note['created_at'][:10]}")
    text = note['text']
    snippet = text[:100] + ("..." if len(text) > 100 else "")
    print(f"Text: {snippet}")
    print("-" * 30)


def _interactive_add() -> int:
    text = input("Add your note here: ").strip()
    if not text:
        print("❌ Note text cannot be empty")
        return 1

    print("\nProcessing note...")
    try:
        metadata, embedding = auto_enrich_note(text)
    except AIServiceError as exc:
        print(f"❌ {exc}")
        return 1

    while True:
        print("\n📝 Generated metadata:")
        print(f"Title: {metadata['title']}")
        print(f"Category: {metadata['category']}")
        print(f"Tags: {', '.join(metadata['tags'])}")

        choice = input("\nConfirm category/tags? (y/n/edit): ").strip().lower()
        if choice == "y":
            break
        if choice == "n":
            print("❌ Note not saved")
            return 1
        if choice == "edit":
            new_category = input("New category (or press enter to keep current): ").strip()
            if new_category:
                metadata["category"] = new_category

            new_tags = input("New tags (comma-separated, or press enter to keep current): ").strip()
            if new_tags:
                metadata["tags"] = [tag.strip() for tag in new_tags.split(",") if tag.strip()]
            continue
        print("Please enter 'y', 'n', or 'edit'.")

    note = build_note(text, metadata, embedding)
    persist_note(note)
    print(f"\n✅ Note saved! ID: {note['id']}")
    return 0


def _direct_add(args: Sequence[str]) -> int:
    if not args:
        return _interactive_add()

    text = args[0]
    print("Processing note...")
    try:
        metadata, embedding = auto_enrich_note(text)
    except AIServiceError as exc:
        print(f"❌ {exc}")
        return 1

    print(f"Title: {metadata['title']}")
    print(f"Category: {metadata['category']}")
    print(f"Tags: {', '.join(metadata['tags'])}")

    note = build_note(text, metadata, embedding)
    persist_note(note)
    print(f"✅ Note saved! ID: {note['id']}")
    return 0


def _handle_list() -> int:
    notes = list_all_notes()
    if not notes:
        print("No notes found.")
        return 0

    print(f"\n📝 Your Notes ({len(notes)} total):")
    print("=" * 50)
    for note in notes:
        _print_note_summary(note)
    return 0


def _handle_search(args: Sequence[str]) -> int:
    if not args:
        print("Usage: mindthread search \"your query\"")
        return 1

    query = args[0]
    matches = search_notes(query)
    if not matches:
        print("No matching notes found.")
        return 0

    print(f"\n🔍 Found {len(matches)} matching notes:")
    print("=" * 50)
    for note in matches:
        _print_note_summary(note)
    return 0


def _handle_show(args: Sequence[str]) -> int:
    if not args:
        print("Usage: mindthread show <note_id>")
        return 1

    note_id = args[0]
    note = get_note(note_id)
    if note is None:
        print(f"Note {note_id} not found.")
        return 1

    print(f"\n📝 {note['title']}")
    print("=" * 50)
    print(f"Category: {note['category']}")
    print(f"Tags: {', '.join(note['tags'])}")
    print(f"Created: {note['created_at']}")
    print(f"\nText:\n{note['text']}")
    return 0


def _handle_related(args: Sequence[str]) -> int:
    if not args:
        print("Usage: mindthread related <note_id>")
        return 1

    note_id = args[0]
    try:
        target, related = find_related_notes(note_id)
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1

    print(f"\n🧠 Related thoughts for: {target['title']}")
    print("=" * 60)
    print(
        f"Target note: {target['text'][:100]}"
        f"{'...' if len(target['text']) > 100 else ''}"
    )
    print("\nMost similar notes:")
    print("-" * 40)

    if not related:
        print("(No other notes with embeddings found.)")
        return 0

    for index, (note, similarity) in enumerate(related, 1):
        print(f"\n{index}. [{note['id']}] {note['title']} (similarity: {similarity:.3f})")
        print(f"   Category: {note['category']}")
        print(f"   Tags: {', '.join(note['tags'])}")
        snippet = note['text'][:150] + ("..." if len(note['text']) > 150 else "")
        print(f"   Text: {snippet}")
        print("-" * 40)
    return 0


def _handle_remove(args: Sequence[str]) -> int:
    if not args:
        print("Usage: mindthread remove <note_id>")
        return 1

    note_id = args[0]
    note = get_note(note_id)
    if note is None:
        print(f"Note {note_id} not found.")
        return 1

    print("\n🗑️  Removing note:")
    _print_note_summary(note)
    confirm = input("\nAre you sure you want to delete this note? (y/N): ").strip().lower()
    if confirm != "y":
        print("❌ Note deletion cancelled.")
        return 1

    if remove_note(note_id):
        print(f"✅ Note {note_id} deleted successfully.")
        return 0

    print(f"❌ Failed to delete note {note_id}.")
    return 1


def _print_help() -> None:
    print("🧠 mindthread-cli - Build your second brain")
    print("\nCommands:")
    print("  add                 - Add a new note (interactive)")
    print("  add \"note text\"     - Add a new note (command line)")
    print("  list                - List all notes")
    print("  search \"query\"      - Search notes")
    print("  show <id>           - Show specific note")
    print("  related <id>        - Find related thoughts using AI embeddings")
    print("  remove <id>         - Remove a note by ID")
    print("  agent-brief         - Print architecture overview for agents")
    print("  help                - Show this help message")


def _handle_agent_brief() -> int:
    print(get_agent_brief())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]

    if not args:
        _print_help()
        return 1

    command, *rest = args

    if command == "add":
        if not _require_api_key():
            return 1
        return _direct_add(rest)

    if command == "list":
        return _handle_list()

    if command == "search":
        return _handle_search(rest)

    if command == "show":
        return _handle_show(rest)

    if command == "related":
        return _handle_related(rest)

    if command == "remove":
        return _handle_remove(rest)

    if command == "help":
        _print_help()
        return 0

    if command == "agent-brief":
        return _handle_agent_brief()

    print(f"Unknown command: {command}")
    print("Use 'mindthread help' to see available commands")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
