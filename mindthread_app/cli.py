"""Command-line interface entry point for mindthread."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Iterable, Sequence, Tuple

from pydoc import pager

from .analytics import render_sparkline, format_tag_heatmap
from .briefing import get_agent_brief
from .catalog import Catalog, load_catalog, save_catalog
from .config import get_settings
from .editor import launch_editor
from .notes import (
    AIServiceError,
    auto_enrich_note,
    find_related_notes,
    get_note,
    list_all_notes,
    build_note,
    persist_note,
    remove_note,
    search_notes,
    note_counts_by_day,
    notes_since,
    rename_category,
    rename_tag,
    update_note_text,
    tag_frequency,
    suggest_related_by_embedding,
)


ELI5_LEVELS = [
    ("5yo", "Like a 5-year-old", "Explain using simple words, friendly analogies, and no jargon."),
    ("middle_school", "Middle schooler", "Use approachable language and basic examples, introduce mild technical terms."),
    ("high_school", "High schooler", "Provide a balanced explanation with concrete examples and some technical detail."),
    ("college", "College student", "Offer a structured explanation with key concepts, assumptions, and implications."),
    ("expert", "Domain expert", "Deliver a precise, technically rich explanation with nuanced insights and edge cases."),
]
from .services.openai_service import (
    generate_embedding,
    generate_metadata,
    generate_recap_summary,
    generate_eli5_explanation,
)


def _require_api_key() -> bool:
    if get_settings().openai_api_key:
        return True
    print("❌ Error: OPENAI_API_KEY not found in environment")
    print("Create a .env file with: OPENAI_API_KEY=sk-...")
    return False


def _format_note_summary(note: dict) -> list[str]:
    note_type = note.get("type", "note")
    type_tile = f"[type: {note_type}]" if note_type != "note" else ""
    category_tile = f"[cat: {note['category']}]" if note.get("category") else ""
    tags = note.get("tags", [])
    tags_tile = f"[tags: {', '.join(tags)}]" if tags else ""
    links = len(note.get("related_ids", []))
    links_tile = f"[links: {links}]" if links else ""
    created_tile = f"[created: {note['created_at'][:10]}]" if note.get("created_at") else ""
    header_tiles = " ".join(tile for tile in [type_tile, category_tile, tags_tile, links_tile, created_tile] if tile)
    header = f"[{note['id']}] {note['title']} {header_tiles}".rstrip()
    text = note.get("text", "")
    snippet = text[:100] + ("..." if len(text) > 100 else "")
    body = f"    {snippet}" if snippet else ""
    return [header] + ([body] if body else [])


def _print_note_summary(note: dict) -> None:
    for line in _format_note_summary(note):
        print(line)


def _interactive_add() -> int:
    text = input("Add your note here: ").strip()
    if not text:
        print("❌ Note text cannot be empty")
        return 1

    print("\nProcessing note...")
    catalog = load_catalog()
    try:
        metadata, embedding = auto_enrich_note(text, catalog.categories, catalog.tags)
    except AIServiceError as exc:
        print(f"❌ {exc}")
        return 1

    metadata = _apply_catalog_defaults(metadata, catalog)
    _display_metadata(metadata, catalog)

    while True:
        choice = input("\nConfirm metadata? (y/n/edit): ").strip().lower()
        if choice == "y":
            break
        if choice == "n":
            print("❌ Note not saved")
            return 1
        if choice == "edit":
            metadata = _edit_metadata(metadata, catalog)
            _display_metadata(metadata, catalog)
        else:
            print("Please enter 'y', 'n', or 'edit'.")

    linked_ids = _prompt_related_links(metadata, embedding)

    note = build_note(text, metadata, embedding, related_ids=linked_ids)
    persist_note(note, linked_ids)
    catalog.add_category(note["category"])
    catalog.add_tags(note["tags"])
    save_catalog(catalog)
    print(f"\n✅ Note saved! ID: {note['id']}")
    return 0


def _direct_add(args: Sequence[str]) -> int:
    if not args:
        return _interactive_add()

    text = args[0]
    print("Processing note...")
    catalog = load_catalog()
    try:
        metadata, embedding = auto_enrich_note(text, catalog.categories, catalog.tags)
    except AIServiceError as exc:
        print(f"❌ {exc}")
        return 1

    metadata = _apply_catalog_defaults(metadata, catalog)
    print(f"Title: {metadata['title']}")
    print(f"Category: {metadata['category']}")
    print(f"Tags: {', '.join(metadata['tags'])}")

    note = build_note(text, metadata, embedding)
    persist_note(note)
    catalog.add_category(note["category"])
    catalog.add_tags(note["tags"])
    save_catalog(catalog)
    print(f"✅ Note saved! ID: {note['id']}")
    return 0


def _extract_flag(args: Sequence[str], flag: str) -> Tuple[bool, list[str]]:
    args_list = list(args)
    found = False
    while flag in args_list:
        args_list.remove(flag)
        found = True
    return found, args_list


def _maybe_page(text: str, use_pager: bool, allow_disable: bool = True) -> None:
    if allow_disable and use_pager:
        pager(text)
        return
    if not allow_disable:
        pager(text)
        return
    print(text)


def _handle_list(args: Sequence[str]) -> int:
    disable_pager, remaining = _extract_flag(args, "--no-pager")
    use_pager = not disable_pager
    notes = list_all_notes()
    if not notes:
        print("No notes found.")
        return 0

    lines = [f"📝 Your Notes ({len(notes)} total):", "=" * 60]
    for note in notes:
        lines.append("")
        lines.extend(_format_note_summary(note))
    _maybe_page("\n".join(lines), use_pager, allow_disable=False if use_pager else True)
    return 0


def _handle_search(args: Sequence[str]) -> int:
    disable_pager, remaining = _extract_flag(args, "--no-pager")
    use_pager = not disable_pager
    args = remaining
    if not args:
        print("Usage: mindthread search \"your query\"")
        return 1

    query = args[0]
    matches = search_notes(query)
    if not matches:
        print("No matching notes found.")
        return 0

    lines = [f"🔍 Found {len(matches)} matching notes:", "=" * 60]
    for note in matches:
        lines.append("")
        lines.extend(_format_note_summary(note))
    _maybe_page("\n".join(lines), use_pager, allow_disable=False if use_pager else True)
    return 0


def _handle_show(args: Sequence[str]) -> int:
    disable_pager, remaining = _extract_flag(args, "--no-pager")
    if not remaining:
        remaining = args

    if not remaining:
        print("Usage: mindthread show <note_id>")
        return 1

    note_id = remaining[0]
    note = get_note(note_id)
    if note is None:
        print(f"Note {note_id} not found.")
        return 1

    while True:
        _maybe_page(_render_note_detail(note), use_pager=not disable_pager, allow_disable=False if not disable_pager else True)

        action = input("\n[h]elp  [e]dit  [q]uit > ").strip().lower()
        if action in {"q", "quit", ""}:
            return 0
        if action in {"h", "help"}:
            print("Commands: h=help, e=edit note text, q=quit show view")
            continue
        if action in {"e", "edit"}:
            if _handle_note_edit(note):
                # Reload note after edit
                refreshed = get_note(note_id)
                if refreshed:
                    note = refreshed
            continue
        print("Unknown command. Use 'h' for help.")


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
    print("  chat                - Start a conversational session with note-aware context")
    print("  eli5               - Ask for an explanation at a chosen level")
    print("  recap [--days N]    - Generate a recap across recent notes")
    print("  stats               - Show note stats and sparkline history")
    print("  tags [limit]        - Display tag frequency heatmap")
    print("  clip                - Save current clipboard as a note")
    print("  agent-brief         - Print architecture overview for agents")
    print("  catalog             - Review and tidy categories/tags")
    print("  ui                  - Launch the prompt_toolkit interface")
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
        return _handle_list(rest)

    if command == "search":
        return _handle_search(rest)

    if command == "show":
        return _handle_show(rest)

    if command == "related":
        return _handle_related(rest)

    if command == "remove":
        return _handle_remove(rest)

    if command == "chat":
        return _handle_chat(rest)

    if command == "eli5":
        return _handle_eli5(rest)

    if command == "recap":
        return _handle_recap(rest)

    if command == "help":
        _print_help()
        return 0

    if command == "agent-brief":
        return _handle_agent_brief()

    if command == "catalog":
        return _handle_catalog()

    if command == "clip":
        return _handle_clip(rest)

    if command == "stats":
        return _handle_stats(rest)

    if command == "tags":
        return _handle_tags(rest)

    if command == "ui":
        try:
            from .promptui import run_ui
        except ImportError as exc:  # pragma: no cover - optional dependency
            print("❌ prompt_toolkit is not installed.")
            print("   Install it with `pip install prompt_toolkit` to enable the UI.")
            return 1
        try:
            run_ui()
        except Exception as exc:  # pragma: no cover - defensive guard
            print("❌ Failed to launch the UI:", exc)
            return 1
        return 0

    print(f"Unknown command: {command}")
    print("Use 'mindthread help' to see available commands")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
def _apply_catalog_defaults(metadata: dict, catalog: Catalog) -> dict:
    """Align metadata with existing categories/tags when possible."""

    updated = metadata.copy()
    category = updated.get("category", "")
    match = catalog.closest_category(category)
    if match:
        updated["category"] = match

    tags = [tag.strip() for tag in updated.get("tags", []) if tag.strip()]
    updated["tags"] = sorted({tag for tag in tags})
    return updated


def _display_metadata(metadata: dict, catalog: Catalog) -> None:
    category = metadata.get("category", "")
    closest = catalog.closest_category(category) if category else None
    tags = metadata.get("tags", [])
    existing_tags = sorted(set(tags) & catalog.tags)
    new_tags = sorted(set(tags) - catalog.tags)

    print("\n📝 Generated metadata:")
    print(f"Title: {metadata.get('title', 'Untitled')}")
    if closest and closest != category:
        print(f"Category: {category} (closest existing: {closest})")
    else:
        print(f"Category: {category}")
    if existing_tags:
        print(f"Tags (existing): {', '.join(existing_tags)}")
    if new_tags:
        print(f"Tags (new): {', '.join(new_tags)}")
    if not tags:
        print("Tags: (none)")


def _prompt_category(current: str, catalog: Catalog) -> str:
    print("\nCategory options:")
    options = sorted(catalog.categories)
    if options:
        for idx, option in enumerate(options, 1):
            print(f"  {idx}. {option}")
    else:
        print("  (no categories yet)")
    prompt = (
        "Enter category name, number from list, or press enter to keep current"
        f" [{current}]: "
    )
    response = input(prompt).strip()
    if not response:
        return current
    if response.isdigit():
        idx = int(response) - 1
        if 0 <= idx < len(options):
            return options[idx]
        print("Invalid selection; keeping current category.")
        return current
    return response


def _prompt_tags(current: Iterable[str], catalog: Catalog) -> list[str]:
    current_set = {tag.strip() for tag in current if tag.strip()}
    print("\nCurrent tags:", ", ".join(sorted(current_set)) or "(none)")
    if catalog.tags:
        print("Existing tags you can reuse:")
        print(", ".join(sorted(catalog.tags)))
    response = input(
        "Enter comma-separated tags to replace current (press enter to keep): "
    ).strip()
    if not response:
        return sorted(current_set)
    new_tags = {tag.strip() for tag in response.split(",") if tag.strip()}
    return sorted(new_tags)


def _edit_metadata(metadata: dict, catalog: Catalog) -> dict:
    updated = metadata.copy()
    while True:
        choice = input("Edit (category/tags/back): ").strip().lower()
        if choice in {"back", "b", ""}:
            break
        if choice.startswith("cat"):
            updated["category"] = _prompt_category(updated.get("category", ""), catalog)
        elif choice.startswith("tag"):
            updated["tags"] = _prompt_tags(updated.get("tags", []), catalog)
        else:
            print("Please choose 'category', 'tags', or 'back'.")
    return _apply_catalog_defaults(updated, catalog)


def _prompt_related_links(metadata: dict, embedding: Sequence[float]) -> list[str]:
    suggestions = suggest_related_by_embedding(embedding)
    if not suggestions:
        return []

    print("\n🔗 Suggested related notes:")
    linked_ids: list[str] = []
    for note, similarity in suggestions:
        snippet = note["text"][:80] + ("..." if len(note["text"]) > 80 else "")
        prompt = (
            f"Link to [{note['id']}] {note['title']} (sim {similarity:.2f})?\n"
            f"   {snippet}\n( y / n / stop ): "
        )
        answer = input(prompt).strip().lower()
        if answer == "stop":
            break
        if answer == "y":
            linked_ids.append(note["id"])
        elif answer and answer != "n":
            print("Please respond with 'y', 'n', or 'stop'.")
    return linked_ids


def _render_note_detail(note: dict) -> str:
    lines = [
        f"📝 {note['title']}",
        "=" * 60,
        f"ID: {note['id']}",
        f"Type: {note.get('type', 'note')}",
        f"Category: {note.get('category', '')}",
        f"Tags: {', '.join(note.get('tags', []))}",
        f"Created: {note.get('created_at', '')}",
    ]
    if note.get("updated_at"):
        lines.append(f"Updated: {note['updated_at']}")

    related = note.get("related_ids", [])
    if related:
        lines.append(f"Links: {', '.join(related)}")

    lines.append("")
    lines.append(note.get("text", ""))
    return "\n".join(lines)


def _handle_note_edit(note: dict) -> bool:
    current_text = note.get("text", "")
    edited = launch_editor(current_text)
    if edited is None:
        print("❌ No editor available or edit cancelled.")
        return False

    if edited == current_text:
        print("No changes detected.")
        return False

    confirm = input("Save changes? (y/N): ").strip().lower()
    if confirm != "y":
        print("❌ Changes discarded.")
        return False

    try:
        update_note_text(note["id"], edited, regenerate_embedding=True)
    except AIServiceError as exc:
        print(f"❌ Failed to update note: {exc}")
        return False

    print("✅ Note updated.")
    return True


def _handle_stats(args: Sequence[str]) -> int:
    disable_pager, remaining = _extract_flag(args, "--no-pager")
    use_pager = not disable_pager
    limit = None
    if remaining and remaining[0].isdigit():
        limit = max(1, int(remaining[0]))

    notes = list_all_notes()
    total_notes = len(notes)
    if total_notes == 0:
        print("No notes captured yet.")
        return 0

    categories = sorted({note.get("category") for note in notes if note.get("category")})
    unique_tags = sorted({tag for note in notes for tag in note.get("tags", []) if tag})
    link_count = sum(len(note.get("related_ids", [])) for note in notes)

    history = note_counts_by_day(limit or 14)
    spark_counts = [count for _, count in history]
    spark = render_sparkline(spark_counts)
    labels = " ".join(date[5:] for date, _ in history)

    lines = [
        "📊 mindthread stats",
        "=" * 40,
        f"Total notes: {total_notes}",
        f"Categories: {len(categories)}",
        f"Tags: {len(unique_tags)}",
        f"Total links: {link_count}",
        "",
        f"History (last {len(history)} days):",
        f"  {labels}" if labels else "  (no dates)",
        f"  {spark}",
    ]

    freq = tag_frequency()[:5]
    if freq:
        lines.append("")
        lines.append("Top tags:")
        for row in format_tag_heatmap(freq, max_width=18):
            lines.append(f"  {row}")

    _maybe_page("\n".join(lines), use_pager, allow_disable=False if use_pager else True)
    return 0


def _handle_tags(args: Sequence[str]) -> int:
    disable_pager, remaining = _extract_flag(args, "--no-pager")
    use_pager = not disable_pager
    limit = None
    if remaining and remaining[0].isdigit():
        limit = max(1, int(remaining[0]))

    freq = tag_frequency()
    if not freq:
        print("No tags recorded yet.")
        return 0

    if limit:
        freq = freq[:limit]

    lines = ["🏷️ Tag heatmap", "=" * 40]
    for row in format_tag_heatmap(freq, max_width=24):
        lines.append(row)

    _maybe_page("\n".join(lines), use_pager, allow_disable=False if use_pager else True)
    return 0


def _print_catalog(catalog: Catalog) -> None:
    print("\n📚 Catalog overview")
    categories = sorted(catalog.categories)
    tags = sorted(catalog.tags)
    if categories:
        print("Categories:")
        for idx, category in enumerate(categories, 1):
            print(f"  {idx}. {category}")
    else:
        print("Categories: (none)")

    print("")
    if tags:
        print("Tags:")
        for idx, tag in enumerate(tags, 1):
            print(f"  {idx}. {tag}")
    else:
        print("Tags: (none)")


def _resolve_selection(selection: str, options: Sequence[str], *, allow_new: bool) -> str | None:
    if not selection:
        return None

    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(options):
            return options[idx]
        if not allow_new:
            return None

    lowered_map = {opt.lower(): opt for opt in options}
    lowered = selection.lower()
    if lowered in lowered_map:
        return lowered_map[lowered]

    return selection if allow_new else None


def _rename_category_flow(catalog: Catalog) -> None:
    categories = sorted(catalog.categories)
    if not categories:
        print("No categories to rename.")
        return

    old_input = input(
        "Select category to rename (number or name, blank to cancel): "
    ).strip()
    old = _resolve_selection(old_input, categories, allow_new=False)
    if not old:
        print("No matching category selected.")
        return

    new_input = input(
        "New category name (number for existing, or type new name): "
    ).strip()
    if not new_input:
        print("No changes made.")
        return
    new = _resolve_selection(new_input, categories, allow_new=True)
    if not new:
        print("Invalid selection.")
        return

    if rename_category(old, new):
        catalog.remove_category(old)
        catalog.add_category(new)
        save_catalog(catalog)
        print(f"Updated category '{old}' → '{new}'.")
    else:
        print("No notes needed updating.")


def _rename_tag_flow(catalog: Catalog) -> None:
    tags = sorted(catalog.tags)
    if not tags:
        print("No tags to rename.")
        return

    old_input = input(
        "Select tag to rename (number or name, blank to cancel): "
    ).strip()
    old = _resolve_selection(old_input, tags, allow_new=False)
    if not old:
        print("No matching tag selected.")
        return

    new_input = input(
        "New tag name (number for existing, type new name, '-' to remove): "
    ).strip()
    if not new_input:
        print("No changes made.")
        return

    if new_input == "-":
        new = ""
    else:
        new = _resolve_selection(new_input, tags, allow_new=True)
        if new is None:
            print("Invalid selection.")
            return

    if rename_tag(old, new or ""):
        catalog.remove_tag(old)
        if new:
            catalog.add_tags([new])
        save_catalog(catalog)
        if new:
            print(f"Updated tag '{old}' → '{new}'.")
        else:
            print(f"Removed tag '{old}'.")
    else:
        print("No notes needed updating.")


def _handle_catalog() -> int:
    catalog = load_catalog()
    while True:
        _print_catalog(catalog)
        action = input(
            "\nAction (rename-cat / rename-tag / refresh / exit): "
        ).strip().lower()

        if action in {"", "exit", "q", "quit"}:
            return 0
        if action in {"rename-cat", "cat", "c"}:
            _rename_category_flow(catalog)
            catalog = load_catalog()
            continue
        if action in {"rename-tag", "tag", "t"}:
            _rename_tag_flow(catalog)
            catalog = load_catalog()
            continue
        if action in {"refresh", "r"}:
            catalog = load_catalog()
            continue
        print("Please choose a valid action.")


def _read_clipboard() -> str | None:
    try:
        result = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    text = result.stdout.strip()
    return text if text else None


def _handle_clip(args: Sequence[str]) -> int:
    text = _read_clipboard()
    if not text:
        print("Clipboard is empty or couldn't be read.")
        return 1

    print("Clipboard contents detected:\n")
    preview = text[:200] + ("..." if len(text) > 200 else "")
    print(preview)

    confirm = input("\nSave this as a note? (y/N): ").strip().lower()
    if confirm != "y":
        print("❌ Clipboard note not saved")
        return 1

    catalog = load_catalog()
    try:
        metadata, embedding = auto_enrich_note(text, catalog.categories, catalog.tags)
    except AIServiceError as exc:
        print(f"❌ {exc}")
        return 1

    metadata = _apply_catalog_defaults(metadata, catalog)
    _display_metadata(metadata, catalog)

    linked_ids = _prompt_related_links(metadata, embedding)

    note = build_note(text, metadata, embedding, related_ids=linked_ids)
    persist_note(note, linked_ids)
    catalog.add_category(note["category"])
    catalog.add_tags(note["tags"])
    save_catalog(catalog)
    print(f"\n✅ Clipboard note saved! ID: {note['id']}")
    return 0


def _parse_days_argument(args: Sequence[str]) -> tuple[int, list[str]]:
    days = 1
    remaining: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--days="):
            value = arg.split("=", 1)[1]
            try:
                days = max(1, int(value))
            except ValueError:
                print("Invalid --days value; using default (1)")
            i += 1
            continue
        if arg == "--days" and i + 1 < len(args):
            value = args[i + 1]
            try:
                days = max(1, int(value))
            except ValueError:
                print("Invalid --days value; using default (1)")
            i += 2
            continue
        if arg.isdigit():
            days = max(1, int(arg))
        else:
            remaining.append(arg)
        i += 1
    return days, remaining


def _parse_model_argument(args: Sequence[str]) -> tuple[str | None, list[str]]:
    model: str | None = None
    remaining: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1].strip()
            i += 1
            continue
        if arg == "--model" and i + 1 < len(args):
            model = args[i + 1].strip()
            i += 2
            continue
        remaining.append(arg)
        i += 1
    return model, remaining


def _handle_recap(args: Sequence[str]) -> int:
    days, _ = _parse_days_argument(args)
    notes = notes_since(days)

    if not notes:
        print(f"No notes found in the last {days} day(s).")
        return 0

    note_ids = [note.get("id") for note in notes if note.get("id")]
    print(
        f"Generating recap across {len(notes)} note(s) from the last {days} day(s): "
        + ", ".join(note_ids)
    )

    try:
        summary = generate_recap_summary(notes)
    except AIServiceError as exc:
        print(f"❌ Failed to generate recap: {exc}")
        return 1

    print("\nRecap Preview\n" + "=" * 60)
    print(summary)
    print("=" * 60)

    confirm = input("Save this recap as a note? (y/N): ").strip().lower()
    if confirm != "y":
        print("❌ Recap discarded.")
        return 0

    try:
        embedding = generate_embedding(summary)
    except AIServiceError as exc:
        print(f"❌ Failed to embed recap: {exc}")
        return 1

    metadata = {
        "title": f"Recap {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "category": "Recap",
        "tags": ["auto", "recap"],
        "type": "recap",
    }

    recap_note = build_note(summary, metadata, embedding, related_ids=note_ids)
    persist_note(recap_note, note_ids)
    print(f"✅ Recap note saved with ID {recap_note['id']}.")
    return 0


def _compose_context_summary(notes: Sequence[dict], subject: str) -> str:
    if not notes:
        return "No prior notes were supplied; respond from general knowledge."

    lines = [
        "Use these prior notes purely as inspiration. Do not copy their language:",
    ]
    for note in notes[:5]:
        note_id = note.get("id", "?")
        title = note.get("title", "Untitled")
        ntype = note.get("type", "note")
        category = note.get("category", "")
        tags = ", ".join(note.get("tags", [])[:5])
        lines.append(
            f"- Note {note_id} [{ntype}] '{title}' (category {category}; tags: {tags})"
        )
    return "\n".join(lines)


def _compose_conversation_note_text(
    context_summary: str,
    history: Sequence[tuple[str, str, str]],
    model_name: str,
) -> str:
    lines = [
        "Context Overview:",
        context_summary if context_summary else "(None)",
        "",
        f"Model: {model_name}",
        "",
        "Transcript:",
    ]

    for timestamp, role, content in history:
        label = "You" if role == "user" else "Assistant"
        lines.append(f"[{timestamp}] {label}: {content}")

    return "\n".join(lines)


def _select_context_notes(
    embedding: Sequence[float],
    manual_threshold: float = 0.6,
    manual_limit: int = 3,
    fallback_threshold: float = 0.7,
    fallback_limit: int = 2,
) -> list[dict]:
    related_pairs = suggest_related_by_embedding(embedding, top_k=10)
    high_similarity = [pair for pair in related_pairs if pair[1] >= manual_threshold]

    manual_context: list[dict] = []
    for note, score in high_similarity:
        if note.get("type", "note") == "note" and score >= manual_threshold:
            manual_context.append(note)
        if len(manual_context) >= manual_limit:
            break

    context_notes = manual_context

    if len(context_notes) < manual_limit:
        supplemental: list[dict] = []
        for note, score in related_pairs:
            if note in context_notes:
                continue
            if score >= fallback_threshold:
                supplemental.append(note)
            if len(supplemental) >= fallback_limit:
                break
        context_notes.extend(supplemental[: manual_limit - len(context_notes)])

    seen_ids = set()
    unique_context: list[dict] = []
    for note in context_notes:
        note_id = note.get("id")
        if note_id and note_id not in seen_ids:
            seen_ids.add(note_id)
            unique_context.append(note)
    return unique_context


def _prompt_eli5_level() -> tuple[str, str, str] | None:
    try:
        from prompt_toolkit.application import Application as PTApplication
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style
    except ImportError:
        print("Select explanation level:")
        for index, (_, label, _) in enumerate(ELI5_LEVELS, start=1):
            print(f"  {index}. {label}")
        response = input("Enter number (blank to cancel): ").strip()
        if not response:
            return None
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(ELI5_LEVELS):
                return ELI5_LEVELS[idx]
        print("Invalid selection.")
        return None

    selected = {"index": 0, "result": None}

    def _render() -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = []
        for idx, (_, label, _) in enumerate(ELI5_LEVELS):
            if idx == selected["index"]:
                fragments.append(("class:option.selected", f"➤ {label}\n"))
            else:
                fragments.append(("class:option", f"  {label}\n"))
        fragments.append(("class:hint", "Use j/k or ↑/↓ to move · Enter select · Esc cancel"))
        return fragments

    control = FormattedTextControl(_render, focusable=True, show_cursor=False)
    window = Window(content=control, always_hide_cursor=True)
    body = HSplit([window])

    kb = KeyBindings()

    @kb.add("down")
    @kb.add("j")
    def _(event) -> None:  # pragma: no cover - UI hook
        selected["index"] = (selected["index"] + 1) % len(ELI5_LEVELS)
        event.app.invalidate()

    @kb.add("up")
    @kb.add("k")
    def _(event) -> None:  # pragma: no cover - UI hook
        selected["index"] = (selected["index"] - 1) % len(ELI5_LEVELS)
        event.app.invalidate()

    @kb.add("enter")
    def _(event) -> None:  # pragma: no cover - UI hook
        selected["result"] = ELI5_LEVELS[selected["index"]]
        event.app.exit()

    @kb.add("escape")
    def _(event) -> None:  # pragma: no cover - UI hook
        selected["result"] = None
        event.app.exit()

    style = Style.from_dict(
        {
            "": "bg:#0b1120",
            "option": "fg:#a5b4fc bg:#0b1120",
            "option.selected": "fg:#0b1120 bg:#38bdf8 bold",
            "hint": "fg:#94a3b8 bg:#0b1120",
        }
    )

    app = PTApplication(
        layout=Layout(body, focused_element=window),
        key_bindings=kb,
        style=style,
        full_screen=True,
    )
    app.run()  # pragma: no cover - UI hook

    return selected["result"]


def _handle_eli5(args: Sequence[str]) -> int:
    subject = input("What do you want explained? ").strip()
    if not subject:
        print("❌ Subject cannot be empty.")
        return 1

    level = _prompt_eli5_level()
    if level is None:
        print("❌ Eli5 request cancelled.")
        return 1

    level_key, level_label, level_instructions = level

    try:
        query_embedding = generate_embedding(subject)
    except AIServiceError as exc:
        print(f"❌ Failed to generate embedding for subject: {exc}")
        return 1

    context_notes = _select_context_notes(query_embedding)
    related_ids = [note.get("id") for note in context_notes if note.get("id")]

    context_summary = _compose_context_summary(context_notes, subject)

    combined_instructions = (
        level_instructions
        + " Treat the context overview as inspiration only—rephrase in fresh language, surface new angles, and avoid copying earlier explanations verbatim."
    )

    try:
        explanation = generate_eli5_explanation(
            subject,
            level_label,
            combined_instructions,
            context_summary,
        )
    except AIServiceError as exc:
        print(f"❌ Failed to generate explanation: {exc}")
        return 1

    print("\nExplanation\n" + "=" * 60)
    print(explanation)
    print("=" * 60)

    confirm = input("Save this explanation as a note? (y/N): ").strip().lower()
    if confirm != "y":
        print("❌ Explanation discarded.")
        return 0

    try:
        embedding = generate_embedding(explanation)
    except AIServiceError as exc:
        print(f"❌ Failed to embed explanation: {exc}")
        return 1

    catalog = load_catalog()
    metadata = generate_metadata(explanation, catalog.categories, catalog.tags)
    metadata = _apply_catalog_defaults(metadata, catalog)
    _display_metadata(metadata, catalog)

    while True:
        choice = input("\nConfirm metadata? (y/n/edit): ").strip().lower()
        if choice == "y":
            break
        if choice == "n":
            print("❌ Explanation not saved.")
            return 0
        if choice == "edit":
            metadata = _edit_metadata(metadata, catalog)
            _display_metadata(metadata, catalog)
            continue
        print("Please enter 'y', 'n', or 'edit'.")

    tags = metadata.get("tags", [])
    tags.extend(["auto", "eli5", f"eli5-level:{level_key}"])
    metadata["tags"] = sorted({tag.strip() for tag in tags if tag.strip()})
    metadata["type"] = "eli5"

    explanation_with_prompt = (
        "Question: " + subject + "\n\nExplanation:\n" + explanation
    )
    note = build_note(explanation_with_prompt, metadata, embedding, related_ids=related_ids)
    persist_note(note, related_ids)
    catalog.add_category(note["category"])
    catalog.add_tags(metadata["tags"])  # keep catalog updated with new tags
    save_catalog(catalog)

    print(f"✅ Eli5 note saved with ID {note['id']}.")

    follow_up = input("\nAsk a follow-up question (press Enter to skip): ").strip()
    if not follow_up:
        return 0

    context_for_followup = context_notes + [note]
    followup_context_summary = _compose_context_summary(context_for_followup, follow_up)
    try:
        followup_answer = generate_eli5_explanation(
            follow_up,
            level_label,
            combined_instructions,
            followup_context_summary,
            previous_answer=explanation,
        )
    except AIServiceError as exc:
        print(f"❌ Failed to generate follow-up: {exc}")
        return 1

    print("\nFollow-up\n" + "=" * 60)
    print(followup_answer)
    print("=" * 60)

    confirm_follow = input("Save this follow-up as a note? (y/N): ").strip().lower()
    if confirm_follow != "y":
        print("❌ Follow-up discarded.")
        return 0

    try:
        followup_embedding = generate_embedding(followup_answer)
    except AIServiceError as exc:
        print(f"❌ Failed to embed follow-up: {exc}")
        return 1

    metadata_follow = generate_metadata(followup_answer, catalog.categories, catalog.tags)
    metadata_follow = _apply_catalog_defaults(metadata_follow, catalog)
    _display_metadata(metadata_follow, catalog)

    while True:
        choice = input("\nConfirm follow-up metadata? (y/n/edit): ").strip().lower()
        if choice == "y":
            break
        if choice == "n":
            print("❌ Follow-up not saved.")
            return 0
        if choice == "edit":
            metadata_follow = _edit_metadata(metadata_follow, catalog)
            _display_metadata(metadata_follow, catalog)
            continue
        print("Please enter 'y', 'n', or 'edit'.")

    follow_tags = metadata_follow.get("tags", [])
    follow_tags.extend(["auto", "eli5", "eli5-followup", f"eli5-level:{level_key}"])
    metadata_follow["tags"] = sorted({tag.strip() for tag in follow_tags if tag.strip()})
    metadata_follow["type"] = "eli5"

    follow_related_ids = sorted({*(related_ids or []), note["id"]})
    followup_with_prompt = (
        "Question: " + follow_up + "\n\nExplanation:\n" + followup_answer
    )

    follow_note = build_note(
        followup_with_prompt,
        metadata_follow,
        followup_embedding,
        related_ids=follow_related_ids,
    )
    persist_note(follow_note, follow_related_ids)
    catalog.add_category(follow_note["category"])
    catalog.add_tags(metadata_follow["tags"])
    save_catalog(catalog)

    print(f"✅ Follow-up note saved with ID {follow_note['id']}.")
    return 0


def _handle_chat(args: Sequence[str]) -> int:
    model_arg, _ = _parse_model_argument(args)
    model_name = model_arg or os.getenv("MINDTHREAD_CHAT_MODEL") or "gpt-4o-mini"

    print("Starting conversation. Press Enter on an empty line to finish, or type :q to abort.")
    first_message = input("You: ").strip()
    if not first_message or first_message == ":q":
        print("❌ Conversation cancelled.")
        return 1

    try:
        query_embedding = generate_embedding(first_message)
    except AIServiceError as exc:
        print(f"❌ Failed to prepare conversation context: {exc}")
        return 1

    context_notes = _select_context_notes(query_embedding, manual_limit=4, fallback_limit=2)
    related_ids = [note.get("id") for note in context_notes if note.get("id")]
    context_summary = _compose_context_summary(context_notes, first_message)

    system_prompt = (
        "You are a collaborative partner that references the user's note context when helpful. "
        "Offer thoughtful insights, ask clarifying questions when needed, and cite note IDs only when they add value."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context overview:\n{context_summary}"},
        {"role": "user", "content": first_message},
    ]

    history: list[tuple[str, str, str]] = []
    history.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user", first_message))

    try:
        assistant_reply = generate_chat_reply(messages, model=model_name)
    except AIServiceError as exc:
        print(f"❌ Failed to generate reply: {exc}")
        return 1

    print("\nAssistant\n" + "=" * 60)
    print(assistant_reply)
    print("=" * 60)

    messages.append({"role": "assistant", "content": assistant_reply})
    history.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "assistant", assistant_reply))

    while True:
        user_input = input("You (blank to finish, :q to abort): ").strip()
        if user_input in (":q", ":quit"):
            print("❌ Conversation aborted.")
            return 0
        if not user_input:
            break

        history.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user", user_input))
        messages.append({"role": "user", "content": user_input})

        try:
            assistant_reply = generate_chat_reply(messages, model=model_name)
        except AIServiceError as exc:
            print(f"❌ Failed to generate reply: {exc}")
            history.pop()
            messages.pop()
            return 1

        print("\nAssistant\n" + "=" * 60)
        print(assistant_reply)
        print("=" * 60)

        messages.append({"role": "assistant", "content": assistant_reply})
        history.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "assistant", assistant_reply))

    assistant_turns = [entry for entry in history if entry[1] == "assistant"]
    if not assistant_turns:
        print("❌ No conversation recorded.")
        return 0

    save = input("Save this conversation as a note? (y/N): ").strip().lower()
    if save != "y":
        print("Conversation discarded.")
        return 0

    conversation_text = _compose_conversation_note_text(context_summary, history, model_name)

    try:
        embedding = generate_embedding(conversation_text)
    except AIServiceError as exc:
        print(f"❌ Failed to embed conversation: {exc}")
        return 1

    catalog = load_catalog()
    metadata = {
        "title": f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "category": "Conversation",
        "tags": ["auto", "convo", f"model:{model_name}"],
        "type": "convo",
    }

    metadata = _apply_catalog_defaults(metadata, catalog)
    _display_metadata(metadata, catalog)

    while True:
        choice = input("\nConfirm metadata? (y/n/edit): ").strip().lower()
        if choice == "y":
            break
        if choice == "n":
            print("❌ Conversation not saved.")
            return 0
        if choice == "edit":
            metadata = _edit_metadata(metadata, catalog)
            _display_metadata(metadata, catalog)
            continue
        print("Please enter 'y', 'n', or 'edit'.")

    note = build_note(conversation_text, metadata, embedding, related_ids=related_ids)
    persist_note(note, related_ids)
    catalog.add_category(note["category"])
    catalog.add_tags(metadata["tags"])
    save_catalog(catalog)

    print(f"✅ Conversation saved with ID {note['id']}.")
    return 0
