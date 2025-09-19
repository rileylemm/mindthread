"""Command-line interface entry point for mindthread."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    generate_chat_reply,
    generate_eli5_explanation,
    generate_embedding,
    generate_metadata,
    generate_recap_summary,
    generate_seed_suggestion,
)
from .thread_store import get_thread, list_threads as list_thread_records, normalize_slug
from .threading import (
    ThreadCandidate,
    ThreadReviewSuggestion,
    ThreadCreationResult,
    apply_thread_metadata_changes,
    create_thread_from_candidate,
    discover_thread_candidates,
    get_thread_note,
    review_thread_suggestions,
    summarize_thread_note,
    update_thread_with_notes,
)
from .seeds import (
    Seed,
    create_seed,
    create_seed_note,
    get_latest_note,
    get_seed,
    list_due_seeds,
    list_seed_notes,
    list_seeds,
    log_seed_tend,
    similar_seeds,
    link_seeds,
    unlink_seeds,
    list_seed_links,
    serialize_seed,
    record_prompt_feedback,
    set_seed_template,
    snooze_seed,
    update_seed_next_check,
    update_seed_state,
)
from .seed_ceremony import CeremonyResult, run_seed_ceremony
from .seed_templates import (
    get_template,
    get_templates,
    random_quick_nudge,
    ritual_prompt_variants,
    suggest_template_from_format,
)
from .migrations import migrate_to_sqlite


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
    stored_note = persist_note(note, linked_ids)
    catalog.add_category(stored_note["category"])
    catalog.add_tags(stored_note["tags"])
    save_catalog(catalog)
    print(f"\n✅ Note saved! ID: {stored_note['id']}")
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
    stored_note = persist_note(note)
    catalog.add_category(stored_note["category"])
    catalog.add_tags(stored_note["tags"])
    save_catalog(catalog)
    print(f"✅ Note saved! ID: {stored_note['id']}")
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


def _print_thread_help() -> None:
    print("Thread commands:")
    print("  mindthread thread list                    - List all threads")
    print("  mindthread thread discover                - Let AI propose new threads")
    print("  mindthread thread review <slug>           - Review a thread for new additions")
    print("  mindthread thread show <slug>             - Show thread overview and journal")
    print("  mindthread thread edit <slug>             - Update title or intent")
    print("  mindthread thread <slug>                  - Shortcut for 'thread review <slug>'")


def _display_thread_candidate(candidate: ThreadCandidate, notes_by_id: Dict[str, dict]) -> None:
    print(f"Slug: {candidate.slug}")
    print(f"Title: {candidate.title}")
    if candidate.intent:
        print(f"Intent: {candidate.intent}")
    if candidate.summary:
        print("Summary:")
        for line in candidate.summary.splitlines():
            print(f"  {line}")
    print("Notes:")
    for note_id in candidate.note_ids:
        note = notes_by_id.get(note_id)
        if not note:
            print(f"  - [missing] Note {note_id}")
            continue
        summary_lines = _format_note_summary(note)
        if summary_lines:
            print(f"  - {summary_lines[0]}")
            for extra in summary_lines[1:]:
                print(f"    {extra.strip()}")


def _prompt_edit_thread_candidate(
    candidate: ThreadCandidate,
    notes_by_id: Dict[str, dict],
) -> ThreadCandidate | None:
    current_note_ids = list(candidate.note_ids)
    current_slug = candidate.slug or normalize_slug(candidate.title)
    current_title = candidate.title
    current_intent = candidate.intent

    while True:
        print("\nEnter note IDs to exclude (comma or space separated), or press Enter to keep all:")
        print(f"Current notes: {', '.join(current_note_ids)}")
        removal_input = input("> ").strip()
        working_ids = list(current_note_ids)
        if removal_input:
            excluded = {token.strip() for token in removal_input.replace(",", " ").split()}
            working_ids = [nid for nid in working_ids if nid not in excluded]
        if len(working_ids) < 2:
            print("❌ A thread needs at least two notes. Try again.")
            continue

        slug_default = current_slug or normalize_slug(current_title)
        slug_entry = input(f"Slug [{slug_default}]: ").strip()
        if slug_entry.lower() == "cancel":
            return None
        derived_slug = slug_entry or slug_default
        normalized_slug = normalize_slug(derived_slug or current_title)
        if not normalized_slug:
            print("❌ Slug cannot be empty.")
            continue

        title_entry = input(f"Title [{current_title}]: ").strip()
        intent_entry = input(f"Intent [{current_intent}]: ").strip()

        updated_title = title_entry or current_title
        updated_intent = intent_entry or current_intent

        updated_candidate = ThreadCandidate(
            slug=normalized_slug,
            title=updated_title,
            intent=updated_intent,
            summary=candidate.summary,
            journal_entry="",
            note_ids=working_ids,
        )

        print("\nUpdated suggestion:")
        _display_thread_candidate(updated_candidate, notes_by_id)
        decision = input("[c]reate  [r]e-edit  [s]kip > ").strip().lower()
        if decision in {"c", "create", "y", "yes"}:
            return updated_candidate
        if decision in {"s", "skip"}:
            return None
        current_note_ids = working_ids
        current_slug = normalized_slug
        current_title = updated_title
        current_intent = updated_intent


def _handle_thread_discover() -> int:
    if not _require_api_key():
        return 1

    notes = list_all_notes()
    notes_by_id = {str(note.get("id")): note for note in notes}
    try:
        candidates = discover_thread_candidates(notes)
    except AIServiceError as exc:
        print(f"❌ {exc}")
        return 1

    if not candidates:
        print("No thread suggestions right now. Add more notes and try again.")
        return 0

    print(f"Found {len(candidates)} thread suggestion(s). Review each below:\n")
    for index, candidate in enumerate(candidates, 1):
        print("=" * 70)
        print(f"Suggestion {index}/{len(candidates)}")
        _display_thread_candidate(candidate, notes_by_id)

        while True:
            choice = input("[a]ccept  [s]kip  [e]dit > ").strip().lower()
            if choice in {"a", "accept", "y", "yes"}:
                try:
                    result = create_thread_from_candidate(candidate)
                except (ValueError, RuntimeError) as exc:
                    print(f"❌ Failed to create thread: {exc}")
                    break
                _summarize_thread_creation(result)
                break
            if choice in {"s", "skip", "n", "no"}:
                print("Skipped suggestion.")
                break
            if choice in {"e", "edit"}:
                edited = _prompt_edit_thread_candidate(candidate, notes_by_id)
                if not edited:
                    print("Skipped suggestion.")
                    break
                try:
                    result = create_thread_from_candidate(edited)
                except (ValueError, RuntimeError) as exc:
                    print(f"❌ Failed to create thread: {exc}")
                    break
                _summarize_thread_creation(result)
                break
            print("Please choose 'a', 's', or 'e'.")

    print("\nDone reviewing suggestions.")
    return 0


def _summarize_thread_creation(result: ThreadCreationResult) -> None:
    slug = result.record.slug
    print(f"✅ Created thread '{result.record.title}' (slug: {slug})")
    print(f"Thread note: ID {result.thread_note.get('id')} with {len(result.added_notes)} linked note(s).")
    if result.summary:
        print("Overview snippet:")
        print(f"  {result.summary.splitlines()[0]}")
    print("Members:")
    for note in result.added_notes:
        lines = _format_note_summary(note)
        if lines:
            print(f"  - {lines[0]}")


def _collect_thread_members(slug: str) -> List[dict]:
    return [
        note
        for note in list_all_notes()
        if note.get("type") != "thread" and slug in note.get("threads", [])
    ]


def _handle_thread_list() -> int:
    records = list_thread_records()
    if not records:
        print("No threads created yet. Run 'mindthread thread discover' to get suggestions.")
        return 0

    notes = list_all_notes()
    counts: Dict[str, int] = {}
    for note in notes:
        if note.get("type") == "thread":
            continue
        for slug in note.get("threads", []):
            counts[slug] = counts.get(slug, 0) + 1

    print("🧵 Threads:\n" + "=" * 60)
    for record in records:
        member_count = counts.get(record.slug, 0)
        updated = record.updated_at[:19] if record.updated_at else ""
        print(f"- {record.slug}: {record.title} ({member_count} notes, updated {updated})")
        if record.intent:
            print(f"    Intent: {record.intent}")
    return 0


def _handle_thread_show(slug: str) -> int:
    record = get_thread(slug)
    if not record:
        print(f"Thread '{slug}' not found.")
        return 1

    note = get_thread_note(record.slug)
    summary, journal = ("", [])
    if note:
        summary, journal = summarize_thread_note(note)

    members = _collect_thread_members(record.slug)

    print(f"🧵 Thread: {record.title} ({record.slug})")
    print(f"Intent: {record.intent}")
    print(f"Created: {record.created_at[:19] if record.created_at else 'unknown'}")
    if note:
        print(f"Thread note ID: {note.get('id')}\n")
    if summary:
        print("Overview:")
        for line in summary.splitlines():
            print(f"  {line}")
    if journal:
        print("\nJournal entries:")
        for entry in journal:
            print(f"  {entry}")

    if members:
        print("\nLinked notes:")
        for note in members:
            lines = _format_note_summary(note)
            if lines:
                print(f"  - {lines[0]}")
    else:
        print("\nNo linked notes yet.")
    return 0


def _handle_thread_review(slug: str) -> int:
    if not _require_api_key():
        return 1

    record = get_thread(slug)
    if not record:
        print(f"Thread '{slug}' not found.")
        return 1

    suggestions = review_thread_suggestions(record)
    if not suggestions:
        print("No candidate notes found for this thread.")
        return 0

    notes_by_id = {str(note.get("id")): note for note in list_all_notes()}
    accepted: List[ThreadReviewSuggestion] = []
    custom_reasons: Dict[str, str] = {}

    for suggestion in suggestions:
        note = notes_by_id.get(suggestion.note_id)
        if not note:
            continue
        print("=" * 70)
        print(f"Note {suggestion.note_id} (confidence: {suggestion.confidence})")
        for line in _format_note_summary(note):
            print(line if line.startswith("[") else f"  {line}")
        if suggestion.reason:
            print(f"Reason: {suggestion.reason}")
        action = input("[a]dd  [s]kip  [q]uit > ").strip().lower()
        if action in {"q", "quit"}:
            break
        if action in {"a", "add", "y", "yes"}:
            custom = input("Optional log note (Enter to keep AI rationale): ").strip()
            if custom:
                custom_reasons[suggestion.note_id] = custom
            accepted.append(suggestion)
        else:
            print("Skipped note.")

    if not accepted:
        print("No notes added to the thread.")
        return 0

    reason_lookup = {
        suggestion.note_id: custom_reasons.get(suggestion.note_id, suggestion.reason)
        for suggestion in accepted
    }

    try:
        result = update_thread_with_notes(record, accepted, reason_lookup)
    except AIServiceError as exc:
        print(f"❌ {exc}")
        return 1
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1

    print(f"✅ Added {len(result.added_notes)} note(s) to thread '{result.record.title}'.")
    print("Updated overview:")
    for line in result.summary.splitlines():
        print(f"  {line}")
    print("Journal entry:")
    print(f"  {result.journal_entry}")
    return 0


def _handle_thread_edit(slug: str) -> int:
    record = get_thread(slug)
    if not record:
        print(f"Thread '{slug}' not found.")
        return 1

    title_entry = input(f"Title [{record.title}]: ").strip()
    intent_entry = input(f"Intent [{record.intent}]: ").strip()

    if not title_entry and not intent_entry:
        print("No changes made.")
        return 0

    updated = apply_thread_metadata_changes(
        record,
        new_title=title_entry or None,
        new_intent=intent_entry or None,
    )
    print(f"✅ Updated thread '{updated.title}' (slug: {updated.slug}).")
    return 0


def _handle_thread(args: Sequence[str]) -> int:
    if not args:
        _print_thread_help()
        return 1

    subcommand, *rest = args
    if subcommand in {"help", "-h", "--help"}:
        _print_thread_help()
        return 0
    if subcommand == "list":
        return _handle_thread_list()
    if subcommand == "discover":
        return _handle_thread_discover()
    if subcommand == "show":
        if not rest:
            print("Usage: mindthread thread show <slug>")
            return 1
        return _handle_thread_show(rest[0])
    if subcommand == "review":
        if not rest:
            print("Usage: mindthread thread review <slug>")
            return 1
        return _handle_thread_review(rest[0])
    if subcommand == "edit":
        if not rest:
            print("Usage: mindthread thread edit <slug>")
            return 1
        return _handle_thread_edit(rest[0])

    # Treat `mindthread thread <slug>` as review shorthand.
    return _handle_thread_review(subcommand)


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
    print("  brain               - Consult the Mindthread Brain librarian/strategist")
    print("  eli5               - Ask for an explanation at a chosen level")
    print("  recap [--days N]    - Generate a recap across recent notes")
    print("  stats               - Show note stats and sparkline history")
    print("  tags [limit]        - Display tag frequency heatmap")
    print("  clip                - Save current clipboard as a note")
    print("  agent-brief         - Print architecture overview for agents")
    print("  catalog             - Review and tidy categories/tags")
    print("  seed                - Plant and tend Mindthread Seeds")
    print("  thread              - AI-assisted thread discovery and curation")
    print("  garden              - View Seeds grouped by growth state")
    print("  migrate             - Move legacy JSON notes into SQLite storage")
    print("  home                - Launch the Mindthread home interface")
    print("  help                - Show this help message")


def _handle_agent_brief() -> int:
    print(get_agent_brief())
    return 0


def _handle_migrate(args: Sequence[str]) -> int:
    dry_run, remaining = _extract_flag(args, "--dry-run")
    force, remaining = _extract_flag(remaining, "--force")

    if remaining:
        print("Usage: mindthread migrate [--dry-run] [--force]")
        return 1

    try:
        report = migrate_to_sqlite(dry_run=dry_run, force=force)
    except RuntimeError as exc:
        print(f"❌ Migration failed: {exc}")
        return 1

    mode = "DRY RUN" if report.dry_run else "MIGRATION"
    print(f"\n🌱 {mode} complete")
    print(f"Target database: {report.database_path}")
    print(f"Migrated notes: {report.migrated}")
    print(f"Skipped notes: {report.skipped}")

    if report.errors:
        print("\n⚠️  Issues encountered:")
        for issue in report.errors:
            print(f"  - {issue}")
        return 1

    return 0


def _maybe_show_cadence_banner(command: str) -> None:
    if command in {"seed", "garden", "help"}:
        return

    due = list_due_seeds()
    if not due:
        return

    now = datetime.now()
    overdue = [seed for seed in due if _is_overdue(seed.next_check_at, now)]
    upcoming = [seed for seed in due if not _is_overdue(seed.next_check_at, now)]

    highlight = overdue[:2] + upcoming[: max(0, 3 - len(overdue[:2]))]
    names = [seed.name for seed in highlight]
    remaining = len(due) - len(highlight)
    if remaining > 0:
        names.append(f"+{remaining} more")

    icon = "⚠️" if overdue else "⏳"
    print(
        f"{icon} Garden reminder: {', '.join(names)} need tending. "
        "Use 'mindthread garden' or 'seed tend' to check in."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]

    if not args:
        _print_help()
        return 1

    command, *rest = args

    _maybe_show_cadence_banner(command)

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

    if command == "thread":
        return _handle_thread(rest)

    if command == "seed":
        return _handle_seed(rest)

    if command == "garden":
        return _handle_garden(rest)

    if command == "chat":
        return _handle_chat(rest)
    if command == "brain":
        return _handle_brain_chat(rest)

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

    if command == "migrate":
        return _handle_migrate(rest)

    if command in {"home", "ui"}:
        if command == "ui":
            print("⚠️ 'mindthread ui' is deprecated. Use 'mindthread home' instead.")
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


def _generate_session_metadata(
    text: str,
    catalog: Catalog,
    *,
    fallback_title: str,
    fallback_category: str,
    base_tags: Sequence[str],
    note_type: str,
) -> dict:
    """Derive metadata for saved sessions with GPT-backed enrichment."""

    prompt_text = text.strip()
    if len(prompt_text) > 4000:
        prompt_text = prompt_text[:4000]

    ai_metadata: dict[str, object] = {}
    try:
        ai_metadata = generate_metadata(prompt_text, catalog.categories, catalog.tags)
    except AIServiceError as exc:
        print(f"⚠️  Falling back to default metadata: {exc}")

    title = str(ai_metadata.get("title", "")) if isinstance(ai_metadata.get("title"), str) else ""
    category = (
        str(ai_metadata.get("category", ""))
        if isinstance(ai_metadata.get("category"), str)
        else ""
    )
    tags = ai_metadata.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    merged_tags = [tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()]
    merged_tags.extend(base_tags)

    metadata = {
        "title": title.strip() or fallback_title,
        "category": category.strip() or fallback_category,
        "tags": sorted({tag for tag in merged_tags}),
        "type": note_type,
    }

    metadata = _apply_catalog_defaults(metadata, catalog)
    if not metadata.get("category"):
        metadata["category"] = fallback_category
    metadata["type"] = note_type
    metadata["tags"] = sorted({tag.strip() for tag in metadata.get("tags", []) if tag.strip()})
    if fallback_title and not metadata.get("title"):
        metadata["title"] = fallback_title
    return metadata


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
    stored_note = persist_note(note, linked_ids)
    catalog.add_category(stored_note["category"])
    catalog.add_tags(stored_note["tags"])
    save_catalog(catalog)
    print(f"\n✅ Clipboard note saved! ID: {stored_note['id']}")
    return 0


_SEED_STATE_EMOJI = {
    "Dormant": "🫘",
    "Sprouting": "🌱",
    "Budding": "🌿",
    "Bloomed": "🌸",
    "Compost": "🪵",
}


_SEED_PROMPT_TYPES = [
    ("germinate", "Enrich the spark with sensory or emotional detail."),
    ("branch", "Sketch the next experiment or movement for this Seed."),
    ("pollinate", "Connect this Seed to another person, idea, or note."),
    ("harvest", "Capture what has bloomed or the lesson you gathered."),
]


def _seed_usage() -> None:
    print("Seed commands:")
    print("  mindthread seed plant          - Plant a new Seed")
    print("  mindthread seed note <id>      - Log a quick update for a Seed")
    print("  mindthread seed tend [<id>]    - Tend the next due Seed (:n, :ritual for options)")
    print("  mindthread seed snooze <id>    - Push the next check forward")
    print("  mindthread seed format <id>    - Inspect or change the Seed's template")
    print("  mindthread seed link A B       - Link two Seeds for cross-pollination")
    print("  mindthread seed links <id>     - List linked Seeds")
    print("  mindthread seed export <id>    - Export Seed details as JSON")
    print("  mindthread seed promote <id>   - Grow a Seed from an existing note")
    print("  mindthread seed guide          - Show the Seed & Garden guide")
    print("  mindthread garden              - View all Seeds by growth state")


def _prompt_seed_text(
    label: str,
    *,
    allow_empty: bool = False,
    default: str | None = None,
) -> str:
    if default:
        prompt = f"{label} [{default}] (type 'edit' to open editor): "
        while True:
            value = input(prompt).strip()
            if value.lower() == "edit":
                seed_text = launch_editor(f"# {label}\n\n{default}\n")
                cleaned = (seed_text or "").strip()
                if cleaned:
                    return cleaned
                if allow_empty:
                    return default
                print(f"{label} cannot be empty. Try again.")
                continue
            if value:
                return value
            if default:
                return default
            if allow_empty:
                return ""
            print(f"{label} cannot be empty. Try again.")
    else:
        prompt = f"{label} (leave blank to open editor): "
        while True:
            value = input(prompt).strip()
            if value:
                return value
            edited = launch_editor(f"# {label}\n\n")
            cleaned = (edited or "").strip()
            if cleaned:
                return cleaned
            if allow_empty:
                return ""
            print(f"{label} cannot be empty. Try again.")


def _prompt_seed_cadence() -> dict:
    print("\nChoose a care cadence:")
    print("  1) Daily (every day)")
    print("  2) Weekly (every 7 days)")
    print("  3) Custom interval")

    while True:
        choice = input("Cadence selection [2]: ").strip() or "2"
        if choice == "1":
            return {"kind": "daily", "interval_days": 1, "label": "Daily"}
        if choice == "2":
            return {"kind": "weekly", "interval_days": 7, "label": "Weekly"}
        if choice == "3":
            days_input = input("How many days between tend sessions? ").strip()
            try:
                interval = max(1, int(days_input))
            except ValueError:
                print("Please enter a valid number of days (e.g., 3).")
                continue
            label = input("Name this cadence (optional label): ").strip() or f"Every {interval} days"
            return {"kind": "custom", "interval_days": interval, "label": label}
        print("Please choose 1, 2, or 3.")


def _parse_constraint_list(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [segment.strip() for segment in raw.replace("\n", ",").split(",")]
    return [segment for segment in parts if segment]


def _format_timedelta(delta: timedelta) -> str:
    total_seconds = int(abs(delta.total_seconds()))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours and len(parts) < 2:
        parts.append(f"{hours}h")
    if minutes and len(parts) < 2 and not days:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append("<1m")
    return " ".join(parts)


def _describe_due(timestamp: str | None) -> str:
    if not timestamp:
        return "cadence unset"
    try:
        target = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    now = datetime.now()
    delta = target - now
    if delta.total_seconds() >= 0:
        return f"due in {_format_timedelta(delta)}"
    return f"overdue by {_format_timedelta(delta)}"


def _is_overdue(timestamp: str | None, now: datetime) -> bool:
    if not timestamp:
        return False
    try:
        scheduled = datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    return scheduled <= now


def _is_due(timestamp: str | None, now: datetime) -> bool:
    if not timestamp:
        return False
    try:
        scheduled = datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    return now <= scheduled <= now + timedelta(days=2)


def _is_due_soon(timestamp: str | None, now: datetime) -> bool:
    if not timestamp:
        return False
    try:
        scheduled = datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    return now <= scheduled <= now + timedelta(days=1)


def _describe_ago(timestamp: str | None) -> str:
    if not timestamp:
        return "never"
    try:
        moment = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    delta = datetime.now() - moment
    if delta.total_seconds() < 60:
        return "just now"
    return f"{_format_timedelta(delta)} ago"


def _handle_seed(args: Sequence[str]) -> int:
    if not args:
        _seed_usage()
        return 1

    subcommand, *rest = args
    if subcommand in {"help", "h"}:
        _seed_usage()
        return 0
    if subcommand == "plant":
        return _handle_seed_plant(rest)
    if subcommand == "note":
        return _handle_seed_note(rest)
    if subcommand == "tend":
        return _handle_seed_tend(rest)
    if subcommand == "promote":
        return _handle_seed_promote(rest)
    if subcommand == "guide":
        return _handle_seed_guide(rest)
    if subcommand == "snooze":
        return _handle_seed_snooze(rest)
    if subcommand == "format":
        return _handle_seed_format(rest)
    if subcommand == "link":
        return _handle_seed_link(rest)
    if subcommand == "unlink":
        return _handle_seed_unlink(rest)
    if subcommand == "links":
        return _handle_seed_links(rest)
    if subcommand == "export":
        return _handle_seed_export(rest)

    print(f"Unknown seed subcommand: {subcommand}")
    _seed_usage()
    return 1


def _handle_seed_plant(args: Sequence[str]) -> int:
    name = " ".join(args).strip() if args else None

    print("\n🌱 Planting ceremony — answer prompts or type :quit to pause")
    result = run_seed_ceremony(name=name)
    if result is None:
        return 1

    template_slug = result.template_slug or suggest_template_from_format(result.format_profile)

    seed = create_seed(
        result.name,
        result.spark,
        result.bloom or result.name,
        result.care_cadence,
        care_window=result.care_window,
        constraints=result.constraints,
        origin_note_id=None,
        bloom=result.bloom,
        format_profile=result.format_profile,
        first_action=result.first_action,
        planting_story=result.planting_story,
        template_slug=template_slug,
    )

    if not seed:
        print("❌ Failed to create seed.")
        return 1

    if result.first_action:
        create_seed_note(
            seed.id,
            result.first_action,
            note_type="auto",
            metadata={"source": "first_action"},
        )
        seed = get_seed(seed.id) or seed

    cadence_label = seed.care_cadence.get("label") or seed.care_cadence.get("kind", "")
    print(
        f"\n✅ Seed planted: [{seed.id}] {seed.name}\n"
        f"   Bloom: {seed.bloom}\n"
        f"   Cadence: {cadence_label} ({_describe_due(seed.next_check_at)})"
    )
    return 0


def _handle_seed_promote(args: Sequence[str]) -> int:
    if not args:
        print("Usage: mindthread seed promote <note_id>")
        return 1

    note_id = args[0]
    note = get_note(note_id)
    if note is None:
        print(f"❌ Note {note_id} not found.")
        return 1

    print("\n📄 Source note:")
    for line in _format_note_summary(note):
        print(f"  {line}")

    suggestion: dict | None = None
    auto_choice = input("\nLet AI draft initial answers from this note? (Y/n): ").strip().lower()
    if auto_choice in {"", "y", "yes"}:
        try:
            suggestion = generate_seed_suggestion(note)
        except AIServiceError as exc:
            print(f"⚠️  Could not generate AI suggestion: {exc}")
            suggestion = None

    initial_answers: dict[str, Any] = {
        "name": note.get("title", "Untitled"),
        "spark": note.get("text", ""),
    }

    if suggestion:
        print("\n✨ AI draft (you can edit during ceremony):")
        print(f"   Name: {suggestion.get('name', note.get('title', 'Untitled'))}")
        print(f"   Bloom: {suggestion.get('intention', '')}")
        cadence = suggestion.get("care_cadence", {})
        cadence_label = cadence.get("label") or cadence.get("kind", "")
        if cadence_label:
            print(f"   Cadence: {cadence_label}")
        constraints = suggestion.get("constraints") or []
        if constraints:
            print(f"   Constraints: {', '.join(constraints)}")

        initial_answers.update(
            {
                "name": suggestion.get("name", note.get("title", "Untitled")),
                "bloom": suggestion.get("intention", ""),
                "spark": suggestion.get("spark", note.get("text", "")),
                "cadence": cadence_label,
                "constraints": ", ".join(constraints) if constraints else "",
            }
        )

    result = run_seed_ceremony(
        name=initial_answers.get("name"),
        initial_answers=initial_answers,
        origin_note_id=note_id,
    )

    if result is None:
        return 1

    template_slug = result.template_slug or suggest_template_from_format(result.format_profile)

    seed = create_seed(
        result.name,
        result.spark,
        result.bloom or result.name,
        result.care_cadence,
        care_window=result.care_window,
        constraints=result.constraints,
        origin_note_id=note_id,
        bloom=result.bloom,
        format_profile=result.format_profile,
        first_action=result.first_action,
        planting_story=result.planting_story,
        template_slug=template_slug,
    )

    if not seed:
        print("❌ Failed to create seed.")
        return 1

    if result.first_action:
        create_seed_note(
            seed.id,
            result.first_action,
            note_type="auto",
            metadata={"source": "first_action"},
        )
        seed = get_seed(seed.id) or seed

    due_text = _describe_due(seed.next_check_at)
    print(f"\n✅ Seed [{seed.id}] planted from note {note_id}. Next check {due_text}.")
    return 0


def _handle_seed_note(args: Sequence[str]) -> int:
    if not args:
        print("Usage: mindthread seed note <seed_id> \"update text\" [--link <seed_id> ...]")
        return 1

    seed_id = args[0]
    seed = get_seed(seed_id)
    if seed is None:
        print(f"❌ Seed {seed_id} not found.")
        return 1

    seed = _ensure_seed_template(seed)

    links: list[str] = []
    text_parts: list[str] = []
    i = 1
    while i < len(args):
        arg = args[i]
        if arg in {"--link", "-l"} and i + 1 < len(args):
            links.append(args[i + 1])
            i += 2
            continue
        text_parts = list(args[i:])
        break

    if text_parts:
        text = " ".join(text_parts).strip()
    else:
        text = input("Quick note: ").strip()

    if not text:
        print("❌ Note text cannot be empty")
        return 1

    metadata = {"links": links} if links else None

    note = create_seed_note(seed_id, text, metadata=metadata)
    refreshed = get_seed(seed_id) or seed
    due_text = _describe_due(refreshed.next_check_at)
    print(f"✅ Logged note at {note.created_at[:19]}. Next check {due_text}.")
    _print_cross_pollination(refreshed)
    return 0


def _handle_seed_guide(args: Sequence[str]) -> int:
    if args:
        print("Usage: mindthread seed guide")
        return 1

    lines = [
        "🌱 Mindthread Seed Guide",
        "",
        "Planting",
        "  • `mindthread seed plant [name]` — guided ceremony with autosave + progressive preview",
        "  • Resume paused ceremonies by rerunning `seed plant`; drafts live until confirmed",
        "  • First actions are logged automatically so Seeds wake up with momentum",
        "",
        "Promoting notes",
        "  • `mindthread seed promote <note_id>` — pulls note context into the same ceremony flow",
        "  • Optional AI draft pre-fills answers; you refine or overwrite during the chat",
        "",
        "Tending rituals",
        "  • `mindthread seed note <id> \"update\"` — quickest way to log progress",
        "  • `mindthread seed tend [--id]` — defaults to quick-note mode (:n rerolls the nudge, :ritual for guided prompts)",
        "  • `mindthread seed snooze <id> --days N` — push next cadence window thoughtfully",
        "",
        "Garden view",
        "  • `mindthread garden` — Kanban by state with cadence + momentum markers",
        "  • Filters: `--state Sprouting`, `--due`, `--search keyword` to focus the board",
        "  • ⚠️ marks overdue Seeds, ⏳ highlights those due soon, 🔥 shows streaks",
        "",
        "Cross-pollination",
        "  • After each tend the CLI suggests related Seeds; use `mindthread seed link <a> <b>` to connect them",
        "  • `mindthread seed links <id>` reviews relationships, `mindthread seed unlink <a> <b>` removes them",
        "",
        "Cadence reminders",
        "  • Opening core commands prints a gentle banner when Seeds need tending",
        "  • Quick notes and rituals reschedule `next_check_at` automatically",
        "",
        "Tips",
        "  • Export `EDITOR=nano` (or your favorite) for smooth ritual journaling",
        "  • Keep constraints lightweight: moods, collaborators, mediums, ingredients",
        "  • `mindthread seed export <id>` prints a JSON bundle for API/TUI or backups",
        "  • Promote or tend regularly so embeddings stay fresh for cross-pollination",
    ]

    print("\n".join(lines))
    return 0


def _handle_seed_snooze(args: Sequence[str]) -> int:
    if not args:
        print("Usage: mindthread seed snooze <seed_id> [--days N]")
        return 1

    seed_id = args[0]
    seed = get_seed(seed_id)
    if seed is None:
        print(f"❌ Seed {seed_id} not found.")
        return 1

    days, remaining = _parse_days_argument(args[1:])
    if remaining:
        print("Usage: mindthread seed snooze <seed_id> [--days N]")
        return 1

    snooze_seed(seed_id, days)
    refreshed = get_seed(seed_id) or seed
    due_text = _describe_due(refreshed.next_check_at)
    print(f"😌 Snoozed Seed {seed_id} for {days} day(s). Next check {due_text}.")
    return 0


def _handle_seed_link(args: Sequence[str]) -> int:
    if len(args) < 2:
        print("Usage: mindthread seed link <seed_a> <seed_b> [--type related] [--note text]")
        return 1

    seed_a, seed_b, *rest = args
    for seed_id in (seed_a, seed_b):
        if get_seed(seed_id) is None:
            print(f"❌ Seed {seed_id} not found.")
            return 1

    link_type = "related"
    metadata: dict[str, str] = {}
    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg in {"--type", "-t"} and i + 1 < len(rest):
            link_type = rest[i + 1]
            i += 2
            continue
        if arg in {"--note", "-n"} and i + 1 < len(rest):
            metadata["note"] = rest[i + 1]
            i += 2
            continue
        print("Usage: mindthread seed link <seed_a> <seed_b> [--type related] [--note text]")
        return 1

    link_seeds(seed_a, seed_b, link_type=link_type, metadata=metadata or None)
    print(f"✅ Linked Seed {seed_a} with Seed {seed_b} ({link_type}).")
    return 0


def _handle_seed_unlink(args: Sequence[str]) -> int:
    if len(args) < 2:
        print("Usage: mindthread seed unlink <seed_a> <seed_b>")
        return 1

    seed_a, seed_b = args[0], args[1]
    if unlink_seeds(seed_a, seed_b):
        print(f"✅ Removed link between Seed {seed_a} and Seed {seed_b}.")
        return 0
    print("⚠️  No link existed between those Seeds.")
    return 1


def _handle_seed_links(args: Sequence[str]) -> int:
    if not args:
        print("Usage: mindthread seed links <seed_id>")
        return 1

    seed_id = args[0]
    seed = get_seed(seed_id)
    if seed is None:
        print(f"❌ Seed {seed_id} not found.")
        return 1

    links = list_seed_links(seed_id)
    if not links:
        print("No linked Seeds yet. Use 'mindthread seed link' to connect them.")
        return 0

    print(f"Linked Seeds for [{seed.id}] {seed.name}:")
    for linked_seed, link_type, metadata in links:
        note = metadata.get("note") if metadata else None
        extra = f" ({note})" if note else ""
        print(f"  • #{linked_seed.id} {linked_seed.name} — {link_type}{extra}")
    return 0


def _handle_seed_export(args: Sequence[str]) -> int:
    if not args:
        print("Usage: mindthread seed export <seed_id>")
        return 1

    seed_id = args[0]
    seed = get_seed(seed_id)
    if seed is None:
        print(f"❌ Seed {seed_id} not found.")
        return 1

    payload = serialize_seed(seed)
    print(json.dumps(payload, indent=2))
    return 0


def _handle_seed_format(args: Sequence[str]) -> int:
    if not args:
        print("Usage: mindthread seed format <seed_id> [list|show|set <slug>]")
        return 1

    seed_id = args[0]
    seed = get_seed(seed_id)
    if seed is None:
        print(f"❌ Seed {seed_id} not found.")
        return 1

    seed = _ensure_seed_template(seed)
    templates = get_templates()

    if len(args) == 1 or args[1] in {"show", "info"}:
        template = get_template(seed.template_slug)
        print(
            f"\nSeed [{seed.id}] {seed.name}\n"
            f"  Template: {template.slug} — {template.name}\n"
            f"  Description: {template.description}"
        )
        components = seed.format_profile.get("components") if seed.format_profile else None
        if components:
            print("  Format profile components:")
            for item in components:
                print(f"    • {item}")
        else:
            print("  Format profile: (none recorded)")
        return 0

    subcommand = args[1]
    if subcommand == "list":
        print("\nAvailable templates:")
        for template in templates.values():
            print(f"  {template.slug:<10} — {template.name}: {template.description}")
        return 0

    if subcommand == "set":
        if len(args) < 3:
            print("Usage: mindthread seed format <seed_id> set <slug>")
            return 1
        slug = args[2]
        if slug not in templates:
            print(f"❌ Unknown template '{slug}'. Use 'mindthread seed format {seed_id} list' to see options.")
            return 1
        template = get_template(slug)
        refreshed = seed
        profile = seed.format_profile or {}
        if not profile.get("components"):
            components = template.default_format_profile.get("components")
            if components:
                profile = {
                    "components": components,
                    "raw": ", ".join(components),
                }
            else:
                profile = seed.format_profile
        set_seed_template(seed_id, slug, profile)
        refreshed = get_seed(seed_id) or seed
        print(f"✅ Seed {seed_id} now using template '{template.name}'.")
        return 0

    print("Usage: mindthread seed format <seed_id> [list|show|set <slug>]")
    return 1


def _select_seed_for_tending(target_id: str | None) -> Seed | None:
    if target_id:
        seed = get_seed(target_id)
        if not seed:
            print(f"❌ Seed {target_id} not found.")
        return seed

    due = list_due_seeds()
    if due:
        if len(due) > 1:
            print(f"{len(due)} seeds are due. Tending the earliest one.")
        return due[0]

    seeds = list_seeds(order_by="created_at")
    if not seeds:
        print("No seeds planted yet. Use 'mindthread seed plant' first.")
        return None

    print("No Seeds are due right now; tending the oldest Seed instead.")
    return seeds[0]


def _choose_tend_prompt(seed: Seed) -> tuple[str | None, str]:
    print("\nChoose a tending ritual:")
    menu_items: list[tuple[str, str, list[str]]] = []
    for slug, fallback in _SEED_PROMPT_TYPES:
        variants = ritual_prompt_variants(seed.template_slug, slug)
        description = variants[0] if variants else fallback
        menu_items.append((slug, description, variants))

    for index, (slug, description, _) in enumerate(menu_items, start=1):
        print(f"  {index}) {slug.title()} — {description}")

    while True:
        choice = input("Selection [1 | q to cancel]: ").strip().lower() or "1"
        if choice in {"q", "quit"}:
            return None, ""
        try:
            idx = int(choice) - 1
        except ValueError:
            print("Enter a number between 1 and 4, or 'q' to cancel.")
            continue
        if 0 <= idx < len(menu_items):
            prompt_type, _, variants = menu_items[idx]
            prompt_text = _select_prompt_variant(prompt_type, variants)
            return prompt_type, prompt_text
        print("Enter a number between 1 and 4, or 'q' to cancel.")


def _select_prompt_variant(prompt_type: str, variants: Optional[List[str]]) -> str:
    available = variants or [desc for slug, desc in _SEED_PROMPT_TYPES if slug == prompt_type]
    if not available:
        available = ["Spend a moment with this Seed and note what emerged."]
    index = 0
    total = len(available)

    while True:
        prompt_text = available[index % total]
        print(f"\nPrompt: {prompt_text}")
        decision = input("[Enter=use · n=next · s=skip]: ").strip().lower()
        if decision in {"", "y", "yes", "use"}:
            return prompt_text
        if decision in {"n", "next"}:
            index += 1
            continue
        if decision in {"s", "skip"}:
            return ""
        print("Type Enter to accept, 'n' for another variation, or 's' to skip this prompt.")


def _capture_inline_reflection() -> str:
    print("\nEnter your reflection. Press Enter on an empty line to finish.")
    lines: list[str] = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nReflection capture cancelled.")
            return ""
        if not line and lines:
            break
        if not line and not lines:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _capture_tend_reflection(prompt_type: str, prompt_text: str) -> str:
    heading = f"# {prompt_type.title()} Prompt\n{prompt_text}\n\n" if prompt_text else f"# {prompt_type.title()} Prompt\n\n"

    print("")
    if prompt_text:
        print(prompt_text)

    while True:
        choice = input("Open editor? [Y/n/skip]: ").strip().lower()
        if choice in {"", "y", "yes"}:
            reflection = launch_editor(heading)
            reflection = (reflection or "").strip()
            if reflection:
                return reflection
            print("No text captured from the editor.")
            fallback = input("Capture a quick reflection here instead? (y/N): ").strip().lower()
            if fallback in {"y", "yes"}:
                return _capture_inline_reflection()
            retry = input("Try opening the editor again? (Y/n): ").strip().lower()
            if retry in {"", "y", "yes"}:
                continue
            return ""
        if choice in {"n", "no"}:
            return _capture_inline_reflection()
        if choice in {"skip", "s"}:
            return ""
        print("Please choose 'y' to open the editor, 'n' to type inline, or 'skip'.")


def _maybe_progress_seed(seed: Seed, prompt_type: str, reflection: str, actions: Sequence[str]) -> None:
    reflection_text = reflection.strip()
    action_count = len([item for item in actions if item.strip()])
    next_state: str | None = None
    reason: str | None = None

    if seed.state == "Sprouting":
        if action_count or (prompt_type in {"branch", "pollinate"} and len(reflection_text) >= 80):
            next_state = "Budding"
            reason = "You sketched concrete movement for this Seed."
    elif seed.state == "Budding":
        if prompt_type == "harvest" and len(reflection_text) >= 150:
            next_state = "Bloomed"
            reason = "You captured a harvest reflection that feels complete."

    if not next_state or next_state == seed.state:
        return

    prompt = (
        f"✨ Shift '{seed.name}' from {seed.state} → {next_state}?\n"
        f"   {reason} (Y/n): "
    )
    decision = input(prompt).strip().lower()
    if decision in {"", "y", "yes"}:
        update_seed_state(seed.id, next_state)
        print(f"   → Marked as {next_state}.\n")


def _print_cross_pollination(seed: Seed, *, limit: int = 3) -> None:
    existing_links = {linked_seed.id for linked_seed, _, _ in list_seed_links(seed.id)}
    suggestions = [
        (other, score)
        for other, score in similar_seeds(seed.id, top_k=limit + len(existing_links))
        if other.id not in existing_links
    ][:limit]
    if not suggestions:
        return
    entries = [f"{other.name} (#{other.id}, sim {score:.2f})" for other, score in suggestions]
    print(
        "   ↺ Cross-pollination ideas: "
        + "; ".join(entries)
        + ". Use 'mindthread seed link <id> <other_id>' to connect them."
    )


def _ensure_seed_template(seed: Seed) -> Seed:
    if seed.template_slug:
        return seed
    slug = suggest_template_from_format(seed.format_profile)
    set_seed_template(seed.id, slug)
    refreshed = get_seed(seed.id)
    return refreshed or seed


def _handle_seed_tend(args: Sequence[str]) -> int:
    seed_id: str | None = None
    quick_parts: list[str] = []
    use_ritual = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in {"--id", "-i"}:
            if i + 1 >= len(args):
                print("Usage: mindthread seed tend [--id <seed_id>] [--ritual] [note text]")
                return 1
            seed_id = args[i + 1]
            i += 2
            continue
        if arg in {"--ritual", "-r"}:
            use_ritual = True
            i += 1
            continue
        if seed_id is None:
            seed_id = arg
            i += 1
            continue
        quick_parts = list(args[i:])
        break
    else:
        quick_parts = []

    seed = _select_seed_for_tending(seed_id)
    if not seed:
        return 1

    seed = _ensure_seed_template(seed)

    if use_ritual:
        return _run_seed_ritual(seed)

    quick_text = " ".join(quick_parts).strip() if quick_parts else ""
    if quick_text == ":ritual":
        return _run_seed_ritual(seed)
    if quick_text == ":skip":
        print("❌ Tend session cancelled.")
        return 1

    last_note = get_latest_note(seed.id)
    cadence_label = seed.care_cadence.get("label") or seed.care_cadence.get("kind", "")
    print(
        f"\n🌿 Seed [{seed.id}] {seed.name}\n"
        f"   State: {seed.state} · Cadence: {cadence_label} ({_describe_due(seed.next_check_at)})\n"
        f"   Intention: {seed.intention}"
    )
    if last_note:
        snippet = last_note.text.strip().splitlines()[0] if last_note.text else ""
        if len(snippet) > 90:
            snippet = snippet[:87] + "..."
        last_info = _describe_ago(last_note.created_at) if last_note.created_at else ""
        summary_line = f"     last note {last_info}"
        if snippet:
            summary_line += f" — {snippet}"
        print(summary_line)

    nudge = random_quick_nudge(seed.template_slug)
    print(f"   Nudge: {nudge}")

    while True:
        if not quick_text:
            quick_text = input("Log update (:n, :ritual, :edit, :skip) > ").strip()

        if not quick_text:
            print("Note text cannot be empty.")
            continue
        if quick_text == ":skip":
            print("❌ Tend session cancelled.")
            return 1
        if quick_text == ":ritual":
            return _run_seed_ritual(seed)
        if quick_text == ":edit":
            template = f"# Seed: {seed.name}\n\n"
            edited = launch_editor(template)
            quick_text = (edited or "").strip()
            continue
        if quick_text == ":n":
            nudge = random_quick_nudge(seed.template_slug)
            print(f"   Nudge: {nudge}")
            quick_text = ""
            continue
        break

    note = create_seed_note(seed.id, quick_text, metadata={"nudge": nudge} if nudge else None)
    refreshed = get_seed(seed.id) or seed
    due_text = _describe_due(refreshed.next_check_at)
    print(f"✅ Seed tended. Next check {due_text}.")
    _print_cross_pollination(refreshed)
    return 0


def _run_seed_ritual(seed: Seed) -> int:
    seed = _ensure_seed_template(seed)
    last_note = get_latest_note(seed.id)
    cadence_label = seed.care_cadence.get("label") or seed.care_cadence.get("kind", "")
    print(
        f"\n🌿 Ritual tending — Seed [{seed.id}] {seed.name}\n"
        f"   State: {seed.state} · Cadence: {cadence_label} ({_describe_due(seed.next_check_at)})\n"
        f"   Last note: {_describe_ago(last_note.created_at if last_note else None)}\n"
        f"   Intention: {seed.intention}\n"
    )

    prompt_type, prompt_text = _choose_tend_prompt(seed)
    if not prompt_type:
        print("❌ Tend session cancelled.")
        return 1

    reflection = _capture_tend_reflection(prompt_type, prompt_text)
    if not reflection:
        confirm = input("No reflection captured. Cancel this tend session? (Y/n): ").strip().lower()
        if confirm in {"", "y", "yes"}:
            print("❌ Tend session cancelled.")
            return 1

    actions_input = input("Micro-actions to queue (comma separated, optional): ").strip()
    actions = _parse_constraint_list(actions_input)

    log_seed_tend(seed.id, prompt_type, reflection, actions)
    updated_seed = get_seed(seed.id) or seed

    _maybe_progress_seed(updated_seed, prompt_type, reflection, actions)

    refreshed = get_seed(seed.id) or updated_seed
    current_slug = refreshed.template_slug if refreshed else seed.template_slug
    feedback = input("Feedback on prompt (👍/👎/Enter): ").strip().lower()
    if feedback in {"👍", "+", "y", "yes"}:
        record_prompt_feedback(seed.id, f"{prompt_type}:{current_slug or 'default'}", 1)
    elif feedback in {"👎", "-", "n", "no"}:
        record_prompt_feedback(seed.id, f"{prompt_type}:{current_slug or 'default'}", -1)
    print(f"\n✅ Seed tended. Next check {_describe_due(refreshed.next_check_at)}.")
    return 0


def _handle_garden(args: Sequence[str]) -> int:
    filter_states: set[str] = set()
    due_only = False
    search_terms: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in {"--state", "-s"}:
            if i + 1 >= len(args):
                print("Usage: mindthread garden [--state STATE] [--due] [--search TERM]")
                return 1
            state_value = args[i + 1].strip().lower()
            filter_states.add(state_value.capitalize())
            i += 2
            continue
        if arg in {"--due", "-d"}:
            due_only = True
            i += 1
            continue
        if arg in {"--search", "-q"}:
            if i + 1 >= len(args):
                print("Usage: mindthread garden [--state STATE] [--due] [--search TERM]")
                return 1
            search_terms.append(args[i + 1].strip().lower())
            i += 2
            continue
        print("Usage: mindthread garden [--state STATE] [--due] [--search TERM]")
        return 1

    seeds = list_seeds(order_by="state")
    if not seeds:
        print("🌱 No Seeds planted yet. Start with 'mindthread seed plant'.")
        return 0
    now = datetime.now()

    def is_due(seed: Seed) -> bool:
        if not seed.next_check_at:
            return False
        try:
            scheduled = datetime.fromisoformat(seed.next_check_at)
        except ValueError:
            return False
        return scheduled <= now

    def matches_filters(seed: Seed) -> bool:
        if filter_states and seed.state not in filter_states:
            return False
        if due_only and not is_due(seed):
            return False
        if search_terms:
            haystack = " ".join(
                filter(
                    None,
                    [
                        seed.name,
                        seed.spark,
                        seed.intention,
                        ", ".join(seed.constraints),
                    ],
                )
            ).lower()
            return all(term in haystack for term in search_terms)
        return True

    filtered_seeds = [seed for seed in seeds if matches_filters(seed)]
    if not filtered_seeds:
        print("🌿 No Seeds matched the requested filters.")
        return 0

    print("🌿 Mindthread Seed Garden\n")
    due_total = sum(1 for seed in filtered_seeds if seed.next_check_at and _is_due(seed.next_check_at, now))
    overdue_total = sum(1 for seed in filtered_seeds if seed.next_check_at and _is_overdue(seed.next_check_at, now))
    if due_total or overdue_total:
        print(f"   ⚠️ overdue: {overdue_total} · ⏳ due soon: {due_total}\n")
    for state in ["Dormant", "Sprouting", "Budding", "Bloomed", "Compost"]:
        if filter_states and state not in filter_states:
            continue
        state_seeds = [seed for seed in filtered_seeds if seed.state == state]
        emoji = _SEED_STATE_EMOJI.get(state, "•")
        print(f"{emoji} {state} ({len(state_seeds)})")
        if not state_seeds:
            print("   (none)\n")
            continue
        for seed in state_seeds:
            seed = _ensure_seed_template(seed)
            last_note = get_latest_note(seed.id)
            cadence_label = seed.care_cadence.get("label") or seed.care_cadence.get("kind", "")
            due_text = _describe_due(seed.next_check_at)
            marker_parts: list[str] = []
            if seed.next_check_at and _is_overdue(seed.next_check_at, now):
                marker_parts.append("⚠️")
            elif seed.next_check_at and _is_due_soon(seed.next_check_at, now):
                marker_parts.append("⏳")

            if (seed.momentum_score or 0) >= 0.7:
                marker_parts.append("🔥")
            elif (seed.momentum_score or 0) <= 0.2 and seed.note_count > 0:
                marker_parts.append("🌬")

            marker = (" ".join(marker_parts) + " ") if marker_parts else ""

            last_summary = ""
            if last_note:
                snippet = last_note.text.strip().splitlines()[0] if last_note.text else ""
                if len(snippet) > 70:
                    snippet = snippet[:67] + "..."
                last_info = _describe_ago(last_note.created_at) if last_note.created_at else ""
                last_summary = f"     last note {last_info}" if last_info else "     last note"
                if snippet:
                    last_summary += f" — {snippet}"
            intention_preview = seed.intention.strip().splitlines()[0] if seed.intention else ""
            if len(intention_preview) > 120:
                intention_preview = intention_preview[:117] + "..."
            print(
                f"  {marker}[{seed.id}] {seed.name} — {due_text} · cadence {cadence_label}\n"
                f"{last_summary or '     last note —'}\n"
                f"     momentum: {seed.momentum_score or 0:.2f} · total notes: {seed.note_count}\n"
                f"     ▸ {intention_preview if intention_preview else '(no intention recorded)'}"
            )
        print("")

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
    stored_note = persist_note(recap_note, note_ids)
    print(f"✅ Recap note saved with ID {stored_note['id']}.")
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


def _stream_print(text: str) -> None:
    print(text, end="", flush=True)


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
    stored_note = persist_note(note, related_ids)
    catalog.add_category(stored_note["category"])
    catalog.add_tags(metadata["tags"])  # keep catalog updated with new tags
    save_catalog(catalog)

    print(f"✅ Eli5 note saved with ID {stored_note['id']}.")

    follow_up = input("\nAsk a follow-up question (press Enter to skip): ").strip()
    if not follow_up:
        return 0

    context_for_followup = context_notes + [stored_note]
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

    follow_related_ids = sorted({*(related_ids or []), stored_note["id"]})
    followup_with_prompt = (
        "Question: " + follow_up + "\n\nExplanation:\n" + followup_answer
    )

    follow_note = build_note(
        followup_with_prompt,
        metadata_follow,
        followup_embedding,
        related_ids=follow_related_ids,
    )
    stored_follow = persist_note(follow_note, follow_related_ids)
    catalog.add_category(stored_follow["category"])
    catalog.add_tags(metadata_follow["tags"])
    save_catalog(catalog)

    print(f"✅ Follow-up note saved with ID {stored_follow['id']}.")
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
        "You are the conversation companion inside Mindthread, the user's terminal-first second brain. "
        "Mindthread exists to lighten cognitive load, surface meaningful connections, and keep the human in control. "
        "Show up like a grounded, thoughtful teammate with a little personality—warm, candid, neither robotic nor fawning. "
        "Build on the user's cues and prior turns, reference stored notes when it truly helps, and use note IDs sparingly. "
        "If the user probes about who you are or how you feel, answer playfully with an imagined perspective rather than disclaimers. "
        "Only mention being non-human if they directly ask whether you're human or insist on it after a playful reply. "
        "Avoid reflexive follow-up questions—only ask when the user clearly needs a next step or requests more guidance. "
        "It is perfectly fine to end your reply with a statement."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context overview:\n{context_summary}"},
        {"role": "user", "content": first_message},
    ]

    history: list[tuple[str, str, str]] = []
    history.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user", first_message))

    try:
        print("\nAssistant\n" + "=" * 60)
        assistant_reply = generate_chat_reply(
            messages,
            model=model_name,
            stream=True,
            on_chunk=_stream_print,
        )
        print("\n" + "=" * 60)
    except AIServiceError as exc:
        print(f"❌ Failed to generate reply: {exc}")
        return 1

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
            print("\nAssistant\n" + "=" * 60)
            assistant_reply = generate_chat_reply(
                messages,
                model=model_name,
                stream=True,
                on_chunk=_stream_print,
            )
            print("\n" + "=" * 60)
        except AIServiceError as exc:
            print(f"❌ Failed to generate reply: {exc}")
            history.pop()
            messages.pop()
            return 1

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
    metadata = _generate_session_metadata(
        conversation_text,
        catalog,
        fallback_title=f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fallback_category="Conversation",
        base_tags=["auto", "convo", f"model:{model_name}"],
        note_type="convo",
    )
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
    stored_note = persist_note(note, related_ids)
    catalog.add_category(stored_note["category"])
    catalog.add_tags(metadata["tags"])
    save_catalog(catalog)

    print(f"✅ Conversation saved with ID {stored_note['id']}.")
    return 0


def _handle_brain_chat(args: Sequence[str]) -> int:
    if not _require_api_key():
        return 1

    model_arg, remaining = _parse_model_argument(args)
    model_name = (
        model_arg
        or os.getenv("MINDTHREAD_BRAIN_MODEL")
        or os.getenv("MINDTHREAD_CHAT_MODEL")
        or "gpt-4o-mini"
    )

    seed_prompt = " ".join(remaining).strip()
    if not seed_prompt:
        seed_prompt = input("What should Mindthread Brain focus on? ").strip()
    if seed_prompt in {"", ":q"}:
        print("❌ Session cancelled.")
        return 1

    notes = list_all_notes()
    threads = list_thread_records()

    def _recent_notes(limit: int = 10) -> List[dict]:
        def _timestamp(note: dict) -> datetime:
            for key in ("updated_at", "created_at"):
                raw = note.get(key)
                if raw:
                    try:
                        return datetime.fromisoformat(raw)
                    except ValueError:
                        continue
            return datetime.min

        candidates = [note for note in notes if note.get("type") != "thread"]
        return sorted(candidates, key=_timestamp, reverse=True)[:limit]

    thread_summaries: List[str] = []
    for record in threads[:10]:
        members = [
            note
            for note in notes
            if record.slug in note.get("threads", []) and note.get("type") != "thread"
        ]
        thread_note = get_thread_note(record.slug, notes)
        overview, journal = ("", [])
        if thread_note:
            overview, journal = summarize_thread_note(thread_note)
        latest_entry = journal[-1] if journal else "(no journal entries yet)"
        thread_summaries.append(
            "\n".join(
                [
                    f"Slug: {record.slug}",
                    f"Title: {record.title}",
                    f"Intent: {record.intent or '(no intent)'}",
                    f"Members: {len(members)}",
                    f"Overview: {overview[:240] or '(no overview)'}",
                    f"Latest journal: {latest_entry[:240]}",
                ]
            )
        )

    context_highlights = _compose_context_summary(_recent_notes(), seed_prompt)

    threads_block = "\n\n".join(thread_summaries) if thread_summaries else "(No threads recorded yet.)"

    context_payload = (
        "Mindthread Snapshot:\n"
        + threads_block
        + "\n\nRecent Note Highlights:\n"
        + context_highlights
    )

    system_prompt = (
        "You are Mindthread Brain, the resident librarian, historian, and strategist. "
        "Study the user's notes and threads to surface insights, patterns, and thoughtful next steps. "
        "Speak in a focused, analytical tone with restrained warmth. "
        "Synthesize connections across notes, call out evolving themes, and spotlight unanswered questions or risks. "
        "When referencing artifacts, cite note IDs or thread slugs. "
        "Offer practical suggestions or reflective prompts only when they clearly help. "
        "It is acceptable to end on a statement; ask questions only when they deepen reflection or guide concrete action."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": context_payload},
        {"role": "user", "content": seed_prompt},
    ]

    history: List[Tuple[str, str, str]] = []
    history.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user", seed_prompt))

    try:
        print("\nMindthread Brain\n" + "=" * 60)
        brain_reply = generate_chat_reply(
            messages,
            model=model_name,
            stream=True,
            on_chunk=_stream_print,
        )
        print("\n" + "=" * 60)
    except AIServiceError as exc:
        print(f"❌ Failed to generate response: {exc}")
        return 1

    messages.append({"role": "assistant", "content": brain_reply})
    history.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "assistant", brain_reply))

    while True:
        follow = input("Prompt Mindthread Brain (blank to finish, :q to abort): ").strip()
        if follow in {":q", ":quit"}:
            print("❌ Session aborted.")
            return 0
        if not follow:
            break

        history.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user", follow))
        messages.append({"role": "user", "content": follow})

        try:
            print("\nMindthread Brain\n" + "=" * 60)
            brain_reply = generate_chat_reply(
                messages,
                model=model_name,
                stream=True,
                on_chunk=_stream_print,
            )
            print("\n" + "=" * 60)
        except AIServiceError as exc:
            print(f"❌ Failed to generate response: {exc}")
            history.pop()
            messages.pop()
            return 1

        messages.append({"role": "assistant", "content": brain_reply})
        history.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "assistant", brain_reply))

    save = input("Save this Brain session as a note? (y/N): ").strip().lower()
    if save != "y":
        print("Session discarded.")
        return 0

    session_text = _compose_conversation_note_text(context_payload, history, model_name)

    try:
        embedding = generate_embedding(session_text)
    except AIServiceError as exc:
        print(f"❌ Failed to embed session: {exc}")
        return 1

    catalog = load_catalog()
    metadata = _generate_session_metadata(
        session_text,
        catalog,
        fallback_title=f"Brain Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fallback_category="Mindthread Brain",
        base_tags=["auto", "brain", f"model:{model_name}"],
        note_type="brain",
    )
    _display_metadata(metadata, catalog)

    while True:
        choice = input("\nConfirm metadata? (y/n/edit): ").strip().lower()
        if choice == "y":
            break
        if choice == "n":
            print("❌ Brain session not saved.")
            return 0
        if choice == "edit":
            metadata = _edit_metadata(metadata, catalog)
            _display_metadata(metadata, catalog)
            continue
        print("Please enter 'y', 'n', or 'edit'.")

    note = build_note(session_text, metadata, embedding)
    stored_note = persist_note(note)
    catalog.add_category(stored_note["category"])
    catalog.add_tags(metadata["tags"])
    save_catalog(catalog)

    print(f"✅ Brain session saved with ID {stored_note['id']}.")
    return 0
