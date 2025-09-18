"""Prompt_toolkit powered TUI for mindthread."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pydoc import pager

from prompt_toolkit import Application
from prompt_toolkit.application import get_app, run_in_terminal
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit, Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from ..analytics import format_tag_heatmap, render_sparkline
from ..editor import launch_editor
from ..notes import (
    AIServiceError,
    get_note,
    list_all_notes,
    note_counts_by_day,
    search_notes,
    tag_frequency,
    update_note_text,
)

NOTE_HISTORY_DAYS = 14
TOP_TAGS = 5


@dataclass
class NoteSelection:
    notes: List[dict]
    selected: int = 0
    query: Optional[str] = None

    def current(self) -> Optional[dict]:
        if 0 <= self.selected < len(self.notes):
            return self.notes[self.selected]
        return None


class MindthreadPromptUI:
    """Tactile prompt_toolkit interface for browsing notes."""

    def __init__(self) -> None:
        self.selection = NoteSelection(notes=list_all_notes())
        self.notes_by_id = {note["id"]: note for note in self.selection.notes}
        self.help_text = "j/k: move · /: search · c: clear · e: edit · v: view · r: refresh · q: quit"

        self.list_control = FormattedTextControl(self._render_note_list, focusable=True, show_cursor=False)
        self.detail_area = TextArea(style="class:detail", read_only=True, scrollbar=True, wrap_lines=True)
        self.status_control = FormattedTextControl(self._render_status_bar)
        self.help_control = FormattedTextControl(lambda: FormattedText([("class:help", self.help_text)]))

        self.list_window = Window(content=self.list_control, style="class:list")
        self.status_window = Window(height=1, content=self.status_control, style="class:status")
        self.help_window = Window(height=1, content=self.help_control, style="class:help")

        self.kb = self._build_key_bindings()
        self.style = Style.from_dict(
            {
                "frame": "bg:#111111 #e5e7eb",
                "frame.border": "#2563eb",
                "list": "bg:#111111 #d1d5db",
                "list.focused": "bg:#0f172a #f9fafb",
                "list.selected": "bg:#2563eb #ffffff",
                "list.unselected": "bg:#111111 #94a3b8",
                "detail": "bg:#0c0c0c #e5e7eb",
                "status": "bg:#111111 #9ca3af",
                "help": "bg:#111111 #64748b",
            }
        )

        body = VSplit(
            [
                Frame(self.list_window, title="NOTES"),
                Frame(self.detail_area, title="DETAIL"),
            ],
            padding=1,
        )
        root = HSplit([body, self.status_window, self.help_window])
        self.layout = Layout(root, focused_element=self.list_window)
        self.app = Application(layout=self.layout, key_bindings=self.kb, style=self.style, full_screen=True, mouse_support=False)

        self._refresh_detail()

    # ------------------------------------------------------------------
    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("q")
        def _(event) -> None:
            event.app.exit()

        @kb.add("j")
        @kb.add("down")
        def _(event) -> None:
            self._move_selection(1)

        @kb.add("k")
        @kb.add("up")
        def _(event) -> None:
            self._move_selection(-1)

        @kb.add("/")
        def _(event) -> None:
            self._prompt_search()

        @kb.add("c")
        def _(event) -> None:
            self.selection.query = None
            self._reload_notes(keep_focus=False)

        @kb.add("e")
        def _(event) -> None:
            self._edit_current_note()

        @kb.add("v")
        def _(event) -> None:
            self._view_current_note()

        @kb.add("r")
        def _(event) -> None:
            self._reload_notes()

        return kb

    # ------------------------------------------------------------------
    def _move_selection(self, delta: int) -> None:
        if not self.selection.notes:
            return
        self.selection.selected = (self.selection.selected + delta) % len(self.selection.notes)
        self._refresh_detail()
        self._invalidate()

    def _render_note_list(self) -> FormattedText:
        if not self.selection.notes:
            return FormattedText([("class:list", " No notes found\n")])

        fragments: list[tuple[str, str]] = []
        for idx, note in enumerate(self.selection.notes):
            style = "class:list.selected" if idx == self.selection.selected else "class:list.unselected"
            tags = note.get("tags", [])
            links = len(note.get("related_ids", []))
            fragments.append(
                (
                    style,
                    f" {note['title']} · {note.get('category', '—')} · {len(tags)} tags · {links} links\n",
                )
            )
        return FormattedText(fragments)

    def _render_status_bar(self) -> FormattedText:
        query_part = f" · filter: '{self.selection.query}'" if self.selection.query else ""
        return FormattedText([("class:status", f"Notes: {len(self.selection.notes)}{query_part}")])

    def _refresh_detail(self) -> None:
        note = self.selection.current()
        if not note:
            self.detail_area.text = "No notes available."
            return

        self.detail_area.text = self._compose_detail_text(note, include_analytics=True)

    def _reload_notes(self, keep_focus: bool = True) -> None:
        previous_id = self.selection.current()["id"] if keep_focus and self.selection.current() else None
        if self.selection.query:
            notes = search_notes(self.selection.query)
        else:
            notes = list_all_notes()
        self.selection.notes = notes
        self.notes_by_id = {note["id"]: note for note in notes}

        if previous_id:
            for idx, note in enumerate(notes):
                if note.get("id") == previous_id:
                    self.selection.selected = idx
                    break
            else:
                self.selection.selected = 0
        else:
            self.selection.selected = 0

        self._refresh_detail()
        self._invalidate()

    def _prompt_search(self) -> None:
        from prompt_toolkit import prompt

        result_holder: list[Optional[str]] = [None]

        def _get_query() -> None:
            try:
                result_holder[0] = prompt("Search: ", default=self.selection.query or "")
            except (EOFError, KeyboardInterrupt):
                result_holder[0] = None

        run_in_terminal(_get_query)
        query = result_holder[0]
        if query is None:
            return
        query = query.strip()
        self.selection.query = query or None
        self._reload_notes(keep_focus=False)
        self._invalidate()

    def _edit_current_note(self) -> None:
        note = self.selection.current()
        if not note:
            return

        result_holder: list[Optional[str]] = [None]

        def _run_editor() -> None:
            result_holder[0] = launch_editor(note.get("text", ""))

        run_in_terminal(_run_editor)
        edited = result_holder[0]
        if edited is None or edited == note.get("text", ""):
            return

        try:
            update_note_text(note["id"], edited, regenerate_embedding=True)
        except AIServiceError as exc:
            self.help_text = f"Edit failed: {exc}"
            self._invalidate()
            return

        # Reload fresh data and keep selection on this note
        self._reload_notes(keep_focus=True)
        self.help_text = "Note saved"
        self._invalidate()

    def _view_current_note(self) -> None:
        note = self.selection.current()
        if not note:
            return

        text = self._compose_detail_text(note, include_analytics=False)

        def _viewer() -> None:
            pager(text)

        run_in_terminal(_viewer)

        def _ask_edit() -> None:
            try:
                response = input("Open note in editor? (y/N): ").strip().lower()
            except EOFError:
                response = ""
            if response == "y":
                self._edit_current_note()

        run_in_terminal(_ask_edit)

    def _compose_detail_text(self, note: dict) -> str:
        return self._compose_detail_text(note, include_analytics=True)

    def _compose_detail_text(self, note: dict, *, include_analytics: bool) -> str:  # type: ignore[override]
        lines = [
            f"Title   : {note['title']}",
            f"Category: {note.get('category', '')}",
            f"Tags    : {', '.join(note.get('tags', [])) or '—'}",
            f"Created : {note.get('created_at', '')}",
        ]
        if note.get("updated_at"):
            lines.append(f"Updated : {note['updated_at']}")
        if note.get("related_ids"):
            lines.append(f"Links   : {', '.join(note['related_ids'])}")
        lines.append("\n" + note.get("text", ""))
        if include_analytics:
            history = note_counts_by_day(NOTE_HISTORY_DAYS)
            spark_counts = [count for _, count in history]
            labels = " ".join(date[5:] for date, _ in history)
            spark = render_sparkline(spark_counts)
            top_tags = list(format_tag_heatmap(tag_frequency()[:TOP_TAGS], max_width=18))

            analytic_lines = [
                "",
                f"History ({len(history)}d):",
                f"  {labels}" if labels else "  (no history)",
                f"  {spark}",
            ]
            if top_tags:
                analytic_lines.append("")
                analytic_lines.append("Top tags:")
                analytic_lines.extend(f"  {line}" for line in top_tags)
            lines.extend(analytic_lines)

        return "\n".join(lines)

    def _invalidate(self) -> None:
        if hasattr(self, "app"):
            self.app.invalidate()

    def run(self) -> None:
        self.app.run()


def run_ui() -> None:
    """Launch the prompt_toolkit-based UI."""

    ui = MindthreadPromptUI()
    ui.run()
