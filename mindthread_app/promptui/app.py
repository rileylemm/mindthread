"""Prompt_toolkit powered TUI for mindthread."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from pydoc import pager

from prompt_toolkit import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.application import get_app, run_in_terminal
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit, Layout
from prompt_toolkit.layout.containers import ScrollOffsets, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from ..analytics import format_tag_heatmap, render_sparkline
from ..editor import launch_editor
from ..notes import (
    AIServiceError,
    list_all_notes,
    note_counts_by_day,
    search_notes,
    tag_frequency,
    update_note_text,
)
from ..seeds import Seed, list_seed_notes, list_seeds

NOTE_HISTORY_DAYS = 14
TOP_TAGS = 5


@dataclass
class NoteSelection:
    notes: List[dict]
    selected: int = 0

    def current(self) -> Optional[dict]:
        if 0 <= self.selected < len(self.notes):
            return self.notes[self.selected]
        return None


@dataclass(frozen=True)
class ViewTab:
    key: str
    title: str
    loader: Callable[["MindthreadPromptUI", List[dict]], List[dict]]


CHAT_NOTE_TYPES = {"convo", "brain"}
ELI5_NOTE_TYPES = {"eli5"}


class MindthreadPromptUI:
    """Tactile prompt_toolkit interface for browsing notes."""

    def __init__(self) -> None:
        self.tabs: List[ViewTab] = [
            ViewTab(key="notes", title="Notes", loader=self._load_notes_tab),
            ViewTab(key="chats", title="Chats", loader=self._load_chats_tab),
            ViewTab(key="eli5", title="ELI5", loader=self._load_eli5_tab),
            ViewTab(key="seeds", title="Seeds", loader=self._load_seeds_tab),
        ]
        self.current_tab_index = 0
        self.query: Optional[str] = None
        self.selections: Dict[str, NoteSelection] = {tab.key: NoteSelection(notes=[]) for tab in self.tabs}
        self.notes_by_id: Dict[str, dict] = {}
        self.help_text = (
            "j/k: move · tab/shift-tab: switch lists · /: search · c: clear · e: edit · v: view · r: refresh · q: quit"
        )

        self.list_control = FormattedTextControl(
            self._render_note_list,
            focusable=True,
            show_cursor=False,
            get_cursor_position=self._get_cursor_position,
        )
        self.detail_area = TextArea(style="class:detail", read_only=True, scrollbar=True, wrap_lines=True)
        self.status_control = FormattedTextControl(self._render_status_bar)
        self.help_control = FormattedTextControl(lambda: FormattedText([("class:help", self.help_text)]))

        self.list_window = Window(
            content=self.list_control,
            style="class:list",
            height=Dimension(preferred=5, max=5, min=5),
            scroll_offsets=ScrollOffsets(top=1, bottom=1),
        )
        self.list_frame = Frame(self.list_window, title=self._current_tab_title())
        self.detail_frame = Frame(self.detail_area, title="DETAIL")
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

        body = VSplit([self.list_frame, self.detail_frame], padding=1)
        root = HSplit([body, self.status_window, self.help_window])
        self.layout = Layout(root, focused_element=self.list_window)
        self.app = Application(layout=self.layout, key_bindings=self.kb, style=self.style, full_screen=True, mouse_support=False)

        self._reload_notes(keep_focus=False)
        self._refresh_detail()

    # ------------------------------------------------------------------
    @property
    def selection(self) -> NoteSelection:
        return self.selections[self.tabs[self.current_tab_index].key]

    def _note_type(self, note: dict) -> str:
        return (note.get("type") or "note").lower()

    def _current_tab(self) -> ViewTab:
        return self.tabs[self.current_tab_index]

    def _current_tab_title(self) -> str:
        return self._current_tab().title.upper()

    def _update_list_frame_title(self) -> None:
        if hasattr(self, "list_frame"):
            self.list_frame.title = self._current_tab_title()

    def _cycle_tab(self, delta: int) -> None:
        if not self.tabs:
            return
        self.current_tab_index = (self.current_tab_index + delta) % len(self.tabs)
        self._update_list_frame_title()
        self._sync_list_scroll()
        self._refresh_detail()
        self._invalidate()

    def _find_selection_index(self, notes: List[dict], note_id: str) -> int:
        for idx, note in enumerate(notes):
            if note.get("id") == note_id:
                return idx
        return 0

    def _get_cursor_position(self) -> Point:
        selection = self.selection
        if not selection.notes:
            return Point(x=0, y=0)
        index = max(0, min(selection.selected, len(selection.notes) - 1))
        return Point(x=0, y=index)

    def _sync_list_scroll(self) -> None:
        selection = self.selection
        if not selection.notes:
            self.list_window.vertical_scroll = 0
            return

        # Keep the selected row inside the 5-line viewport by scrolling
        # once the cursor moves beyond the bottom slot.
        target = max(0, selection.selected - 4)
        self.list_window.vertical_scroll = target

    def _load_notes_tab(self, notes: List[dict]) -> List[dict]:
        return [
            note
            for note in notes
            if self._note_type(note) not in CHAT_NOTE_TYPES | ELI5_NOTE_TYPES
        ]

    def _load_chats_tab(self, notes: List[dict]) -> List[dict]:
        return [note for note in notes if self._note_type(note) in CHAT_NOTE_TYPES]

    def _load_eli5_tab(self, notes: List[dict]) -> List[dict]:
        return [note for note in notes if self._note_type(note) in ELI5_NOTE_TYPES]

    def _load_seeds_tab(self, _notes: List[dict]) -> List[dict]:
        seeds = list_seeds(order_by="state")
        query = (self.query or "").strip().lower()
        items: List[dict] = []
        for seed in seeds:
            haystack = " ".join(
                filter(
                    None,
                    [
                        seed.name,
                        seed.bloom,
                        seed.spark,
                        seed.intention,
                        seed.state,
                        seed.first_action,
                    ],
                )
            ).lower()
            if query and query not in haystack:
                continue
            items.append(self._seed_to_entry(seed))
        return items

    def _seed_to_entry(self, seed: Seed) -> dict:
        intention = (seed.intention or seed.spark or "").strip()
        subtitle = intention.splitlines()[0] if intention else ""
        tags = []
        cadence = seed.care_cadence or {}
        rhythm = cadence.get("kind") or cadence.get("rhythm")
        if rhythm:
            tags.append(str(rhythm))
        if seed.bloom:
            tags.append(seed.bloom)
        return {
            "id": f"seed:{seed.id}",
            "type": "seed",
            "title": seed.name or f"Seed {seed.id}",
            "category": seed.state,
            "tags": tags,
            "text": subtitle,
            "created_at": seed.created_at,
            "updated_at": seed.updated_at,
            "related_ids": [],
            "_seed": seed,
        }

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

        @kb.add("tab")
        def _(event) -> None:
            self._cycle_tab(1)

        @kb.add("s-tab")
        def _(event) -> None:
            self._cycle_tab(-1)

        @kb.add("/")
        def _(event) -> None:
            self._prompt_search()

        @kb.add("c")
        def _(event) -> None:
            self.query = None
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
        selection = self.selection
        if not selection.notes:
            return

        new_index = selection.selected + delta
        new_index = max(0, min(new_index, len(selection.notes) - 1))
        selection.selected = new_index
        self._sync_list_scroll()
        self._refresh_detail()
        self._invalidate()

    def _render_note_list(self) -> FormattedText:
        selection = self.selection
        if not selection.notes:
            return FormattedText([("class:list", " No notes found\n")])

        fragments: list[tuple[str, str]] = []
        for idx, note in enumerate(selection.notes):
            style = "class:list.selected" if idx == selection.selected else "class:list.unselected"
            tags = note.get("tags", [])
            links = len(note.get("related_ids", []))
            note_type = note.get("type", "note")
            type_label = f"{note_type}" if note_type != "note" else ""
            type_segment = f"{type_label} · " if type_label else ""
            fragments.append(
                (
                    style,
                    f" {note['title']} · {type_segment}{note.get('category', '—')} · {len(tags)} tags · {links} links\n",
                )
            )
        return FormattedText(fragments)

    def _render_status_bar(self) -> FormattedText:
        query_part = f" · filter: '{self.query}'" if self.query else ""
        tab = self._current_tab()
        count = len(self.selection.notes)
        return FormattedText([("class:status", f"{tab.title}: {count}{query_part}")])

    def _refresh_detail(self) -> None:
        note = self.selection.current()
        if not note:
            self.detail_area.text = "No notes available."
            return

        self.detail_area.text = self._compose_detail_text(note, include_analytics=True)

    def _reload_notes(self, keep_focus: bool = True) -> None:
        previous_ids: Dict[str, Optional[str]] = {}
        if keep_focus:
            for key, selection in self.selections.items():
                current = selection.current()
                previous_ids[key] = current.get("id") if current else None

        if self.query:
            all_notes = search_notes(self.query)
        else:
            all_notes = list_all_notes()

        self.notes_by_id = {note["id"]: note for note in all_notes if note.get("id") is not None}

        for tab in self.tabs:
            tab_notes = tab.loader(all_notes)
            selection = self.selections.setdefault(tab.key, NoteSelection(notes=[]))
            selection.notes = tab_notes

            previous_id = previous_ids.get(tab.key)
            if previous_id:
                selection.selected = self._find_selection_index(tab_notes, previous_id)
            elif tab_notes:
                selection.selected = min(selection.selected, len(tab_notes) - 1)
            else:
                selection.selected = 0

        self._update_list_frame_title()
        self._sync_list_scroll()
        self._refresh_detail()
        self._invalidate()

    def _prompt_search(self) -> None:
        from prompt_toolkit import prompt

        result_holder: list[Optional[str]] = [None]

        def _get_query() -> None:
            try:
                result_holder[0] = prompt("Search: ", default=self.query or "")
            except (EOFError, KeyboardInterrupt):
                result_holder[0] = None

        run_in_terminal(_get_query)
        query = result_holder[0]
        if query is None:
            return
        query = query.strip()
        self.query = query or None
        self._reload_notes(keep_focus=False)
        self._invalidate()

    def _edit_current_note(self) -> None:
        note = self.selection.current()
        if not note:
            return
        if self._note_type(note) == "seed":
            self.help_text = "Seed details are read-only here. Use 'mindthread seed' to update."
            self._invalidate()
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

        if self._note_type(note) == "seed":
            return

        def _ask_edit() -> None:
            try:
                response = input("Open note in editor? (y/N): ").strip().lower()
            except EOFError:
                response = ""
            if response == "y":
                self._edit_current_note()

        run_in_terminal(_ask_edit)

    def _compose_detail_text(self, note: dict, include_analytics: bool = True) -> str:
        note_type = self._note_type(note)
        if note_type == "seed" and note.get("_seed"):
            return self._compose_seed_detail(note["_seed"])  # type: ignore[arg-type]

        lines = [
            f"Title   : {note['title']}",
            f"Type    : {note.get('type', 'note')}",
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

    def _compose_seed_detail(self, seed: Seed) -> str:
        cadence = seed.care_cadence or {}
        cadence_desc = cadence.get("kind") or cadence.get("rhythm") or "—"
        window = seed.care_window if isinstance(seed.care_window, dict) else None
        lines = [
            f"Name      : {seed.name}",
            f"State     : {seed.state}",
            f"Cadence   : {cadence_desc}",
            f"Next check: {seed.next_check_at or '—'}",
            f"Momentum  : {seed.momentum_score if seed.momentum_score is not None else '—'}",
            f"Notes     : {seed.note_count}",
            f"Bloom     : {seed.bloom or '—'}",
        ]
        if window:
            window_bits = ", ".join(f"{key}={value}" for key, value in window.items())
            lines.append(f"Care window: {window_bits or '—'}")
        if seed.intention:
            lines.append("")
            lines.append("Intention:")
            lines.append(f"  {seed.intention.strip()}")
        elif seed.spark:
            lines.append("")
            lines.append("Spark:")
            lines.append(f"  {seed.spark.strip()}")

        recent = list_seed_notes(seed.id, limit=3)
        if recent:
            lines.append("")
            lines.append("Recent notes:")
            for note in recent:
                preview = note.text.strip().splitlines()
                snippet = preview[0][:80] if preview else ""
                suffix = "…" if preview and (len(preview[0]) > 80 or len(preview) > 1) else ""
                lines.append(
                    f"  [{note.note_type}] {note.created_at[:16]} {snippet}{suffix}"
                )

        lines.append("")
        lines.append("Use 'mindthread seed' or 'mindthread garden' for full seed actions.")
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
