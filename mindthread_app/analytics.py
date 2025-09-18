"""Rendering helpers for sparklines and tag heatmaps."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

SPARKLINE_SYMBOLS = " .:-=+*#"


def render_sparkline(counts: Sequence[int]) -> str:
    """Return a simple sparkline string for a sequence of counts."""

    if not counts:
        return "(no data)"
    max_count = max(counts)
    if max_count == 0:
        return SPARKLINE_SYMBOLS[1] * len(counts)
    scale = len(SPARKLINE_SYMBOLS) - 1
    return "".join(SPARKLINE_SYMBOLS[int((count / max_count) * scale)] for count in counts)


def format_tag_heatmap(
    frequencies: Sequence[Tuple[str, int]],
    *,
    max_width: int = 24,
) -> Iterable[str]:
    """Yield formatted lines to represent a tag frequency heatmap."""

    if not frequencies:
        return []

    max_count = max(freq for _, freq in frequencies) or 1

    rows = []
    for tag, count in frequencies:
        scale = count / max_count if max_count else 0
        bar_len = max(1, int(scale * max_width))
        bar = "#" * bar_len
        rows.append(f"{tag:<20} {bar} ({count})")
    return rows


__all__ = ["render_sparkline", "format_tag_heatmap"]
