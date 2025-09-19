"""Simple catalog tracking categories and tags."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Set

from .config import get_settings
from .notes import list_all_notes


CATALOG_FILENAME = "catalog.json"


@dataclass
class Catalog:
    categories: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)

    def add_category(self, category: str) -> None:
        if category:
            self.categories.add(category)

    def add_categories(self, categories: Iterable[str]) -> None:
        for category in categories:
            self.add_category(category)

    def add_tags(self, tags: Iterable[str]) -> None:
        for tag in tags:
            cleaned = tag.strip()
            if cleaned:
                self.tags.add(cleaned)

    def remove_category(self, category: str) -> None:
        self.categories.discard(category)

    def remove_tag(self, tag: str) -> None:
        self.tags.discard(tag)

    def closest_category(self, target: str) -> str | None:
        if not target or not self.categories:
            return None
        best_match = None
        best_ratio = 0.0
        for candidate in self.categories:
            ratio = SequenceMatcher(None, candidate.lower(), target.lower()).ratio()
            if ratio > best_ratio:
                best_match = candidate
                best_ratio = ratio
        return best_match if best_ratio >= 0.6 else None

    def to_payload(self) -> dict:
        return {
            "categories": sorted(self.categories),
            "tags": sorted(self.tags),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "Catalog":
        return cls(set(payload.get("categories", [])), set(payload.get("tags", [])))


def _catalog_path() -> Path:
    settings = get_settings()
    return settings.data_dir / CATALOG_FILENAME


def load_catalog() -> Catalog:
    path = _catalog_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return Catalog.from_payload(json.load(handle))

    # Build from existing notes as fallback
    notes = list_all_notes()
    categories = {note.get("category", "").strip() for note in notes if note.get("category")}
    tags = set()
    for note in notes:
        for tag in note.get("tags", []):
            cleaned = tag.strip()
            if cleaned:
                tags.add(cleaned)
    catalog = Catalog(categories, tags)
    save_catalog(catalog)
    return catalog


def save_catalog(catalog: Catalog) -> None:
    path = _catalog_path()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(catalog.to_payload(), handle, indent=2)


__all__ = ["Catalog", "load_catalog", "save_catalog"]
