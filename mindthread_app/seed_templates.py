"""Seed format templates and nudge helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SeedTemplate:
    slug: str
    name: str
    description: str
    default_format_profile: dict
    quick_nudges: List[str]
    ritual_prompts: Dict[str, List[str]]


_TEMPLATES: List[SeedTemplate] = [
    SeedTemplate(
        slug="playlist",
        name="Curated Playlist",
        description="Track list with alignment notes and energy arcs.",
        default_format_profile={
            "components": [
                "curated track list",
                "alignment notes",
                "energy arc narrative",
            ]
        },
        quick_nudges=[
            "Add a track and note why it belongs.",
            "Refine the arc: do you need a bridge between phases?",
            "Write one sentence about the vibe you're curating right now.",
        ],
        ritual_prompts={
            "germinate": [
                "Describe the scene this track underscores inside the mythology.",
                "What textures or field recordings would deepen this moment?",
            ],
            "branch": [
                "Sketch how the next three songs evolve the energy arc.",
                "List collaborators or references you want to weave in next.",
            ],
            "pollinate": [
                "Borrow from another Seed or story—who lends energy here?",
                "Note a person to share this draft with for resonance feedback.",
            ],
            "harvest": [
                "Document the emotional shift this sequence creates for listeners.",
                "Reflect on what mythology thread you advanced today.",
            ],
        },
    ),
    SeedTemplate(
        slug="novel",
        name="Narrative Manuscript",
        description="Scenes, character arcs, and structural beats.",
        default_format_profile={
            "components": ["scene list", "character notes", "act structure"]
        },
        quick_nudges=[
            "Log a beat you discovered today.",
            "Capture a line of dialogue you don't want to lose.",
            "Note how the protagonist changed in this scene.",
        ],
        ritual_prompts={
            "germinate": [
                "Paint the sensory details anchoring this scene.",
                "What emotional question is the chapter asking?",
            ],
            "branch": [
                "Outline the next obstacle or reveal.",
                "List two ways the setting can echo the theme.",
            ],
            "pollinate": [
                "Which character arc parallels this moment?",
                "Note research or inspiration sources you need here.",
            ],
            "harvest": [
                "Summarize the scene's impact in one paragraph.",
                "Capture what you learned about the story's spine today.",
            ],
        },
    ),
    SeedTemplate(
        slug="recipe",
        name="Recipe / Culinary Lab",
        description="Ingredient experiments, tasting notes, ritual plating.",
        default_format_profile={
            "components": ["ingredient experiments", "tasting notes", "ritual plating"]
        },
        quick_nudges=[
            "Document a tweak you tried (ingredient, technique, time).",
            "Capture tasting notes or feedback from testers.",
            "Note the presentation ritual you envision for serving.",
        ],
        ritual_prompts={
            "germinate": [
                "Describe the sensory memory this dish invokes.",
                "Which seasonal element should shine right now?",
            ],
            "branch": [
                "Outline the next experiment or variation to run.",
                "List collaborators or tasters to invite for feedback.",
            ],
            "pollinate": [
                "Pair it with a music, drink, or story that heightens the experience.",
                "What cultural nod or memory do you want to honor?",
            ],
            "harvest": [
                "Write the final plating instructions with intention cues.",
                "Summarize what you learned from today's tasting.",
            ],
        },
    ),
    SeedTemplate(
        slug="default",
        name="General Seed",
        description="Open format for untamed ideas.",
        default_format_profile={"components": []},
        quick_nudges=[
            "Log the most recent thing that moved this Seed forward.",
            "Name one next step that feels alive.",
            "Capture how this Seed is making you feel right now.",
        ],
        ritual_prompts={
            "germinate": [
                "Revisit the spark—what details feel vivid today?",
                "Describe the world this Seed wants to inhabit.",
            ],
            "branch": [
                "List experiments or drafts you could run next.",
                "Sketch a small prototype you can complete soon.",
            ],
            "pollinate": [
                "Who or what could breathe energy into this Seed?",
                "Link it to another Seed or note that resonates.",
            ],
            "harvest": [
                "Record what emerged or shifted during this session.",
                "What story will you tell about this Seed today?",
            ],
        },
    ),
]


@lru_cache(maxsize=1)
def get_templates() -> Dict[str, SeedTemplate]:
    return {template.slug: template for template in _TEMPLATES}


def get_template(slug: str | None) -> SeedTemplate:
    templates = get_templates()
    if slug and slug in templates:
        return templates[slug]
    return templates["default"]


def suggest_template_from_format(format_profile: dict | None) -> str:
    if not format_profile:
        return "default"
    components = ", ".join(format_profile.get("components", []))
    text = f"{format_profile.get('raw', '')} {components}".lower()
    for slug, keywords in {
        "playlist": ["track", "playlist", "dj", "energy arc"],
        "novel": ["chapter", "scene", "character", "manuscript"],
        "recipe": ["ingredient", "recipe", "tasting", "plating"],
    }.items():
        if any(keyword in text for keyword in keywords):
            return slug
    return "default"


def random_quick_nudge(template_slug: str | None) -> str:
    template = get_template(template_slug)
    return random.choice(template.quick_nudges)


def ritual_prompt_variants(template_slug: str | None, prompt_type: str) -> List[str]:
    template = get_template(template_slug)
    prompts = template.ritual_prompts.get(prompt_type)
    if not prompts:
        default_template = get_template("default")
        return default_template.ritual_prompts.get(prompt_type, [])
    return prompts


__all__ = [
    "SeedTemplate",
    "get_template",
    "get_templates",
    "suggest_template_from_format",
    "random_quick_nudge",
    "ritual_prompt_variants",
]

