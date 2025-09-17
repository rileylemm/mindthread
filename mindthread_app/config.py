"""Configuration helpers for mindthread CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load .env from the project root (if present) regardless of current working dir
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    openai_api_key: Optional[str]
    embedding_model: str
    gpt_model: str
    storage_type: str
    data_dir: Path

    @property
    def notes_file(self) -> Path:
        return self.data_dir / "notes.json"

    @property
    def catalog_file(self) -> Path:
        return self.data_dir / "catalog.json"


def _resolve_data_dir(raw_value: str | None) -> Path:
    if not raw_value:
        return PROJECT_ROOT

    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    data_dir = _resolve_data_dir(os.getenv("DATA_DIR"))
    data_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        gpt_model=os.getenv("GPT_MODEL", "gpt-4"),
        storage_type=os.getenv("STORAGE_TYPE", "json").lower(),
        data_dir=data_dir,
    )


__all__ = ["Settings", "get_settings", "PROJECT_ROOT"]
