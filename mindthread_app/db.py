"""SQLite database helpers for mindthread."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import get_settings


SCHEMA_VERSION = 2

SCHEMA_STATEMENTS = [
    # Schema version tracking
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        version INTEGER NOT NULL
    )
    """,
    """
    INSERT INTO schema_meta (id, version)
    VALUES (1, ?)
    ON CONFLICT(id) DO UPDATE SET version = excluded.version
    """,
    # Notes table
    """
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY,
        type TEXT NOT NULL DEFAULT 'note',
        title TEXT NOT NULL,
        body TEXT NOT NULL,
        category TEXT,
        tags TEXT,
        threads TEXT,
        related_ids TEXT,
        embedding BLOB,
        metadata TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_notes_category ON notes(category)",
    # Seeds table
    """
    CREATE TABLE IF NOT EXISTS seeds (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        bloom TEXT,
        spark TEXT,
        intention TEXT,
        format_profile TEXT,
        care_cadence TEXT,
        care_window TEXT,
        first_action TEXT,
        constraints TEXT,
        planting_story TEXT,
        template_slug TEXT,
        state TEXT NOT NULL DEFAULT 'Dormant',
        momentum_score REAL,
        embedding BLOB,
        created_at TEXT NOT NULL,
        updated_at TEXT,
        next_check_at TEXT,
        origin_note_id INTEGER,
        FOREIGN KEY (origin_note_id) REFERENCES notes(id) ON DELETE SET NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_seeds_state ON seeds(state)",
    "CREATE INDEX IF NOT EXISTS idx_seeds_next_check ON seeds(next_check_at)",
    "CREATE INDEX IF NOT EXISTS idx_seeds_template ON seeds(template_slug)",
    "CREATE INDEX IF NOT EXISTS idx_seeds_momentum ON seeds(momentum_score)",
    # Seed notes
    """
    CREATE TABLE IF NOT EXISTS seed_notes (
        id INTEGER PRIMARY KEY,
        seed_id INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        note_type TEXT NOT NULL,
        text TEXT NOT NULL,
        next_action TEXT,
        metadata TEXT,
        embedding BLOB,
        FOREIGN KEY (seed_id) REFERENCES seeds(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_seed_notes_seed_created ON seed_notes(seed_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_seed_notes_created ON seed_notes(created_at)",
    # Seed tend log (legacy)
    """
    CREATE TABLE IF NOT EXISTS seed_tends (
        id INTEGER PRIMARY KEY,
        seed_id INTEGER NOT NULL,
        performed_at TEXT NOT NULL,
        prompt_type TEXT,
        reflection TEXT,
        actions TEXT,
        ai_assist TEXT,
        FOREIGN KEY (seed_id) REFERENCES seeds(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_seed_tends_seed_id ON seed_tends(seed_id)",
    # Prompt preferences
    """
    CREATE TABLE IF NOT EXISTS seed_prompt_preferences (
        id INTEGER PRIMARY KEY,
        seed_id INTEGER NOT NULL,
        prompt_key TEXT NOT NULL,
        score INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (seed_id) REFERENCES seeds(id) ON DELETE CASCADE
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_seed_prompt_pref_unique ON seed_prompt_preferences(seed_id, prompt_key)",
    # Seed links
    """
    CREATE TABLE IF NOT EXISTS seed_links (
        id INTEGER PRIMARY KEY,
        a_seed_id INTEGER NOT NULL,
        b_seed_id INTEGER NOT NULL,
        link_type TEXT NOT NULL DEFAULT 'related',
        metadata TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (a_seed_id) REFERENCES seeds(id) ON DELETE CASCADE,
        FOREIGN KEY (b_seed_id) REFERENCES seeds(id) ON DELETE CASCADE,
        CHECK (a_seed_id < b_seed_id)
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_seed_links_pair ON seed_links(a_seed_id, b_seed_id)",
    # Notifications queue
    """
    CREATE TABLE IF NOT EXISTS seed_notifications (
        id INTEGER PRIMARY KEY,
        seed_id INTEGER NOT NULL,
        type TEXT NOT NULL,
        scheduled_for TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'pending',
        payload TEXT,
        FOREIGN KEY (seed_id) REFERENCES seeds(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_seed_notifications_state ON seed_notifications(state, scheduled_for)",
]


def _apply_schema(connection: sqlite3.Connection) -> None:
    """Create base tables/indexes if they do not exist."""

    cursor = connection.cursor()
    try:
        for statement in SCHEMA_STATEMENTS:
            if statement.count("?"):
                cursor.execute(statement, (SCHEMA_VERSION,))
            else:
                cursor.execute(statement)
            if "CREATE TABLE IF NOT EXISTS seeds" in statement:
                _ensure_seed_columns(connection)
        _ensure_seed_columns(connection)
        connection.commit()
    finally:
        cursor.close()


def _ensure_seed_columns(connection: sqlite3.Connection) -> None:
    """Add new columns to seeds table when migrating from older schema."""

    existing = {row["name"] for row in connection.execute("PRAGMA table_info(seeds)")}

    def add_column(name: str, sql_fragment: str) -> None:
        if name not in existing:
            connection.execute(f"ALTER TABLE seeds ADD COLUMN {sql_fragment}")
            existing.add(name)

    add_column("bloom", "bloom TEXT")
    add_column("format_profile", "format_profile TEXT")
    add_column("first_action", "first_action TEXT")
    add_column("planting_story", "planting_story TEXT")
    add_column("template_slug", "template_slug TEXT")
    add_column("momentum_score", "momentum_score REAL")

    existing_seed_notes = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='seed_notes'").fetchone()
    if not existing_seed_notes:
        connection.execute(
            """
            CREATE TABLE seed_notes (
                id INTEGER PRIMARY KEY,
                seed_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                note_type TEXT NOT NULL,
                text TEXT NOT NULL,
                next_action TEXT,
                metadata TEXT,
                embedding BLOB,
                FOREIGN KEY (seed_id) REFERENCES seeds(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute("CREATE INDEX idx_seed_notes_seed_created ON seed_notes(seed_id, created_at)")
        connection.execute("CREATE INDEX idx_seed_notes_created ON seed_notes(created_at)")

    existing_prompt_pref = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='seed_prompt_preferences'"
    ).fetchone()
    if not existing_prompt_pref:
        connection.execute(
            """
            CREATE TABLE seed_prompt_preferences (
                id INTEGER PRIMARY KEY,
                seed_id INTEGER NOT NULL,
                prompt_key TEXT NOT NULL,
                score INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (seed_id) REFERENCES seeds(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            "CREATE UNIQUE INDEX idx_seed_prompt_pref_unique ON seed_prompt_preferences(seed_id, prompt_key)"
        )

    existing_links = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='seed_links'"
    ).fetchone()
    if not existing_links:
        connection.execute(
            """
            CREATE TABLE seed_links (
                id INTEGER PRIMARY KEY,
                a_seed_id INTEGER NOT NULL,
                b_seed_id INTEGER NOT NULL,
                link_type TEXT NOT NULL DEFAULT 'related',
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (a_seed_id) REFERENCES seeds(id) ON DELETE CASCADE,
                FOREIGN KEY (b_seed_id) REFERENCES seeds(id) ON DELETE CASCADE,
                CHECK (a_seed_id < b_seed_id)
            )
            """
        )
        connection.execute(
            "CREATE UNIQUE INDEX idx_seed_links_pair ON seed_links(a_seed_id, b_seed_id)"
        )


def get_database_path() -> Path:
    """Return the path to the configured SQLite database file."""

    settings = get_settings()
    db_path = settings.database_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def connect(readonly: bool = False) -> sqlite3.Connection:
    """Return a SQLite connection with sensible defaults."""

    path = get_database_path()
    if readonly:
        uri = f"file:{path}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
    else:
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def get_connection(readonly: bool = False) -> Iterator[sqlite3.Connection]:
    """Context manager that yields a connection and closes it afterwards."""

    connection = connect(readonly=readonly)
    try:
        yield connection
    finally:
        connection.close()


def ensure_schema(connection: sqlite3.Connection | None = None) -> None:
    """Ensure the database schema exists, creating tables if needed."""

    if connection is not None:
        _apply_schema(connection)
        return

    with get_connection() as conn:
        _apply_schema(conn)
