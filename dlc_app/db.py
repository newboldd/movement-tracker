"""SQLite database schema and connection helpers."""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import DB_PATH

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    stage TEXT NOT NULL DEFAULT 'created',
    iteration INTEGER NOT NULL DEFAULT 1,
    camera_name TEXT,
    dlc_dir TEXT,
    video_pattern TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    log_path TEXT,
    progress_pct REAL DEFAULT 0,
    remote_host TEXT,
    pid INTEGER,
    tmux_session TEXT,
    error_msg TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    finished_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS label_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    iteration INTEGER NOT NULL DEFAULT 1,
    session_type TEXT NOT NULL DEFAULT 'initial',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    committed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS frame_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES label_sessions(id),
    frame_num INTEGER NOT NULL,
    trial_idx INTEGER NOT NULL DEFAULT 0,
    side TEXT NOT NULL DEFAULT 'OS',
    keypoints TEXT NOT NULL DEFAULT '{}',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, frame_num, trial_idx, side)
);

CREATE TABLE IF NOT EXISTS job_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type    TEXT NOT NULL,
    subject_ids TEXT NOT NULL,
    resource    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    job_id      INTEGER,
    position    INTEGER NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    error_msg   TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_subject ON jobs(subject_id);
CREATE INDEX IF NOT EXISTS idx_labels_session ON frame_labels(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_subject ON label_sessions(subject_id);
CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status, resource);
"""


def dict_factory(cursor, row):
    """Row factory that returns dicts instead of tuples."""
    fields = [col[0] for col in cursor.description]
    return dict(zip(fields, row))


def get_db() -> sqlite3.Connection:
    """Get a database connection with dict row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = dict_factory
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db_ctx():
    """Context manager for database connections."""
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _get_table_columns(conn, table_name: str) -> list[str]:
    """Get column names for a table."""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    return [row["name"] for row in cursor.fetchall()]


def _migrate_frame_labels(conn):
    """Migrate frame_labels from old thumb_x/thumb_y/index_x/index_y to keypoints JSON."""
    columns = _get_table_columns(conn, "frame_labels")
    if "thumb_x" not in columns:
        return  # Already migrated or fresh DB

    logger.info("Migrating frame_labels to JSON keypoints...")

    # Read old data
    rows = conn.execute(
        "SELECT id, thumb_x, thumb_y, index_x, index_y FROM frame_labels"
    ).fetchall()

    # Add keypoints column if missing
    if "keypoints" not in columns:
        conn.execute("ALTER TABLE frame_labels ADD COLUMN keypoints TEXT NOT NULL DEFAULT '{}'")

    # Convert each row
    for row in rows:
        kp = {}
        if row["thumb_x"] is not None and row["thumb_y"] is not None:
            kp["thumb"] = [row["thumb_x"], row["thumb_y"]]
        if row["index_x"] is not None and row["index_y"] is not None:
            kp["index"] = [row["index_x"], row["index_y"]]
        conn.execute(
            "UPDATE frame_labels SET keypoints = ? WHERE id = ?",
            (json.dumps(kp), row["id"]),
        )

    # Recreate table without old columns
    conn.execute("""
        CREATE TABLE frame_labels_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES label_sessions(id),
            frame_num INTEGER NOT NULL,
            trial_idx INTEGER NOT NULL DEFAULT 0,
            side TEXT NOT NULL DEFAULT 'OS',
            keypoints TEXT NOT NULL DEFAULT '{}',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, frame_num, trial_idx, side)
        )
    """)
    conn.execute("""
        INSERT INTO frame_labels_new (id, session_id, frame_num, trial_idx, side, keypoints, updated_at)
        SELECT id, session_id, frame_num, trial_idx, side, keypoints, updated_at FROM frame_labels
    """)
    conn.execute("DROP TABLE frame_labels")
    conn.execute("ALTER TABLE frame_labels_new RENAME TO frame_labels")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_session ON frame_labels(session_id)")

    logger.info(f"Migrated {len(rows)} label rows to JSON keypoints")


def _migrate_add_tmux_session(conn):
    """Add tmux_session column to jobs table if missing."""
    columns = _get_table_columns(conn, "jobs")
    if "tmux_session" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN tmux_session TEXT")
        logger.info("Added tmux_session column to jobs table")


def _migrate_relative_dlc_dir(conn):
    """Convert absolute dlc_dir paths to relative (subject name only)."""
    rows = conn.execute("SELECT id, dlc_dir FROM subjects WHERE dlc_dir IS NOT NULL").fetchall()
    for row in rows:
        dlc_dir = row["dlc_dir"]
        if not dlc_dir:
            continue
        # If it looks like an absolute path, extract just the last component
        p = Path(dlc_dir)
        if p.is_absolute() or "/" in dlc_dir or "\\" in dlc_dir:
            name = p.name
            conn.execute(
                "UPDATE subjects SET dlc_dir = ? WHERE id = ?",
                (name, row["id"]),
            )
            logger.info(f"Migrated dlc_dir: {dlc_dir} -> {name}")


def init_db():
    """Create tables if they don't exist, run migrations."""
    conn = get_db()

    # Check if frame_labels exists with old schema before running CREATE TABLE
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    if "frame_labels" in tables:
        _migrate_frame_labels(conn)
        conn.commit()

    if "jobs" in tables:
        _migrate_add_tmux_session(conn)
        conn.commit()

    if "subjects" in tables:
        _migrate_relative_dlc_dir(conn)
        conn.commit()

    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
