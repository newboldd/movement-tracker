"""SQLite database schema and connection helpers.

The schema is a subset of the full Movement Tracker's, on the same
``dlc_app.db`` filename and with identical column definitions for the
tables both apps use.  That is deliberate: pointing this fork at an
existing Movement Tracker data directory must not migrate, rewrite or
lose anything.  Tables this fork doesn't use are left untouched, and the
few migrations below exist only to bring an *older* database up to the
shape the code here expects.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager

from .config import DB_PATH

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    stage TEXT NOT NULL DEFAULT 'created',
    iteration INTEGER NOT NULL DEFAULT 1,
    camera_mode TEXT DEFAULT 'stereo',
    camera_name TEXT,
    no_face_videos TEXT,
    dlc_dir TEXT,
    video_pattern TEXT,
    notes TEXT,
    hand_size_left REAL,
    hand_size_right REAL,
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
    pid INTEGER,
    error_msg TEXT,
    epoch_info TEXT,
    params_json TEXT,
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
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type            TEXT NOT NULL,
    subject_ids         TEXT NOT NULL,
    resource            TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'queued',
    job_id              INTEGER,
    position            INTEGER NOT NULL,
    execution_target    TEXT NOT NULL DEFAULT 'local-cpu',
    progress_pct        REAL DEFAULT 0,
    extra_params_json   TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at          TIMESTAMP,
    finished_at         TIMESTAMP,
    error_msg           TEXT
);

CREATE TABLE IF NOT EXISTS mp_crop_boxes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    trial_idx INTEGER NOT NULL,
    camera_name TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT 'default',
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject_id, trial_idx, camera_name, model_name)
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    trial_label TEXT NOT NULL,
    source_path TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    camera_name TEXT,
    frame_offset INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject_id, trial_label, camera_name)
);

CREATE INDEX IF NOT EXISTS idx_jobs_subject ON jobs(subject_id);
CREATE INDEX IF NOT EXISTS idx_labels_session ON frame_labels(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_subject ON label_sessions(subject_id);
CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status, resource);
CREATE INDEX IF NOT EXISTS idx_job_queue_status_target
    ON job_queue(status, execution_target);
CREATE INDEX IF NOT EXISTS idx_mp_crop_boxes_model
    ON mp_crop_boxes(subject_id, trial_idx, model_name);
CREATE INDEX IF NOT EXISTS idx_segments_subject ON segments(subject_id);
"""


def dict_factory(cursor, row):
    """Row factory that returns dicts instead of tuples."""
    fields = [col[0] for col in cursor.description]
    return dict(zip(fields, row))


def get_db() -> sqlite3.Connection:
    """Get a database connection with dict row factory."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = dict_factory
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
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


def _table_names(conn) -> set[str]:
    return {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


def _columns(conn, table: str) -> list[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _add_missing_columns(conn, table: str, columns: dict[str, str]):
    """ALTER TABLE ... ADD COLUMN for every column not already present."""
    existing = _columns(conn, table)
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
            logger.info("Added %s.%s", table, name)


def _migrate_frame_labels(conn):
    """Convert pre-JSON frame_labels (thumb_x/thumb_y/index_x/index_y).

    Only databases written by very early Movement Tracker builds have
    those columns; everything since stores a ``keypoints`` JSON blob that
    can hold an arbitrary bodypart set.
    """
    columns = _columns(conn, "frame_labels")
    if "thumb_x" not in columns:
        return  # Already migrated or fresh DB

    logger.info("Migrating frame_labels to JSON keypoints...")
    rows = conn.execute(
        "SELECT id, thumb_x, thumb_y, index_x, index_y FROM frame_labels"
    ).fetchall()

    if "keypoints" not in columns:
        conn.execute(
            "ALTER TABLE frame_labels ADD COLUMN keypoints TEXT NOT NULL DEFAULT '{}'")

    for row in rows:
        kp = {}
        if row["thumb_x"] is not None and row["thumb_y"] is not None:
            kp["thumb"] = [row["thumb_x"], row["thumb_y"]]
        if row["index_x"] is not None and row["index_y"] is not None:
            kp["index"] = [row["index_x"], row["index_y"]]
        conn.execute("UPDATE frame_labels SET keypoints = ? WHERE id = ?",
                     (json.dumps(kp), row["id"]))

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
        INSERT INTO frame_labels_new
            (id, session_id, frame_num, trial_idx, side, keypoints, updated_at)
        SELECT id, session_id, frame_num, trial_idx, side, keypoints, updated_at
        FROM frame_labels
    """)
    conn.execute("DROP TABLE frame_labels")
    conn.execute("ALTER TABLE frame_labels_new RENAME TO frame_labels")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_labels_session ON frame_labels(session_id)")

    logger.info("Migrated %d label rows to JSON keypoints", len(rows))


def _migrate_crop_box_model(conn):
    """Add mp_crop_boxes.model_name and widen its UNIQUE constraint.

    Early builds keyed a crop box on (subject, trial, camera) only.  The
    per-model column lets the saved MediaPipe bbox and any future
    per-model box coexist for the same trial.
    """
    columns = _columns(conn, "mp_crop_boxes")
    if not columns or "model_name" in columns:
        return

    logger.info("Migrating mp_crop_boxes to include model_name...")
    conn.execute("ALTER TABLE mp_crop_boxes RENAME TO mp_crop_boxes_old")
    conn.execute("""
        CREATE TABLE mp_crop_boxes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER NOT NULL REFERENCES subjects(id),
            trial_idx INTEGER NOT NULL,
            camera_name TEXT NOT NULL,
            model_name TEXT NOT NULL DEFAULT 'default',
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(subject_id, trial_idx, camera_name, model_name)
        )
    """)
    conn.execute("""
        INSERT INTO mp_crop_boxes
            (id, subject_id, trial_idx, camera_name, model_name,
             x1, y1, x2, y2, updated_at)
        SELECT id, subject_id, trial_idx, camera_name, 'default',
               x1, y1, x2, y2, updated_at
        FROM mp_crop_boxes_old
    """)
    conn.execute("DROP TABLE mp_crop_boxes_old")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mp_crop_boxes_model "
                 "ON mp_crop_boxes(subject_id, trial_idx, model_name)")


def init_db():
    """Create tables if they don't exist, then run migrations."""
    conn = get_db()
    try:
        tables = _table_names(conn)

        # Migrations that reshape an existing table must run before the
        # CREATE TABLE IF NOT EXISTS pass, which would otherwise be a
        # silent no-op against the old shape.
        if "frame_labels" in tables:
            _migrate_frame_labels(conn)
            conn.commit()

        if "mp_crop_boxes" in tables:
            _migrate_crop_box_model(conn)
            conn.commit()

        if "subjects" in tables:
            _add_missing_columns(conn, "subjects", {
                "camera_mode": "TEXT DEFAULT 'stereo'",
                "camera_name": "TEXT",
                "no_face_videos": "TEXT",
                "notes": "TEXT",
                "hand_size_left": "REAL",
                "hand_size_right": "REAL",
            })
            conn.commit()

        if "jobs" in tables:
            _add_missing_columns(conn, "jobs", {
                "epoch_info": "TEXT",
                "params_json": "TEXT",
            })
            conn.commit()

        if "job_queue" in tables:
            _add_missing_columns(conn, "job_queue", {
                "execution_target": "TEXT NOT NULL DEFAULT 'local-cpu'",
                "progress_pct": "REAL DEFAULT 0",
                "extra_params_json": "TEXT",
                "error_msg": "TEXT",
            })
            conn.commit()

        if "segments" in tables:
            _add_missing_columns(conn, "segments", {
                "camera_name": "TEXT",
                "frame_offset": "INTEGER DEFAULT 0",
            })
            conn.commit()

        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
