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
    camera_mode TEXT DEFAULT 'stereo',
    camera_name TEXT,
    no_face_videos TEXT,
    dlc_dir TEXT,
    video_pattern TEXT,
    diagnosis TEXT DEFAULT 'Control',
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
    epoch_info TEXT,
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
    execution_target    TEXT NOT NULL DEFAULT 'remote',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at          TIMESTAMP,
    finished_at         TIMESTAMP,
    error_msg           TEXT
);

CREATE TABLE IF NOT EXISTS subject_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    event_type TEXT NOT NULL,
    frame_num INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject_id, event_type, frame_num)
);

CREATE INDEX IF NOT EXISTS idx_jobs_subject ON jobs(subject_id);
CREATE INDEX IF NOT EXISTS idx_labels_session ON frame_labels(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_subject ON label_sessions(subject_id);
CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status, resource);
CREATE INDEX IF NOT EXISTS idx_job_queue_status_target ON job_queue(status, execution_target);
CREATE INDEX IF NOT EXISTS idx_subject_events ON subject_events(subject_id, event_type);

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

CREATE INDEX IF NOT EXISTS idx_segments_subject ON segments(subject_id);

CREATE TABLE IF NOT EXISTS camera_setups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    mode TEXT NOT NULL DEFAULT 'stereo',
    camera_count INTEGER NOT NULL DEFAULT 2,
    camera_names TEXT NOT NULL DEFAULT '[]',
    calibration_path TEXT,
    checkerboard_rows INTEGER,
    checkerboard_cols INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mp_crop_boxes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    trial_idx INTEGER NOT NULL,
    camera_name TEXT NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject_id, trial_idx, camera_name)
);

CREATE INDEX IF NOT EXISTS idx_mp_crop_boxes ON mp_crop_boxes(subject_id, trial_idx);

CREATE TABLE IF NOT EXISTS blur_specs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    trial_idx INTEGER NOT NULL,
    spot_type TEXT NOT NULL DEFAULT 'face',
    x REAL NOT NULL,
    y REAL NOT NULL,
    radius REAL NOT NULL,
    width REAL,
    height REAL,
    offset_x REAL DEFAULT 0,
    offset_y REAL DEFAULT 0,
    frame_start INTEGER NOT NULL,
    frame_end INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_blur_specs ON blur_specs(subject_id, trial_idx);

CREATE TABLE IF NOT EXISTS blur_hand_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    trial_idx INTEGER NOT NULL,
    hand_mask_enabled INTEGER NOT NULL DEFAULT 1,
    hand_mask_radius INTEGER NOT NULL DEFAULT 30,
    hand_frame_start INTEGER,
    hand_frame_end INTEGER,
    UNIQUE(subject_id, trial_idx)
);
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


def _migrate_add_no_face_videos(conn):
    """Add no_face_videos column to subjects table if missing."""
    columns = _get_table_columns(conn, "subjects")
    if "no_face_videos" not in columns:
        conn.execute("ALTER TABLE subjects ADD COLUMN no_face_videos TEXT")
        logger.info("Added no_face_videos column to subjects table")


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


def _migrate_add_execution_target(conn):
    """Add execution_target column to job_queue table if missing."""
    columns = _get_table_columns(conn, "job_queue")
    if "execution_target" not in columns:
        conn.execute("ALTER TABLE job_queue ADD COLUMN execution_target TEXT NOT NULL DEFAULT 'remote'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_job_queue_status_target ON job_queue(status, execution_target)")
        logger.info("Added execution_target column to job_queue table")


def _migrate_add_diagnosis(conn):
    """Add diagnosis column to subjects table if missing."""
    columns = _get_table_columns(conn, "subjects")
    if "diagnosis" not in columns:
        conn.execute("ALTER TABLE subjects ADD COLUMN diagnosis TEXT DEFAULT 'Control'")
        logger.info("Added diagnosis column to subjects table")


def _migrate_add_subject_events(conn):
    """Create subject_events table if missing."""
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "subject_events" not in tables:
        conn.execute("""
            CREATE TABLE subject_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL REFERENCES subjects(id),
                event_type TEXT NOT NULL,
                frame_num INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject_id, event_type, frame_num)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_subject_events ON subject_events(subject_id, event_type)")
        logger.info("Created subject_events table")


def _migrate_add_epoch_info(conn):
    """Add epoch_info column to jobs table if missing."""
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
    if "epoch_info" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN epoch_info TEXT")
        logger.info("Added epoch_info column to jobs")


def _migrate_add_camera_mode(conn):
    """Add camera_mode column to subjects table if missing."""
    columns = _get_table_columns(conn, "subjects")
    if "camera_mode" not in columns:
        conn.execute("ALTER TABLE subjects ADD COLUMN camera_mode TEXT DEFAULT 'stereo'")
        logger.info("Added camera_mode column to subjects table")


def _migrate_add_frame_offset(conn):
    """Add frame_offset column to segments table if missing."""
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "segments" in tables:
        columns = _get_table_columns(conn, "segments")
        if "frame_offset" not in columns:
            conn.execute("ALTER TABLE segments ADD COLUMN frame_offset INTEGER DEFAULT 0")
            logger.info("Added frame_offset column to segments table")


def _migrate_add_segments(conn):
    """Create segments table if missing."""
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "segments" not in tables:
        conn.execute("""
            CREATE TABLE segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL REFERENCES subjects(id),
                trial_label TEXT NOT NULL,
                source_path TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                camera_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject_id, trial_label, camera_name)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_subject ON segments(subject_id)")
        logger.info("Created segments table")


def _migrate_add_camera_setups(conn):
    """Create camera_setups table if missing."""
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "camera_setups" not in tables:
        conn.execute("""
            CREATE TABLE camera_setups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                mode TEXT NOT NULL DEFAULT 'stereo',
                camera_count INTEGER NOT NULL DEFAULT 2,
                camera_names TEXT NOT NULL DEFAULT '[]',
                calibration_path TEXT,
                checkerboard_rows INTEGER,
                checkerboard_cols INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created camera_setups table")


def _migrate_add_blur_specs(conn):
    """Create blur_specs and blur_hand_settings tables if missing."""
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "blur_specs" not in tables:
        conn.execute("""
            CREATE TABLE blur_specs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL REFERENCES subjects(id),
                trial_idx INTEGER NOT NULL,
                spot_type TEXT NOT NULL DEFAULT 'face',
                x REAL NOT NULL,
                y REAL NOT NULL,
                radius REAL NOT NULL,
                frame_start INTEGER NOT NULL,
                frame_end INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_blur_specs ON blur_specs(subject_id, trial_idx)")
        logger.info("Created blur_specs table")
    else:
        # Add width/height/offset columns if missing
        columns = _get_table_columns(conn, "blur_specs")
        if "width" not in columns:
            conn.execute("ALTER TABLE blur_specs ADD COLUMN width REAL")
            conn.execute("ALTER TABLE blur_specs ADD COLUMN height REAL")
            conn.execute("ALTER TABLE blur_specs ADD COLUMN offset_x REAL DEFAULT 0")
            conn.execute("ALTER TABLE blur_specs ADD COLUMN offset_y REAL DEFAULT 0")
            logger.info("Added width/height/offset columns to blur_specs")
    if "blur_hand_settings" not in tables:
        conn.execute("""
            CREATE TABLE blur_hand_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL REFERENCES subjects(id),
                trial_idx INTEGER NOT NULL,
                hand_mask_enabled INTEGER NOT NULL DEFAULT 1,
                hand_mask_radius INTEGER NOT NULL DEFAULT 30,
                UNIQUE(subject_id, trial_idx)
            )
        """)
        logger.info("Created blur_hand_settings table")
    else:
        # Add frame range columns if missing
        columns = _get_table_columns(conn, "blur_hand_settings")
        if "hand_frame_start" not in columns:
            conn.execute("ALTER TABLE blur_hand_settings ADD COLUMN hand_frame_start INTEGER")
            conn.execute("ALTER TABLE blur_hand_settings ADD COLUMN hand_frame_end INTEGER")
            logger.info("Added hand_frame_start/end columns to blur_hand_settings")


def _migrate_add_mp_crop_boxes(conn):
    """Create mp_crop_boxes table if missing."""
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "mp_crop_boxes" not in tables:
        conn.execute("""
            CREATE TABLE mp_crop_boxes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL REFERENCES subjects(id),
                trial_idx INTEGER NOT NULL,
                camera_name TEXT NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject_id, trial_idx, camera_name)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mp_crop_boxes ON mp_crop_boxes(subject_id, trial_idx)")
        logger.info("Created mp_crop_boxes table")


def _migrate_add_face_detections(conn):
    """Create face_detections table if missing, add side column if needed."""
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "face_detections" not in tables:
        conn.execute("""
            CREATE TABLE face_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL REFERENCES subjects(id),
                trial_idx INTEGER NOT NULL,
                frame_num INTEGER NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                confidence REAL DEFAULT 1.0,
                side TEXT DEFAULT 'full'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_face_detections ON face_detections(subject_id, trial_idx)")
        logger.info("Created face_detections table")
    else:
        # Add side column if missing (migration for existing tables)
        columns = _get_table_columns(conn, "face_detections")
        if "side" not in columns:
            conn.execute("ALTER TABLE face_detections ADD COLUMN side TEXT DEFAULT 'full'")
            logger.info("Added side column to face_detections table")


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
        _migrate_add_no_face_videos(conn)
        _migrate_relative_dlc_dir(conn)
        _migrate_add_diagnosis(conn)
        _migrate_add_camera_mode(conn)
        conn.commit()

    if "job_queue" in tables:
        _migrate_add_execution_target(conn)
        conn.commit()

    _migrate_add_subject_events(conn)
    conn.commit()

    if "jobs" in tables:
        _migrate_add_epoch_info(conn)
        conn.commit()

    _migrate_add_camera_setups(conn)
    conn.commit()

    _migrate_add_segments(conn)
    conn.commit()

    _migrate_add_frame_offset(conn)
    conn.commit()

    _migrate_add_mp_crop_boxes(conn)
    conn.commit()

    _migrate_add_blur_specs(conn)
    conn.commit()

    _migrate_add_face_detections(conn)
    conn.commit()

    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
