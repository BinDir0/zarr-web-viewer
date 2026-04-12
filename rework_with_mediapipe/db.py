from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 15000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS clips (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        episode_id TEXT NOT NULL,
        episode_name TEXT,
        dataset_name TEXT,
        source_type TEXT NOT NULL,
        source_json TEXT NOT NULL,
        clip_start INTEGER NOT NULL,
        clip_end INTEGER NOT NULL,
        num_frames INTEGER NOT NULL,
        dirty_reason TEXT NOT NULL,
        status TEXT NOT NULL,
        bundle_relpath TEXT,
        review_payload_json TEXT,
        fit_payload_json TEXT,
        export_payload_json TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(episode_id, clip_start, clip_end, dirty_reason)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS candidate_chains (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
        chain_index INTEGER NOT NULL,
        role_hint TEXT,
        score REAL NOT NULL,
        start_frame INTEGER NOT NULL,
        end_frame INTEGER NOT NULL,
        num_frames INTEGER NOT NULL,
        preview_frame INTEGER,
        hidden INTEGER NOT NULL DEFAULT 0,
        chain_json TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(clip_id, chain_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS vendor_reviews (
        clip_id INTEGER PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
        left_choice TEXT NOT NULL,
        right_choice TEXT NOT NULL,
        merge_answers_json TEXT NOT NULL,
        review_confidence TEXT,
        review_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fit_jobs (
        clip_id INTEGER PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        fit_json TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS exports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        export_name TEXT NOT NULL,
        status TEXT NOT NULL,
        export_json TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
]


def init_db(conn: sqlite3.Connection) -> None:
    for statement in SCHEMA:
        conn.execute(statement)
    _ensure_column(conn, "vendor_reviews", "review_json", "TEXT NOT NULL DEFAULT '{}'")
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_sql: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {str(row["name"]) for row in rows}
    if column_name in existing:
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")


@contextmanager
def open_db(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        init_db(conn)
        yield conn
    finally:
        conn.close()


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _loads(payload: Optional[str], default: Any = None) -> Any:
    if not payload:
        return default
    return json.loads(payload)


def upsert_clip(
    conn: sqlite3.Connection,
    *,
    episode_id: str,
    episode_name: str,
    dataset_name: str,
    source_type: str,
    source_json: Dict[str, Any],
    clip_start: int,
    clip_end: int,
    num_frames: int,
    dirty_reason: str,
    status: str,
) -> int:
    conn.execute(
        """
        INSERT INTO clips (
            episode_id, episode_name, dataset_name, source_type, source_json,
            clip_start, clip_end, num_frames, dirty_reason, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(episode_id, clip_start, clip_end, dirty_reason) DO UPDATE SET
            episode_name = excluded.episode_name,
            dataset_name = excluded.dataset_name,
            source_type = excluded.source_type,
            source_json = excluded.source_json,
            num_frames = excluded.num_frames,
            status = excluded.status,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            episode_id,
            episode_name,
            dataset_name,
            source_type,
            _dumps(source_json),
            clip_start,
            clip_end,
            num_frames,
            dirty_reason,
            status,
        ),
    )
    row = conn.execute(
        """
        SELECT id FROM clips
        WHERE episode_id = ? AND clip_start = ? AND clip_end = ? AND dirty_reason = ?
        """,
        (episode_id, clip_start, clip_end, dirty_reason),
    ).fetchone()
    conn.commit()
    assert row is not None
    return int(row["id"])


def update_clip_status(
    conn: sqlite3.Connection,
    clip_id: int,
    *,
    status: str,
    bundle_relpath: Optional[str] = None,
    review_payload: Optional[Dict[str, Any]] = None,
    fit_payload: Optional[Dict[str, Any]] = None,
    export_payload: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    conn.execute(
        """
        UPDATE clips
        SET status = ?,
            bundle_relpath = COALESCE(?, bundle_relpath),
            review_payload_json = COALESCE(?, review_payload_json),
            fit_payload_json = COALESCE(?, fit_payload_json),
            export_payload_json = COALESCE(?, export_payload_json),
            error_message = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            status,
            bundle_relpath,
            _dumps(review_payload) if review_payload is not None else None,
            _dumps(fit_payload) if fit_payload is not None else None,
            _dumps(export_payload) if export_payload is not None else None,
            error_message,
            clip_id,
        ),
    )
    conn.commit()


def replace_candidate_chains(conn: sqlite3.Connection, clip_id: int, chains: Iterable[Dict[str, Any]]) -> None:
    conn.execute("DELETE FROM candidate_chains WHERE clip_id = ?", (clip_id,))
    for chain in chains:
        track_id = int(chain.get("track_id", chain.get("chain_index", 0)))
        conn.execute(
            """
            INSERT INTO candidate_chains (
                clip_id, chain_index, role_hint, score, start_frame, end_frame,
                num_frames, preview_frame, hidden, chain_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clip_id,
                track_id,
                chain.get("role_hint"),
                float(chain["score"]),
                int(chain["start_frame"]),
                int(chain["end_frame"]),
                int(chain["num_frames"]),
                int(chain.get("preview_frame", chain["start_frame"])),
                int(bool(chain.get("hidden", False))),
                _dumps(chain),
            ),
        )
    conn.commit()


def list_clips_by_status(conn: sqlite3.Connection, status: str, limit: int = 100) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM clips WHERE status = ? ORDER BY id LIMIT ?",
        (status, limit),
    ).fetchall()


def list_clips_by_statuses(conn: sqlite3.Connection, statuses: Sequence[str], limit: int = 100) -> List[sqlite3.Row]:
    if not statuses:
        return []
    placeholders = ",".join(["?"] * len(statuses))
    return conn.execute(
        f"SELECT * FROM clips WHERE status IN ({placeholders}) ORDER BY id LIMIT ?",
        list(statuses) + [limit],
    ).fetchall()


def count_clips_by_status(conn: sqlite3.Connection) -> Dict[str, int]:
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM clips GROUP BY status").fetchall()
    return {str(row["status"]): int(row["n"]) for row in rows}


def get_clip(conn: sqlite3.Connection, clip_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    if row is None:
        return None
    clip = dict(row)
    clip["source_json"] = _loads(clip["source_json"], {})
    clip["review_payload_json"] = _loads(clip["review_payload_json"], {})
    clip["fit_payload_json"] = _loads(clip["fit_payload_json"], {})
    clip["export_payload_json"] = _loads(clip["export_payload_json"], {})
    chains = conn.execute(
        "SELECT * FROM candidate_chains WHERE clip_id = ? ORDER BY chain_index",
        (clip_id,),
    ).fetchall()
    clip["tracks"] = [
        {
            **dict(chain),
            "track_id": int(chain["chain_index"]),
            "track_json": _loads(chain["chain_json"], {}),
        }
        for chain in chains
    ]
    review = conn.execute("SELECT * FROM vendor_reviews WHERE clip_id = ?", (clip_id,)).fetchone()
    clip["vendor_review"] = dict(review) if review is not None else None
    if clip["vendor_review"] is not None:
        clip["vendor_review"]["merge_answers_json"] = _loads(clip["vendor_review"]["merge_answers_json"], [])
        clip["vendor_review"]["review_json"] = _loads(clip["vendor_review"].get("review_json"), {})
    return clip


def save_vendor_review(
    conn: sqlite3.Connection,
    clip_id: int,
    review_payload: Dict[str, Any],
) -> None:
    left_track_ids = [int(item) for item in review_payload.get("left_track_ids", [])]
    right_track_ids = [int(item) for item in review_payload.get("right_track_ids", [])]
    left_missing_box = bool(review_payload.get("left_missing_box", False))
    right_missing_box = bool(review_payload.get("right_missing_box", False))
    should_exclude = left_missing_box or right_missing_box
    payload = {
        "left_track_ids": left_track_ids,
        "right_track_ids": right_track_ids,
        "left_missing_box": left_missing_box,
        "right_missing_box": right_missing_box,
    }
    conn.execute(
        """
        INSERT INTO vendor_reviews (clip_id, left_choice, right_choice, merge_answers_json, review_confidence, review_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(clip_id) DO UPDATE SET
            left_choice = excluded.left_choice,
            right_choice = excluded.right_choice,
            merge_answers_json = excluded.merge_answers_json,
            review_confidence = excluded.review_confidence,
            review_json = excluded.review_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            clip_id,
            _dumps(left_track_ids),
            _dumps(right_track_ids),
            "[]",
            "missing_box" if should_exclude else "normal",
            _dumps(payload),
        ),
    )
    conn.execute(
        """
        UPDATE clips
        SET status = ?,
            review_payload_json = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        ("qa_needed_missing_box" if should_exclude else "reviewed_ready_for_fit", _dumps(payload), clip_id),
    )
    if should_exclude:
        conn.execute(
            """
            INSERT INTO fit_jobs (clip_id, status, fit_json)
            VALUES (?, 'qa_needed_missing_box', ?)
            ON CONFLICT(clip_id) DO UPDATE SET
                status = 'qa_needed_missing_box',
                fit_json = excluded.fit_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                clip_id,
                _dumps({"status": "qa_needed_missing_box", "message": "Reviewer marked at least one side as missing box."}),
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO fit_jobs (clip_id, status)
            VALUES (?, 'queued_fit')
            ON CONFLICT(clip_id) DO UPDATE SET
                status = 'queued_fit',
                updated_at = CURRENT_TIMESTAMP
            """,
            (clip_id,),
        )
    conn.commit()


def list_fit_jobs(conn: sqlite3.Connection, status: str = "queued_fit", limit: int = 20) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM fit_jobs WHERE status = ? ORDER BY updated_at LIMIT ?",
        (status, limit),
    ).fetchall()


def save_fit_result(conn: sqlite3.Connection, clip_id: int, status: str, fit_json: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO fit_jobs (clip_id, status, fit_json)
        VALUES (?, ?, ?)
        ON CONFLICT(clip_id) DO UPDATE SET
            status = excluded.status,
            fit_json = excluded.fit_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (clip_id, status, _dumps(fit_json)),
    )
    if status.startswith("fit_ok"):
        clip_status = "fit_ok"
    elif status.startswith("qa_needed"):
        clip_status = status
    else:
        clip_status = "fit_failed"
    conn.execute(
        """
        UPDATE clips
        SET status = ?, fit_payload_json = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (clip_status, _dumps(fit_json), clip_id),
    )
    conn.commit()


def record_export(conn: sqlite3.Connection, export_name: str, export_json: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO exports (export_name, status, export_json)
        VALUES (?, ?, ?)
        """,
        (export_name, "ok", _dumps(export_json)),
    )
    conn.commit()
