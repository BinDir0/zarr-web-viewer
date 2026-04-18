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
    init_db(conn)
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
        review_unit TEXT NOT NULL DEFAULT 'clip',
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
    _ensure_column(conn, "clips", "review_unit", "TEXT NOT NULL DEFAULT 'clip'")
    _backfill_clip_review_units(conn)
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
        yield conn
    finally:
        conn.close()


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _loads(payload: Optional[str], default: Any = None) -> Any:
    if not payload:
        return default
    return json.loads(payload)


def _infer_review_unit(
    *,
    clip_start: int,
    clip_end: int,
    num_frames: int,
    source_json: Optional[str] = None,
    source_payload: Optional[Dict[str, Any]] = None,
) -> str:
    payload = source_payload
    if payload is None and source_json:
        try:
            payload = _loads(source_json, {})
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
    payload = payload or {}
    episode = payload.get("episode") or {}
    try:
        episode_num_frames = int(episode.get("num_frames", 0))
    except (TypeError, ValueError):
        episode_num_frames = 0
    try:
        payload_clip_start = int(payload.get("clip_start", -1))
    except (TypeError, ValueError):
        payload_clip_start = -1
    try:
        payload_clip_end = int(payload.get("clip_end", -1))
    except (TypeError, ValueError):
        payload_clip_end = -1
    if (
        int(clip_start) == 0
        and int(clip_end) > 0
        and int(num_frames) == int(clip_end)
        and episode_num_frames > 0
        and int(clip_end) == episode_num_frames
        and payload_clip_start == 0
        and payload_clip_end == episode_num_frames
    ):
        return "episode"
    return "clip"


def _episode_review_predicate(table_alias: Optional[str] = None) -> str:
    prefix = f"{table_alias}." if table_alias else ""
    return f"{prefix}review_unit = 'episode'"


def _backfill_clip_review_units(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT id, clip_start, clip_end, num_frames, source_json, review_unit
        FROM clips
        """
    ).fetchall()
    updates: List[tuple[str, int]] = []
    for row in rows:
        inferred = _infer_review_unit(
            clip_start=int(row["clip_start"]),
            clip_end=int(row["clip_end"]),
            num_frames=int(row["num_frames"]),
            source_json=str(row["source_json"] or ""),
        )
        if str(row["review_unit"] or "clip") != inferred:
            updates.append((inferred, int(row["id"])))
    if updates:
        conn.executemany("UPDATE clips SET review_unit = ? WHERE id = ?", updates)


def _episode_definition_changed(
    row: sqlite3.Row,
    *,
    source_json: Dict[str, Any],
    clip_start: int,
    clip_end: int,
    num_frames: int,
    dirty_reason: str,
) -> bool:
    previous_source = str(row["source_json"] or "")
    current_source = _dumps(source_json)
    return any(
        [
            int(row["clip_start"]) != int(clip_start),
            int(row["clip_end"]) != int(clip_end),
            int(row["num_frames"]) != int(num_frames),
            str(row["dirty_reason"] or "") != str(dirty_reason),
            previous_source != current_source,
        ]
    )


def _reset_clip_runtime_state(conn: sqlite3.Connection, clip_id: int, *, next_status: str) -> None:
    conn.execute("DELETE FROM candidate_chains WHERE clip_id = ?", (clip_id,))
    conn.execute("DELETE FROM vendor_reviews WHERE clip_id = ?", (clip_id,))
    conn.execute("DELETE FROM fit_jobs WHERE clip_id = ?", (clip_id,))
    conn.execute(
        """
        UPDATE clips
        SET status = ?,
            bundle_relpath = NULL,
            review_payload_json = NULL,
            fit_payload_json = NULL,
            export_payload_json = NULL,
            error_message = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (next_status, clip_id),
    )


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
    review_unit = _infer_review_unit(
        clip_start=clip_start,
        clip_end=clip_end,
        num_frames=num_frames,
        source_payload=source_json,
    )
    if review_unit == "episode":
        existing = conn.execute(
            """
            SELECT *
            FROM clips
            WHERE episode_id = ? AND review_unit = 'episode'
            ORDER BY id
            LIMIT 1
            """,
            (episode_id,),
        ).fetchone()
        if existing is not None:
            clip_id = int(existing["id"])
            definition_changed = _episode_definition_changed(
                existing,
                source_json=source_json,
                clip_start=clip_start,
                clip_end=clip_end,
                num_frames=num_frames,
                dirty_reason=dirty_reason,
            )
            conn.execute(
                """
                UPDATE clips
                SET episode_name = ?,
                    dataset_name = ?,
                    source_type = ?,
                    source_json = ?,
                    clip_start = ?,
                    clip_end = ?,
                    num_frames = ?,
                    review_unit = ?,
                    dirty_reason = ?,
                    status = CASE
                        WHEN clips.status IN ('queued_preprocess', 'preprocessing', 'failed')
                            THEN ?
                        ELSE clips.status
                    END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    episode_name,
                    dataset_name,
                    source_type,
                    _dumps(source_json),
                    clip_start,
                    clip_end,
                    num_frames,
                    review_unit,
                    dirty_reason,
                    status,
                    clip_id,
                ),
            )
            if definition_changed:
                _reset_clip_runtime_state(conn, clip_id, next_status=status)
            conn.commit()
            return clip_id
    conn.execute(
        """
        INSERT INTO clips (
            episode_id, episode_name, dataset_name, source_type, source_json,
            clip_start, clip_end, num_frames, review_unit, dirty_reason, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(episode_id, clip_start, clip_end, dirty_reason) DO UPDATE SET
            episode_name = excluded.episode_name,
            dataset_name = excluded.dataset_name,
            source_type = excluded.source_type,
            source_json = excluded.source_json,
            num_frames = excluded.num_frames,
            review_unit = excluded.review_unit,
            status = CASE
                WHEN clips.status IN ('queued_preprocess', 'preprocessing', 'failed')
                    THEN excluded.status
                ELSE clips.status
            END,
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
            review_unit,
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
        f"SELECT * FROM clips WHERE status = ? AND {_episode_review_predicate()} ORDER BY id LIMIT ?",
        (status, limit),
    ).fetchall()


def get_clip_by_status_offset(conn: sqlite3.Connection, status: str, offset: int = 0) -> Optional[sqlite3.Row]:
    return conn.execute(
        f"SELECT * FROM clips WHERE status = ? AND {_episode_review_predicate()} ORDER BY id LIMIT 1 OFFSET ?",
        (status, max(0, int(offset))),
    ).fetchone()


def get_preprocessed_clip_by_offset(conn: sqlite3.Connection, offset: int = 0) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM clips
        WHERE bundle_relpath IS NOT NULL
          AND review_unit = 'episode'
        ORDER BY id
        LIMIT 1 OFFSET ?
        """,
        (max(0, int(offset)),),
    ).fetchone()


def get_next_clip_by_status_after_id(conn: sqlite3.Connection, status: str, after_clip_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        f"SELECT * FROM clips WHERE status = ? AND id > ? AND {_episode_review_predicate()} ORDER BY id LIMIT 1",
        (status, int(after_clip_id)),
    ).fetchone()


def list_clips_by_statuses(conn: sqlite3.Connection, statuses: Sequence[str], limit: int = 100) -> List[sqlite3.Row]:
    if not statuses:
        return []
    placeholders = ",".join(["?"] * len(statuses))
    return conn.execute(
        f"SELECT * FROM clips WHERE status IN ({placeholders}) AND {_episode_review_predicate()} ORDER BY id LIMIT ?",
        list(statuses) + [limit],
    ).fetchall()


def count_clips_by_status(conn: sqlite3.Connection) -> Dict[str, int]:
    rows = conn.execute(
        f"SELECT status, COUNT(*) AS n FROM clips WHERE {_episode_review_predicate()} GROUP BY status"
    ).fetchall()
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
    review_version = str(review_payload.get("review_version") or "")
    if review_version == "frame_review_v2":
        frame_reviews = list(review_payload.get("frame_reviews", []))
        left_track_ids = sorted(
            {
                int(item["left_track_id"])
                for item in frame_reviews
                if item.get("left_mode") == "track" and item.get("left_track_id") is not None
            }
        )
        right_track_ids = sorted(
            {
                int(item["right_track_id"])
                for item in frame_reviews
                if item.get("right_mode") == "track" and item.get("right_track_id") is not None
            }
        )
        payload = {
            "review_version": "frame_review_v2",
            "frame_reviews": frame_reviews,
        }
        review_confidence = "frame_recovery_v2"
        should_exclude = False
    elif review_version == "keyframe_v1":
        keyframe_reviews = list(review_payload.get("keyframe_reviews", []))
        left_track_ids = sorted(
            {
                int(item["left_track_id"])
                for item in keyframe_reviews
                if item.get("left_track_id") is not None
            }
        )
        right_track_ids = sorted(
            {
                int(item["right_track_id"])
                for item in keyframe_reviews
                if item.get("right_track_id") is not None
            }
        )
        left_missing_box = any(bool(item.get("left_missing_box", False)) for item in keyframe_reviews)
        right_missing_box = any(bool(item.get("right_missing_box", False)) for item in keyframe_reviews)
        has_missing_box = left_missing_box or right_missing_box
        payload = {
            "review_version": "keyframe_v1",
            "keyframe_reviews": keyframe_reviews,
        }
        review_confidence = "partial_missing_box" if has_missing_box else "keyframe_v1"
        should_exclude = False
    else:
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
        review_confidence = "missing_box" if should_exclude else "normal"
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
            review_confidence,
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
        f"""
        SELECT fit_jobs.*
        FROM fit_jobs
        INNER JOIN clips ON clips.id = fit_jobs.clip_id
        WHERE fit_jobs.status = ?
          AND {_episode_review_predicate('clips')}
        ORDER BY fit_jobs.updated_at DESC, fit_jobs.clip_id DESC
        LIMIT ?
        """,
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
    elif status == "fit_dropped":
        clip_status = "fit_dropped"
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
