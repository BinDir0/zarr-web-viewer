from __future__ import annotations

import importlib.util
import json
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, render_template, request, send_from_directory

from .config import MediaPipeReviewConfig, load_config
from .db import (
    count_clips_by_status,
    get_clip,
    get_next_clip_by_status_after_id,
    list_clips_by_statuses,
    list_fit_jobs,
    open_db,
    save_vendor_review,
)

mediapipe_review_bp = Blueprint(
    "mediapipe_review",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/mediapipe-static",
)

_CFG: Optional[MediaPipeReviewConfig] = None
_PREPROCESSED_RANK_CACHE: Dict[tuple[str, int], Dict[str, Any]] = {}
_COMPACT_BUNDLE_CACHE: OrderedDict[tuple[str, int, int], Dict[str, Any]] = OrderedDict()
_COMPACT_BUNDLE_CACHE_MAX = 128


def _parse_start_rank(raw_value: Any) -> int:
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return 1
    return max(1, parsed)


def get_cfg() -> MediaPipeReviewConfig:
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def init_runtime() -> MediaPipeReviewConfig:
    cfg = get_cfg()
    with open_db(cfg.db_path):
        pass
    return cfg


def _dependency_status(cfg: MediaPipeReviewConfig) -> Dict[str, Any]:
    ultralytics_available = importlib.util.find_spec("ultralytics") is not None or cfg.ultralytics_repo_root.exists()
    mediapipe_available = importlib.util.find_spec("mediapipe") is not None or cfg.mediapipe_repo_root.exists()
    return {
        "python3": True,
        "ultralytics": ultralytics_available,
        "mediapipe": mediapipe_available,
        "torch": importlib.util.find_spec("torch") is not None,
        "cv2": importlib.util.find_spec("cv2") is not None,
        "pillow": importlib.util.find_spec("PIL") is not None,
        "ultralytics_repo_root": cfg.ultralytics_repo_root.exists(),
        "yolo_model_path": cfg.yolo_model_path.exists(),
        "hand_landmarker_task": cfg.hand_landmarker_task.exists(),
    }


def _preprocessed_rank_signature(conn: sqlite3.Connection, min_num_frames: int) -> tuple[int, int, int]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS n,
            COALESCE(MAX(id), 0) AS max_id,
            COALESCE(SUM(id), 0) AS sum_id
        FROM clips
        WHERE bundle_relpath IS NOT NULL
          AND review_unit = 'episode'
          AND num_frames >= ?
        """,
        (max(0, int(min_num_frames)),),
    ).fetchone()
    return int(row["n"]), int(row["max_id"]), int(row["sum_id"])


def _preprocessed_rank_ids(conn: sqlite3.Connection, cfg: MediaPipeReviewConfig) -> List[int]:
    min_num_frames = int(cfg.min_export_episode_frames)
    cache_key = (str(cfg.db_path), min_num_frames)
    signature = _preprocessed_rank_signature(conn, min_num_frames)
    cached = _PREPROCESSED_RANK_CACHE.get(cache_key)
    if cached and cached.get("signature") == signature:
        return list(cached["ids"])
    rows = conn.execute(
        """
        SELECT id
        FROM clips
        WHERE bundle_relpath IS NOT NULL
          AND review_unit = 'episode'
          AND num_frames >= ?
        ORDER BY id
        """,
        (max(0, min_num_frames),),
    ).fetchall()
    ids = [int(row["id"]) for row in rows]
    _PREPROCESSED_RANK_CACHE[cache_key] = {"signature": signature, "ids": ids}
    return ids


def _preprocessed_clip_by_rank(
    conn: sqlite3.Connection,
    cfg: MediaPipeReviewConfig,
    start_rank: int,
) -> Optional[sqlite3.Row]:
    ids = _preprocessed_rank_ids(conn, cfg)
    index = max(0, int(start_rank) - 1)
    if index >= len(ids):
        return None
    return conn.execute("SELECT * FROM clips WHERE id = ?", (ids[index],)).fetchone()


def _summary_payload(cfg: MediaPipeReviewConfig, *, start_rank: int = 1) -> Dict[str, Any]:
    with open_db(cfg.db_path) as conn:
        counts = count_clips_by_status(conn, min_num_frames=cfg.min_export_episode_frames)
        start_clip = _preprocessed_clip_by_rank(conn, cfg, start_rank)
        reviewed = list_clips_by_statuses(
            conn,
            ["reviewed_ready_for_fit", "fit_ok", "exported"],
            limit=5,
            min_num_frames=cfg.min_export_episode_frames,
        )
        queued_fit = list_fit_jobs(
            conn,
            status="queued_fit",
            limit=5,
            min_num_frames=cfg.min_export_episode_frames,
        )
    return {
        "counts": counts,
        "requested_start_rank": int(start_rank),
        "next_clip_id": int(start_clip["id"]) if start_clip else None,
        "next_clip_status": str(start_clip["status"]) if start_clip else None,
        "recent_reviewed_count": len(reviewed),
        "queued_fit_count": len(queued_fit),
        "paths": {
            "db_path": str(cfg.db_path),
            "artifacts_dir": str(cfg.artifacts_dir),
            "source_annotations_db": str(cfg.source_annotations_db),
            "ultralytics_repo_root": str(cfg.ultralytics_repo_root),
            "yolo_model_path": str(cfg.raw.get("yolo_model_path", "") or ""),
            "hand_landmarker_task": str(cfg.hand_landmarker_task),
        },
        "dependencies": _dependency_status(cfg),
    }


def _load_bundle(cfg: MediaPipeReviewConfig, bundle_relpath: str) -> Dict[str, Any]:
    bundle_path = cfg.artifact_abspath(bundle_relpath) / "bundle.json"
    with bundle_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compact_bundle_cache_key(bundle_path: Path) -> tuple[str, int, int]:
    stat = bundle_path.stat()
    return (str(bundle_path), int(stat.st_mtime_ns), int(stat.st_size))


def _load_compact_bundle(cfg: MediaPipeReviewConfig, bundle_relpath: str) -> Dict[str, Any]:
    bundle_path = cfg.artifact_abspath(bundle_relpath) / "bundle.json"
    cache_key = _compact_bundle_cache_key(bundle_path)
    cached = _COMPACT_BUNDLE_CACHE.get(cache_key)
    if cached is not None:
        _COMPACT_BUNDLE_CACHE.move_to_end(cache_key)
        return cached
    with bundle_path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)
    compact = _compact_bundle_for_review(bundle)
    _COMPACT_BUNDLE_CACHE[cache_key] = compact
    _COMPACT_BUNDLE_CACHE.move_to_end(cache_key)
    while len(_COMPACT_BUNDLE_CACHE) > _COMPACT_BUNDLE_CACHE_MAX:
        _COMPACT_BUNDLE_CACHE.popitem(last=False)
    return compact


def _compact_bundle_for_review(bundle: Dict[str, Any]) -> Dict[str, Any]:
    if int(bundle.get("bundle_version", 0)) != 3:
        raise ValueError(f"Unsupported bundle_version: {bundle.get('bundle_version')}")
    review_frames = list(bundle.get("review_frames", []))
    review_indices = {int(item["frame_idx"]) for item in review_frames}
    frames = [item for item in bundle.get("frames", []) if int(item["frame_idx"]) in review_indices]
    frame_proposals = [item for item in bundle.get("frame_proposals", []) if int(item["frame_idx"]) in review_indices]
    frames.sort(key=lambda item: int(item["frame_idx"]))
    frame_proposals.sort(key=lambda item: int(item["frame_idx"]))
    return {
        "bundle_version": 3,
        "clip_id": bundle.get("clip_id"),
        "review_unit": bundle.get("review_unit"),
        "episode_id": bundle.get("episode_id"),
        "episode_name": bundle.get("episode_name"),
        "dataset_name": bundle.get("dataset_name"),
        "clip_start": bundle.get("clip_start"),
        "clip_end": bundle.get("clip_end"),
        "num_frames": len(bundle.get("frames", [])),
        "dirty_reason": bundle.get("dirty_reason"),
        "dirty_frame_indices": list(bundle.get("dirty_frame_indices", [])),
        "detector_meta": dict(bundle.get("detector_meta", {})),
        "review_frames": review_frames,
        "frames": frames,
        "frame_proposals": frame_proposals,
    }


def _normalize_manual_bbox(raw_bbox: Any, *, frame_idx: int, side: str) -> List[float]:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise ValueError(f"Frame {frame_idx} {side}_manual_bbox_xyxy_orig must be a 4-element array")
    values: List[float] = []
    for item in raw_bbox:
        try:
            values.append(float(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Frame {frame_idx} {side}_manual_bbox_xyxy_orig must contain numbers") from exc
    x1, y1, x2, y2 = values
    if not (x2 > x1 and y2 > y1):
        raise ValueError(f"Frame {frame_idx} {side}_manual_bbox_xyxy_orig is invalid")
    return values


def _normalize_frame_review_payload_v3(payload: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
    if int(bundle.get("bundle_version", 0)) != 3:
        raise ValueError("Only bundle_version=3 can be reviewed in the new UI")
    raw_reviews = payload.get("frame_reviews", [])
    if not isinstance(raw_reviews, list):
        raise ValueError("frame_reviews must be an array")

    review_frame_lookup = {int(item["frame_idx"]): item for item in bundle.get("review_frames", [])}
    if not review_frame_lookup:
        raise ValueError("bundle has no review_frames")
    proposal_lookup = {
        int(item["frame_idx"]): {
            str(proposal["proposal_id"]): proposal for proposal in item.get("proposals", [])
        }
        for item in bundle.get("frame_proposals", [])
    }

    seen_frames = set()
    normalized: List[Dict[str, Any]] = []
    for item in raw_reviews:
        if not isinstance(item, dict):
            raise ValueError("each frame review must be an object")
        try:
            frame_idx = int(item.get("frame_idx"))
        except (TypeError, ValueError) as exc:
            raise ValueError("frame_idx must be an integer") from exc
        if frame_idx not in review_frame_lookup:
            raise ValueError(f"Unexpected frame_idx: {frame_idx}")
        if frame_idx in seen_frames:
            raise ValueError(f"Duplicate frame review: {frame_idx}")
        seen_frames.add(frame_idx)
        if not bool(item.get("confirmed", False)):
            raise ValueError(f"Frame {frame_idx} is not confirmed")

        frame_proposals = proposal_lookup.get(frame_idx, {})
        normalized_item: Dict[str, Any] = {
            "frame_idx": frame_idx,
            "confirmed": True,
        }
        for side in ("left", "right"):
            mode = str(item.get(f"{side}_mode", "absent") or "absent")
            if mode not in {"proposal", "manual_box", "absent", "unusable"}:
                raise ValueError(f"Frame {frame_idx} {side}_mode is invalid: {mode}")
            normalized_item[f"{side}_mode"] = mode
            normalized_item[f"{side}_proposal_id"] = None
            normalized_item[f"{side}_manual_bbox_xyxy_orig"] = None
            if mode == "proposal":
                proposal_id = str(item.get(f"{side}_proposal_id") or "")
                if not proposal_id:
                    raise ValueError(f"Frame {frame_idx} {side}_proposal_id is required")
                if proposal_id not in frame_proposals:
                    raise ValueError(f"Frame {frame_idx} {side}_proposal_id {proposal_id} is not visible")
                normalized_item[f"{side}_proposal_id"] = proposal_id
            elif mode == "manual_box":
                normalized_item[f"{side}_manual_bbox_xyxy_orig"] = _normalize_manual_bbox(
                    item.get(f"{side}_manual_bbox_xyxy_orig"),
                    frame_idx=frame_idx,
                    side=side,
                )
        if (
            normalized_item["left_mode"] == "proposal"
            and normalized_item["right_mode"] == "proposal"
            and normalized_item["left_proposal_id"] == normalized_item["right_proposal_id"]
        ):
            raise ValueError(f"Frame {frame_idx} left and right cannot share the same proposal")
        normalized.append(normalized_item)

    missing = sorted(set(review_frame_lookup) - seen_frames)
    if missing:
        raise ValueError(f"Missing frame reviews: {missing}")
    normalized.sort(key=lambda item: int(item["frame_idx"]))
    return {
        "review_version": "frame_review_v3",
        "frame_reviews": normalized,
    }


@mediapipe_review_bp.get("/mediapipe")
def dashboard():
    cfg = init_runtime()
    start_rank = _parse_start_rank(request.args.get("start_rank", 1))
    summary = _summary_payload(cfg, start_rank=start_rank)
    return render_template("mediapipe_dashboard.html", summary=summary, start_rank=start_rank)


@mediapipe_review_bp.get("/mediapipe/clip/<int:clip_id>")
def clip_page(clip_id: int):
    init_runtime()
    start_rank = _parse_start_rank(request.args.get("start_rank", 1))
    return render_template("mediapipe_clip_review.html", clip_id=clip_id, start_rank=start_rank)


@mediapipe_review_bp.get("/mediapipe/assets/<path:relpath>")
def asset(relpath: str):
    cfg = init_runtime()
    return send_from_directory(str(cfg.artifacts_dir), relpath, max_age=86400)


@mediapipe_review_bp.get("/api/mediapipe/summary")
def api_summary():
    cfg = init_runtime()
    start_rank = _parse_start_rank(request.args.get("start_rank", 1))
    return jsonify({"success": True, **_summary_payload(cfg, start_rank=start_rank)})


@mediapipe_review_bp.get("/api/mediapipe/clips/next")
def api_next_clip():
    cfg = init_runtime()
    start_rank = _parse_start_rank(request.args.get("start_rank", 1))
    after_clip_id = request.args.get("after_clip_id", default=None, type=int)
    with open_db(cfg.db_path) as conn:
        if after_clip_id is not None:
            row = get_next_clip_by_status_after_id(
                conn,
                "ready_for_review",
                after_clip_id=after_clip_id,
                min_num_frames=cfg.min_export_episode_frames,
            )
        else:
            row = _preprocessed_clip_by_rank(conn, cfg, start_rank)
    return jsonify(
        {
            "success": True,
            "requested_start_rank": int(start_rank),
            "clip_id": int(row["id"]) if row else None,
        }
    )


@mediapipe_review_bp.get("/api/mediapipe/clips/<int:clip_id>")
def api_clip(clip_id: int):
    cfg = init_runtime()
    with open_db(cfg.db_path) as conn:
        row = conn.execute(
            """
            SELECT id, status, bundle_relpath, review_payload_json
            FROM clips
            WHERE id = ?
            """,
            (clip_id,),
        ).fetchone()
        clip = (
            {
                "id": int(row["id"]),
                "status": str(row["status"]),
                "bundle_relpath": str(row["bundle_relpath"] or ""),
                "review_payload_json": json.loads(str(row["review_payload_json"] or "{}")),
            }
            if row is not None
            else None
        )
    if clip is None:
        return jsonify({"success": False, "message": f"Unknown clip_id: {clip_id}"}), 404
    if not clip.get("bundle_relpath"):
        return jsonify({"success": True, "clip": clip, "bundle": None})
    try:
        compact_bundle = _load_compact_bundle(cfg, clip["bundle_relpath"])
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    return jsonify({"success": True, "clip": clip, "bundle": compact_bundle})


@mediapipe_review_bp.post("/api/mediapipe/clips/<int:clip_id>/submit-review")
def api_submit_review(clip_id: int):
    cfg = init_runtime()
    payload = request.get_json(force=True)
    with open_db(cfg.db_path) as conn:
        clip = get_clip(conn, clip_id)
        if clip is None:
            return jsonify({"success": False, "message": f"Unknown clip_id: {clip_id}"}), 404
        if not clip.get("bundle_relpath"):
            return jsonify({"success": False, "message": "episode bundle is missing"}), 400
        try:
            bundle = _load_bundle(cfg, clip["bundle_relpath"])
            review_payload = _normalize_frame_review_payload_v3(payload, bundle)
        except Exception as exc:
            return jsonify({"success": False, "message": str(exc)}), 400
        save_vendor_review(
            conn,
            clip_id=clip_id,
            review_payload=review_payload,
        )
        next_row = get_next_clip_by_status_after_id(
            conn,
            "ready_for_review",
            after_clip_id=clip_id,
            min_num_frames=cfg.min_export_episode_frames,
        )
    return jsonify({"success": True, "next_clip_id": int(next_row["id"]) if next_row else None})
