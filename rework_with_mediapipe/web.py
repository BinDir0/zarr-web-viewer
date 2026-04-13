from __future__ import annotations

import importlib.util
import json
from typing import Any, Dict, List, Optional

from flask import Blueprint, abort, jsonify, render_template, request, send_from_directory

from .config import MediaPipeReviewConfig, load_config
from .db import (
    count_clips_by_status,
    get_clip,
    list_clips_by_status,
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
    return {
        "python3": True,
        "mediapipe": importlib.util.find_spec("mediapipe") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "cv2": importlib.util.find_spec("cv2") is not None,
        "pillow": importlib.util.find_spec("PIL") is not None,
        "hand_landmarker_task": cfg.hand_landmarker_task.exists(),
        "mano_models_root": cfg.mano_models_root.exists(),
    }


def _summary_payload(cfg: MediaPipeReviewConfig) -> Dict[str, Any]:
    with open_db(cfg.db_path) as conn:
        counts = count_clips_by_status(conn)
        next_ready = list_clips_by_status(conn, "ready_for_review", limit=1)
        reviewed = list_clips_by_statuses(conn, ["reviewed_ready_for_fit", "fit_ok", "exported"], limit=5)
        queued_fit = list_fit_jobs(conn, status="queued_fit", limit=5)
    return {
        "counts": counts,
        "next_clip_id": int(next_ready[0]["id"]) if next_ready else None,
        "recent_reviewed_count": len(reviewed),
        "queued_fit_count": len(queued_fit),
        "paths": {
            "db_path": str(cfg.db_path),
            "artifacts_dir": str(cfg.artifacts_dir),
            "source_annotations_db": str(cfg.source_annotations_db),
            "hand_landmarker_task": str(cfg.hand_landmarker_task),
            "mano_models_root": str(cfg.mano_models_root),
        },
        "dependencies": _dependency_status(cfg),
    }


def _load_bundle(cfg: MediaPipeReviewConfig, bundle_relpath: str) -> Dict[str, Any]:
    bundle_path = cfg.artifact_abspath(bundle_relpath) / "bundle.json"
    with bundle_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_legacy_review_payload(payload: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
    left_track_ids = payload.get("left_track_ids", [])
    right_track_ids = payload.get("right_track_ids", [])
    left_missing_box = bool(payload.get("left_missing_box", False))
    right_missing_box = bool(payload.get("right_missing_box", False))
    if not isinstance(left_track_ids, list) or not isinstance(right_track_ids, list):
        raise ValueError("left_track_ids and right_track_ids must be arrays")

    def _int_set(raw_items: List[Any]) -> List[int]:
        values = []
        for item in raw_items:
            try:
                values.append(int(item))
            except (TypeError, ValueError) as exc:
                raise ValueError("track ids must be integers") from exc
        return sorted(set(values))

    left_ids = _int_set(left_track_ids)
    right_ids = _int_set(right_track_ids)
    overlap = sorted(set(left_ids) & set(right_ids))
    if overlap:
        raise ValueError("Left and right cannot share the same track")
    valid_track_ids = {int(item["track_id"]) for item in bundle.get("tracks", [])}
    invalid = [item for item in left_ids + right_ids if item not in valid_track_ids]
    if invalid:
        raise ValueError(f"Unknown track ids: {sorted(set(invalid))}")
    return {
        "left_track_ids": left_ids,
        "right_track_ids": right_ids,
        "left_missing_box": left_missing_box,
        "right_missing_box": right_missing_box,
    }


def _normalize_keyframe_review_payload(payload: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
    raw_keyframes = payload.get("keyframe_reviews", [])
    if not isinstance(raw_keyframes, list):
        raise ValueError("keyframe_reviews must be an array")

    expected_keyframes = list(bundle.get("keyframes", []))
    if not expected_keyframes:
        raise ValueError("bundle has no keyframes")
    expected_by_frame = {int(item["frame_idx"]): item for item in expected_keyframes}
    if len(expected_by_frame) != len(expected_keyframes):
        raise ValueError("bundle keyframes contain duplicate frame_idx")
    seen_frames = set()
    normalized: List[Dict[str, Any]] = []
    frame_tracks = {
        int(item["frame_idx"]): {int(track["track_id"]) for track in item.get("tracks", [])}
        for item in bundle.get("frame_tracks", [])
    }
    for item in raw_keyframes:
        if not isinstance(item, dict):
            raise ValueError("each keyframe review must be an object")
        try:
            frame_idx = int(item.get("frame_idx"))
        except (TypeError, ValueError) as exc:
            raise ValueError("frame_idx must be an integer") from exc
        if frame_idx not in expected_by_frame:
            raise ValueError(f"Unexpected keyframe frame_idx: {frame_idx}")
        if frame_idx in seen_frames:
            raise ValueError(f"Duplicate keyframe review: {frame_idx}")
        seen_frames.add(frame_idx)
        confirmed = bool(item.get("confirmed", False))
        if not confirmed:
            raise ValueError(f"Keyframe {frame_idx} is not confirmed")
        left_track_id = item.get("left_track_id")
        right_track_id = item.get("right_track_id")
        left_track_id = None if left_track_id is None else int(left_track_id)
        right_track_id = None if right_track_id is None else int(right_track_id)
        left_missing_box = bool(item.get("left_missing_box", False))
        right_missing_box = bool(item.get("right_missing_box", False))
        if left_missing_box and left_track_id is not None:
            raise ValueError(f"Keyframe {frame_idx} left_missing_box conflicts with left_track_id")
        if right_missing_box and right_track_id is not None:
            raise ValueError(f"Keyframe {frame_idx} right_missing_box conflicts with right_track_id")
        if left_track_id is not None and left_track_id == right_track_id:
            raise ValueError(f"Keyframe {frame_idx} left and right cannot share the same track")
        visible_track_ids = frame_tracks.get(frame_idx, set())
        if left_track_id is not None and left_track_id not in visible_track_ids:
            raise ValueError(f"Keyframe {frame_idx} left_track_id {left_track_id} is not visible")
        if right_track_id is not None and right_track_id not in visible_track_ids:
            raise ValueError(f"Keyframe {frame_idx} right_track_id {right_track_id} is not visible")
        normalized.append(
            {
                "frame_idx": frame_idx,
                "confirmed": True,
                "left_track_id": left_track_id,
                "right_track_id": right_track_id,
                "left_missing_box": left_missing_box,
                "right_missing_box": right_missing_box,
            }
        )
    missing = sorted(set(expected_by_frame) - seen_frames)
    if missing:
        raise ValueError(f"Missing keyframe reviews: {missing}")
    normalized.sort(key=lambda item: int(item["frame_idx"]))
    return {
        "review_version": "keyframe_v1",
        "keyframe_reviews": normalized,
    }


@mediapipe_review_bp.get("/mediapipe")
def dashboard():
    cfg = init_runtime()
    summary = _summary_payload(cfg)
    return render_template("mediapipe_dashboard.html", summary=summary)


@mediapipe_review_bp.get("/mediapipe/clip/<int:clip_id>")
def clip_page(clip_id: int):
    init_runtime()
    return render_template("mediapipe_clip_review.html", clip_id=clip_id)


@mediapipe_review_bp.get("/mediapipe/assets/<path:relpath>")
def asset(relpath: str):
    cfg = init_runtime()
    return send_from_directory(str(cfg.artifacts_dir), relpath)


@mediapipe_review_bp.get("/api/mediapipe/summary")
def api_summary():
    cfg = init_runtime()
    return jsonify({"success": True, **_summary_payload(cfg)})


@mediapipe_review_bp.get("/api/mediapipe/clips/next")
def api_next_clip():
    cfg = init_runtime()
    with open_db(cfg.db_path) as conn:
        rows = list_clips_by_status(conn, "ready_for_review", limit=1)
    return jsonify({"success": True, "clip_id": int(rows[0]["id"]) if rows else None})


@mediapipe_review_bp.get("/api/mediapipe/clips/<int:clip_id>")
def api_clip(clip_id: int):
    cfg = init_runtime()
    with open_db(cfg.db_path) as conn:
        clip = get_clip(conn, clip_id)
    if clip is None:
        abort(404)
    if not clip.get("bundle_relpath"):
        return jsonify({"success": True, "clip": clip, "bundle": None})
    bundle = _load_bundle(cfg, clip["bundle_relpath"])
    return jsonify({"success": True, "clip": clip, "bundle": bundle})


@mediapipe_review_bp.post("/api/mediapipe/clips/<int:clip_id>/submit-review")
def api_submit_review(clip_id: int):
    cfg = init_runtime()
    payload = request.get_json(force=True)
    with open_db(cfg.db_path) as conn:
        clip = get_clip(conn, clip_id)
        if clip is None:
            abort(404)
        if not clip.get("bundle_relpath"):
            return jsonify({"success": False, "message": "clip bundle is missing"}), 400
        bundle = _load_bundle(cfg, clip["bundle_relpath"])
        try:
            if str(payload.get("review_version") or "") == "keyframe_v1":
                review_payload = _normalize_keyframe_review_payload(payload, bundle)
            else:
                review_payload = _normalize_legacy_review_payload(payload, bundle)
        except ValueError as exc:
            return jsonify({"success": False, "message": str(exc)}), 400
        save_vendor_review(
            conn,
            clip_id=clip_id,
            review_payload=review_payload,
        )
        next_rows = list_clips_by_status(conn, "ready_for_review", limit=1)
    return jsonify({"success": True, "next_clip_id": int(next_rows[0]["id"]) if next_rows else None})
