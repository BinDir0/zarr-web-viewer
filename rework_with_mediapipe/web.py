from __future__ import annotations

import importlib.util
import json
from typing import Any, Dict, List, Optional

from flask import Blueprint, abort, jsonify, render_template, request, send_from_directory

from .config import MediaPipeReviewConfig, load_config
from .db import (
    count_clips_by_status,
    get_clip,
    get_clip_by_status_offset,
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
    return {
        "python3": True,
        "ultralytics": ultralytics_available,
        "torch": importlib.util.find_spec("torch") is not None,
        "cv2": importlib.util.find_spec("cv2") is not None,
        "pillow": importlib.util.find_spec("PIL") is not None,
        "ultralytics_repo_root": cfg.ultralytics_repo_root.exists(),
        "yolo_model_path": cfg.yolo_model_path.exists(),
    }


def _summary_payload(cfg: MediaPipeReviewConfig, *, start_rank: int = 1) -> Dict[str, Any]:
    queue_offset = max(0, int(start_rank) - 1)
    with open_db(cfg.db_path) as conn:
        counts = count_clips_by_status(conn)
        next_ready = get_clip_by_status_offset(conn, "ready_for_review", offset=queue_offset)
        reviewed = list_clips_by_statuses(conn, ["reviewed_ready_for_fit", "fit_ok", "exported"], limit=5)
        queued_fit = list_fit_jobs(conn, status="queued_fit", limit=5)
    return {
        "counts": counts,
        "requested_start_rank": int(start_rank),
        "next_clip_id": int(next_ready["id"]) if next_ready else None,
        "recent_reviewed_count": len(reviewed),
        "queued_fit_count": len(queued_fit),
        "paths": {
            "db_path": str(cfg.db_path),
            "artifacts_dir": str(cfg.artifacts_dir),
            "source_annotations_db": str(cfg.source_annotations_db),
            "ultralytics_repo_root": str(cfg.ultralytics_repo_root),
            "yolo_model_path": str(cfg.raw.get("yolo_model_path", "") or ""),
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


def _normalize_frame_review_payload(payload: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
    raw_reviews = payload.get("frame_reviews", [])
    if not isinstance(raw_reviews, list):
        raise ValueError("frame_reviews must be an array")

    frame_tracks = {
        int(item["frame_idx"]): {int(track["track_id"]) for track in item.get("tracks", [])}
        for item in bundle.get("frame_tracks", [])
    }
    segment_by_frame = {
        int(item["frame_idx"]): int(item.get("segment_id", -1))
        for item in bundle.get("frames", [])
    }
    keyframe_by_frame = {int(item["frame_idx"]): item for item in bundle.get("keyframes", [])}
    if not keyframe_by_frame:
        raise ValueError("bundle has no keyframes")

    segment_keyframes: Dict[int, set[int]] = {}
    for keyframe in bundle.get("keyframes", []):
        segment_id = int(keyframe.get("segment_id", segment_by_frame.get(int(keyframe["frame_idx"]), -1)))
        segment_keyframes.setdefault(segment_id, set()).add(int(keyframe["frame_idx"]))

    allowed_frames = set(keyframe_by_frame)
    segment_required_frames: Dict[int, set[int]] = {}
    for segment in bundle.get("segments", []):
        segment_id = int(segment["segment_id"])
        required = set(int(item) for item in segment.get("keyframe_frame_indices", []))
        if not required:
            required.update(segment_keyframes.get(segment_id, set()))
        required.update(int(item) for item in segment.get("recovery_candidate_frames", []))
        segment_required_frames[segment_id] = required
        allowed_frames.update(required)

    seen_frames = set()
    normalized: List[Dict[str, Any]] = []
    segment_needs_recovery: Dict[int, bool] = {}
    for item in raw_reviews:
        if not isinstance(item, dict):
            raise ValueError("each frame review must be an object")
        try:
            frame_idx = int(item.get("frame_idx"))
        except (TypeError, ValueError) as exc:
            raise ValueError("frame_idx must be an integer") from exc
        if frame_idx not in allowed_frames:
            raise ValueError(f"Unexpected frame_idx: {frame_idx}")
        if frame_idx in seen_frames:
            raise ValueError(f"Duplicate frame review: {frame_idx}")
        seen_frames.add(frame_idx)
        confirmed = bool(item.get("confirmed", False))
        if not confirmed:
            raise ValueError(f"Frame {frame_idx} is not confirmed")
        segment_id = int(segment_by_frame.get(frame_idx, -1))
        if segment_id < 0:
            raise ValueError(f"Frame {frame_idx} is missing segment info")
        frame_visible_track_ids = frame_tracks.get(frame_idx, set())
        normalized_item: Dict[str, Any] = {
            "frame_idx": frame_idx,
            "confirmed": True,
        }
        frame_requires_recovery = False
        for side in ("left", "right"):
            raw_mode = item.get(f"{side}_mode")
            if raw_mode is None:
                if item.get(f"{side}_track_id") is not None:
                    mode = "track"
                elif bool(item.get(f"{side}_missing_box", False)):
                    mode = "visible_unrecoverable"
                else:
                    mode = "not_visible"
            else:
                mode = str(raw_mode)
            if mode not in {"track", "manual_box", "not_visible", "visible_unrecoverable"}:
                raise ValueError(f"Frame {frame_idx} {side}_mode is invalid: {mode}")
            normalized_item[f"{side}_mode"] = mode
            normalized_item[f"{side}_track_id"] = None
            normalized_item[f"{side}_manual_bbox_xyxy_orig"] = None
            if mode == "track":
                try:
                    track_id = int(item.get(f"{side}_track_id"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Frame {frame_idx} {side}_track_id must be an integer") from exc
                if track_id not in frame_visible_track_ids:
                    raise ValueError(f"Frame {frame_idx} {side}_track_id {track_id} is not visible")
                normalized_item[f"{side}_track_id"] = track_id
            elif mode == "manual_box":
                normalized_item[f"{side}_manual_bbox_xyxy_orig"] = _normalize_manual_bbox(
                    item.get(f"{side}_manual_bbox_xyxy_orig"),
                    frame_idx=frame_idx,
                    side=side,
                )
                frame_requires_recovery = True
            elif mode == "visible_unrecoverable":
                frame_requires_recovery = True
        if (
            normalized_item["left_mode"] == "track"
            and normalized_item["right_mode"] == "track"
            and normalized_item["left_track_id"] == normalized_item["right_track_id"]
        ):
            raise ValueError(f"Frame {frame_idx} left and right cannot share the same track")
        normalized.append(normalized_item)
        if frame_requires_recovery:
            segment_needs_recovery[segment_id] = True

    missing_keyframes = sorted(set(keyframe_by_frame) - seen_frames)
    if missing_keyframes:
        raise ValueError(f"Missing keyframe reviews: {missing_keyframes}")

    normalized_by_frame = {int(item["frame_idx"]): item for item in normalized}
    for segment_id, needs_recovery in segment_needs_recovery.items():
        if not needs_recovery:
            continue
        required_frames = segment_required_frames.get(segment_id, set())
        missing_frames = sorted(frame_idx for frame_idx in required_frames if frame_idx not in normalized_by_frame)
        if missing_frames:
            raise ValueError(f"Segment {segment_id} is missing required recovery frames: {missing_frames}")

    normalized.sort(key=lambda item: int(item["frame_idx"]))
    return {
        "review_version": "frame_review_v2",
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
    return send_from_directory(str(cfg.artifacts_dir), relpath)


@mediapipe_review_bp.get("/api/mediapipe/summary")
def api_summary():
    cfg = init_runtime()
    start_rank = _parse_start_rank(request.args.get("start_rank", 1))
    return jsonify({"success": True, **_summary_payload(cfg, start_rank=start_rank)})


@mediapipe_review_bp.get("/api/mediapipe/clips/next")
def api_next_clip():
    cfg = init_runtime()
    start_rank = _parse_start_rank(request.args.get("start_rank", 1))
    queue_offset = max(0, start_rank - 1)
    after_clip_id = request.args.get("after_clip_id", default=None, type=int)
    with open_db(cfg.db_path) as conn:
        if after_clip_id is not None:
            row = get_next_clip_by_status_after_id(conn, "ready_for_review", after_clip_id=after_clip_id)
        else:
            row = get_clip_by_status_offset(conn, "ready_for_review", offset=queue_offset)
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
            review_version = str(payload.get("review_version") or "")
            if review_version == "frame_review_v2":
                review_payload = _normalize_frame_review_payload(payload, bundle)
            elif review_version == "keyframe_v1":
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
        next_row = get_next_clip_by_status_after_id(conn, "ready_for_review", after_clip_id=clip_id)
    return jsonify({"success": True, "next_clip_id": int(next_row["id"]) if next_row else None})
