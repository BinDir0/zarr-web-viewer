from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, abort, jsonify, render_template, request, send_from_directory

from .config import MediaPipeReviewConfig, load_config
from .db import (
    connect,
    count_clips_by_status,
    get_clip,
    init_db,
    list_clips_by_status,
    list_fit_jobs,
    open_db,
    record_export,
    replace_candidate_chains,
    save_fit_result,
    save_vendor_review,
    update_clip_status,
    upsert_clip,
)
from .discover import discover_dirty_clips
from .export import export_clip_manifest
from .fit import fit_reviewed_clip, save_fit_artifacts
from .precompute import preprocess_clip
from .types import ClipRef, EpisodeRef

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
        reviewed = list_clips_by_status(conn, "reviewed", limit=5)
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


def _run_discover(cfg: MediaPipeReviewConfig) -> Dict[str, Any]:
    discovered = 0
    clips = discover_dirty_clips(cfg)
    with open_db(cfg.db_path) as conn:
        for clip in clips:
            upsert_clip(
                conn,
                episode_id=clip.episode.episode_id,
                episode_name=clip.episode.episode_name,
                dataset_name=clip.episode.dataset_name,
                source_type=clip.episode.source_type,
                source_json=clip.to_dict(),
                clip_start=clip.clip_start,
                clip_end=clip.clip_end,
                num_frames=clip.num_frames,
                dirty_reason=clip.dirty_reason,
                status="queued_preprocess",
            )
            discovered += 1
    return {"discovered": discovered}


def _run_process(cfg: MediaPipeReviewConfig, limit: int) -> Dict[str, Any]:
    processed = 0
    failed: List[Dict[str, Any]] = []
    with open_db(cfg.db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, source_json FROM clips
            WHERE status IN ('queued_preprocess', 'failed')
            ORDER BY id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        for row in rows:
            payload = json.loads(row["source_json"])
            clip = ClipRef(
                episode=EpisodeRef(**payload["episode"]),
                clip_start=int(payload["clip_start"]),
                clip_end=int(payload["clip_end"]),
                dirty_reason=str(payload["dirty_reason"]),
                bad_frames=list(payload.get("bad_frames", [])),
            )
            clip_id = int(row["id"])
            update_clip_status(conn, clip_id, status="preprocessing")
            try:
                result = preprocess_clip(clip, clip_id, cfg)
                replace_candidate_chains(conn, clip_id, result["chains"])
                next_status = "ready_for_review" if result["bundle"]["chains"] else "qa_needed"
                update_clip_status(conn, clip_id, status=next_status, bundle_relpath=result["bundle_relpath"])
                processed += 1
            except Exception as exc:
                update_clip_status(conn, clip_id, status="failed", error_message=str(exc))
                failed.append({"clip_id": clip_id, "message": str(exc)})
    return {"processed": processed, "failed": failed[:20]}


def _run_fit(cfg: MediaPipeReviewConfig, limit: int) -> Dict[str, Any]:
    fitted = 0
    failed: List[Dict[str, Any]] = []
    conn = connect(cfg.db_path)
    try:
        jobs = list_fit_jobs(conn, status="queued_fit", limit=limit)
        for job in jobs:
            clip_id = int(job["clip_id"])
            clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
            review = conn.execute("SELECT * FROM vendor_reviews WHERE clip_id = ?", (clip_id,)).fetchone()
            if clip is None or review is None or not clip["bundle_relpath"]:
                save_fit_result(conn, clip_id, "qa_needed", {"status": "qa_needed", "message": "Missing review/bundle"})
                continue
            bundle_dir = cfg.artifact_abspath(str(clip["bundle_relpath"]))
            with (bundle_dir / "bundle.json").open("r", encoding="utf-8") as f:
                bundle = json.load(f)
            review_payload = {
                "left_choice": review["left_choice"],
                "right_choice": review["right_choice"],
                "merge_answers": json.loads(review["merge_answers_json"]),
            }
            try:
                fit_payload = fit_reviewed_clip(bundle, review_payload, cfg)
                save_fit_artifacts(bundle_dir, fit_payload)
                save_fit_result(conn, clip_id, fit_payload["status"], fit_payload)
                fitted += 1
            except Exception as exc:
                save_fit_result(conn, clip_id, "qa_needed", {"status": "qa_needed", "message": str(exc)})
                failed.append({"clip_id": clip_id, "message": str(exc)})
    finally:
        conn.close()
    return {"fitted": fitted, "failed": failed[:20]}


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
    left_choice = str(payload.get("left_choice", "")).strip()
    right_choice = str(payload.get("right_choice", "")).strip()
    merge_answers = payload.get("merge_answers", [])
    review_confidence = str(payload.get("review_confidence", "")).strip() or "normal"
    if not left_choice or not right_choice:
        return jsonify({"success": False, "message": "left_choice and right_choice are required"}), 400
    concrete_left = left_choice not in {"none", "unsure"}
    concrete_right = right_choice not in {"none", "unsure"}
    if concrete_left and concrete_right and left_choice == right_choice:
        return jsonify({"success": False, "message": "Left and right cannot point to the same candidate"}), 400
    with open_db(cfg.db_path) as conn:
        if get_clip(conn, clip_id) is None:
            abort(404)
        save_vendor_review(
            conn,
            clip_id=clip_id,
            left_choice=left_choice,
            right_choice=right_choice,
            merge_answers=merge_answers if isinstance(merge_answers, list) else [],
            review_confidence=review_confidence,
        )
        next_rows = list_clips_by_status(conn, "ready_for_review", limit=1)
    return jsonify({"success": True, "next_clip_id": int(next_rows[0]["id"]) if next_rows else None})


@mediapipe_review_bp.post("/api/mediapipe/jobs/discover")
def api_run_discover():
    cfg = init_runtime()
    try:
        return jsonify({"success": True, **_run_discover(cfg), **_summary_payload(cfg)})
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@mediapipe_review_bp.post("/api/mediapipe/jobs/process")
def api_run_process():
    cfg = init_runtime()
    limit = max(1, int((request.get_json(silent=True) or {}).get("limit", 25)))
    try:
        return jsonify({"success": True, **_run_process(cfg, limit), **_summary_payload(cfg)})
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@mediapipe_review_bp.post("/api/mediapipe/jobs/fit")
def api_run_fit():
    cfg = init_runtime()
    limit = max(1, int((request.get_json(silent=True) or {}).get("limit", 10)))
    try:
        return jsonify({"success": True, **_run_fit(cfg, limit), **_summary_payload(cfg)})
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@mediapipe_review_bp.post("/api/mediapipe/jobs/export")
def api_run_export():
    cfg = init_runtime()
    try:
        payload = export_clip_manifest(cfg)
        return jsonify({"success": True, "export": payload, **_summary_payload(cfg)})
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500
