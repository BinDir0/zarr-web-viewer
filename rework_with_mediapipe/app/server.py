from __future__ import annotations

import json
import os
from typing import Any, Dict

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from app.config import AppConfig, load_config
from app.db import get_clip, list_clips_by_status, open_db, save_vendor_review


def _load_bundle(cfg: AppConfig, bundle_relpath: str) -> Dict[str, Any]:
    bundle_path = cfg.app_root / bundle_relpath / "bundle.json"
    with open(bundle_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_app(config_path: str | None = None) -> Flask:
    cfg = load_config(config_path)
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["RWM_CONFIG"] = cfg

    @app.get("/")
    def index():
        with open_db(cfg.app_db) as conn:
            next_ready = list_clips_by_status(conn, "ready_for_review", limit=1)
            reviewed = list_clips_by_status(conn, "reviewed", limit=10)
        return render_template(
            "index.html",
            next_clip_id=(int(next_ready[0]["id"]) if next_ready else None),
            reviewed_count=len(reviewed),
        )

    @app.get("/clip/<int:clip_id>")
    def clip_page(clip_id: int):
        return render_template("clip_review.html", clip_id=clip_id)

    @app.get("/assets/<path:relpath>")
    def asset(relpath: str):
        return send_from_directory(str(cfg.app_root), relpath)

    @app.get("/api/clips/next")
    def api_next_clip():
        with open_db(cfg.app_db) as conn:
            rows = list_clips_by_status(conn, "ready_for_review", limit=1)
        return jsonify({"clip_id": int(rows[0]["id"]) if rows else None})

    @app.get("/api/clips/<int:clip_id>")
    def api_clip(clip_id: int):
        with open_db(cfg.app_db) as conn:
            clip = get_clip(conn, clip_id)
        if clip is None:
            abort(404)
        if not clip.get("bundle_relpath"):
            return jsonify({"clip": clip, "bundle": None})
        bundle = _load_bundle(cfg, clip["bundle_relpath"])
        return jsonify({"clip": clip, "bundle": bundle})

    @app.post("/api/clips/<int:clip_id>/submit-review")
    def api_submit_review(clip_id: int):
        payload = request.get_json(force=True)
        left_choice = str(payload.get("left_choice", "")).strip()
        right_choice = str(payload.get("right_choice", "")).strip()
        merge_answers = payload.get("merge_answers", [])
        review_confidence = str(payload.get("review_confidence", "")).strip()
        if not left_choice or not right_choice:
            return jsonify({"success": False, "message": "left_choice and right_choice are required"}), 400
        concrete_left = left_choice not in {"none", "unsure"}
        concrete_right = right_choice not in {"none", "unsure"}
        if concrete_left and concrete_right and left_choice == right_choice:
            return jsonify({"success": False, "message": "Left and right cannot point to the same candidate"}), 400
        with open_db(cfg.app_db) as conn:
            if get_clip(conn, clip_id) is None:
                abort(404)
            save_vendor_review(
                conn,
                clip_id=clip_id,
                left_choice=left_choice,
                right_choice=right_choice,
                merge_answers=merge_answers,
                review_confidence=review_confidence or "normal",
            )
        return jsonify({"success": True})

    return app


app = create_app(os.environ.get("RWM_CONFIG"))


if __name__ == "__main__":
    cfg = load_config(os.environ.get("RWM_CONFIG"))
    app.run(
        host=str(cfg.server.get("host", "127.0.0.1")),
        port=int(cfg.server.get("port", 9481)),
        debug=bool(cfg.server.get("debug", False)),
    )
