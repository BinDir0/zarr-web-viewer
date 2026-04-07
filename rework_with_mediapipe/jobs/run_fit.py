from __future__ import annotations

import argparse
import json
import traceback

from app.config import load_config
from app.db import connect, list_fit_jobs, save_fit_result
from export.export_zarr import export_fit_ok_clips
from mano.fit_clip import fit_reviewed_clip


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit MANO after review and export zarr.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = connect(cfg.app_db)
    try:
        jobs = list_fit_jobs(conn, status="queued_fit", limit=args.limit)
        for job in jobs:
            clip_id = int(job["clip_id"])
            clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
            review = conn.execute("SELECT * FROM vendor_reviews WHERE clip_id = ?", (clip_id,)).fetchone()
            if clip is None or review is None or not clip["bundle_relpath"]:
                save_fit_result(conn, clip_id, "qa_needed", {"status": "qa_needed", "message": "Missing review/bundle"})
                continue
            bundle_path = cfg.app_root / clip["bundle_relpath"] / "bundle.json"
            with open(bundle_path, "r", encoding="utf-8") as f:
                bundle = json.load(f)
            review_payload = {
                "left_choice": review["left_choice"],
                "right_choice": review["right_choice"],
                "merge_answers": json.loads(review["merge_answers_json"]),
            }
            try:
                fit = fit_reviewed_clip(bundle, review_payload, cfg)
                save_fit_result(conn, clip_id, fit["status"], fit)
                print(f"[fit] clip={clip_id} status={fit['status']}")
            except Exception as exc:
                traceback.print_exc()
                save_fit_result(conn, clip_id, "qa_needed", {"status": "qa_needed", "message": str(exc)})
                print(f"[fit] clip={clip_id} failed: {exc}")
    finally:
        conn.close()
    if args.export:
        summary = export_fit_ok_clips(cfg)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
