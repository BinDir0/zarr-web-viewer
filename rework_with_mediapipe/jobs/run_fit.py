from __future__ import annotations

import argparse
import json
import traceback

from ..config import load_config
from ..db import connect, list_fit_jobs, save_fit_result
from ..export import export_clip_manifest
from ..fit import fit_reviewed_clip, save_fit_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit MANO after review and export manifest.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = connect(cfg.db_path)
    try:
        jobs = list_fit_jobs(conn, status="queued_fit", limit=args.limit)
        for job in jobs:
            clip_id = int(job["clip_id"])
            clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
            if clip is None or not clip["bundle_relpath"] or not clip["review_payload_json"]:
                save_fit_result(conn, clip_id, "qa_needed", {"status": "qa_needed", "message": "Missing review/bundle"})
                continue
            bundle_dir = cfg.artifact_abspath(str(clip["bundle_relpath"]))
            bundle_path = bundle_dir / "bundle.json"
            with bundle_path.open("r", encoding="utf-8") as f:
                bundle = json.load(f)
            review_payload = json.loads(str(clip["review_payload_json"]))
            try:
                fit_payload = fit_reviewed_clip(bundle, review_payload, cfg)
                save_fit_artifacts(bundle_dir, fit_payload)
                save_fit_result(conn, clip_id, fit_payload["status"], fit_payload)
                print(f"[fit] clip={clip_id} status={fit_payload['status']}")
            except Exception as exc:
                traceback.print_exc()
                save_fit_result(conn, clip_id, "qa_needed", {"status": "qa_needed", "message": str(exc)})
                print(f"[fit] clip={clip_id} failed: {exc}")
    finally:
        conn.close()

    if args.export:
        print(json.dumps(export_clip_manifest(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
