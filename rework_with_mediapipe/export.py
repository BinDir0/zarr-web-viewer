from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .config import MediaPipeReviewConfig
from .db import connect, record_export


def export_clip_manifest(cfg: MediaPipeReviewConfig) -> Dict:
    output_dir = cfg.exports_dir / datetime.utcnow().strftime("export_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"

    conn = connect(cfg.db_path)
    rows = conn.execute(
        """
        SELECT * FROM clips
        WHERE status = 'fit_ok'
          AND fit_payload_json IS NOT NULL
          AND review_unit = 'episode'
        ORDER BY id
        """
    ).fetchall()
    items: List[Dict] = []
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            bundle_relpath = str(row["bundle_relpath"] or "")
            fit_payload = json.loads(str(row["fit_payload_json"] or "{}"))
            subepisodes = list(fit_payload.get("subepisodes", []))
            if not subepisodes:
                subepisodes = [
                    {
                        "subepisode_index": 0,
                        "start_frame": int(row["clip_start"]),
                        "end_frame": int(row["clip_end"]),
                        "num_frames": int(row["num_frames"]),
                    }
                ]
            for subepisode in subepisodes:
                item = {
                    "clip_id": int(row["id"]),
                    "parent_clip_id": int(row["id"]),
                    "review_unit": "episode",
                    "subepisode_index": int(subepisode.get("subepisode_index", 0)),
                    "episode_id": str(row["episode_id"]),
                    "episode_name": str(row["episode_name"] or row["episode_id"]),
                    "dataset_name": str(row["dataset_name"] or ""),
                    "clip_start": int(row["clip_start"]),
                    "clip_end": int(row["clip_end"]),
                    "frame_start": int(subepisode.get("start_frame", row["clip_start"])),
                    "frame_end": int(subepisode.get("end_frame", row["clip_end"])),
                    "num_frames": int(subepisode.get("num_frames", row["num_frames"])),
                    "status": str(row["status"]),
                    "dirty_reason": str(row["dirty_reason"]),
                    "bundle_json": str(cfg.artifact_abspath(bundle_relpath) / "bundle.json"),
                    "proposals_npz": str(cfg.artifact_abspath(bundle_relpath) / "proposals.npz"),
                    "fit_json": str(cfg.artifact_abspath(bundle_relpath) / "fit.json"),
                    "fit_npz": str(cfg.artifact_abspath(bundle_relpath) / "fit.npz"),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                items.append(item)

    summary = {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "num_clips": len(rows),
        "num_episodes": len(rows),
        "num_items": len(items),
        "items": items[:10],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if items:
        payload = {
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
            "exported_at": datetime.utcnow().isoformat(),
            "num_clips": len(rows),
            "num_episodes": len(rows),
            "num_items": len(items),
        }
        conn.executemany(
            """
            UPDATE clips
            SET status = 'exported',
                export_payload_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            [(json.dumps(payload, ensure_ascii=False), item["clip_id"]) for item in items],
        )
        conn.commit()
        record_export(conn, output_dir.name, payload)

    conn.close()
    return summary
