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
        "SELECT * FROM clips WHERE status = 'fit_ok' AND fit_payload_json IS NOT NULL ORDER BY id"
    ).fetchall()
    items: List[Dict] = []
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            bundle_relpath = str(row["bundle_relpath"] or "")
            item = {
                "clip_id": int(row["id"]),
                "episode_id": str(row["episode_id"]),
                "episode_name": str(row["episode_name"] or row["episode_id"]),
                "dataset_name": str(row["dataset_name"] or ""),
                "clip_start": int(row["clip_start"]),
                "clip_end": int(row["clip_end"]),
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
        "num_clips": len(items),
        "items": items[:10],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if items:
        payload = {
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
            "exported_at": datetime.utcnow().isoformat(),
            "num_clips": len(items),
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
