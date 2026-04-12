from __future__ import annotations

import argparse
import json
import traceback

from ..config import load_config
from ..db import open_db, replace_candidate_chains, update_clip_status, upsert_clip
from ..discover import discover_dirty_clips
from ..precompute import preprocess_clip
from ..types import ClipRef, EpisodeRef


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover dirty clips and precompute MediaPipe review bundles.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--process", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    discovered = 0
    processed = 0
    with open_db(cfg.db_path) as conn:
        if args.discover:
            for clip in discover_dirty_clips(cfg):
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
        if args.process:
            rows = conn.execute(
                """
                SELECT id, source_json FROM clips
                WHERE status IN ('queued_preprocess', 'failed')
                ORDER BY id
                LIMIT ?
                """,
                (args.limit,),
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
                    replace_candidate_chains(conn, clip_id, result["tracks"])
                    next_status = (
                        "qa_needed_dense_scene"
                        if result["bundle"].get("too_dense_for_review")
                        else "ready_for_review"
                    )
                    update_clip_status(conn, clip_id, status=next_status, bundle_relpath=result["bundle_relpath"])
                    processed += 1
                    print(f"[preprocess] clip={clip_id} status={next_status}")
                except Exception as exc:
                    traceback.print_exc()
                    update_clip_status(conn, clip_id, status="failed", error_message=str(exc))
                    print(f"[preprocess] clip={clip_id} failed: {exc}")
    print(json.dumps({"discovered": discovered, "processed": processed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
