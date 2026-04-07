from __future__ import annotations

import argparse
import json
import traceback

from app.config import load_config
from app.db import open_db, replace_candidate_chains, update_clip_status, upsert_clip
from pipeline.discover import discover_dirty_clips
from pipeline.precompute import preprocess_clip
from pipeline.types import ClipRef, EpisodeRef


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover dirty clips and precompute review bundles.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--process", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    discovered_ids = []
    with open_db(cfg.app_db) as conn:
        if args.discover:
            clips = discover_dirty_clips(
                cfg.zarr_viewer_config,
                cfg.zarr_viewer_db,
                clip_merge_gap=int(cfg.review.get("clip_merge_gap", 15)),
                clip_context_frames=int(cfg.review.get("clip_context_frames", 60)),
                max_clip_frames=int(cfg.review.get("max_clip_frames", 240)),
                overlap_frames=int(cfg.review.get("overlap_frames", 30)),
            )
            for clip in clips:
                clip_id = upsert_clip(
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
                discovered_ids.append((clip_id, clip))
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
                episode = EpisodeRef(**payload["episode"])
                clip = ClipRef(
                    episode=episode,
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
                    update_clip_status(
                        conn,
                        clip_id,
                        status=next_status,
                        bundle_relpath=result["bundle_relpath"],
                    )
                    print(f"[preprocess] clip={clip_id} status={next_status}")
                except Exception as exc:
                    traceback.print_exc()
                    update_clip_status(conn, clip_id, status="failed", error_message=str(exc))
                    print(f"[preprocess] clip={clip_id} failed: {exc}")
    if args.discover:
        print(f"Discovered {len(discovered_ids)} clips.")


if __name__ == "__main__":
    main()
