from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from pipeline.types import ClipRef, EpisodeRef

LEGACY_BUILDAI_EPISODE_RE = re.compile(r"^factory_(?P<factory>\d+)_worker_(?P<worker>\d+)_.+$")


def _read_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_bad_frames(content: str) -> List[int]:
    if content.startswith("BAD_FRAMES:"):
        raw = content.split(":", 1)[1]
        return [int(x) for x in raw.split(",") if x.strip().isdigit()]
    if content.startswith("BAD_FRAME:"):
        raw = content.split(":", 1)[1]
        return [int(x) for x in raw.split(",") if x.strip().isdigit()]
    if content.startswith("REWORK_V1:"):
        payload = json.loads(content.split(":", 1)[1])
        bad = list(payload.get("bad_box", [])) + list(payload.get("not_clear", []))
        return sorted({int(x) for x in bad if isinstance(x, int) or str(x).isdigit()})
    return []


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _load_annotation_rows(db_path: Path) -> List[sqlite3.Row]:
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    try:
        return conn.execute(
            """
            SELECT episode_id, episode_name, dataset_name, content
            FROM annotations
            WHERE content LIKE 'BAD_FRAMES:%' OR content LIKE 'BAD_FRAME:%' OR content LIKE 'REWORK_V1:%'
            ORDER BY episode_id
            """
        ).fetchall()
    finally:
        conn.close()


def _load_source_rows(cfg: Dict, local_db_path: Path) -> List[Dict]:
    """Load dirty rows from the parent overlay db and optional source dbs.

    This mirrors the practical zarr-viewer setup where the local annotations.db
    may only store overlays while the real BAD_FRAMES source queue lives in
    source_factory_annotations_db or legacy_buildai_annotations_db.
    """
    candidates = [
        local_db_path,
        Path(str(cfg.get("source_factory_annotations_db"))) if cfg.get("source_factory_annotations_db") else None,
        Path(str(cfg.get("legacy_buildai_annotations_db"))) if cfg.get("legacy_buildai_annotations_db") else None,
    ]
    merged: Dict[str, Dict] = {}
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        rows = _load_annotation_rows(candidate)
        for row in rows:
            row_dict = dict(row)
            # Prefer local overlay rows when episode_id overlaps.
            merged.setdefault(str(row_dict["episode_id"]), row_dict)
            if candidate == local_db_path:
                merged[str(row_dict["episode_id"])] = row_dict
    return [merged[key] for key in sorted(merged.keys())]


def _load_factory_lookup(zarr_viewer_config: Path) -> Dict[str, EpisodeRef]:
    cfg = _read_yaml(zarr_viewer_config)
    factory_base = cfg.get("factory_base")
    start = int(cfg.get("factory_start", 1))
    end = int(cfg.get("factory_end", 0))
    if not factory_base:
        return {}
    base = Path(factory_base)
    lookup: Dict[str, EpisodeRef] = {}
    for fid in range(start, end + 1):
        factory_dir = base / f"factory{fid:03d}"
        index_path = factory_dir / "_video_index.json"
        if not index_path.exists():
            continue
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for episode_id, info in sorted(index.get("videos", {}).items()):
            frames = info.get("frames") or []
            if not frames:
                continue
            if isinstance(frames[0], dict):
                frame_names = [frame["name"] for frame in frames]
                frame_offsets = [[int(frame["offset"]), int(frame["size"])] for frame in frames]
            else:
                frame_names = list(frames)
                frame_offsets = None
            lookup[episode_id] = EpisodeRef(
                episode_id=episode_id,
                episode_name=info.get("video_name", episode_id),
                dataset_name=f"factory{fid:03d}",
                source_type="factory_tar",
                num_frames=len(frames),
                shard_path=str(factory_dir / info["shard"]),
                seq_folder=str(factory_dir / "outputs" / episode_id),
                frame_names=frame_names,
                frame_offsets=frame_offsets,
            )
    return lookup


def _resolve_legacy_episode(episode_id: str, cfg: Dict, row: sqlite3.Row) -> Optional[EpisodeRef]:
    root = cfg.get("legacy_buildai_dataset_root") or cfg.get("buildai_dataset_root")
    if not root:
        return None
    match = LEGACY_BUILDAI_EPISODE_RE.match(episode_id)
    if match is None:
        return None
    crop_dir = (
        Path(root)
        / f"factory_{match.group('factory')}"
        / f"worker_{match.group('worker')}"
        / "processed"
        / episode_id
    )
    extracted = crop_dir / "extracted_images"
    if not extracted.is_dir():
        return None
    num_frames = len(list(extracted.glob("*.jpg")))
    return EpisodeRef(
        episode_id=episode_id,
        episode_name=row["episode_name"] or episode_id,
        dataset_name=row["dataset_name"] or cfg.get("legacy_buildai_dataset_name", "BuildAI-10k"),
        source_type="legacy_raw",
        num_frames=num_frames,
        crop_dir=str(crop_dir),
    )


def _merge_spans(frame_indices: Iterable[int], max_gap: int) -> List[Tuple[int, int]]:
    ordered = sorted({int(idx) for idx in frame_indices if int(idx) >= 0})
    if not ordered:
        return []
    spans: List[Tuple[int, int]] = []
    start = ordered[0]
    prev = ordered[0]
    for idx in ordered[1:]:
        if idx - prev <= max_gap:
            prev = idx
            continue
        spans.append((start, prev + 1))
        start = idx
        prev = idx
    spans.append((start, prev + 1))
    return spans


def _slice_spans(
    spans: List[Tuple[int, int]],
    *,
    num_frames: int,
    context: int,
    max_clip_frames: int,
    overlap: int,
) -> List[Tuple[int, int]]:
    clips: List[Tuple[int, int]] = []
    for start, end in spans:
        clip_start = max(0, start - context)
        clip_end = min(num_frames, end + context)
        while clip_end - clip_start > max_clip_frames:
            window_end = min(clip_start + max_clip_frames, num_frames)
            clips.append((clip_start, window_end))
            clip_start = max(0, window_end - overlap)
        clips.append((clip_start, clip_end))
    deduped: List[Tuple[int, int]] = []
    seen = set()
    for item in clips:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def discover_dirty_clips(
    zarr_viewer_config: Path,
    zarr_viewer_db: Path,
    *,
    clip_merge_gap: int,
    clip_context_frames: int,
    max_clip_frames: int,
    overlap_frames: int,
) -> List[ClipRef]:
    cfg = _read_yaml(zarr_viewer_config)
    factory_lookup = _load_factory_lookup(zarr_viewer_config)
    rows = _load_source_rows(cfg, zarr_viewer_db)
    clips: List[ClipRef] = []
    for row in rows:
        bad_frames = _parse_bad_frames(str(row["content"]))
        if not bad_frames:
            continue
        episode = factory_lookup.get(str(row["episode_id"]))
        if episode is None:
            episode = _resolve_legacy_episode(str(row["episode_id"]), cfg, row)
        if episode is None or episode.num_frames <= 0:
            continue
        spans = _merge_spans(bad_frames, clip_merge_gap)
        windows = _slice_spans(
            spans,
            num_frames=episode.num_frames,
            context=clip_context_frames,
            max_clip_frames=max_clip_frames,
            overlap=overlap_frames,
        )
        dirty_reason = "legacy_bad_frames"
        if str(row["content"]).startswith("REWORK_V1:"):
            dirty_reason = "rework_v1"
        for clip_start, clip_end in windows:
            clip_bad = [idx for idx in bad_frames if clip_start <= idx < clip_end]
            clips.append(
                ClipRef(
                    episode=episode,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    dirty_reason=dirty_reason,
                    bad_frames=clip_bad,
                )
            )
    return clips
