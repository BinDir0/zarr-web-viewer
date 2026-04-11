from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import MediaPipeReviewConfig
from .types import ClipRef, EpisodeRef

SANITY_REASON_V1_PREFIX = "SANITY_REASON_V1:"
SANITY_REASON_V2_PREFIX = "SANITY_REASON_V2:"
SANITY_REASON_V3_PREFIX = "SANITY_REASON_V3:"


def is_factory_dataset_name(dataset_name: str) -> bool:
    return str(dataset_name).startswith("factory")


def normalize_reason_payload(bad_box, not_clear) -> Dict[str, List[int]]:
    def _ints(seq) -> List[int]:
        out: List[int] = []
        if not isinstance(seq, list):
            return out
        for x in seq:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out

    bb = sorted(set(_ints(bad_box)))
    nc = sorted(set(i for i in _ints(not_clear) if i not in bb))
    return {"bad_box": bb, "not_clear": nc}


def normalize_sanity_reason_payload(
    wrong_annotation=None,
    missing_annotation=None,
    non_visible_wrong_annotation=None,
    visible_box_wrong_annotation=None,
    bad_box=None,
    not_clear=None,
) -> Dict[str, List[int]]:
    def _ints(seq) -> List[int]:
        out: List[int] = []
        if not isinstance(seq, list):
            return out
        for x in seq:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out

    ma_src = missing_annotation if missing_annotation is not None else not_clear
    nb_src = non_visible_wrong_annotation
    vb_src = visible_box_wrong_annotation if visible_box_wrong_annotation is not None else wrong_annotation
    ma = sorted(set(_ints(ma_src)))
    nb = sorted(set(i for i in _ints(nb_src) if i not in ma))
    vb = sorted(set(i for i in _ints(vb_src if vb_src is not None else bad_box) if i not in ma and i not in nb))
    return {
        "missing_annotation": ma,
        "non_visible_wrong_annotation": nb,
        "visible_box_wrong_annotation": vb,
    }


def _parse_json_payload(content: str, prefix: str) -> Optional[Dict]:
    if not content or not content.startswith(prefix):
        return None
    try:
        payload = json.loads(content[len(prefix) :])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_rework_v1_payload(content: str) -> Optional[Dict[str, List[int]]]:
    payload = _parse_json_payload(content, "REWORK_V1:")
    if payload is None:
        return None
    return normalize_reason_payload(payload.get("bad_box"), payload.get("not_clear"))


def parse_sanity_reason_payload(content: str) -> Optional[Dict[str, List[int]]]:
    for prefix in (SANITY_REASON_V3_PREFIX, SANITY_REASON_V2_PREFIX, SANITY_REASON_V1_PREFIX):
        payload = _parse_json_payload(content, prefix)
        if payload is not None:
            return normalize_sanity_reason_payload(
                wrong_annotation=payload.get("wrong_annotation"),
                missing_annotation=payload.get("missing_annotation"),
                non_visible_wrong_annotation=payload.get("non_visible_wrong_annotation"),
                visible_box_wrong_annotation=payload.get("visible_box_wrong_annotation"),
                bad_box=payload.get("bad_box"),
                not_clear=payload.get("not_clear"),
            )
    return None


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _load_annotation_rows(db_path: Path) -> List[sqlite3.Row]:
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    try:
        has_reason_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'annotation_reasons'"
        ).fetchone() is not None
        if has_reason_table:
            return conn.execute(
                """
                SELECT
                    a.episode_id,
                    a.episode_name,
                    a.dataset_name,
                    a.content,
                    r.content AS reason_content
                FROM annotations AS a
                LEFT JOIN annotation_reasons AS r
                  ON r.episode_id = a.episode_id
                WHERE a.dataset_name LIKE 'factory%'
                  AND (
                    a.content LIKE 'BAD_FRAMES:%'
                    OR a.content LIKE 'BAD_FRAME:%'
                    OR a.content LIKE 'REWORK_V1:%'
                  )
                ORDER BY a.episode_id
                """
            ).fetchall()
        return conn.execute(
            """
            SELECT
                episode_id,
                episode_name,
                dataset_name,
                content,
                '' AS reason_content
            FROM annotations
            WHERE dataset_name LIKE 'factory%'
              AND (
                content LIKE 'BAD_FRAMES:%'
                OR content LIKE 'BAD_FRAME:%'
                OR content LIKE 'REWORK_V1:%'
              )
            ORDER BY episode_id
            """
        ).fetchall()
    finally:
        conn.close()


def _load_factory_lookup(cfg: MediaPipeReviewConfig) -> Dict[str, EpisodeRef]:
    lookup: Dict[str, EpisodeRef] = {}
    for fid in range(cfg.factory_start, cfg.factory_end + 1):
        factory_dir = cfg.factory_base / f"factory{fid:03d}"
        index_path = factory_dir / "_video_index.json"
        if not index_path.exists():
            continue
        try:
            with index_path.open("r", encoding="utf-8") as f:
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


def _parse_dirty_frames(row: sqlite3.Row) -> Tuple[str, List[int]]:
    content = str(row["content"] or "")
    if content.startswith("REWORK_V1:"):
        payload = parse_rework_v1_payload(content)
        if payload is None:
            return "", []
        bad_frames = sorted(set(payload["bad_box"] + payload["not_clear"]))
        return "rework_box_issue", bad_frames

    if content.startswith("BAD_FRAMES:") or content.startswith("BAD_FRAME:"):
        reason_payload = parse_sanity_reason_payload(str(row["reason_content"] or ""))
        if reason_payload is None:
            return "", []
        bad_frames = sorted(
            set(
                reason_payload["missing_annotation"]
                + reason_payload["non_visible_wrong_annotation"]
                + reason_payload["visible_box_wrong_annotation"]
            )
        )
        return "sanity_box_issue", bad_frames

    return "", []


def discover_dirty_clips(cfg: MediaPipeReviewConfig) -> List[ClipRef]:
    factory_lookup = _load_factory_lookup(cfg)
    rows = _load_annotation_rows(cfg.source_annotations_db)
    clips: List[ClipRef] = []
    for row in rows:
        if not is_factory_dataset_name(row["dataset_name"]):
            continue
        dirty_reason, bad_frames = _parse_dirty_frames(row)
        if not bad_frames:
            continue
        episode = factory_lookup.get(str(row["episode_id"]))
        if episode is None or episode.num_frames <= 0:
            continue
        spans = _merge_spans(bad_frames, cfg.clip_merge_gap)
        windows = _slice_spans(
            spans,
            num_frames=episode.num_frames,
            context=cfg.clip_context_frames,
            max_clip_frames=cfg.max_clip_frames,
            overlap=cfg.overlap_frames,
        )
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
