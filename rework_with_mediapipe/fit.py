from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import MediaPipeReviewConfig


@dataclass
class SideObservations:
    track_id: int
    frame_indices: List[int]
    bbox_xyxy: np.ndarray
    bbox_xyxy_orig: np.ndarray
    score: np.ndarray
    source: str = "track"


def _track_lookup(bundle: Dict) -> Dict[int, Dict]:
    return {int(track["track_id"]): track for track in bundle.get("tracks", [])}


def _build_side_observation(bundle: Dict, track_id: int) -> Optional[SideObservations]:
    track = _track_lookup(bundle).get(int(track_id))
    if track is None:
        return None
    proposals = sorted(track["proposals"], key=lambda item: int(item["frame_idx"]))
    if not proposals:
        return None
    return SideObservations(
        track_id=int(track["track_id"]),
        frame_indices=[int(item["frame_idx"]) for item in proposals],
        bbox_xyxy=np.asarray([item["bbox_xyxy"] for item in proposals], dtype=np.float32),
        bbox_xyxy_orig=np.asarray([item.get("bbox_xyxy_orig", item["bbox_xyxy"]) for item in proposals], dtype=np.float32),
        score=np.asarray([item.get("score", 0.0) for item in proposals], dtype=np.float32),
    )


def _track_proposal_lookup(bundle: Dict) -> Dict[int, Dict[int, Dict]]:
    lookup: Dict[int, Dict[int, Dict]] = {}
    for track in bundle.get("tracks", []):
        proposal_lookup = {
            int(item["frame_idx"]): item
            for item in sorted(track.get("proposals", []), key=lambda item: int(item["frame_idx"]))
        }
        lookup[int(track["track_id"])] = proposal_lookup
    return lookup


def _build_side_observation_from_frames(
    proposal_lookup: Dict[int, Dict[int, Dict]],
    track_id: int,
    frame_indices: Sequence[int],
) -> Optional[SideObservations]:
    frame_list = [int(frame_idx) for frame_idx in frame_indices]
    track_proposals = proposal_lookup.get(int(track_id), {})
    proposals = [track_proposals[frame_idx] for frame_idx in frame_list if frame_idx in track_proposals]
    if not proposals:
        return None
    return SideObservations(
        track_id=int(track_id),
        frame_indices=[int(item["frame_idx"]) for item in proposals],
        bbox_xyxy=np.asarray([item["bbox_xyxy"] for item in proposals], dtype=np.float32),
        bbox_xyxy_orig=np.asarray([item.get("bbox_xyxy_orig", item["bbox_xyxy"]) for item in proposals], dtype=np.float32),
        score=np.asarray([item.get("score", 0.0) for item in proposals], dtype=np.float32),
    )


def _normalize_keyframe_state(review: Dict) -> Dict[str, Optional[int] | bool]:
    return {
        "left_track_id": None if review.get("left_track_id") is None else int(review["left_track_id"]),
        "right_track_id": None if review.get("right_track_id") is None else int(review["right_track_id"]),
        "left_missing_box": bool(review.get("left_missing_box", False)),
        "right_missing_box": bool(review.get("right_missing_box", False)),
    }


def _state_signature(state: Dict[str, Optional[int] | bool]) -> tuple:
    return (
        state["left_track_id"],
        state["right_track_id"],
        bool(state["left_missing_box"]),
        bool(state["right_missing_box"]),
    )


def _frame_assignments_from_keyframes(bundle: Dict, review: Dict) -> Dict[str, object]:
    ordered_frames = [int(item["frame_idx"]) for item in bundle.get("frames", [])]
    if not ordered_frames:
        return {
            "left_missing_box": False,
            "right_missing_box": False,
            "left_track_ids": [],
            "right_track_ids": [],
            "assignment_segments": [],
            "kept_segments": [],
            "dropped_segments": [],
            "valid_frame_ranges": [],
            "frame_roles": {},
        }
    frame_to_pos = {frame_idx: pos for pos, frame_idx in enumerate(ordered_frames)}
    keyframe_reviews = {
        int(item["frame_idx"]): item
        for item in sorted(review.get("keyframe_reviews", []), key=lambda item: int(item["frame_idx"]))
    }
    bundle_keyframes = sorted(bundle.get("keyframes", []), key=lambda item: int(item["frame_idx"]))
    segments = sorted(bundle.get("segments", []), key=lambda item: int(item["start_frame"]))
    if not segments:
        segments = [
            {
                "segment_id": 0,
                "start_frame": int(ordered_frames[0]),
                "end_frame": int(ordered_frames[-1]),
            }
        ]
        if not bundle_keyframes:
            bundle_keyframes = [{"frame_idx": int(ordered_frames[0]), "segment_id": 0, "kind": "anchor"}]

    default_role = {
        "left_track_id": None,
        "right_track_id": None,
        "left_missing_box": False,
        "right_missing_box": False,
        "dropped": False,
    }
    frame_roles: Dict[int, Dict[str, Optional[int] | bool]] = {
        int(frame_idx): {**default_role} for frame_idx in ordered_frames
    }
    assignment_segments: List[Dict] = []
    for segment in segments:
        segment_id = int(segment["segment_id"])
        segment_start = int(segment["start_frame"])
        segment_end = int(segment["end_frame"])
        segment_keyframes = [
            keyframe_reviews[int(item["frame_idx"])]
            for item in bundle_keyframes
            if int(item.get("segment_id", segment_id)) == segment_id and int(item["frame_idx"]) in keyframe_reviews
        ]
        if not segment_keyframes:
            continue
        segment_keyframes.sort(key=lambda item: int(item["frame_idx"]))
        current_state = _normalize_keyframe_state(segment_keyframes[0])
        current_start_pos = frame_to_pos[segment_start]
        for next_review in segment_keyframes[1:]:
            next_frame = int(next_review["frame_idx"])
            next_state = _normalize_keyframe_state(next_review)
            if _state_signature(next_state) == _state_signature(current_state):
                continue
            next_pos = frame_to_pos[next_frame]
            chunk_end_pos = max(current_start_pos, next_pos - 1)
            assignment_segments.append(
                {
                    "segment_id": segment_id,
                    "start_frame": int(ordered_frames[current_start_pos]),
                    "end_frame": int(ordered_frames[chunk_end_pos]),
                    **current_state,
                }
            )
            current_start_pos = next_pos
            current_state = next_state
        assignment_segments.append(
            {
                "segment_id": segment_id,
                "start_frame": int(ordered_frames[current_start_pos]),
                "end_frame": segment_end,
                **current_state,
            }
        )

    dropped_segments: List[Dict] = []
    kept_segments: List[Dict] = []
    valid_frame_ranges: List[List[int]] = []
    for item in assignment_segments:
        is_dropped = bool(item["left_missing_box"]) or bool(item["right_missing_box"])
        segment_payload = {**item, "dropped": is_dropped}
        if is_dropped:
            dropped_segments.append(segment_payload)
        else:
            kept_segments.append(segment_payload)
            valid_frame_ranges.append([int(item["start_frame"]), int(item["end_frame"])])
        start_pos = frame_to_pos[int(item["start_frame"])]
        end_pos = frame_to_pos[int(item["end_frame"])]
        for pos in range(start_pos, end_pos + 1):
            frame_roles[int(ordered_frames[pos])] = {
                "left_track_id": item["left_track_id"],
                "right_track_id": item["right_track_id"],
                "left_missing_box": bool(item["left_missing_box"]),
                "right_missing_box": bool(item["right_missing_box"]),
                "dropped": is_dropped,
            }

    left_missing_box = any(bool(item["left_missing_box"]) for item in assignment_segments)
    right_missing_box = any(bool(item["right_missing_box"]) for item in assignment_segments)
    left_track_ids = sorted(
        {
            int(item["left_track_id"])
            for item in assignment_segments
            if item.get("left_track_id") is not None
        }
    )
    right_track_ids = sorted(
        {
            int(item["right_track_id"])
            for item in assignment_segments
            if item.get("right_track_id") is not None
        }
    )
    return {
        "left_missing_box": left_missing_box,
        "right_missing_box": right_missing_box,
        "left_track_ids": left_track_ids,
        "right_track_ids": right_track_ids,
        "assignment_segments": assignment_segments,
        "kept_segments": kept_segments,
        "dropped_segments": dropped_segments,
        "valid_frame_ranges": valid_frame_ranges,
        "frame_roles": frame_roles,
    }


def _build_side_observations_from_frame_roles(
    bundle: Dict,
    frame_roles: Dict[int, Dict[str, Optional[int] | bool]],
    side: str,
) -> List[SideObservations]:
    proposal_lookup = _track_proposal_lookup(bundle)
    ordered_frames = [int(item["frame_idx"]) for item in bundle.get("frames", [])]
    observations: List[SideObservations] = []
    current_track_id: Optional[int] = None
    current_frames: List[int] = []
    role_key = f"{side}_track_id"
    for frame_idx in ordered_frames:
        role = frame_roles.get(int(frame_idx), {})
        if bool(role.get("dropped", False)):
            track_id = None
        else:
            track_id = role.get(role_key)
        track_id = None if track_id is None else int(track_id)
        if current_track_id is not None and track_id == current_track_id:
            current_frames.append(int(frame_idx))
            continue
        if current_track_id is not None and current_frames:
            obs = _build_side_observation_from_frames(proposal_lookup, current_track_id, current_frames)
            if obs is not None:
                observations.append(obs)
        current_track_id = track_id
        current_frames = [int(frame_idx)] if track_id is not None else []
    if current_track_id is not None and current_frames:
        obs = _build_side_observation_from_frames(proposal_lookup, current_track_id, current_frames)
        if obs is not None:
            observations.append(obs)
    return observations


def _build_side_observations(bundle: Dict, track_ids: Sequence[int]) -> List[SideObservations]:
    observations: List[SideObservations] = []
    for track_id in track_ids:
        obs = _build_side_observation(bundle, int(track_id))
        if obs is not None:
            observations.append(obs)
    return observations


def fit_side(side: str, obs: SideObservations, _cfg: MediaPipeReviewConfig) -> Dict:
    boxes = obs.bbox_xyxy
    boxes_orig = obs.bbox_xyxy_orig
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
    centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
    return {
        "side": side,
        "track_id": obs.track_id,
        "source": obs.source,
        "frame_indices": obs.frame_indices,
        "bbox_xyxy": boxes.astype(np.float32).tolist(),
        "bbox_xyxy_orig": boxes_orig.astype(np.float32).tolist(),
        "score": obs.score.astype(np.float32).tolist(),
        "center_xy": np.stack([centers_x, centers_y], axis=1).astype(np.float32).tolist(),
        "box_size_wh": np.stack([widths, heights], axis=1).astype(np.float32).tolist(),
        "valid_ratio": 1.0,
    }


def _frame_lookup(bundle: Dict) -> Dict[int, Dict[str, Any]]:
    return {int(item["frame_idx"]): item for item in bundle.get("frames", [])}


def _preview_box_from_orig(frame_info: Optional[Dict[str, Any]], bbox_xyxy_orig: Sequence[float]) -> List[float]:
    if not frame_info:
        return [float(item) for item in bbox_xyxy_orig]
    orig_size = frame_info.get("orig_size") or []
    preview_size = frame_info.get("preview_size") or []
    if len(orig_size) != 2 or len(preview_size) != 2:
        return [float(item) for item in bbox_xyxy_orig]
    orig_w, orig_h = max(float(orig_size[0]), 1.0), max(float(orig_size[1]), 1.0)
    preview_w, preview_h = float(preview_size[0]), float(preview_size[1])
    scale_x = preview_w / orig_w
    scale_y = preview_h / orig_h
    x1, y1, x2, y2 = [float(item) for item in bbox_xyxy_orig]
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def _make_frame_box_payload(
    *,
    frame_idx: int,
    bbox_xyxy_orig: Sequence[float],
    score: float,
    frame_lookup: Dict[int, Dict[str, Any]],
    track_id: Optional[int],
    source: str,
) -> Dict[str, Any]:
    bbox_orig = [float(item) for item in bbox_xyxy_orig]
    return {
        "frame_idx": int(frame_idx),
        "bbox_xyxy_orig": bbox_orig,
        "bbox_xyxy": _preview_box_from_orig(frame_lookup.get(int(frame_idx)), bbox_orig),
        "score": float(score),
        "track_id": None if track_id is None else int(track_id),
        "source": str(source),
        "visible": True,
    }


def _make_observation_from_payloads(frame_payloads: List[Dict[str, Any]]) -> SideObservations:
    first = frame_payloads[0]
    return SideObservations(
        track_id=-1 if first.get("track_id") is None else int(first["track_id"]),
        frame_indices=[int(item["frame_idx"]) for item in frame_payloads],
        bbox_xyxy=np.asarray([item["bbox_xyxy"] for item in frame_payloads], dtype=np.float32),
        bbox_xyxy_orig=np.asarray([item["bbox_xyxy_orig"] for item in frame_payloads], dtype=np.float32),
        score=np.asarray([float(item.get("score", 0.0)) for item in frame_payloads], dtype=np.float32),
        source=str(first.get("source", "track")),
    )


def _interpolate_bbox(box_a: Sequence[float], box_b: Sequence[float], alpha: float) -> List[float]:
    return [
        float(box_a[0]) + (float(box_b[0]) - float(box_a[0])) * alpha,
        float(box_a[1]) + (float(box_b[1]) - float(box_a[1])) * alpha,
        float(box_a[2]) + (float(box_b[2]) - float(box_a[2])) * alpha,
        float(box_a[3]) + (float(box_b[3]) - float(box_a[3])) * alpha,
    ]


def _anchor_side_state(review: Dict[str, Any], side: str) -> Dict[str, Any]:
    mode = str(review.get(f"{side}_mode", "not_visible"))
    return {
        "mode": mode,
        "track_id": None if review.get(f"{side}_track_id") is None else int(review[f"{side}_track_id"]),
        "manual_bbox_xyxy_orig": None
        if review.get(f"{side}_manual_bbox_xyxy_orig") is None
        else [float(item) for item in review[f"{side}_manual_bbox_xyxy_orig"]],
    }


def _anchor_concrete_box(
    state: Dict[str, Any],
    *,
    frame_idx: int,
    proposal_lookup: Dict[int, Dict[int, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if state["mode"] == "manual_box":
        return {
            "bbox_xyxy_orig": list(state["manual_bbox_xyxy_orig"] or []),
            "score": 1.0,
            "track_id": None,
            "source": "manual_box",
        }
    if state["mode"] == "track" and state["track_id"] is not None:
        proposal = proposal_lookup.get(int(state["track_id"]), {}).get(int(frame_idx))
        if proposal is None:
            return None
        return {
            "bbox_xyxy_orig": [float(item) for item in proposal.get("bbox_xyxy_orig", proposal["bbox_xyxy"])],
            "score": float(proposal.get("score", 0.0)),
            "track_id": int(state["track_id"]),
            "source": "track_anchor",
        }
    return None


def _resolve_track_interval(
    *,
    track_id: int,
    interval_frames: Sequence[int],
    proposal_lookup: Dict[int, Dict[int, Dict[str, Any]]],
    frame_lookup: Dict[int, Dict[str, Any]],
    cfg: MediaPipeReviewConfig,
) -> Dict[str, Any]:
    payloads: Dict[int, Dict[str, Any]] = {}
    invalid_frames = set()
    track_proposals = proposal_lookup.get(int(track_id), {})
    known_positions = [idx for idx, frame_idx in enumerate(interval_frames) if int(frame_idx) in track_proposals]
    if not known_positions:
        return {"payloads": payloads, "invalid_frames": set(int(item) for item in interval_frames), "reason": "track_missing"}

    for pos in known_positions:
        frame_idx = int(interval_frames[pos])
        proposal = track_proposals[frame_idx]
        payloads[frame_idx] = _make_frame_box_payload(
            frame_idx=frame_idx,
            bbox_xyxy_orig=proposal.get("bbox_xyxy_orig", proposal["bbox_xyxy"]),
            score=float(proposal.get("score", 0.0)),
            frame_lookup=frame_lookup,
            track_id=int(track_id),
            source="track",
        )

    first_known = known_positions[0]
    last_known = known_positions[-1]
    invalid_frames.update(int(item) for item in interval_frames[:first_known])
    invalid_frames.update(int(item) for item in interval_frames[last_known + 1 :])
    max_gap = max(1, int(cfg.recovery_anchor_stride_frames))
    for left_pos, right_pos in zip(known_positions, known_positions[1:]):
        gap = right_pos - left_pos - 1
        if gap <= 0:
            continue
        left_frame = int(interval_frames[left_pos])
        right_frame = int(interval_frames[right_pos])
        left_payload = payloads[left_frame]
        right_payload = payloads[right_frame]
        if gap <= max_gap:
            for inner_offset in range(1, gap + 1):
                alpha = inner_offset / float(gap + 1)
                frame_idx = int(interval_frames[left_pos + inner_offset])
                payloads[frame_idx] = _make_frame_box_payload(
                    frame_idx=frame_idx,
                    bbox_xyxy_orig=_interpolate_bbox(
                        left_payload["bbox_xyxy_orig"],
                        right_payload["bbox_xyxy_orig"],
                        alpha,
                    ),
                    score=min(float(left_payload["score"]), float(right_payload["score"])),
                    frame_lookup=frame_lookup,
                    track_id=int(track_id),
                    source="track_gap_interp",
                )
        else:
            invalid_frames.update(int(item) for item in interval_frames[left_pos + 1 : right_pos])
    return {"payloads": payloads, "invalid_frames": invalid_frames, "reason": "track"}


def _resolve_concrete_interval(
    *,
    start_box: Dict[str, Any],
    end_box: Dict[str, Any],
    interval_frames: Sequence[int],
    frame_lookup: Dict[int, Dict[str, Any]],
    cfg: MediaPipeReviewConfig,
) -> Dict[str, Any]:
    payloads: Dict[int, Dict[str, Any]] = {}
    max_gap = max(1, int(cfg.recovery_anchor_stride_frames))
    if not interval_frames:
        return {"payloads": payloads, "invalid_frames": set(), "reason": "empty"}
    if len(interval_frames) == 1:
        frame_idx = int(interval_frames[0])
        only_box = start_box or end_box
        if only_box is None:
            return {"payloads": {}, "invalid_frames": {frame_idx}, "reason": "missing_anchor_box"}
        payloads[frame_idx] = _make_frame_box_payload(
            frame_idx=frame_idx,
            bbox_xyxy_orig=only_box["bbox_xyxy_orig"],
            score=float(only_box.get("score", 1.0)),
            frame_lookup=frame_lookup,
            track_id=only_box.get("track_id"),
            source=str(only_box.get("source", "manual_box")),
        )
        return {"payloads": payloads, "invalid_frames": set(), "reason": "single_frame"}
    if start_box is None or end_box is None:
        return {"payloads": payloads, "invalid_frames": set(int(item) for item in interval_frames), "reason": "missing_anchor_box"}
    if len(interval_frames) - 1 > max_gap:
        return {"payloads": payloads, "invalid_frames": set(int(item) for item in interval_frames), "reason": "anchor_gap_too_large"}
    uses_manual = start_box.get("source") == "manual_box" or end_box.get("source") == "manual_box"
    source = "manual_interp" if uses_manual else "anchor_interp"
    for offset, frame_idx in enumerate(interval_frames):
        alpha = offset / float(max(len(interval_frames) - 1, 1))
        payloads[int(frame_idx)] = _make_frame_box_payload(
            frame_idx=int(frame_idx),
            bbox_xyxy_orig=_interpolate_bbox(start_box["bbox_xyxy_orig"], end_box["bbox_xyxy_orig"], alpha),
            score=min(float(start_box.get("score", 1.0)), float(end_box.get("score", 1.0))),
            frame_lookup=frame_lookup,
            track_id=None,
            source=source,
        )
    return {"payloads": payloads, "invalid_frames": set(), "reason": source}


def _resolve_side_interval(
    *,
    start_review: Dict[str, Any],
    end_review: Dict[str, Any],
    side: str,
    interval_frames: Sequence[int],
    proposal_lookup: Dict[int, Dict[int, Dict[str, Any]]],
    frame_lookup: Dict[int, Dict[str, Any]],
    cfg: MediaPipeReviewConfig,
) -> Dict[str, Any]:
    start_state = _anchor_side_state(start_review, side)
    end_state = _anchor_side_state(end_review, side)
    if start_state["mode"] == "visible_unrecoverable" or end_state["mode"] == "visible_unrecoverable":
        return {"payloads": {}, "invalid_frames": set(int(item) for item in interval_frames), "reason": "visible_unrecoverable"}
    if start_state["mode"] == "not_visible" and end_state["mode"] == "not_visible":
        return {"payloads": {}, "invalid_frames": set(), "reason": "not_visible"}
    if (
        start_state["mode"] == "track"
        and end_state["mode"] == "track"
        and start_state["track_id"] is not None
        and start_state["track_id"] == end_state["track_id"]
    ):
        return _resolve_track_interval(
            track_id=int(start_state["track_id"]),
            interval_frames=interval_frames,
            proposal_lookup=proposal_lookup,
            frame_lookup=frame_lookup,
            cfg=cfg,
        )
    start_box = _anchor_concrete_box(start_state, frame_idx=int(start_review["frame_idx"]), proposal_lookup=proposal_lookup)
    end_box = _anchor_concrete_box(end_state, frame_idx=int(end_review["frame_idx"]), proposal_lookup=proposal_lookup)
    if start_box is None and end_box is None:
        return {"payloads": {}, "invalid_frames": set(int(item) for item in interval_frames), "reason": "unsupported_transition"}
    return _resolve_concrete_interval(
        start_box=start_box,
        end_box=end_box,
        interval_frames=interval_frames,
        frame_lookup=frame_lookup,
        cfg=cfg,
    )


def _frame_reviews_by_segment(bundle: Dict, review: Dict) -> Dict[int, List[Dict[str, Any]]]:
    frame_to_segment = {int(item["frame_idx"]): int(item.get("segment_id", -1)) for item in bundle.get("frames", [])}
    by_segment: Dict[int, List[Dict[str, Any]]] = {}
    for item in review.get("frame_reviews", []):
        frame_idx = int(item["frame_idx"])
        segment_id = int(frame_to_segment.get(frame_idx, -1))
        by_segment.setdefault(segment_id, []).append({**item})
    for reviews in by_segment.values():
        reviews.sort(key=lambda item: int(item["frame_idx"]))
    return by_segment


def _build_frame_review_v2_result(bundle: Dict, review: Dict, cfg: MediaPipeReviewConfig) -> Dict[str, Any]:
    ordered_frames = [int(item["frame_idx"]) for item in bundle.get("frames", [])]
    frame_to_pos = {frame_idx: pos for pos, frame_idx in enumerate(ordered_frames)}
    frame_lookup = _frame_lookup(bundle)
    proposal_lookup = _track_proposal_lookup(bundle)
    segment_reviews = _frame_reviews_by_segment(bundle, review)
    per_frame: Dict[int, Dict[str, Any]] = {
        int(frame_idx): {
            "frame_idx": int(frame_idx),
            "segment_id": int(frame_lookup.get(int(frame_idx), {}).get("segment_id", -1)),
            "left": {"visible": False, "source": "not_visible", "track_id": None, "score": 0.0},
            "right": {"visible": False, "source": "not_visible", "track_id": None, "score": 0.0},
            "invalid_reasons": set(),
        }
        for frame_idx in ordered_frames
    }
    assignment_segments: List[Dict[str, Any]] = []
    for segment in sorted(bundle.get("segments", []), key=lambda item: int(item["start_frame"])):
        segment_id = int(segment["segment_id"])
        reviews = list(segment_reviews.get(segment_id, []))
        if not reviews:
            continue
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
        if int(reviews[0]["frame_idx"]) > start_frame:
            reviews.insert(0, {**reviews[0], "frame_idx": start_frame, "synthetic_boundary": True})
        if int(reviews[-1]["frame_idx"]) < end_frame:
            reviews.append({**reviews[-1], "frame_idx": end_frame, "synthetic_boundary": True})
        if len(reviews) == 1:
            reviews.append({**reviews[0], "frame_idx": int(reviews[0]["frame_idx"]), "synthetic_boundary": True})
        for pair_idx in range(len(reviews) - 1):
            start_review = reviews[pair_idx]
            end_review = reviews[pair_idx + 1]
            start_pos = frame_to_pos[int(start_review["frame_idx"])]
            end_pos = frame_to_pos[int(end_review["frame_idx"])]
            interval_frames = ordered_frames[start_pos : end_pos + 1]
            if pair_idx > 0:
                interval_frames = interval_frames[1:]
            if not interval_frames:
                continue
            left_result = _resolve_side_interval(
                start_review=start_review,
                end_review=end_review,
                side="left",
                interval_frames=interval_frames,
                proposal_lookup=proposal_lookup,
                frame_lookup=frame_lookup,
                cfg=cfg,
            )
            right_result = _resolve_side_interval(
                start_review=start_review,
                end_review=end_review,
                side="right",
                interval_frames=interval_frames,
                proposal_lookup=proposal_lookup,
                frame_lookup=frame_lookup,
                cfg=cfg,
            )
            for frame_idx in interval_frames:
                frame_state = per_frame[int(frame_idx)]
                left_payload = left_result["payloads"].get(int(frame_idx))
                if left_payload is not None:
                    frame_state["left"] = left_payload
                right_payload = right_result["payloads"].get(int(frame_idx))
                if right_payload is not None:
                    frame_state["right"] = right_payload
                if int(frame_idx) in left_result["invalid_frames"]:
                    frame_state["invalid_reasons"].add(str(left_result["reason"]))
                if int(frame_idx) in right_result["invalid_frames"]:
                    frame_state["invalid_reasons"].add(str(right_result["reason"]))
            assignment_segments.append(
                {
                    "segment_id": segment_id,
                    "start_frame": int(interval_frames[0]),
                    "end_frame": int(interval_frames[-1]),
                    "left_mode_start": str(start_review.get("left_mode", "not_visible")),
                    "left_mode_end": str(end_review.get("left_mode", "not_visible")),
                    "right_mode_start": str(start_review.get("right_mode", "not_visible")),
                    "right_mode_end": str(end_review.get("right_mode", "not_visible")),
                    "left_reason": str(left_result["reason"]),
                    "right_reason": str(right_result["reason"]),
                }
            )

    usable_frames = [
        frame_idx
        for frame_idx in ordered_frames
        if not per_frame[frame_idx]["invalid_reasons"]
        and (bool(per_frame[frame_idx]["left"].get("visible")) or bool(per_frame[frame_idx]["right"].get("visible")))
    ]
    usable_set = set(int(item) for item in usable_frames)
    valid_spans: List[Dict[str, int]] = []
    short_spans: List[Dict[str, int]] = []
    current_start_pos: Optional[int] = None
    previous_pos: Optional[int] = None

    def _close_usable_span(end_pos: int) -> None:
        nonlocal current_start_pos
        if current_start_pos is None:
            return
        span = {
            "start_pos": int(current_start_pos),
            "end_pos": int(end_pos),
            "start_frame": int(ordered_frames[current_start_pos]),
            "end_frame": int(ordered_frames[end_pos]),
            "num_frames": int(end_pos - current_start_pos + 1),
        }
        if span["num_frames"] >= int(cfg.min_export_episode_frames):
            valid_spans.append(span)
        else:
            short_spans.append(span)
        current_start_pos = None

    for pos, frame_idx in enumerate(ordered_frames):
        if frame_idx in usable_set:
            if current_start_pos is None:
                current_start_pos = int(pos)
            previous_pos = int(pos)
            continue
        if current_start_pos is not None and previous_pos is not None:
            _close_usable_span(previous_pos)
        previous_pos = None
    if current_start_pos is not None and previous_pos is not None:
        _close_usable_span(previous_pos)

    dropped_segments: List[Dict[str, Any]] = []
    invalid_start_pos: Optional[int] = None
    invalid_reasons: set[str] = set()
    previous_invalid_pos: Optional[int] = None
    for pos, frame_idx in enumerate(ordered_frames):
        frame_invalid = bool(per_frame[frame_idx]["invalid_reasons"])
        if frame_invalid:
            if invalid_start_pos is None:
                invalid_start_pos = int(pos)
                invalid_reasons = set(str(item) for item in per_frame[frame_idx]["invalid_reasons"])
            else:
                invalid_reasons.update(str(item) for item in per_frame[frame_idx]["invalid_reasons"])
            previous_invalid_pos = int(pos)
        elif invalid_start_pos is not None and previous_invalid_pos is not None:
            dropped_segments.append(
                {
                    "start_frame": int(ordered_frames[invalid_start_pos]),
                    "end_frame": int(ordered_frames[previous_invalid_pos]),
                    "num_frames": int(previous_invalid_pos - invalid_start_pos + 1),
                    "reason": " / ".join(sorted(invalid_reasons)) or "invalid_interval",
                }
            )
            invalid_start_pos = None
            previous_invalid_pos = None
            invalid_reasons = set()
    if invalid_start_pos is not None and previous_invalid_pos is not None:
        dropped_segments.append(
            {
                "start_frame": int(ordered_frames[invalid_start_pos]),
                "end_frame": int(ordered_frames[previous_invalid_pos]),
                "num_frames": int(previous_invalid_pos - invalid_start_pos + 1),
                "reason": " / ".join(sorted(invalid_reasons)) or "invalid_interval",
            }
        )
    for span in short_spans:
        dropped_segments.append(
            {
                "start_frame": int(span["start_frame"]),
                "end_frame": int(span["end_frame"]),
                "num_frames": int(span["num_frames"]),
                "reason": "short_subepisode",
            }
        )

    def _build_fragments_for_range(side: str, frame_indices: Sequence[int]) -> List[Dict[str, Any]]:
        fragments: List[Dict[str, Any]] = []
        current_payloads: List[Dict[str, Any]] = []
        current_key: Optional[tuple] = None
        for frame_idx in frame_indices:
            side_payload = per_frame[int(frame_idx)][side]
            visible = bool(side_payload.get("visible", False))
            key = None
            if visible:
                key = (
                    side_payload.get("track_id"),
                    str(side_payload.get("source", "")),
                )
            if visible and current_key == key:
                current_payloads.append(side_payload)
                continue
            if current_payloads:
                fragments.append(fit_side(side, _make_observation_from_payloads(current_payloads), cfg))
            current_payloads = [side_payload] if visible else []
            current_key = key
        if current_payloads:
            fragments.append(fit_side(side, _make_observation_from_payloads(current_payloads), cfg))
        return fragments

    subepisodes: List[Dict[str, Any]] = []
    all_left_fragments: List[Dict[str, Any]] = []
    all_right_fragments: List[Dict[str, Any]] = []
    for subepisode_index, span in enumerate(valid_spans):
        frame_indices = ordered_frames[int(span["start_pos"]) : int(span["end_pos"]) + 1]
        left_fragments = _build_fragments_for_range("left", frame_indices)
        right_fragments = _build_fragments_for_range("right", frame_indices)
        subepisode = {
            "subepisode_index": int(subepisode_index),
            "start_frame": int(span["start_frame"]),
            "end_frame": int(span["end_frame"]),
            "num_frames": int(span["num_frames"]),
            "sides": {
                "left": {"fragments": left_fragments},
                "right": {"fragments": right_fragments},
            },
        }
        subepisodes.append(subepisode)
        all_left_fragments.extend(left_fragments)
        all_right_fragments.extend(right_fragments)

    left_track_ids = sorted(
        {
            int(item["left_track_id"])
            for item in review.get("frame_reviews", [])
            if item.get("left_mode") == "track" and item.get("left_track_id") is not None
        }
    )
    right_track_ids = sorted(
        {
            int(item["right_track_id"])
            for item in review.get("frame_reviews", [])
            if item.get("right_mode") == "track" and item.get("right_track_id") is not None
        }
    )
    left_unrecoverable = any(str(item.get("left_mode", "")) == "visible_unrecoverable" for item in review.get("frame_reviews", []))
    right_unrecoverable = any(str(item.get("right_mode", "")) == "visible_unrecoverable" for item in review.get("frame_reviews", []))
    mean_scores = [float(fragment_score) for fragment in all_left_fragments + all_right_fragments for fragment_score in fragment["score"]]
    num_box_frames = sum(len(fragment["frame_indices"]) for fragment in all_left_fragments + all_right_fragments)
    fit = {
        "clip_id": bundle["clip_id"],
        "review_version": "frame_review_v2",
        "box_only": True,
        "track_generation": bundle.get("track_generation", ""),
        "left_track_ids": left_track_ids,
        "right_track_ids": right_track_ids,
        "left_missing_box": left_unrecoverable,
        "right_missing_box": right_unrecoverable,
        "assignment_segments": assignment_segments,
        "kept_segments": [
            {
                "start_frame": int(item["start_frame"]),
                "end_frame": int(item["end_frame"]),
                "num_frames": int(item["num_frames"]),
                "subepisode_index": int(item["subepisode_index"]),
            }
            for item in subepisodes
        ],
        "dropped_segments": sorted(dropped_segments, key=lambda item: (int(item["start_frame"]), int(item["end_frame"]))),
        "valid_frame_ranges": [[int(item["start_frame"]), int(item["end_frame"])] for item in valid_spans],
        "subepisodes": subepisodes,
        "sides": {
            "left": {"missing_box": left_unrecoverable, "fragments": all_left_fragments},
            "right": {"missing_box": right_unrecoverable, "fragments": all_right_fragments},
        },
        "num_box_frames": int(num_box_frames),
        "mean_box_score": float(np.mean(mean_scores)) if mean_scores else 0.0,
        "median_reproj_error": 0.0,
        "p95_reproj_error": 0.0,
    }
    if not subepisodes:
        fit["status"] = "fit_dropped"
        fit["message"] = "No valid subepisode reached the minimum export length."
    else:
        fit["status"] = "fit_ok"
        if dropped_segments:
            fit["message"] = "Some frame ranges were dropped during conservative recovery."
    return fit


def fit_reviewed_clip(bundle: Dict, review: Dict, cfg: MediaPipeReviewConfig) -> Dict:
    review_version = str(review.get("review_version") or "")
    if review_version == "frame_review_v2":
        return _build_frame_review_v2_result(bundle, review, cfg)

    assignment_segments: List[Dict] = []
    if review_version == "keyframe_v1":
        derived = _frame_assignments_from_keyframes(bundle, review)
        left_track_ids = list(derived["left_track_ids"])
        right_track_ids = list(derived["right_track_ids"])
        left_missing_box = bool(derived["left_missing_box"])
        right_missing_box = bool(derived["right_missing_box"])
        assignment_segments = list(derived["assignment_segments"])
        kept_segments = list(derived["kept_segments"])
        dropped_segments = list(derived["dropped_segments"])
        valid_frame_ranges = list(derived["valid_frame_ranges"])
        frame_roles = dict(derived["frame_roles"])
        left_observations = _build_side_observations_from_frame_roles(bundle, frame_roles, "left")
        right_observations = _build_side_observations_from_frame_roles(bundle, frame_roles, "right")
    else:
        left_track_ids = [int(item) for item in review.get("left_track_ids", [])]
        right_track_ids = [int(item) for item in review.get("right_track_ids", [])]
        left_missing_box = bool(review.get("left_missing_box", False))
        right_missing_box = bool(review.get("right_missing_box", False))
        kept_segments = []
        dropped_segments = []
        valid_frame_ranges = []
        left_observations = _build_side_observations(bundle, left_track_ids)
        right_observations = _build_side_observations(bundle, right_track_ids)

    fit = {
        "clip_id": bundle["clip_id"],
        "review_version": review_version or "legacy_track_v0",
        "box_only": True,
        "track_generation": bundle.get("track_generation", ""),
        "left_track_ids": left_track_ids,
        "right_track_ids": right_track_ids,
        "left_missing_box": left_missing_box,
        "right_missing_box": right_missing_box,
        "assignment_segments": assignment_segments,
        "kept_segments": kept_segments,
        "dropped_segments": dropped_segments,
        "valid_frame_ranges": valid_frame_ranges,
        "sides": {
            "left": {"missing_box": left_missing_box, "fragments": []},
            "right": {"missing_box": right_missing_box, "fragments": []},
        },
    }
    for obs in left_observations:
        fit["sides"]["left"]["fragments"].append(fit_side("left", obs, cfg))
    for obs in right_observations:
        fit["sides"]["right"]["fragments"].append(fit_side("right", obs, cfg))

    num_box_frames = 0
    score_values: List[float] = []
    for side_name in ("left", "right"):
        for fragment in fit["sides"][side_name]["fragments"]:
            num_box_frames += len(fragment["frame_indices"])
            score_values.extend(float(item) for item in fragment["score"])
    fit["num_box_frames"] = int(num_box_frames)
    fit["mean_box_score"] = float(np.mean(score_values)) if score_values else 0.0
    fit["median_reproj_error"] = 0.0
    fit["p95_reproj_error"] = 0.0

    if not fit["sides"]["left"]["fragments"] and not fit["sides"]["right"]["fragments"]:
        if dropped_segments:
            fit["status"] = "qa_needed_missing_box"
            fit["message"] = "All usable frames were dropped because at least one governing keyframe had a missing proposal."
        else:
            fit["status"] = "fit_ok_negative"
            fit["message"] = "No wearer hand box was selected in this clip."
        return fit

    fit["status"] = "fit_ok"
    if dropped_segments:
        fit["message"] = "Some governed frame ranges were dropped because a keyframe was marked missing_box."
    return fit


def _empty_fragment_arrays(prefix: str, arrays: Dict[str, np.ndarray]) -> None:
    arrays[prefix + "fragment_track_ids"] = np.zeros((0,), dtype=np.int32)
    arrays[prefix + "fragment_source_codes"] = np.zeros((0,), dtype=np.int8)
    arrays[prefix + "fragment_offsets"] = np.zeros((1,), dtype=np.int32)
    arrays[prefix + "frame_indices"] = np.zeros((0,), dtype=np.int32)
    arrays[prefix + "bbox_xyxy"] = np.zeros((0, 4), dtype=np.float32)
    arrays[prefix + "bbox_xyxy_orig"] = np.zeros((0, 4), dtype=np.float32)
    arrays[prefix + "score"] = np.zeros((0,), dtype=np.float32)


def _flatten_side_fragments(side_payload: Optional[Dict], prefix: str, arrays: Dict[str, np.ndarray]) -> None:
    fragments = list((side_payload or {}).get("fragments", []))
    arrays[prefix + "missing_box"] = np.asarray([1 if bool((side_payload or {}).get("missing_box", False)) else 0], dtype=np.int8)
    if not fragments:
        _empty_fragment_arrays(prefix, arrays)
        return

    track_ids: List[int] = []
    source_codes: List[int] = []
    offsets = [0]
    frame_indices: List[np.ndarray] = []
    bbox_xyxy: List[np.ndarray] = []
    bbox_xyxy_orig: List[np.ndarray] = []
    score: List[np.ndarray] = []
    total = 0
    source_code_lookup = {
        "track": 0,
        "track_gap_interp": 1,
        "manual_box": 2,
        "manual_interp": 3,
        "anchor_interp": 4,
        "track_anchor": 5,
    }
    for fragment in fragments:
        frames = np.asarray(fragment["frame_indices"], dtype=np.int32)
        total += len(frames)
        offsets.append(total)
        track_ids.append(int(fragment["track_id"]))
        source_codes.append(int(source_code_lookup.get(str(fragment.get("source", "track")), 127)))
        frame_indices.append(frames)
        bbox_xyxy.append(np.asarray(fragment["bbox_xyxy"], dtype=np.float32))
        bbox_xyxy_orig.append(np.asarray(fragment["bbox_xyxy_orig"], dtype=np.float32))
        score.append(np.asarray(fragment["score"], dtype=np.float32))

    arrays[prefix + "fragment_track_ids"] = np.asarray(track_ids, dtype=np.int32)
    arrays[prefix + "fragment_source_codes"] = np.asarray(source_codes, dtype=np.int8)
    arrays[prefix + "fragment_offsets"] = np.asarray(offsets, dtype=np.int32)
    arrays[prefix + "frame_indices"] = np.concatenate(frame_indices, axis=0) if frame_indices else np.zeros((0,), dtype=np.int32)
    arrays[prefix + "bbox_xyxy"] = np.concatenate(bbox_xyxy, axis=0) if bbox_xyxy else np.zeros((0, 4), dtype=np.float32)
    arrays[prefix + "bbox_xyxy_orig"] = (
        np.concatenate(bbox_xyxy_orig, axis=0) if bbox_xyxy_orig else np.zeros((0, 4), dtype=np.float32)
    )
    arrays[prefix + "score"] = np.concatenate(score, axis=0) if score else np.zeros((0,), dtype=np.float32)


def _ranges_array(ranges: Sequence[Sequence[int]]) -> np.ndarray:
    if not ranges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray([[int(item[0]), int(item[1])] for item in ranges], dtype=np.int32)


def save_fit_artifacts(bundle_dir: Path, fit_payload: Dict) -> Dict[str, str]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    json_path = bundle_dir / "fit.json"
    npz_path = bundle_dir / "fit.npz"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(fit_payload, f, ensure_ascii=False, indent=2)

    arrays = {}
    for side in ("left", "right"):
        prefix = f"{side}_"
        _flatten_side_fragments(fit_payload.get("sides", {}).get(side), prefix, arrays)
    arrays["valid_frame_ranges"] = _ranges_array(fit_payload.get("valid_frame_ranges", []))
    arrays["dropped_frame_ranges"] = _ranges_array(
        [[int(item["start_frame"]), int(item["end_frame"])] for item in fit_payload.get("dropped_segments", [])]
    )
    arrays["subepisode_ranges"] = _ranges_array(
        [[int(item["start_frame"]), int(item["end_frame"])] for item in fit_payload.get("subepisodes", [])]
    )
    arrays["subepisode_lengths"] = np.asarray(
        [int(item.get("num_frames", 0)) for item in fit_payload.get("subepisodes", [])],
        dtype=np.int32,
    )
    arrays["num_box_frames"] = np.asarray([fit_payload.get("num_box_frames", 0)], dtype=np.int32)
    arrays["mean_box_score"] = np.asarray([fit_payload.get("mean_box_score", 0.0)], dtype=np.float32)
    arrays["median_reproj_error"] = np.asarray([fit_payload.get("median_reproj_error", 0.0)], dtype=np.float32)
    arrays["p95_reproj_error"] = np.asarray([fit_payload.get("p95_reproj_error", 0.0)], dtype=np.float32)
    np.savez_compressed(npz_path, **arrays)
    return {"fit_json": str(json_path), "fit_npz": str(npz_path)}
