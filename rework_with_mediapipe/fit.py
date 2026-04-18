from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MediaPipeReviewConfig


def _frame_lookup(bundle: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(item["frame_idx"]): item for item in bundle.get("frames", [])}


def _proposal_lookup(bundle: Dict[str, Any]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    return {
        int(item["frame_idx"]): {str(proposal["proposal_id"]): proposal for proposal in item.get("proposals", [])}
        for item in bundle.get("frame_proposals", [])
    }


def _ordered_frame_indices(bundle: Dict[str, Any]) -> List[int]:
    return [int(item["frame_idx"]) for item in bundle.get("frames", [])]


def _bbox_area_xyxy(bbox_xyxy: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(item) for item in bbox_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center_xyxy(bbox_xyxy: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(item) for item in bbox_xyxy]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _bbox_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(item) for item in box_a]
    bx1, by1, bx2, by2 = [float(item) for item in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = _bbox_area_xyxy((inter_x1, inter_y1, inter_x2, inter_y2))
    if inter_area <= 0.0:
        return 0.0
    union = _bbox_area_xyxy(box_a) + _bbox_area_xyxy(box_b) - inter_area
    return inter_area / max(union, 1e-6)


def _lerp_box(box_a: Sequence[float], box_b: Sequence[float], t: float) -> List[float]:
    return [float(a) + (float(b) - float(a)) * float(t) for a, b in zip(box_a, box_b)]


def _scale_bbox_orig_to_preview(frame_meta: Dict[str, Any], bbox_xyxy_orig: Sequence[float]) -> List[float]:
    orig_width, orig_height = [int(item) for item in frame_meta.get("orig_size", [1, 1])]
    preview_width, preview_height = [int(item) for item in frame_meta.get("preview_size", [orig_width, orig_height])]
    scale_x = preview_width / max(float(orig_width), 1.0)
    scale_y = preview_height / max(float(orig_height), 1.0)
    x1, y1, x2, y2 = [float(item) for item in bbox_xyxy_orig]
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def _state_from_review(
    review: Dict[str, Any],
    side: str,
    frame_lookup: Dict[int, Dict[str, Any]],
    proposal_lookup: Dict[int, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    frame_idx = int(review["frame_idx"])
    mode = str(review.get(f"{side}_mode", "absent"))
    state = {
        "frame_idx": frame_idx,
        "mode": mode,
        "visible": False,
        "bbox_xyxy": None,
        "bbox_xyxy_orig": None,
        "score": 0.0,
        "proposal_id": None,
        "source": mode,
    }
    if mode == "proposal":
        proposal_id = str(review.get(f"{side}_proposal_id") or "")
        proposal = proposal_lookup.get(frame_idx, {}).get(proposal_id)
        if proposal is None:
            raise RuntimeError(f"Frame {frame_idx} {side} proposal {proposal_id} is missing from bundle")
        state.update(
            {
                "visible": True,
                "bbox_xyxy": list(proposal["bbox_xyxy"]),
                "bbox_xyxy_orig": list(proposal["bbox_xyxy_orig"]),
                "score": float(proposal.get("score", 0.0)),
                "proposal_id": proposal_id,
                "source": "proposal_anchor",
            }
        )
    elif mode == "manual_box":
        frame_meta = frame_lookup.get(frame_idx)
        if frame_meta is None:
            raise RuntimeError(f"Frame {frame_idx} metadata is missing")
        bbox_xyxy_orig = [float(item) for item in review[f"{side}_manual_bbox_xyxy_orig"]]
        state.update(
            {
                "visible": True,
                "bbox_xyxy_orig": bbox_xyxy_orig,
                "bbox_xyxy": _scale_bbox_orig_to_preview(frame_meta, bbox_xyxy_orig),
                "score": 1.0,
                "proposal_id": None,
                "source": "manual_anchor",
            }
        )
    elif mode == "absent":
        state["source"] = "absent"
    elif mode == "unusable":
        state["source"] = "unusable"
    else:
        raise RuntimeError(f"Unsupported review mode: {mode}")
    return state


def _proposal_state(frame_idx: int, proposal: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "frame_idx": int(frame_idx),
        "mode": "proposal",
        "visible": True,
        "bbox_xyxy": list(proposal["bbox_xyxy"]),
        "bbox_xyxy_orig": list(proposal["bbox_xyxy_orig"]),
        "score": float(proposal.get("score", 0.0)),
        "proposal_id": str(proposal.get("proposal_id", "")),
        "source": "propagated_proposal",
    }


def _null_state(frame_idx: int) -> Dict[str, Any]:
    return {
        "frame_idx": int(frame_idx),
        "mode": "absent",
        "visible": False,
        "bbox_xyxy": None,
        "bbox_xyxy_orig": None,
        "score": 0.0,
        "proposal_id": None,
        "source": "absent",
    }


def _transition_score(prev_state: Dict[str, Any], curr_state: Dict[str, Any], image_diag: float) -> float:
    if not prev_state["visible"] and not curr_state["visible"]:
        return 0.0
    if not prev_state["visible"] or not curr_state["visible"]:
        return -0.45
    iou = _bbox_iou_xyxy(prev_state["bbox_xyxy_orig"], curr_state["bbox_xyxy_orig"])
    prev_center = _bbox_center_xyxy(prev_state["bbox_xyxy_orig"])
    curr_center = _bbox_center_xyxy(curr_state["bbox_xyxy_orig"])
    center_delta = math.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1]) / max(image_diag, 1.0)
    prev_area = max(_bbox_area_xyxy(prev_state["bbox_xyxy_orig"]), 1.0)
    curr_area = max(_bbox_area_xyxy(curr_state["bbox_xyxy_orig"]), 1.0)
    area_delta = abs(math.log(curr_area / prev_area))
    return (2.4 * iou) - (0.8 * center_delta) - (0.35 * area_delta)


def _emission_score(
    state: Dict[str, Any],
    *,
    frame_idx: int,
    start_state: Dict[str, Any],
    end_state: Dict[str, Any],
    image_diag: float,
) -> float:
    target_visible = bool(start_state["visible"]) or bool(end_state["visible"])
    if not state["visible"]:
        return -0.7 if target_visible else 0.0
    score = float(state.get("score", 0.0)) * 1.6
    if start_state["visible"] and end_state["visible"] and int(end_state["frame_idx"]) != int(start_state["frame_idx"]):
        t = (float(frame_idx) - float(start_state["frame_idx"])) / max(float(end_state["frame_idx"] - start_state["frame_idx"]), 1.0)
        target_box = _lerp_box(start_state["bbox_xyxy_orig"], end_state["bbox_xyxy_orig"], t)
        score += 0.9 * _bbox_iou_xyxy(state["bbox_xyxy_orig"], target_box)
    elif start_state["visible"]:
        center = _bbox_center_xyxy(state["bbox_xyxy_orig"])
        anchor_center = _bbox_center_xyxy(start_state["bbox_xyxy_orig"])
        dist = math.hypot(center[0] - anchor_center[0], center[1] - anchor_center[1]) / max(image_diag, 1.0)
        score += max(-0.5, 0.4 - dist)
    elif end_state["visible"]:
        center = _bbox_center_xyxy(state["bbox_xyxy_orig"])
        anchor_center = _bbox_center_xyxy(end_state["bbox_xyxy_orig"])
        dist = math.hypot(center[0] - anchor_center[0], center[1] - anchor_center[1]) / max(image_diag, 1.0)
        score += max(-0.5, 0.4 - dist)
    return float(score)


def _resolve_interval_path(
    *,
    inner_frames: Sequence[int],
    start_state: Dict[str, Any],
    end_state: Dict[str, Any],
    proposal_lookup: Dict[int, Dict[str, Dict[str, Any]]],
    image_diag: float,
    cfg: MediaPipeReviewConfig,
) -> Tuple[Dict[int, Dict[str, Any]], Optional[str]]:
    if not inner_frames:
        return {}, None
    if start_state["mode"] == "unusable" or end_state["mode"] == "unusable":
        return {}, "unusable_anchor"
    if not start_state["visible"] and not end_state["visible"]:
        return {int(frame_idx): _null_state(int(frame_idx)) for frame_idx in inner_frames}, None

    candidate_sets: List[List[Dict[str, Any]]] = []
    for frame_idx in inner_frames:
        frame_candidates = [_null_state(frame_idx)]
        for proposal in proposal_lookup.get(int(frame_idx), {}).values():
            frame_candidates.append(_proposal_state(int(frame_idx), proposal))
        candidate_sets.append(frame_candidates)

    scores: List[List[float]] = []
    backpointers: List[List[int]] = []
    for layer_idx, states in enumerate(candidate_sets):
        frame_idx = int(inner_frames[layer_idx])
        layer_scores = [float("-inf")] * len(states)
        layer_back = [-1] * len(states)
        for state_idx, state in enumerate(states):
            emission = _emission_score(
                state,
                frame_idx=frame_idx,
                start_state=start_state,
                end_state=end_state,
                image_diag=image_diag,
            )
            if layer_idx == 0:
                layer_scores[state_idx] = emission + _transition_score(start_state, state, image_diag)
                continue
            best_score = float("-inf")
            best_prev = -1
            for prev_idx, prev_state in enumerate(candidate_sets[layer_idx - 1]):
                candidate_score = scores[layer_idx - 1][prev_idx] + _transition_score(prev_state, state, image_diag) + emission
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_prev = prev_idx
            layer_scores[state_idx] = best_score
            layer_back[state_idx] = best_prev
        scores.append(layer_scores)
        backpointers.append(layer_back)

    best_last_score = float("-inf")
    best_last_idx = -1
    for state_idx, state in enumerate(candidate_sets[-1]):
        candidate_score = scores[-1][state_idx] + _transition_score(state, end_state, image_diag)
        if candidate_score > best_last_score:
            best_last_score = candidate_score
            best_last_idx = state_idx
    if best_last_idx < 0:
        return {}, "empty_path"

    path: List[Dict[str, Any]] = []
    pointer = best_last_idx
    for layer_idx in range(len(candidate_sets) - 1, -1, -1):
        path.append(candidate_sets[layer_idx][pointer])
        pointer = backpointers[layer_idx][pointer]
        if layer_idx > 0 and pointer < 0:
            return {}, "broken_path"
    path.reverse()

    average_score = best_last_score / max(len(path), 1)
    min_average_score = float(cfg.raw.get("propagation_min_average_score", 0.15))
    if (start_state["visible"] or end_state["visible"]) and average_score < min_average_score:
        return {}, "low_confidence_path"

    resolved = {int(frame_idx): state for frame_idx, state in zip(inner_frames, path)}
    return resolved, None


def _propagate_side(
    bundle: Dict[str, Any],
    review_payload: Dict[str, Any],
    side: str,
    cfg: MediaPipeReviewConfig,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, set[str]], List[Dict[str, Any]]]:
    frame_lookup = _frame_lookup(bundle)
    proposal_lookup = _proposal_lookup(bundle)
    ordered_frames = _ordered_frame_indices(bundle)
    review_items = sorted(review_payload.get("frame_reviews", []), key=lambda item: int(item["frame_idx"]))
    anchors = [_state_from_review(item, side, frame_lookup, proposal_lookup) for item in review_items]
    if not anchors:
        raise RuntimeError("review payload is missing frame anchors")

    image_diag = 1.0
    if ordered_frames:
        first_frame = frame_lookup.get(int(ordered_frames[0]), {})
        preview_width, preview_height = [int(item) for item in first_frame.get("orig_size", [1, 1])]
        image_diag = max(math.hypot(float(preview_width), float(preview_height)), 1.0)

    frame_results: Dict[int, Dict[str, Any]] = {}
    invalid_reasons: Dict[int, set[str]] = defaultdict(set)
    assignment_segments: List[Dict[str, Any]] = []

    for anchor in anchors:
        frame_results[int(anchor["frame_idx"])] = dict(anchor)

    frame_to_pos = {int(frame_idx): pos for pos, frame_idx in enumerate(ordered_frames)}
    for start_anchor, end_anchor in zip(anchors[:-1], anchors[1:]):
        start_pos = frame_to_pos[int(start_anchor["frame_idx"])]
        end_pos = frame_to_pos[int(end_anchor["frame_idx"])]
        segment_frames = ordered_frames[start_pos : end_pos + 1]
        inner_frames = segment_frames[1:-1]
        interval_payloads, interval_error = _resolve_interval_path(
            inner_frames=inner_frames,
            start_state=start_anchor,
            end_state=end_anchor,
            proposal_lookup=proposal_lookup,
            image_diag=image_diag,
            cfg=cfg,
        )
        if interval_error is not None:
            for frame_idx in segment_frames:
                invalid_reasons[int(frame_idx)].add(interval_error)
            assignment_segments.append(
                {
                    "side": side,
                    "start_frame": int(segment_frames[0]),
                    "end_frame": int(segment_frames[-1]),
                    "reason": interval_error,
                    "status": "dropped",
                }
            )
            continue
        for frame_idx, payload in interval_payloads.items():
            frame_results[int(frame_idx)] = payload
        assignment_segments.append(
            {
                "side": side,
                "start_frame": int(segment_frames[0]),
                "end_frame": int(segment_frames[-1]),
                "reason": "propagated",
                "status": "ok",
            }
        )

    return frame_results, invalid_reasons, assignment_segments


def _default_side_payload(frame_idx: int) -> Dict[str, Any]:
    payload = _null_state(frame_idx)
    payload["source"] = "absent"
    return payload


def _build_side_fragments(
    *,
    side_outputs: Dict[int, Dict[str, Any]],
    valid_frame_set: set[int],
    ordered_frames: Sequence[int],
) -> List[Dict[str, Any]]:
    fragments: List[Dict[str, Any]] = []
    current_items: List[Dict[str, Any]] = []
    for frame_idx in ordered_frames:
        payload = side_outputs.get(int(frame_idx), _default_side_payload(int(frame_idx)))
        if int(frame_idx) in valid_frame_set and bool(payload.get("visible", False)):
            current_items.append(payload)
            continue
        if current_items:
            fragments.append(_materialize_fragment(current_items))
            current_items = []
    if current_items:
        fragments.append(_materialize_fragment(current_items))
    return fragments


def _materialize_fragment(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    frame_indices = [int(item["frame_idx"]) for item in items]
    scores = [float(item.get("score", 0.0)) for item in items]
    sources = {str(item.get("source", "propagated_proposal")) for item in items}
    return {
        "track_id": -1,
        "source": next(iter(sources)) if len(sources) == 1 else "mixed",
        "proposal_ids": [item.get("proposal_id") for item in items],
        "frame_indices": frame_indices,
        "bbox_xyxy": [item["bbox_xyxy"] for item in items],
        "bbox_xyxy_orig": [item["bbox_xyxy_orig"] for item in items],
        "score": scores,
    }


def fit_reviewed_clip(bundle: Dict[str, Any], review: Dict[str, Any], cfg: MediaPipeReviewConfig) -> Dict[str, Any]:
    if int(bundle.get("bundle_version", 0)) != 3:
        raise RuntimeError("Only bundle_version=3 is supported by the new fit pipeline")
    if str(review.get("review_version") or "") != "frame_review_v3":
        raise RuntimeError("Only frame_review_v3 is supported by the new fit pipeline")

    ordered_frames = _ordered_frame_indices(bundle)
    left_outputs, left_invalid_reasons, left_segments = _propagate_side(bundle, review, "left", cfg)
    right_outputs, right_invalid_reasons, right_segments = _propagate_side(bundle, review, "right", cfg)

    frame_records: Dict[int, Dict[str, Any]] = {}
    for frame_idx in ordered_frames:
        left_payload = left_outputs.get(int(frame_idx), _default_side_payload(int(frame_idx)))
        right_payload = right_outputs.get(int(frame_idx), _default_side_payload(int(frame_idx)))
        reasons = set()
        reasons.update(left_invalid_reasons.get(int(frame_idx), set()))
        reasons.update(right_invalid_reasons.get(int(frame_idx), set()))
        frame_records[int(frame_idx)] = {
            "frame_idx": int(frame_idx),
            "left": left_payload,
            "right": right_payload,
            "excluded_reasons": reasons,
        }

    min_frames = int(cfg.min_export_episode_frames)
    subepisodes: List[Dict[str, Any]] = []
    dropped_segments: List[Dict[str, Any]] = []
    current_valid: List[int] = []
    for frame_idx in ordered_frames:
        record = frame_records[int(frame_idx)]
        usable = bool(record["left"]["visible"]) or bool(record["right"]["visible"])
        if usable:
            current_valid.append(int(frame_idx))
            continue
        if current_valid:
            if len(current_valid) >= min_frames:
                subepisodes.append(
                    {
                        "subepisode_index": len(subepisodes),
                        "start_frame": int(current_valid[0]),
                        "end_frame": int(current_valid[-1]),
                        "num_frames": int(len(current_valid)),
                    }
                )
            else:
                dropped_segments.append(
                    {
                        "start_frame": int(current_valid[0]),
                        "end_frame": int(current_valid[-1]),
                        "num_frames": int(len(current_valid)),
                        "reason": "short_subepisode",
                    }
                )
            current_valid = []
        reasons = sorted(record["excluded_reasons"]) or ["no_visible_box"]
        if dropped_segments and int(dropped_segments[-1]["end_frame"]) + 1 == int(frame_idx) and dropped_segments[-1]["reason"] == " / ".join(reasons):
            dropped_segments[-1]["end_frame"] = int(frame_idx)
            dropped_segments[-1]["num_frames"] = int(dropped_segments[-1]["end_frame"]) - int(dropped_segments[-1]["start_frame"]) + 1
        else:
            dropped_segments.append(
                {
                    "start_frame": int(frame_idx),
                    "end_frame": int(frame_idx),
                    "num_frames": 1,
                    "reason": " / ".join(reasons),
                }
            )
    if current_valid:
        if len(current_valid) >= min_frames:
            subepisodes.append(
                {
                    "subepisode_index": len(subepisodes),
                    "start_frame": int(current_valid[0]),
                    "end_frame": int(current_valid[-1]),
                    "num_frames": int(len(current_valid)),
                }
            )
        else:
            dropped_segments.append(
                {
                    "start_frame": int(current_valid[0]),
                    "end_frame": int(current_valid[-1]),
                    "num_frames": int(len(current_valid)),
                    "reason": "short_subepisode",
                }
            )

    valid_frame_set = {
        frame_idx
        for subepisode in subepisodes
        for frame_idx in range(int(subepisode["start_frame"]), int(subepisode["end_frame"]) + 1)
    }
    left_fragments = _build_side_fragments(
        side_outputs=left_outputs,
        valid_frame_set=valid_frame_set,
        ordered_frames=ordered_frames,
    )
    right_fragments = _build_side_fragments(
        side_outputs=right_outputs,
        valid_frame_set=valid_frame_set,
        ordered_frames=ordered_frames,
    )

    all_scores = [
        float(score)
        for fragment in left_fragments + right_fragments
        for score in fragment.get("score", [])
    ]
    fit_payload = {
        "clip_id": int(bundle["clip_id"]),
        "review_version": "frame_review_v3",
        "box_only": True,
        "bundle_version": 3,
        "assignment_segments": left_segments + right_segments,
        "kept_segments": [
            {
                "start_frame": int(item["start_frame"]),
                "end_frame": int(item["end_frame"]),
                "num_frames": int(item["num_frames"]),
                "subepisode_index": int(item["subepisode_index"]),
            }
            for item in subepisodes
        ],
        "dropped_segments": dropped_segments,
        "valid_frame_ranges": [[int(item["start_frame"]), int(item["end_frame"])] for item in subepisodes],
        "subepisodes": subepisodes,
        "sides": {
            "left": {"missing_box": False, "fragments": left_fragments},
            "right": {"missing_box": False, "fragments": right_fragments},
        },
        "num_box_frames": int(sum(len(fragment["frame_indices"]) for fragment in left_fragments + right_fragments)),
        "mean_box_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "median_reproj_error": 0.0,
        "p95_reproj_error": 0.0,
    }
    if not subepisodes:
        fit_payload["status"] = "fit_dropped"
        fit_payload["message"] = "No valid subepisode reached the minimum export length."
    else:
        fit_payload["status"] = "fit_ok"
        fit_payload["message"] = "Episode boxes were propagated from review anchors over frame proposals."
    return fit_payload


def _empty_fragment_arrays(prefix: str, arrays: Dict[str, np.ndarray]) -> None:
    arrays[prefix + "fragment_track_ids"] = np.zeros((0,), dtype=np.int32)
    arrays[prefix + "fragment_source_codes"] = np.zeros((0,), dtype=np.int8)
    arrays[prefix + "fragment_offsets"] = np.zeros((1,), dtype=np.int32)
    arrays[prefix + "frame_indices"] = np.zeros((0,), dtype=np.int32)
    arrays[prefix + "bbox_xyxy"] = np.zeros((0, 4), dtype=np.float32)
    arrays[prefix + "bbox_xyxy_orig"] = np.zeros((0, 4), dtype=np.float32)
    arrays[prefix + "score"] = np.zeros((0,), dtype=np.float32)


def _flatten_side_fragments(side_payload: Optional[Dict[str, Any]], prefix: str, arrays: Dict[str, np.ndarray]) -> None:
    fragments = list((side_payload or {}).get("fragments", []))
    arrays[prefix + "missing_box"] = np.asarray([1 if bool((side_payload or {}).get("missing_box", False)) else 0], dtype=np.int8)
    if not fragments:
        _empty_fragment_arrays(prefix, arrays)
        return

    offsets = [0]
    fragment_ids: List[int] = []
    source_codes: List[int] = []
    frame_indices: List[np.ndarray] = []
    bbox_xyxy: List[np.ndarray] = []
    bbox_xyxy_orig: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    source_code_lookup = {
        "proposal_anchor": 0,
        "manual_anchor": 1,
        "propagated_proposal": 2,
        "mixed": 3,
    }
    total = 0
    for fragment_index, fragment in enumerate(fragments):
        frames = np.asarray(fragment["frame_indices"], dtype=np.int32)
        total += len(frames)
        offsets.append(total)
        fragment_ids.append(int(fragment_index))
        source_codes.append(int(source_code_lookup.get(str(fragment.get("source", "mixed")), 127)))
        frame_indices.append(frames)
        bbox_xyxy.append(np.asarray(fragment["bbox_xyxy"], dtype=np.float32))
        bbox_xyxy_orig.append(np.asarray(fragment["bbox_xyxy_orig"], dtype=np.float32))
        scores.append(np.asarray(fragment["score"], dtype=np.float32))

    arrays[prefix + "fragment_track_ids"] = np.asarray(fragment_ids, dtype=np.int32)
    arrays[prefix + "fragment_source_codes"] = np.asarray(source_codes, dtype=np.int8)
    arrays[prefix + "fragment_offsets"] = np.asarray(offsets, dtype=np.int32)
    arrays[prefix + "frame_indices"] = np.concatenate(frame_indices, axis=0) if frame_indices else np.zeros((0,), dtype=np.int32)
    arrays[prefix + "bbox_xyxy"] = np.concatenate(bbox_xyxy, axis=0) if bbox_xyxy else np.zeros((0, 4), dtype=np.float32)
    arrays[prefix + "bbox_xyxy_orig"] = np.concatenate(bbox_xyxy_orig, axis=0) if bbox_xyxy_orig else np.zeros((0, 4), dtype=np.float32)
    arrays[prefix + "score"] = np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)


def _ranges_array(ranges: Sequence[Sequence[int]]) -> np.ndarray:
    if not ranges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray([[int(item[0]), int(item[1])] for item in ranges], dtype=np.int32)


def save_fit_artifacts(bundle_dir: Path, fit_payload: Dict[str, Any]) -> Dict[str, str]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    json_path = bundle_dir / "fit.json"
    npz_path = bundle_dir / "fit.npz"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(fit_payload, f, ensure_ascii=False, indent=2)

    arrays: Dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        _flatten_side_fragments(fit_payload.get("sides", {}).get(side), f"{side}_", arrays)
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
