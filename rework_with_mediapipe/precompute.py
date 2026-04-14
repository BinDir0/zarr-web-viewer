from __future__ import annotations

import json
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image

from .config import MediaPipeReviewConfig
from .frame_sources import make_frame_source
from .types import ClipRef, Proposal


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _import_mediapipe_modules():
    import mediapipe as mp  # type: ignore

    errors: List[str] = []
    try:
        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        return mp, mp_python, vision
    except ImportError as exc:
        errors.append(f"from mediapipe.tasks import python failed: {exc}")

    try:
        import mediapipe.tasks.python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        return mp, mp_python, vision
    except ImportError as exc:
        errors.append(f"import mediapipe.tasks.python failed: {exc}")

    raise ImportError("; ".join(errors))


def _ensure_mediapipe(cfg: MediaPipeReviewConfig):
    first_error: Exception | None = None

    if cfg.prefer_installed_mediapipe:
        try:
            return _import_mediapipe_modules()
        except ImportError as exc:
            first_error = exc

    repo_root = str(cfg.mediapipe_repo_root)
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        return _import_mediapipe_modules()
    except ImportError as exc:
        if first_error is not None:
            raise RuntimeError(
                "缺少可用的 mediapipe Python 依赖。"
                "已先尝试环境内安装版本，再尝试 mediapipe_repo_root，均失败。"
                f" 当前解释器: {sys.executable}。"
            ) from exc
        raise RuntimeError(
            "缺少 mediapipe Python 依赖，无法运行预计算。"
            f" 当前解释器: {sys.executable}。"
        ) from exc


def _compute_bbox(landmarks: List[List[float]], width: int, height: int) -> List[float]:
    xs = [point[0] * width for point in landmarks]
    ys = [point[1] * height for point in landmarks]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width - 1), max(xs))
    y2 = min(float(height - 1), max(ys))
    pad_x = (x2 - x1) * 0.1
    pad_y = (y2 - y1) * 0.1
    return [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.zeros_like(vec)
    return vec / norm


def _derive_palm_geometry(world_points: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
    pts = np.asarray(world_points, dtype=np.float32)
    wrist = pts[0]
    index_mcp = pts[5]
    pinky_mcp = pts[17]
    middle_mcp = pts[9]
    x_axis = _normalize(index_mcp - pinky_mcp)
    y_axis = _normalize(middle_mcp - wrist)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if float(np.linalg.norm(z_axis)) < 1e-6:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return z_axis.tolist(), np.stack([x_axis, y_axis, z_axis], axis=0).tolist()


def _roi_rotation(landmarks_2d: List[List[float]], width: int, height: int) -> float:
    wrist = np.array(landmarks_2d[0][:2], dtype=np.float32) * np.array([width, height], dtype=np.float32)
    middle_mcp = np.array(landmarks_2d[9][:2], dtype=np.float32) * np.array([width, height], dtype=np.float32)
    vec = middle_mcp - wrist
    return float(math.atan2(float(vec[1]), float(vec[0])))


def _resolve_video_running_mode(vision):
    running_mode = getattr(vision, "RunningMode", None)
    if running_mode is None:
        return None
    return getattr(running_mode, "VIDEO", None)


def _role_hint_from_label(label: str) -> str:
    lowered = str(label).lower()
    if lowered.startswith("left"):
        return "mediapipe_left"
    if lowered.startswith("right"):
        return "mediapipe_right"
    return "mediapipe_unknown"


def _build_mediapipe_tracks(
    per_frame_proposals: Dict[int, List[Proposal]],
    *,
    bad_frame_indices: List[int],
    max_gap: int,
    min_frames: int,
) -> List[Dict]:
    bad_frames = {int(item) for item in bad_frame_indices}
    active: Dict[Tuple[str, int], Dict] = {}
    finished: List[Dict] = []
    next_track_id = 0

    for frame_idx in sorted(per_frame_proposals.keys()):
        proposals = per_frame_proposals[frame_idx]
        present_keys = set()
        for proposal in proposals:
            key = (proposal.handedness_label, int(proposal.side_rank))
            present_keys.add(key)
            current = active.get(key)
            if current is None or frame_idx - int(current["last_frame_idx"]) > max_gap + 1:
                if current is not None:
                    finished.append(current)
                active[key] = {
                    "track_id": next_track_id,
                    "handedness_label": proposal.handedness_label,
                    "side_rank": int(proposal.side_rank),
                    "proposals": [proposal],
                    "last_frame_idx": frame_idx,
                }
                next_track_id += 1
                continue
            current["proposals"].append(proposal)
            current["last_frame_idx"] = frame_idx

        stale_keys = [
            key
            for key, track in active.items()
            if key not in present_keys and frame_idx - int(track["last_frame_idx"]) > max_gap
        ]
        for key in stale_keys:
            finished.append(active.pop(key))

    finished.extend(active.values())
    tracks: List[Dict] = []
    for track in finished:
        proposals = list(track["proposals"])
        frames = [int(item.frame_idx) for item in proposals]
        num_frames = len(frames)
        intersects_bad_frames = any(frame_idx in bad_frames for frame_idx in frames)
        if num_frames < max(1, min_frames) and not intersects_bad_frames:
            continue
        if num_frames < 2 and not intersects_bad_frames:
            continue
        preview_idx = frames[len(frames) // 2]
        mean_score = float(sum(float(item.score) for item in proposals) / max(1, len(proposals)))
        tracks.append(
            {
                "track_id": int(track["track_id"]),
                "role_hint": _role_hint_from_label(track["handedness_label"]),
                "score": float(num_frames) + mean_score + (3.0 if intersects_bad_frames else 0.0),
                "start_frame": min(frames),
                "end_frame": max(frames),
                "num_frames": num_frames,
                "preview_frame": preview_idx,
                "frames": frames,
                "intersects_bad_frames": intersects_bad_frames,
                "handedness_label": track["handedness_label"],
                "side_rank": int(track["side_rank"]),
                "proposals": [proposal.to_dict() for proposal in proposals],
            }
        )
    tracks.sort(
        key=lambda item: (
            not bool(item["intersects_bad_frames"]),
            int(item["start_frame"]),
            str(item["handedness_label"]),
            int(item["side_rank"]),
            int(item["track_id"]),
        )
    )
    return tracks


def _wrap_angle_diff_deg(prev_angle: float, curr_angle: float) -> float:
    delta = abs(curr_angle - prev_angle)
    while delta > math.pi:
        delta -= 2.0 * math.pi
    return abs(math.degrees(delta))


def _bbox_center(bbox_xyxy: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _bbox_area(bbox_xyxy: List[float]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _build_frame_tracks(
    frame_indices: List[int],
    tracks: List[Dict],
) -> Tuple[List[Dict], Dict[int, Set[int]]]:
    per_frame: Dict[int, List[Dict]] = {int(frame_idx): [] for frame_idx in frame_indices}
    visible_ids: Dict[int, Set[int]] = {int(frame_idx): set() for frame_idx in frame_indices}
    for track in tracks:
        track_id = int(track["track_id"])
        for proposal in track["proposals"]:
            frame_idx = int(proposal["frame_idx"])
            if frame_idx not in per_frame:
                continue
            item = {
                "track_id": track_id,
                "bbox_xyxy": proposal["bbox_xyxy"],
                "score": float(proposal["score"]),
                "handedness_label": str(proposal["handedness_label"]),
                "side_rank": int(proposal["side_rank"]),
                "roi_rotation_2d": float(proposal["roi_rotation_2d"]),
            }
            per_frame[frame_idx].append(item)
            visible_ids[frame_idx].add(track_id)
    frame_tracks: List[Dict] = []
    for frame_idx in frame_indices:
        proposals = sorted(per_frame.get(frame_idx, []), key=lambda item: int(item["track_id"]))
        frame_tracks.append(
            {
                "frame_idx": int(frame_idx),
                "visible_track_ids": [int(item["track_id"]) for item in proposals],
                "tracks": proposals,
            }
        )
    return frame_tracks, visible_ids


def _compute_motion_scores(
    tracks: List[Dict],
    frame_indices: List[int],
    image_size: Tuple[int, int],
    cfg: MediaPipeReviewConfig,
) -> Tuple[Dict[int, float], Dict[int, Set[str]]]:
    frame_scores = {int(frame_idx): 0.0 for frame_idx in frame_indices}
    boundary_reasons: Dict[int, Set[str]] = defaultdict(set)
    width, height = image_size
    diag = max(math.hypot(float(width), float(height)), 1.0)
    for track in tracks:
        proposals = sorted(track["proposals"], key=lambda item: int(item["frame_idx"]))
        prev = None
        for proposal in proposals:
            frame_idx = int(proposal["frame_idx"])
            if prev is None:
                prev = proposal
                continue
            prev_center = _bbox_center(prev["bbox_xyxy"])
            curr_center = _bbox_center(proposal["bbox_xyxy"])
            center_ratio = math.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1]) / diag
            prev_area = max(_bbox_area(prev["bbox_xyxy"]), 1.0)
            curr_area = max(_bbox_area(proposal["bbox_xyxy"]), 1.0)
            area_ratio = curr_area / prev_area
            rot_delta_deg = _wrap_angle_diff_deg(float(prev["roi_rotation_2d"]), float(proposal["roi_rotation_2d"]))
            score = center_ratio + abs(math.log(area_ratio)) + (rot_delta_deg / 180.0)
            frame_scores[frame_idx] = max(float(frame_scores.get(frame_idx, 0.0)), float(score))
            if (
                center_ratio >= cfg.keyframe_motion_center_ratio
                or area_ratio >= cfg.keyframe_motion_area_ratio_high
                or area_ratio <= cfg.keyframe_motion_area_ratio_low
                or rot_delta_deg >= cfg.keyframe_motion_rotation_deg
            ):
                boundary_reasons[frame_idx].add("motion_jump")
            prev = proposal
    return frame_scores, boundary_reasons


def _build_segments_and_keyframes(
    *,
    clip: ClipRef,
    frame_indices: List[int],
    frame_tracks: List[Dict],
    tracks: List[Dict],
    visible_track_ids: Dict[int, Set[int]],
    motion_scores: Dict[int, float],
    motion_reasons: Dict[int, Set[str]],
    cfg: MediaPipeReviewConfig,
) -> Tuple[List[Dict], List[Dict], Dict[int, int]]:
    if not frame_indices:
        return [], [], {}

    frame_track_map = {int(item["frame_idx"]): item for item in frame_tracks}
    boundary_reasons: Dict[int, Set[str]] = defaultdict(set)
    boundary_reasons[int(frame_indices[0])].add("clip_start")
    boundary_reasons[int(frame_indices[-1])].add("clip_end")

    bad_frames = {int(frame_idx) for frame_idx in clip.bad_frames}
    frame_to_pos = {int(frame_idx): pos for pos, frame_idx in enumerate(frame_indices)}
    previous_visible: Set[int] | None = None
    previous_signature: Tuple[Tuple[str, int], ...] | None = None
    previous_frame_idx: int | None = None
    for frame_idx in frame_indices:
        if frame_idx in bad_frames:
            boundary_reasons[frame_idx].add("bad_frame")
        visible = set(visible_track_ids.get(frame_idx, set()))
        signature = tuple(
            sorted(
                (str(track["handedness_label"]), int(track["side_rank"]))
                for track in frame_track_map.get(frame_idx, {}).get("tracks", [])
            )
        )
        if previous_visible is not None and visible != previous_visible:
            boundary_reasons[frame_idx].add("track_set_changed")
            if previous_frame_idx is not None:
                boundary_reasons[previous_frame_idx].add("track_set_changed_before")
        if previous_signature is not None and signature != previous_signature:
            boundary_reasons[frame_idx].add("hand_signature_changed")
            if previous_frame_idx is not None:
                boundary_reasons[previous_frame_idx].add("hand_signature_changed_before")
        previous_visible = visible
        previous_signature = signature
        previous_frame_idx = int(frame_idx)

    for track in tracks:
        start_frame = int(track["start_frame"])
        end_frame = int(track["end_frame"])
        if start_frame in frame_to_pos:
            boundary_reasons[start_frame].add("track_appeared")
        if end_frame in frame_to_pos:
            boundary_reasons[end_frame].add("track_disappeared")
        prev_pos = frame_to_pos.get(start_frame)
        if prev_pos is not None and prev_pos > 0:
            boundary_reasons[frame_indices[prev_pos - 1]].add("before_track_appeared")
        next_pos = frame_to_pos.get(end_frame)
        if next_pos is not None and next_pos + 1 < len(frame_indices):
            boundary_reasons[frame_indices[next_pos + 1]].add("after_track_disappeared")

    for frame_idx, reasons in motion_reasons.items():
        boundary_reasons[int(frame_idx)].update(reasons)

    last_position = max(len(frame_indices) - 1, 0)
    boundary_positions = sorted(
        {
            frame_to_pos[frame_idx]
            for frame_idx in boundary_reasons.keys()
            if frame_idx in frame_to_pos and frame_to_pos[frame_idx] < last_position
        }
    )
    if 0 not in boundary_positions:
        boundary_positions.insert(0, 0)

    segments: List[Dict] = []
    keyframes: List[Dict] = []
    frame_to_segment: Dict[int, int] = {}
    for segment_id, start_pos in enumerate(boundary_positions):
        end_pos = boundary_positions[segment_id + 1] - 1 if segment_id + 1 < len(boundary_positions) else len(frame_indices) - 1
        segment_frames = frame_indices[start_pos : end_pos + 1]
        start_frame = int(segment_frames[0])
        end_frame = int(segment_frames[-1])
        for frame_idx in segment_frames:
            frame_to_segment[int(frame_idx)] = segment_id
        visible = sorted(visible_track_ids.get(start_frame, set()))
        segment = {
            "segment_id": segment_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "anchor_frame_idx": start_frame,
            "visible_track_ids": visible,
            "boundary_reasons": sorted(boundary_reasons.get(start_frame, set())),
        }
        segments.append(segment)
        keyframes.append(
            {
                "frame_idx": start_frame,
                "relpath": "",
                "segment_id": segment_id,
                "kind": "anchor",
                "reasons": sorted(boundary_reasons.get(start_frame, set())),
                "visible_track_ids": visible,
            }
        )

        candidate_frames = [int(frame_idx) for frame_idx in segment_frames[1:] if visible_track_ids.get(int(frame_idx), set())]
        if candidate_frames:
            motion_pick = max(candidate_frames, key=lambda item: (float(motion_scores.get(item, 0.0)), -item))
            if float(motion_scores.get(motion_pick, 0.0)) > 0.0:
                keyframes.append(
                    {
                        "frame_idx": motion_pick,
                        "relpath": "",
                        "segment_id": segment_id,
                        "kind": "coverage",
                        "reasons": ["motion_coverage"],
                        "visible_track_ids": sorted(visible_track_ids.get(motion_pick, set())),
                    }
                )
        if len(segment_frames) > cfg.keyframe_extra_random_min_frames:
            pool = [int(frame_idx) for frame_idx in segment_frames[1:] if visible_track_ids.get(int(frame_idx), set())]
            taken = {int(item["frame_idx"]) for item in keyframes if int(item["segment_id"]) == segment_id}
            pool = [item for item in pool if item not in taken]
            if pool:
                seed = f"{clip.episode.episode_id}|{clip.clip_start}|{clip.clip_end}|{clip.dirty_reason}|{segment_id}"
                pick_index = int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16) % len(pool)
                random_pick = int(pool[pick_index])
                keyframes.append(
                    {
                        "frame_idx": random_pick,
                        "relpath": "",
                        "segment_id": segment_id,
                        "kind": "coverage",
                        "reasons": ["deterministic_sample"],
                        "visible_track_ids": sorted(visible_track_ids.get(random_pick, set())),
                    }
                )

    deduped_keyframes: Dict[int, Dict] = {}
    for item in keyframes:
        frame_idx = int(item["frame_idx"])
        existing = deduped_keyframes.get(frame_idx)
        if existing is None:
            deduped_keyframes[frame_idx] = item
            continue
        merged_reasons = sorted(set(existing.get("reasons", [])) | set(item.get("reasons", [])))
        kind = "anchor" if existing.get("kind") == "anchor" or item.get("kind") == "anchor" else "coverage"
        deduped_keyframes[frame_idx] = {
            **existing,
            **item,
            "kind": kind,
            "reasons": merged_reasons,
        }

    ordered_keyframes = sorted(
        deduped_keyframes.values(),
        key=lambda item: (int(item["frame_idx"]), 0 if item["kind"] == "anchor" else 1),
    )
    return segments, ordered_keyframes, frame_to_segment


class MediaPipeProposalDetector:
    def __init__(self, cfg: MediaPipeReviewConfig):
        mp, mp_python, vision = _ensure_mediapipe(cfg)
        if not cfg.hand_landmarker_task.exists():
            raise FileNotFoundError(
                f"缺少 hand_landmarker.task: {cfg.hand_landmarker_task}"
            )
        self._mp = mp
        self._running_mode_video = _resolve_video_running_mode(vision)
        base_options = mp_python.BaseOptions(model_asset_path=str(cfg.hand_landmarker_task))
        option_kwargs = dict(
            base_options=base_options,
            num_hands=cfg.detector_max_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        if self._running_mode_video is not None:
            option_kwargs["running_mode"] = self._running_mode_video
        options = vision.HandLandmarkerOptions(**option_kwargs)
        self._detector = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._detector.close()

    def detect(
        self,
        image: Image.Image,
        frame_idx: int,
        *,
        output_size: Tuple[int, int] | None = None,
        timestamp_ms: int | None = None,
    ) -> List[Proposal]:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.asarray(image))
        if timestamp_ms is not None and hasattr(self._detector, "detect_for_video"):
            result = self._detector.detect_for_video(mp_image, int(timestamp_ms))
        else:
            result = self._detector.detect(mp_image)
        width, height = output_size or image.size
        proposals: List[Proposal] = []
        label_counts: Dict[str, int] = defaultdict(int)
        for det_idx, (norm_lms, world_lms, handedness) in enumerate(
            zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness)
        ):
            landmarks_2d = [
                [_safe_float(lm.x), _safe_float(lm.y), _safe_float(getattr(lm, "visibility", None), 1.0)]
                for lm in norm_lms
            ]
            landmarks_3d = [[_safe_float(lm.x), _safe_float(lm.y), _safe_float(lm.z)] for lm in world_lms]
            bbox_xyxy = _compute_bbox(landmarks_2d, width, height)
            palm_normal, wrist_frame = _derive_palm_geometry(landmarks_3d)
            handedness_score = 0.0
            handedness_label = "unknown"
            if handedness:
                category = handedness[0]
                handedness_score = _safe_float(category.score)
                handedness_label = str(category.category_name or "unknown")
                if str(category.category_name).lower().startswith("left"):
                    handedness_score *= -1.0
            side_rank = int(label_counts[handedness_label])
            label_counts[handedness_label] += 1
            proposals.append(
                Proposal(
                    frame_idx=frame_idx,
                    det_idx=det_idx,
                    bbox_xyxy=bbox_xyxy,
                    score=float(abs(handedness_score)),
                    handedness_score=handedness_score,
                    handedness_label=handedness_label,
                    side_rank=side_rank,
                    landmarks_2d=landmarks_2d,
                    landmarks_3d_rel=landmarks_3d,
                    roi_rotation_2d=_roi_rotation(landmarks_2d, width, height),
                    palm_normal=palm_normal,
                    wrist_frame=wrist_frame,
                )
            )
        return proposals


def _resize_for_review(image: Image.Image, preview_width: int) -> Image.Image:
    width, height = image.size
    if width <= preview_width:
        return image
    scale = preview_width / float(width)
    return image.resize((preview_width, int(height * scale)), Image.Resampling.BILINEAR)


def _save_review_frame(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=80)


def _save_proposals_npz(
    bundle_dir: Path,
    per_frame_proposals: Dict[int, List[Proposal]],
    chain_lookup: Dict[Tuple[int, int], int],
) -> Path:
    records: List[Proposal] = []
    chain_indices: List[int] = []
    for frame_idx in sorted(per_frame_proposals.keys()):
        for proposal in per_frame_proposals[frame_idx]:
            records.append(proposal)
            chain_indices.append(chain_lookup.get((proposal.frame_idx, proposal.det_idx), -1))

    npz_path = bundle_dir / "proposals.npz"
    if not records:
        np.savez_compressed(
            npz_path,
            frame_idx=np.zeros((0,), dtype=np.int32),
            det_idx=np.zeros((0,), dtype=np.int32),
            chain_index=np.zeros((0,), dtype=np.int32),
            bbox_xyxy=np.zeros((0, 4), dtype=np.float32),
            score=np.zeros((0,), dtype=np.float32),
            handedness_score=np.zeros((0,), dtype=np.float32),
            landmarks_2d=np.zeros((0, 21, 3), dtype=np.float32),
            landmarks_3d_rel=np.zeros((0, 21, 3), dtype=np.float32),
            roi_rotation_2d=np.zeros((0,), dtype=np.float32),
            palm_normal=np.zeros((0, 3), dtype=np.float32),
            wrist_frame=np.zeros((0, 3, 3), dtype=np.float32),
        )
        return npz_path

    np.savez_compressed(
        npz_path,
        frame_idx=np.asarray([item.frame_idx for item in records], dtype=np.int32),
        det_idx=np.asarray([item.det_idx for item in records], dtype=np.int32),
        chain_index=np.asarray(chain_indices, dtype=np.int32),
        bbox_xyxy=np.asarray([item.bbox_xyxy for item in records], dtype=np.float32),
        score=np.asarray([item.score for item in records], dtype=np.float32),
        handedness_score=np.asarray([item.handedness_score for item in records], dtype=np.float32),
        landmarks_2d=np.asarray([item.landmarks_2d for item in records], dtype=np.float32),
        landmarks_3d_rel=np.asarray([item.landmarks_3d_rel for item in records], dtype=np.float32),
        roi_rotation_2d=np.asarray([item.roi_rotation_2d for item in records], dtype=np.float32),
        palm_normal=np.asarray([item.palm_normal for item in records], dtype=np.float32),
        wrist_frame=np.asarray([item.wrist_frame for item in records], dtype=np.float32),
    )
    return npz_path


def preprocess_clip(clip: ClipRef, clip_id: int, cfg: MediaPipeReviewConfig) -> Dict:
    frame_source = make_frame_source(clip.episode)
    bundle_dir = cfg.bundles_dir / f"clip_{clip_id:06d}"
    frames_dir = bundle_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    per_frame_proposals: Dict[int, List[Proposal]] = {}
    frames_relpaths: Dict[int, str] = {}
    image_size = (0, 0)

    detector = MediaPipeProposalDetector(cfg)
    try:
        for packet in frame_source.iter_frames(clip.clip_start, clip.clip_end):
            review_image = _resize_for_review(packet.image, cfg.preview_width)
            image_size = review_image.size
            relpath = f"bundles/clip_{clip_id:06d}/frames/{packet.frame_idx:06d}.jpg"
            _save_review_frame(review_image, frames_dir / f"{packet.frame_idx:06d}.jpg")
            frames_relpaths[packet.frame_idx] = relpath
            per_frame_proposals[packet.frame_idx] = detector.detect(
                packet.image,
                packet.frame_idx,
                output_size=review_image.size,
                timestamp_ms=max(0, (packet.frame_idx - clip.clip_start) * 33),
            )
    finally:
        detector.close()

    tracks = _build_mediapipe_tracks(
        per_frame_proposals,
        bad_frame_indices=clip.bad_frames,
        max_gap=cfg.track_max_gap,
        min_frames=cfg.min_chain_frames,
    )
    visible_tracks = list(tracks)
    too_dense_for_review = len(visible_tracks) > cfg.max_review_tracks

    chain_lookup: Dict[Tuple[int, int], int] = {}
    for track in tracks:
        for proposal in track["proposals"]:
            chain_lookup[(int(proposal["frame_idx"]), int(proposal["det_idx"]))] = int(track["track_id"])

    proposals_npz_path = _save_proposals_npz(bundle_dir, per_frame_proposals, chain_lookup)
    bad_frame_lookup = {int(frame_idx) for frame_idx in clip.bad_frames}
    ordered_frame_indices = sorted(frames_relpaths.keys())
    frame_tracks, visible_track_ids = _build_frame_tracks(ordered_frame_indices, visible_tracks)
    motion_scores, motion_reasons = _compute_motion_scores(visible_tracks, ordered_frame_indices, image_size, cfg)
    segments, keyframes, frame_to_segment = _build_segments_and_keyframes(
        clip=clip,
        frame_indices=ordered_frame_indices,
        frame_tracks=frame_tracks,
        tracks=visible_tracks,
        visible_track_ids=visible_track_ids,
        motion_scores=motion_scores,
        motion_reasons=motion_reasons,
        cfg=cfg,
    )
    for item in frame_tracks:
        item["segment_id"] = int(frame_to_segment.get(int(item["frame_idx"]), -1))
    for item in keyframes:
        item["relpath"] = frames_relpaths[int(item["frame_idx"])]
    bundle = {
        "clip_id": clip_id,
        "episode_id": clip.episode.episode_id,
        "episode_name": clip.episode.episode_name,
        "dataset_name": clip.episode.dataset_name,
        "clip_start": clip.clip_start,
        "clip_end": clip.clip_end,
        "dirty_reason": clip.dirty_reason,
        "bad_frame_indices": sorted(bad_frame_lookup),
        "too_dense_for_review": too_dense_for_review,
        "track_generation": "mediapipe_video_segments",
        "proposals_npz_relpath": str(proposals_npz_path.relative_to(cfg.artifacts_dir)),
        "frames": [
            {
                "frame_idx": frame_idx,
                "relpath": frames_relpaths[frame_idx],
                "is_bad": frame_idx in bad_frame_lookup,
                "segment_id": int(frame_to_segment.get(frame_idx, -1)),
                "visible_track_ids": sorted(visible_track_ids.get(frame_idx, set())),
            }
            for frame_idx in ordered_frame_indices
        ],
        "frame_tracks": frame_tracks,
        "segments": segments,
        "keyframes": keyframes,
        "tracks": visible_tracks,
    }
    with (bundle_dir / "bundle.json").open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    return {
        "bundle_dir": str(bundle_dir),
        "bundle_relpath": str(bundle_dir.relative_to(cfg.artifacts_dir)),
        "bundle": bundle,
        "tracks": tracks,
    }
