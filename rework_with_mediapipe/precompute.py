from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import sys
import time
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from PIL import Image
import yaml

from .config import MediaPipeReviewConfig
from .frame_sources import make_frame_source
from .types import ClipRef, Proposal


@contextmanager
def _torchvision_metadata_compat():
    original_version = importlib.metadata.version

    def _patched_version(name: str) -> str:
        try:
            return original_version(name)
        except importlib.metadata.PackageNotFoundError:
            if name != "torchvision":
                raise
            module = importlib.import_module("torchvision")
            module_version = getattr(module, "__version__", None)
            if not module_version:
                raise
            return str(module_version)

    importlib.metadata.version = _patched_version
    try:
        yield
    finally:
        importlib.metadata.version = original_version


def _import_ultralytics_runtime():
    with _torchvision_metadata_compat():
        from ultralytics import YOLO  # type: ignore
        from ultralytics.engine.results import Boxes  # type: ignore
        from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore
        try:
            from ultralytics.trackers.bot_sort import BOTSORT  # type: ignore
        except Exception:  # pragma: no cover - older ultralytics builds may miss BoT-SORT
            BOTSORT = None

    return YOLO, Boxes, BYTETracker, BOTSORT


def _ensure_ultralytics_runtime(cfg: MediaPipeReviewConfig):
    first_error: Exception | None = None
    if cfg.prefer_installed_ultralytics:
        try:
            return _import_ultralytics_runtime()
        except Exception as exc:  # pragma: no cover - import surface differs by env
            first_error = exc

    repo_root = str(cfg.ultralytics_repo_root)
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        return _import_ultralytics_runtime()
    except Exception as exc:  # pragma: no cover - import surface differs by env
        root_cause = exc.__cause__ or exc
        if isinstance(root_cause, importlib.metadata.PackageNotFoundError) and "torchvision" in str(root_cause):
            raise RuntimeError(
                "当前环境导入 ultralytics 时读取不到 torchvision 的包 metadata。"
                "如果 torchvision 已经能 import，这个兼容层本应自动兜底；"
                "若仍失败，请先在同一个环境里安装/重装 torchvision。"
                f" 当前解释器: {sys.executable}。"
            ) from exc
        if first_error is not None:
            raise RuntimeError(
                "缺少可用的 ultralytics/YOLO Python 依赖。"
                "已先尝试环境内安装版本，再尝试 ultralytics_repo_root，均失败。"
                f" 当前解释器: {sys.executable}。"
            ) from exc
        raise RuntimeError(
            "缺少 ultralytics/YOLO Python 依赖，无法运行预计算。"
            f" 当前解释器: {sys.executable}。"
        ) from exc


def _resolve_tracker_config_path(cfg: MediaPipeReviewConfig) -> Path:
    raw = str(cfg.yolo_tracker_config or "").strip()
    candidates: List[Path] = []
    if raw:
        base = Path(raw)
        candidates.append(base)
        if not base.suffix:
            candidates.append(base.with_suffix(".yaml"))
        candidates.append(cfg.ultralytics_repo_root / "ultralytics" / "cfg" / "trackers" / base.name)
        if not base.suffix:
            candidates.append(cfg.ultralytics_repo_root / "ultralytics" / "cfg" / "trackers" / f"{base.name}.yaml")
    else:
        candidates.append(cfg.ultralytics_repo_root / "ultralytics" / "cfg" / "trackers" / "botsort.yaml")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_tracker_args(cfg: MediaPipeReviewConfig) -> SimpleNamespace:
    tracker_path = _resolve_tracker_config_path(cfg)
    if not tracker_path.exists():
        raise FileNotFoundError(f"缺少 YOLO tracker 配置文件: {tracker_path}")
    with tracker_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"YOLO tracker 配置格式不合法: {tracker_path}")
    raw.setdefault("tracker_type", "botsort")
    return SimpleNamespace(**raw)


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


def _resize_for_review(image: Image.Image, preview_width: int) -> Image.Image:
    width, height = image.size
    if width <= preview_width:
        return image
    scale = preview_width / float(width)
    return image.resize((preview_width, int(height * scale)), Image.Resampling.BILINEAR)


def _preview_size(orig_size: Tuple[int, int], preview_width: int) -> Tuple[int, int]:
    width, height = int(orig_size[0]), int(orig_size[1])
    if width <= preview_width:
        return width, height
    scale = preview_width / float(width)
    return preview_width, int(height * scale)


def _save_review_frame(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=80)


class UltralyticsProposalDetector:
    def __init__(self, cfg: MediaPipeReviewConfig):
        YOLO, Boxes, BYTETracker, BOTSORT = _ensure_ultralytics_runtime(cfg)
        if not str(cfg.raw.get("yolo_model_path", "") or "").strip():
            raise FileNotFoundError("缺少 yolo_model_path 配置，无法运行 YOLO 预处理。")
        if not cfg.yolo_model_path.exists():
            raise FileNotFoundError(f"缺少 YOLO 权重文件: {cfg.yolo_model_path}")
        self._cfg = cfg
        self._Boxes = Boxes
        self._model = YOLO(str(cfg.yolo_model_path))
        self._tracker_args = _load_tracker_args(cfg)
        self._tracker_type = str(getattr(self._tracker_args, "tracker_type", "botsort")).strip().lower()
        tracker_classes: Dict[str, Any] = {
            "bytetrack": BYTETracker,
        }
        if BOTSORT is not None:
            tracker_classes["botsort"] = BOTSORT
        tracker_cls = tracker_classes.get(self._tracker_type)
        if tracker_cls is None:
            supported = ", ".join(sorted(tracker_classes.keys()))
            raise RuntimeError(f"不支持的 YOLO tracker_type: {self._tracker_type}。当前支持: {supported}")
        self._track_generation = f"ultralytics_{self._tracker_type}"
        self._tracker = tracker_cls(args=self._tracker_args, frame_rate=30)

    def close(self) -> None:
        self._tracker = None
        return None

    @property
    def track_generation(self) -> str:
        return self._track_generation

    def _build_predict_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "stream": False,
            "verbose": False,
            "conf": self._cfg.yolo_confidence,
            "iou": self._cfg.yolo_iou,
            "max_det": self._cfg.yolo_max_det,
            "imgsz": self._cfg.yolo_imgsz,
        }
        if self._cfg.yolo_device:
            kwargs["device"] = self._cfg.yolo_device
        return kwargs

    def _predict_batch(self, frames: List[Dict[str, Any]]) -> List[Any]:
        kwargs = self._build_predict_kwargs()
        kwargs["source"] = [item["image_np"] for item in frames]
        results = self._model.predict(**kwargs)
        if len(results) != len(frames):
            raise RuntimeError(f"YOLO predict 返回帧数不匹配: expected={len(frames)} actual={len(results)}")
        return list(results)

    def _make_boxes(self, result: Any, orig_shape: Tuple[int, int]):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            data = np.zeros((0, 6), dtype=np.float32)
            return self._Boxes(data, orig_shape)
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
        conf = (
            boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
            if boxes.conf is not None
            else np.zeros((len(boxes),), dtype=np.float32)
        )
        cls = (
            boxes.cls.detach().cpu().numpy().astype(np.float32, copy=False)
            if boxes.cls is not None
            else np.zeros((len(boxes),), dtype=np.float32)
        )
        data = np.concatenate([xyxy, conf[:, None], cls[:, None]], axis=1)
        return self._Boxes(data, orig_shape)

    @staticmethod
    def _rows_from_tracker_state(tracker: Any) -> np.ndarray:
        current_rows: List[List[float]] = []
        for track in getattr(tracker, "tracked_stracks", []):
            if int(getattr(track, "frame_id", -1)) != int(getattr(tracker, "frame_id", -2)):
                continue
            result = getattr(track, "result", None)
            if result is None:
                continue
            current_rows.append([float(value) for value in result])
        if not current_rows:
            return np.zeros((0, 8), dtype=np.float32)
        return np.asarray(current_rows, dtype=np.float32)

    def detect_batch(self, frames: List[Dict[str, Any]]) -> Tuple[Dict[int, List[Proposal]], Dict[str, float]]:
        if not frames:
            return {}, {"detector_infer_seconds": 0.0, "tracker_association_seconds": 0.0}
        infer_start = time.perf_counter()
        results = self._predict_batch(frames)
        infer_seconds = time.perf_counter() - infer_start

        proposals_by_frame: Dict[int, List[Proposal]] = {}
        tracking_start = time.perf_counter()
        for frame_item, result in zip(frames, results):
            frame_idx = int(frame_item["frame_idx"])
            preview_width, preview_height = frame_item["preview_size"]
            orig_width, orig_height = frame_item["orig_size"]
            scale_x = preview_width / max(float(orig_width), 1.0)
            scale_y = preview_height / max(float(orig_height), 1.0)
            names = getattr(result, "names", {}) or {}
            det_boxes = self._make_boxes(result, (int(orig_height), int(orig_width)))
            _ = self._tracker.update(det_boxes, frame_item["image_np"])
            track_rows = self._rows_from_tracker_state(self._tracker)
            if len(track_rows) == 0:
                proposals_by_frame[frame_idx] = []
                continue

            frame_proposals: List[Proposal] = []
            for row_idx, row in enumerate(track_rows):
                x1, y1, x2, y2, track_id, score, cls_id, det_idx = row.tolist()
                class_id = int(cls_id)
                class_name = str(names.get(class_id, class_id))
                bbox_orig = [x1, y1, x2, y2]
                bbox_preview = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                frame_proposals.append(
                    Proposal(
                        frame_idx=frame_idx,
                        det_idx=int(det_idx) if det_idx >= 0 else row_idx,
                        track_id=int(track_id),
                        bbox_xyxy=bbox_preview,
                        bbox_xyxy_orig=bbox_orig,
                        score=float(score),
                        class_id=class_id,
                        class_name=class_name,
                        roi_rotation_2d=0.0,
                    )
                )
            proposals_by_frame[frame_idx] = frame_proposals
        tracking_seconds = time.perf_counter() - tracking_start
        return proposals_by_frame, {
            "detector_infer_seconds": float(infer_seconds),
            "tracker_association_seconds": float(tracking_seconds),
        }


def _build_yolo_tracks(
    per_frame_proposals: Dict[int, List[Proposal]],
    *,
    bad_frame_indices: List[int],
    min_frames: int,
) -> List[Dict]:
    bad_frames = {int(item) for item in bad_frame_indices}
    grouped: Dict[int, List[Proposal]] = defaultdict(list)
    for proposals in per_frame_proposals.values():
        for proposal in proposals:
            grouped[int(proposal.track_id)].append(proposal)

    tracks: List[Dict] = []
    for track_id, proposals in grouped.items():
        proposals.sort(key=lambda item: int(item.frame_idx))
        frames = [int(item.frame_idx) for item in proposals]
        num_frames = len(frames)
        intersects_bad_frames = any(frame_idx in bad_frames for frame_idx in frames)
        if num_frames < max(1, min_frames) and not intersects_bad_frames:
            continue
        preview_idx = frames[len(frames) // 2]
        mean_score = float(sum(float(item.score) for item in proposals) / max(1, len(proposals)))
        class_ids = [int(item.class_id) for item in proposals]
        class_id = max(set(class_ids), key=class_ids.count)
        class_name = next((str(item.class_name) for item in proposals if int(item.class_id) == class_id), str(class_id))
        tracks.append(
            {
                "track_id": int(track_id),
                "role_hint": f"class_{class_name}",
                "score": float(num_frames) + mean_score + (3.0 if intersects_bad_frames else 0.0),
                "start_frame": min(frames),
                "end_frame": max(frames),
                "num_frames": num_frames,
                "preview_frame": preview_idx,
                "frames": frames,
                "intersects_bad_frames": intersects_bad_frames,
                "class_id": class_id,
                "class_name": class_name,
                "proposals": [proposal.to_dict() for proposal in proposals],
            }
        )
    tracks.sort(
        key=lambda item: (
            not bool(item["intersects_bad_frames"]),
            int(item["start_frame"]),
            int(item["track_id"]),
        )
    )
    return tracks


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
                "bbox_xyxy_orig": proposal["bbox_xyxy_orig"],
                "score": float(proposal["score"]),
                "class_id": int(proposal["class_id"]),
                "class_name": str(proposal["class_name"]),
                "roi_rotation_2d": float(proposal.get("roi_rotation_2d", 0.0)),
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
            rot_delta_deg = _wrap_angle_diff_deg(
                float(prev.get("roi_rotation_2d", 0.0)),
                float(proposal.get("roi_rotation_2d", 0.0)),
            )
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

    boundary_reasons: Dict[int, Set[str]] = defaultdict(set)
    boundary_reasons[int(frame_indices[0])].add("clip_start")
    boundary_reasons[int(frame_indices[-1])].add("clip_end")

    bad_frames = {int(frame_idx) for frame_idx in clip.bad_frames}
    frame_to_pos = {int(frame_idx): pos for pos, frame_idx in enumerate(frame_indices)}
    previous_visible: Set[int] | None = None
    previous_frame_idx: int | None = None
    for frame_idx in frame_indices:
        if frame_idx in bad_frames:
            boundary_reasons[frame_idx].add("bad_frame")
        visible = set(visible_track_ids.get(frame_idx, set()))
        if previous_visible is not None and visible != previous_visible:
            boundary_reasons[frame_idx].add("track_set_changed")
            if previous_frame_idx is not None:
                boundary_reasons[previous_frame_idx].add("track_set_changed_before")
        previous_visible = visible
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
        recovery_candidate_frames = []
        stride = max(1, int(cfg.recovery_anchor_stride_frames))
        for pos in range(stride, max(len(segment_frames) - 1, 0), stride):
            recovery_candidate_frames.append(int(segment_frames[pos]))
        segment = {
            "segment_id": segment_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "anchor_frame_idx": start_frame,
            "visible_track_ids": visible,
            "boundary_reasons": sorted(boundary_reasons.get(start_frame, set())),
            "recovery_stride_frames": stride,
            "recovery_candidate_frames": recovery_candidate_frames,
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
    keyframes_by_segment: Dict[int, Set[int]] = defaultdict(set)
    for item in ordered_keyframes:
        keyframes_by_segment[int(item["segment_id"])].add(int(item["frame_idx"]))
    for segment in segments:
        segment_id = int(segment["segment_id"])
        keyframe_frames = set(keyframes_by_segment.get(segment_id, set()))
        recovery_frames = [
            int(frame_idx)
            for frame_idx in segment.get("recovery_candidate_frames", [])
            if int(frame_idx) not in keyframe_frames
        ]
        end_frame = int(segment["end_frame"])
        if end_frame not in keyframe_frames and end_frame not in recovery_frames:
            recovery_frames.append(end_frame)
        segment["keyframe_frame_indices"] = sorted(keyframe_frames)
        segment["recovery_candidate_frames"] = sorted(set(int(frame_idx) for frame_idx in recovery_frames))
    return segments, ordered_keyframes, frame_to_segment


def _save_proposals_npz(bundle_dir: Path, per_frame_proposals: Dict[int, List[Proposal]]) -> Path:
    records: List[Proposal] = []
    for frame_idx in sorted(per_frame_proposals.keys()):
        records.extend(per_frame_proposals[frame_idx])

    npz_path = bundle_dir / "proposals.npz"
    if not records:
        np.savez_compressed(
            npz_path,
            frame_idx=np.zeros((0,), dtype=np.int32),
            det_idx=np.zeros((0,), dtype=np.int32),
            track_id=np.zeros((0,), dtype=np.int32),
            bbox_xyxy=np.zeros((0, 4), dtype=np.float32),
            bbox_xyxy_orig=np.zeros((0, 4), dtype=np.float32),
            score=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=np.int32),
        )
        return npz_path

    np.savez_compressed(
        npz_path,
        frame_idx=np.asarray([item.frame_idx for item in records], dtype=np.int32),
        det_idx=np.asarray([item.det_idx for item in records], dtype=np.int32),
        track_id=np.asarray([item.track_id for item in records], dtype=np.int32),
        bbox_xyxy=np.asarray([item.bbox_xyxy for item in records], dtype=np.float32),
        bbox_xyxy_orig=np.asarray([item.bbox_xyxy_orig for item in records], dtype=np.float32),
        score=np.asarray([item.score for item in records], dtype=np.float32),
        class_id=np.asarray([item.class_id for item in records], dtype=np.int32),
    )
    return npz_path


def preprocess_clip(clip: ClipRef, clip_id: int, cfg: MediaPipeReviewConfig) -> Dict:
    total_start = time.perf_counter()
    frame_source = make_frame_source(clip.episode)
    bundle_dir = cfg.bundles_dir / f"clip_{clip_id:06d}"
    frames_dir = bundle_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames_relpaths: Dict[int, str] = {}
    frame_sizes: Dict[int, Dict[str, List[int]]] = {}
    ordered_frame_indices: List[int] = []
    per_frame_proposals: Dict[int, List[Proposal]] = {}
    detector_inputs: List[Dict[str, Any]] = []
    detector_infer_seconds = 0.0
    tracker_association_seconds = 0.0
    first_pass_start = time.perf_counter()

    detector = UltralyticsProposalDetector(cfg)
    try:
        for packet in frame_source.iter_frames(clip.clip_start, clip.clip_end):
            orig_size = packet.image.size
            preview_size = _preview_size(orig_size, cfg.preview_width)
            detector_inputs.append(
                {
                    "frame_idx": int(packet.frame_idx),
                    "image_np": np.ascontiguousarray(np.asarray(packet.image)),
                    "orig_size": orig_size,
                    "preview_size": preview_size,
                }
            )
            ordered_frame_indices.append(int(packet.frame_idx))
            frame_sizes[int(packet.frame_idx)] = {
                "orig_size": [int(orig_size[0]), int(orig_size[1])],
                "preview_size": [int(preview_size[0]), int(preview_size[1])],
            }
            if len(detector_inputs) < cfg.yolo_predict_batch_size:
                continue
            batch_proposals, batch_timing = detector.detect_batch(detector_inputs)
            per_frame_proposals.update(batch_proposals)
            detector_infer_seconds += float(batch_timing["detector_infer_seconds"])
            tracker_association_seconds += float(batch_timing["tracker_association_seconds"])
            detector_inputs = []
        if detector_inputs:
            batch_proposals, batch_timing = detector.detect_batch(detector_inputs)
            per_frame_proposals.update(batch_proposals)
            detector_infer_seconds += float(batch_timing["detector_infer_seconds"])
            tracker_association_seconds += float(batch_timing["tracker_association_seconds"])
    finally:
        detector.close()
    first_pass_seconds = time.perf_counter() - first_pass_start
    decode_seconds = max(0.0, first_pass_seconds - detector_infer_seconds - tracker_association_seconds)

    tracks = _build_yolo_tracks(
        per_frame_proposals,
        bad_frame_indices=clip.bad_frames,
        min_frames=cfg.min_chain_frames,
    )
    visible_tracks = list(tracks)
    too_dense_for_review = len(visible_tracks) > cfg.max_review_tracks

    bad_frame_lookup = {int(frame_idx) for frame_idx in clip.bad_frames}
    frame_tracks, visible_track_ids = _build_frame_tracks(ordered_frame_indices, visible_tracks)
    image_size = tuple(frame_sizes[ordered_frame_indices[0]]["preview_size"]) if ordered_frame_indices else (0, 0)
    keyframe_build_start = time.perf_counter()
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
    keyframe_build_seconds = time.perf_counter() - keyframe_build_start
    for item in frame_tracks:
        item["segment_id"] = int(frame_to_segment.get(int(item["frame_idx"]), -1))

    review_frame_indices = {
        int(item["frame_idx"])
        for item in keyframes
    }
    for segment in segments:
        review_frame_indices.update(int(frame_idx) for frame_idx in segment.get("recovery_candidate_frames", []))

    artifact_write_start = time.perf_counter()
    for packet in frame_source.iter_frame_indices(sorted(review_frame_indices)):
        review_image = _resize_for_review(packet.image, cfg.preview_width)
        relpath = f"bundles/clip_{clip_id:06d}/frames/{packet.frame_idx:06d}.jpg"
        _save_review_frame(review_image, frames_dir / f"{packet.frame_idx:06d}.jpg")
        frames_relpaths[int(packet.frame_idx)] = relpath
    for item in keyframes:
        item["relpath"] = frames_relpaths[int(item["frame_idx"])]
    proposals_npz_path = _save_proposals_npz(bundle_dir, per_frame_proposals)
    artifact_write_seconds = time.perf_counter() - artifact_write_start

    timing = {
        "frame_decode_seconds": float(decode_seconds),
        "detector_infer_seconds": float(detector_infer_seconds),
        "tracker_association_seconds": float(tracker_association_seconds),
        "keyframe_build_seconds": float(keyframe_build_seconds),
        "artifact_write_seconds": float(artifact_write_seconds),
        "total_seconds": float(time.perf_counter() - total_start),
    }

    bundle = {
        "clip_id": clip_id,
        "review_unit": "episode",
        "episode_id": clip.episode.episode_id,
        "episode_name": clip.episode.episode_name,
        "dataset_name": clip.episode.dataset_name,
        "clip_start": clip.clip_start,
        "clip_end": clip.clip_end,
        "dirty_reason": clip.dirty_reason,
        "bad_frame_indices": sorted(bad_frame_lookup),
        "too_dense_for_review": too_dense_for_review,
        "track_generation": detector.track_generation,
        "detector_model_path": str(cfg.yolo_model_path),
        "proposals_npz_relpath": str(proposals_npz_path.relative_to(cfg.artifacts_dir)),
        "timing": timing,
        "frames": [
            {
                "frame_idx": frame_idx,
                "relpath": frames_relpaths.get(frame_idx),
                "is_bad": frame_idx in bad_frame_lookup,
                "segment_id": int(frame_to_segment.get(frame_idx, -1)),
                "visible_track_ids": sorted(visible_track_ids.get(frame_idx, set())),
                **frame_sizes.get(frame_idx, {}),
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
