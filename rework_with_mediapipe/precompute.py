from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

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

    return YOLO


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


def _import_mediapipe_runtime():
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python as mp_python  # type: ignore
    from mediapipe.tasks.python import vision  # type: ignore

    return mp, mp_python, vision


def _ensure_mediapipe_runtime(cfg: MediaPipeReviewConfig):
    first_error: Exception | None = None
    if cfg.prefer_installed_mediapipe:
        try:
            return _import_mediapipe_runtime()
        except Exception as exc:  # pragma: no cover - import surface differs by env
            first_error = exc

    repo_root = str(cfg.mediapipe_repo_root)
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        return _import_mediapipe_runtime()
    except Exception as exc:  # pragma: no cover - import surface differs by env
        if first_error is not None:
            raise RuntimeError(
                "缺少可用的 mediapipe Python 依赖。"
                "已先尝试环境内安装版本，再尝试 mediapipe_repo_root，均失败。"
                f" 当前解释器: {sys.executable}。"
            ) from exc
        raise RuntimeError(
            "缺少 mediapipe Python 依赖，无法运行 fallback 检测。"
            f" 当前解释器: {sys.executable}。"
        ) from exc


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
    inter = _bbox_area_xyxy((inter_x1, inter_y1, inter_x2, inter_y2))
    if inter <= 0.0:
        return 0.0
    union = _bbox_area_xyxy(box_a) + _bbox_area_xyxy(box_b) - inter
    return inter / max(union, 1e-6)


def _preview_size(orig_size: Tuple[int, int], preview_width: int) -> Tuple[int, int]:
    width, height = int(orig_size[0]), int(orig_size[1])
    if width <= preview_width:
        return width, height
    scale = preview_width / float(width)
    return preview_width, int(height * scale)


def _resize_for_review(image: Image.Image, preview_width: int) -> Image.Image:
    width, height = image.size
    if width <= preview_width:
        return image
    scale = preview_width / float(width)
    return image.resize((preview_width, int(height * scale)), Image.Resampling.BILINEAR)


def _save_review_frame(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=80)


def _local_config_int(cfg: MediaPipeReviewConfig, key: str, default: int) -> int:
    try:
        return int(cfg.raw.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _local_config_float(cfg: MediaPipeReviewConfig, key: str, default: float) -> float:
    try:
        return float(cfg.raw.get(key, default))
    except (TypeError, ValueError):
        return float(default)


class PreprocessRuntime:
    def __init__(self, cfg: MediaPipeReviewConfig):
        self._cfg = cfg
        self._primary_detector: UltralyticsFrameProposalDetector | None = None
        self._fallback_detector: MediaPipeImageProposalDetector | None = None

    def primary_detector(self) -> "UltralyticsFrameProposalDetector":
        if self._primary_detector is None:
            self._primary_detector = UltralyticsFrameProposalDetector(self._cfg)
        return self._primary_detector

    def fallback_detector(self) -> "MediaPipeImageProposalDetector":
        if self._fallback_detector is None:
            self._fallback_detector = MediaPipeImageProposalDetector(self._cfg)
        return self._fallback_detector

    def close(self) -> None:
        if self._fallback_detector is not None:
            self._fallback_detector.close()
            self._fallback_detector = None
        self._primary_detector = None


class UltralyticsFrameProposalDetector:
    def __init__(self, cfg: MediaPipeReviewConfig):
        YOLO = _ensure_ultralytics_runtime(cfg)
        if not str(cfg.raw.get("yolo_model_path", "") or "").strip():
            raise FileNotFoundError("缺少 yolo_model_path 配置，无法运行 YOLO 预处理。")
        if not cfg.yolo_model_path.exists():
            raise FileNotFoundError(f"缺少 YOLO 权重文件: {cfg.yolo_model_path}")
        self._cfg = cfg
        self._model = YOLO(str(cfg.yolo_model_path))

    def _predict_batch(self, frames: List[Dict[str, Any]]) -> List[Any]:
        kwargs: Dict[str, Any] = {
            "source": [item["image_np"] for item in frames],
            "stream": False,
            "verbose": False,
            "conf": self._cfg.yolo_confidence,
            "iou": self._cfg.yolo_iou,
            "max_det": self._cfg.yolo_max_det,
            "imgsz": self._cfg.yolo_imgsz,
        }
        if self._cfg.yolo_device:
            kwargs["device"] = self._cfg.yolo_device
        results = self._model.predict(**kwargs)
        if len(results) != len(frames):
            raise RuntimeError(f"YOLO predict 返回帧数不匹配: expected={len(frames)} actual={len(results)}")
        return list(results)

    def detect_batch(self, frames: List[Dict[str, Any]]) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, float]]:
        if not frames:
            return {}, {"detector_infer_seconds": 0.0}
        start = time.perf_counter()
        results = self._predict_batch(frames)
        infer_seconds = time.perf_counter() - start
        proposals_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        top_k = max(1, _local_config_int(self._cfg, "frame_proposal_top_k", min(self._cfg.yolo_max_det, 6)))
        for frame_item, result in zip(frames, results):
            frame_idx = int(frame_item["frame_idx"])
            preview_width, preview_height = frame_item["preview_size"]
            orig_width, orig_height = frame_item["orig_size"]
            scale_x = preview_width / max(float(orig_width), 1.0)
            scale_y = preview_height / max(float(orig_height), 1.0)
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            frame_props: List[Dict[str, Any]] = []
            if boxes is not None and len(boxes) > 0:
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
                order = np.argsort(-conf)
                for local_rank, det_idx in enumerate(order[:top_k]):
                    x1, y1, x2, y2 = [float(item) for item in xyxy[int(det_idx)].tolist()]
                    class_id = int(cls[int(det_idx)])
                    class_name = str(names.get(class_id, class_id))
                    frame_props.append(
                        {
                            "frame_idx": frame_idx,
                            "det_idx": int(local_rank),
                            "bbox_xyxy_orig": [x1, y1, x2, y2],
                            "bbox_xyxy": [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                            "score": float(conf[int(det_idx)]),
                            "class_id": class_id,
                            "class_name": class_name,
                            "detector_name": "yolo",
                        }
                    )
            proposals_by_frame[frame_idx] = frame_props
        return proposals_by_frame, {"detector_infer_seconds": float(infer_seconds)}


class MediaPipeImageProposalDetector:
    def __init__(self, cfg: MediaPipeReviewConfig):
        if not cfg.hand_landmarker_task.exists():
            raise FileNotFoundError(f"缺少 MediaPipe hand_landmarker.task: {cfg.hand_landmarker_task}")
        mp, mp_python, vision = _ensure_mediapipe_runtime(cfg)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(cfg.hand_landmarker_task)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max(1, cfg.detector_max_hands),
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._mp = mp
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._cfg = cfg

    def close(self) -> None:
        if hasattr(self._detector, "close"):
            self._detector.close()

    def detect_batch(self, frames: List[Dict[str, Any]]) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, float]]:
        if not frames:
            return {}, {"detector_infer_seconds": 0.0}
        start = time.perf_counter()
        proposals_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        top_k = max(1, _local_config_int(self._cfg, "frame_proposal_top_k", 6))
        bbox_pad = _local_config_float(self._cfg, "mediapipe_bbox_pad_ratio", 0.04)
        for frame_item in frames:
            frame_idx = int(frame_item["frame_idx"])
            orig_width, orig_height = frame_item["orig_size"]
            preview_width, preview_height = frame_item["preview_size"]
            scale_x = preview_width / max(float(orig_width), 1.0)
            scale_y = preview_height / max(float(orig_height), 1.0)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_item["image_np"])
            result = self._detector.detect(mp_image)
            handedness = getattr(result, "handedness", []) or []
            frame_props: List[Dict[str, Any]] = []
            for hand_idx, hand_landmarks in enumerate(getattr(result, "hand_landmarks", []) or []):
                xs = [float(lm.x) for lm in hand_landmarks]
                ys = [float(lm.y) for lm in hand_landmarks]
                if not xs or not ys:
                    continue
                x1 = max(0.0, min(xs) - bbox_pad)
                y1 = max(0.0, min(ys) - bbox_pad)
                x2 = min(1.0, max(xs) + bbox_pad)
                y2 = min(1.0, max(ys) + bbox_pad)
                if not (x2 > x1 and y2 > y1):
                    continue
                handedness_label = "hand"
                handedness_score = 0.5
                if hand_idx < len(handedness) and handedness[hand_idx]:
                    category = handedness[hand_idx][0]
                    handedness_label = str(getattr(category, "category_name", "") or getattr(category, "display_name", "") or "hand")
                    handedness_score = float(getattr(category, "score", 0.5) or 0.5)
                bbox_orig = [
                    x1 * float(orig_width),
                    y1 * float(orig_height),
                    x2 * float(orig_width),
                    y2 * float(orig_height),
                ]
                frame_props.append(
                    {
                        "frame_idx": frame_idx,
                        "det_idx": int(hand_idx),
                        "bbox_xyxy_orig": bbox_orig,
                        "bbox_xyxy": [
                            bbox_orig[0] * scale_x,
                            bbox_orig[1] * scale_y,
                            bbox_orig[2] * scale_x,
                            bbox_orig[3] * scale_y,
                        ],
                        "score": float(handedness_score),
                        "class_id": 0 if handedness_label.lower() == "left" else 1 if handedness_label.lower() == "right" else -1,
                        "class_name": handedness_label,
                        "detector_name": "mediapipe_image",
                    }
                )
            frame_props.sort(key=lambda item: float(item["score"]), reverse=True)
            proposals_by_frame[frame_idx] = frame_props[:top_k]
        infer_seconds = time.perf_counter() - start
        return proposals_by_frame, {"detector_infer_seconds": float(infer_seconds)}


def _proposal_score_key(proposal: Dict[str, Any]) -> Tuple[float, float]:
    detector_bonus = 0.0 if str(proposal.get("detector_name", "")) == "yolo" else -0.03
    return (float(proposal.get("score", 0.0)) + detector_bonus, float(_bbox_area_xyxy(proposal.get("bbox_xyxy_orig", [0, 0, 0, 0]))))


def _merge_frame_proposals(
    *,
    frame_idx: int,
    orig_size: Tuple[int, int],
    preview_size: Tuple[int, int],
    primary_props: Sequence[Dict[str, Any]],
    fallback_props: Sequence[Dict[str, Any]],
    cfg: MediaPipeReviewConfig,
) -> List[Proposal]:
    candidates = list(primary_props) + list(fallback_props)
    if not candidates:
        return []
    nms_iou = _local_config_float(cfg, "proposal_merge_iou", 0.45)
    top_k = max(1, _local_config_int(cfg, "frame_proposal_top_k", min(cfg.yolo_max_det, 6)))
    kept: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=_proposal_score_key, reverse=True):
        if any(_bbox_iou_xyxy(candidate["bbox_xyxy_orig"], existing["bbox_xyxy_orig"]) >= nms_iou for existing in kept):
            continue
        kept.append(candidate)
        if len(kept) >= top_k:
            break
    finalized: List[Proposal] = []
    for det_idx, candidate in enumerate(kept):
        proposal_id = f"{frame_idx}:{candidate['detector_name']}:{det_idx}"
        finalized.append(
            Proposal(
                frame_idx=int(frame_idx),
                det_idx=int(det_idx),
                proposal_id=proposal_id,
                bbox_xyxy=[float(item) for item in candidate["bbox_xyxy"]],
                bbox_xyxy_orig=[float(item) for item in candidate["bbox_xyxy_orig"]],
                score=float(candidate["score"]),
                class_id=int(candidate.get("class_id", -1)),
                class_name=str(candidate.get("class_name", "hand")),
                detector_name=str(candidate.get("detector_name", "unknown")),
            )
        )
    return finalized


def _select_fallback_frames(
    ordered_frame_indices: Sequence[int],
    primary_by_frame: Dict[int, List[Dict[str, Any]]],
    clip: ClipRef,
    cfg: MediaPipeReviewConfig,
) -> List[int]:
    score_threshold = _local_config_float(cfg, "fallback_score_threshold", 0.45)
    min_good = max(1, _local_config_int(cfg, "fallback_min_good_proposals", 2))
    context_radius = max(0, _local_config_int(cfg, "fallback_dirty_context_radius", 1))
    fallback_frames: set[int] = set()
    bad_frames = {int(item) for item in clip.bad_frames}
    ordered_set = set(int(item) for item in ordered_frame_indices)
    for frame_idx in ordered_frame_indices:
        props = list(primary_by_frame.get(int(frame_idx), []))
        high_score = sum(1 for item in props if float(item.get("score", 0.0)) >= score_threshold)
        if not props or high_score < min_good:
            fallback_frames.add(int(frame_idx))
    for bad_frame in bad_frames:
        for delta in range(-context_radius, context_radius + 1):
            candidate = int(bad_frame) + delta
            if candidate in ordered_set:
                fallback_frames.add(candidate)
    return sorted(fallback_frames)


def _frame_best_box(frame_proposals: Sequence[Proposal]) -> List[float] | None:
    if not frame_proposals:
        return None
    return list(frame_proposals[0].bbox_xyxy_orig)


def _compute_frame_uncertainty(
    ordered_frame_indices: Sequence[int],
    merged_proposals: Dict[int, List[Proposal]],
    clip: ClipRef,
    cfg: MediaPipeReviewConfig,
) -> Tuple[Dict[int, float], Dict[int, set[str]], set[int]]:
    uncertainty: Dict[int, float] = {}
    reasons: Dict[int, set[str]] = defaultdict(set)
    forced_review_frames: set[int] = set()
    bad_frames = {int(item) for item in clip.bad_frames}
    dirty_context_radius = max(0, _local_config_int(cfg, "review_dirty_context_radius", 1))
    geometry_center_threshold = _local_config_float(cfg, "review_geometry_center_threshold", 0.12)
    geometry_area_log_threshold = _local_config_float(cfg, "review_geometry_area_log_threshold", 0.55)
    ambiguous_gap_threshold = _local_config_float(cfg, "review_ambiguous_score_gap", 0.18)

    prev_frame_idx: int | None = None
    prev_props: List[Proposal] = []
    prev_best_box: List[float] | None = None
    prev_best_score = 0.0
    image_diag = 1.0

    for pos, frame_idx in enumerate(ordered_frame_indices):
        props = list(merged_proposals.get(int(frame_idx), []))
        best_box = _frame_best_box(props)
        best_score = float(props[0].score) if props else 0.0
        score = 0.0
        if pos == 0:
            reasons[frame_idx].add("episode_start")
            forced_review_frames.add(frame_idx)
        if pos == len(ordered_frame_indices) - 1:
            reasons[frame_idx].add("episode_end")
            forced_review_frames.add(frame_idx)
        if frame_idx in bad_frames:
            reasons[frame_idx].add("dirty_frame")
            forced_review_frames.add(frame_idx)
            score += 4.0
        for delta in range(1, dirty_context_radius + 1):
            if frame_idx - delta in bad_frames or frame_idx + delta in bad_frames:
                reasons[frame_idx].add("dirty_context")
                forced_review_frames.add(frame_idx)
                score += 1.5
                break
        if not props:
            reasons[frame_idx].add("no_proposal")
            score += 1.5
        else:
            score += max(0.0, 0.75 - best_score)
        if len(props) >= 2:
            gap = float(props[0].score) - float(props[1].score)
            if gap < ambiguous_gap_threshold:
                reasons[frame_idx].add("ambiguous_scores")
                score += (ambiguous_gap_threshold - gap) + 0.5

        if prev_frame_idx is not None:
            if bool(props) != bool(prev_props):
                reasons[frame_idx].add("proposal_presence_changed")
                reasons[prev_frame_idx].add("proposal_presence_changed")
                forced_review_frames.add(frame_idx)
                forced_review_frames.add(prev_frame_idx)
                score += 1.8
            if len(props) != len(prev_props):
                reasons[frame_idx].add("proposal_count_changed")
                score += 0.8
            if best_box is not None and prev_best_box is not None:
                center = _bbox_center_xyxy(best_box)
                prev_center = _bbox_center_xyxy(prev_best_box)
                area = max(_bbox_area_xyxy(best_box), 1.0)
                prev_area = max(_bbox_area_xyxy(prev_best_box), 1.0)
                if image_diag <= 1.0:
                    image_diag = max(math.hypot(best_box[2], best_box[3]), 1.0)
                center_delta = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1]) / max(image_diag, 1.0)
                area_delta = abs(math.log(area / prev_area))
                if center_delta >= geometry_center_threshold or area_delta >= geometry_area_log_threshold:
                    reasons[frame_idx].add("geometry_jump")
                    forced_review_frames.add(frame_idx)
                    score += (center_delta * 3.0) + area_delta
            if best_score < prev_best_score * 0.55 and best_score < 0.4:
                reasons[frame_idx].add("confidence_drop")
                score += 0.8

        uncertainty[frame_idx] = float(score)
        prev_frame_idx = int(frame_idx)
        prev_props = props
        prev_best_box = best_box
        prev_best_score = best_score

    return uncertainty, reasons, forced_review_frames


def _select_review_frames(
    ordered_frame_indices: Sequence[int],
    uncertainty_by_frame: Dict[int, float],
    reason_map: Dict[int, set[str]],
    forced_review_frames: set[int],
    cfg: MediaPipeReviewConfig,
) -> List[Dict[str, Any]]:
    if not ordered_frame_indices:
        return []
    max_frames = max(1, _local_config_int(cfg, "review_frame_max_count", 40))
    cover_stride = max(1, _local_config_int(cfg, "review_cover_stride_frames", 48))
    min_score = _local_config_float(cfg, "review_extra_frame_min_uncertainty", 0.65)
    selected = set(int(item) for item in forced_review_frames if int(item) in set(int(frame) for frame in ordered_frame_indices))
    ordered = [int(item) for item in ordered_frame_indices]

    for start in range(0, len(ordered), cover_stride):
        window = ordered[start : start + cover_stride]
        if not window:
            continue
        if any(frame_idx in selected for frame_idx in window):
            continue
        best_frame = max(window, key=lambda frame_idx: float(uncertainty_by_frame.get(frame_idx, 0.0)))
        if float(uncertainty_by_frame.get(best_frame, 0.0)) > 0.0:
            selected.add(int(best_frame))
            reason_map[int(best_frame)].add("coverage_window")

    ranked = [
        frame_idx
        for frame_idx in sorted(
            ordered,
            key=lambda item: (float(uncertainty_by_frame.get(item, 0.0)), -item),
            reverse=True,
        )
        if frame_idx not in selected and float(uncertainty_by_frame.get(frame_idx, 0.0)) >= min_score
    ]
    for frame_idx in ranked:
        if len(selected) >= max_frames:
            break
        selected.add(int(frame_idx))
        reason_map[int(frame_idx)].add("uncertainty_peak")

    if len(selected) > max_frames:
        must_keep = {ordered[0], ordered[-1]}
        must_keep.update(int(item) for item in forced_review_frames)
        kept = list(sorted(must_keep & selected))
        extras = [frame_idx for frame_idx in sorted(selected - set(kept), key=lambda item: float(uncertainty_by_frame.get(item, 0.0)), reverse=True)]
        for frame_idx in extras:
            if len(kept) >= max_frames:
                break
            kept.append(int(frame_idx))
        selected = set(kept)

    review_frames: List[Dict[str, Any]] = []
    for frame_idx in sorted(selected):
        reasons = sorted(reason_map.get(int(frame_idx), set()))
        review_frames.append(
            {
                "frame_idx": int(frame_idx),
                "relpath": "",
                "kind": "required" if int(frame_idx) in forced_review_frames else "uncertainty",
                "reasons": reasons,
                "uncertainty": float(uncertainty_by_frame.get(int(frame_idx), 0.0)),
            }
        )
    return review_frames


def _save_proposals_npz(bundle_dir: Path, frame_proposals: Dict[int, List[Proposal]]) -> Path:
    records: List[Proposal] = []
    for frame_idx in sorted(frame_proposals.keys()):
        records.extend(frame_proposals[frame_idx])

    npz_path = bundle_dir / "proposals.npz"
    if not records:
        np.savez_compressed(
            npz_path,
            frame_idx=np.zeros((0,), dtype=np.int32),
            det_idx=np.zeros((0,), dtype=np.int32),
            proposal_id=np.asarray([], dtype=np.str_),
            bbox_xyxy=np.zeros((0, 4), dtype=np.float32),
            bbox_xyxy_orig=np.zeros((0, 4), dtype=np.float32),
            score=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=np.int32),
            class_name=np.asarray([], dtype=np.str_),
            detector_name=np.asarray([], dtype=np.str_),
        )
        return npz_path

    np.savez_compressed(
        npz_path,
        frame_idx=np.asarray([item.frame_idx for item in records], dtype=np.int32),
        det_idx=np.asarray([item.det_idx for item in records], dtype=np.int32),
        proposal_id=np.asarray([item.proposal_id for item in records], dtype=np.str_),
        bbox_xyxy=np.asarray([item.bbox_xyxy for item in records], dtype=np.float32),
        bbox_xyxy_orig=np.asarray([item.bbox_xyxy_orig for item in records], dtype=np.float32),
        score=np.asarray([item.score for item in records], dtype=np.float32),
        class_id=np.asarray([item.class_id for item in records], dtype=np.int32),
        class_name=np.asarray([item.class_name for item in records], dtype=np.str_),
        detector_name=np.asarray([item.detector_name for item in records], dtype=np.str_),
    )
    return npz_path


def preprocess_clip(
    clip: ClipRef,
    clip_id: int,
    cfg: MediaPipeReviewConfig,
    runtime: PreprocessRuntime | None = None,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    frame_source = make_frame_source(clip.episode)
    bundle_dir = cfg.bundles_dir / f"clip_{clip_id:06d}"
    frames_dir = bundle_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    ordered_frame_indices: List[int] = []
    frame_sizes: Dict[int, Dict[str, List[int]]] = {}
    detector_inputs: List[Dict[str, Any]] = []
    primary_raw_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    primary_infer_seconds = 0.0

    primary_detector = runtime.primary_detector() if runtime is not None else UltralyticsFrameProposalDetector(cfg)
    primary_start = time.perf_counter()
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
        batch_props, timing = primary_detector.detect_batch(detector_inputs)
        primary_raw_by_frame.update(batch_props)
        primary_infer_seconds += float(timing["detector_infer_seconds"])
        detector_inputs = []
    if detector_inputs:
        batch_props, timing = primary_detector.detect_batch(detector_inputs)
        primary_raw_by_frame.update(batch_props)
        primary_infer_seconds += float(timing["detector_infer_seconds"])
    primary_seconds = time.perf_counter() - primary_start

    fallback_frames = _select_fallback_frames(ordered_frame_indices, primary_raw_by_frame, clip, cfg)
    fallback_raw_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    fallback_infer_seconds = 0.0
    if fallback_frames:
        fallback_detector = runtime.fallback_detector() if runtime is not None else MediaPipeImageProposalDetector(cfg)
        fallback_inputs: List[Dict[str, Any]] = []
        for packet in frame_source.iter_frame_indices(fallback_frames):
            sizes = frame_sizes[int(packet.frame_idx)]
            fallback_inputs.append(
                {
                    "frame_idx": int(packet.frame_idx),
                    "image_np": np.ascontiguousarray(np.asarray(packet.image)),
                    "orig_size": (int(sizes["orig_size"][0]), int(sizes["orig_size"][1])),
                    "preview_size": (int(sizes["preview_size"][0]), int(sizes["preview_size"][1])),
                }
            )
        batch_props, timing = fallback_detector.detect_batch(fallback_inputs)
        fallback_raw_by_frame.update(batch_props)
        fallback_infer_seconds += float(timing["detector_infer_seconds"])
        if runtime is None:
            fallback_detector.close()

    merge_start = time.perf_counter()
    merged_proposals: Dict[int, List[Proposal]] = {}
    frame_proposals_payload: List[Dict[str, Any]] = []
    for frame_idx in ordered_frame_indices:
        sizes = frame_sizes[int(frame_idx)]
        merged = _merge_frame_proposals(
            frame_idx=int(frame_idx),
            orig_size=(int(sizes["orig_size"][0]), int(sizes["orig_size"][1])),
            preview_size=(int(sizes["preview_size"][0]), int(sizes["preview_size"][1])),
            primary_props=primary_raw_by_frame.get(int(frame_idx), []),
            fallback_props=fallback_raw_by_frame.get(int(frame_idx), []),
            cfg=cfg,
        )
        merged_proposals[int(frame_idx)] = merged
        frame_proposals_payload.append(
            {
                "frame_idx": int(frame_idx),
                "proposals": [item.to_dict() for item in merged],
            }
        )
    merge_seconds = time.perf_counter() - merge_start

    keyframe_start = time.perf_counter()
    uncertainty_by_frame, reason_map, forced_review_frames = _compute_frame_uncertainty(
        ordered_frame_indices,
        merged_proposals,
        clip,
        cfg,
    )
    review_frames = _select_review_frames(
        ordered_frame_indices,
        uncertainty_by_frame,
        reason_map,
        forced_review_frames,
        cfg,
    )
    keyframe_seconds = time.perf_counter() - keyframe_start

    review_frame_indices = sorted({int(item["frame_idx"]) for item in review_frames})
    frames_relpaths: Dict[int, str] = {}
    artifact_start = time.perf_counter()
    for packet in frame_source.iter_frame_indices(review_frame_indices):
        review_image = _resize_for_review(packet.image, cfg.preview_width)
        relpath = f"bundles/clip_{clip_id:06d}/frames/{packet.frame_idx:06d}.jpg"
        _save_review_frame(review_image, frames_dir / f"{packet.frame_idx:06d}.jpg")
        frames_relpaths[int(packet.frame_idx)] = relpath
    for item in review_frames:
        item["relpath"] = frames_relpaths[int(item["frame_idx"])]
    proposals_npz_path = _save_proposals_npz(bundle_dir, merged_proposals)
    artifact_seconds = time.perf_counter() - artifact_start

    frames_payload = []
    for frame_idx in ordered_frame_indices:
        frame_meta = {
            "frame_idx": int(frame_idx),
            "relpath": frames_relpaths.get(int(frame_idx)),
            "is_bad": int(frame_idx) in {int(item) for item in clip.bad_frames},
            "proposal_ids": [item.proposal_id for item in merged_proposals.get(int(frame_idx), [])],
            "uncertainty": float(uncertainty_by_frame.get(int(frame_idx), 0.0)),
            **frame_sizes.get(int(frame_idx), {}),
        }
        frames_payload.append(frame_meta)

    timing = {
        "primary_total_seconds": float(primary_seconds),
        "primary_infer_seconds": float(primary_infer_seconds),
        "fallback_infer_seconds": float(fallback_infer_seconds),
        "merge_seconds": float(merge_seconds),
        "keyframe_build_seconds": float(keyframe_seconds),
        "artifact_write_seconds": float(artifact_seconds),
        "total_seconds": float(time.perf_counter() - total_start),
    }

    bundle = {
        "bundle_version": 3,
        "clip_id": clip_id,
        "review_unit": "episode",
        "episode_id": clip.episode.episode_id,
        "episode_name": clip.episode.episode_name,
        "dataset_name": clip.episode.dataset_name,
        "clip_start": clip.clip_start,
        "clip_end": clip.clip_end,
        "dirty_reason": clip.dirty_reason,
        "dirty_frame_indices": sorted(int(item) for item in clip.bad_frames),
        "detector_meta": {
            "primary_detector": "yolo",
            "fallback_detector": "mediapipe_image",
            "yolo_model_path": str(cfg.yolo_model_path),
            "hand_landmarker_task": str(cfg.hand_landmarker_task),
            "fallback_frame_count": len(fallback_frames),
        },
        "timing": timing,
        "frames": frames_payload,
        "frame_proposals": frame_proposals_payload,
        "review_frames": review_frames,
        "uncertainty_by_frame": {str(frame_idx): float(score) for frame_idx, score in uncertainty_by_frame.items()},
        "proposals_npz_relpath": str(proposals_npz_path.relative_to(cfg.artifacts_dir)),
    }
    with (bundle_dir / "bundle.json").open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    return {
        "bundle_dir": str(bundle_dir),
        "bundle_relpath": str(bundle_dir.relative_to(cfg.artifacts_dir)),
        "bundle": bundle,
        "tracks": [],
    }
