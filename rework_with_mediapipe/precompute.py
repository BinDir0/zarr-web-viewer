from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from .chains import build_candidate_chains, build_merge_questions
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


class MediaPipeProposalDetector:
    def __init__(self, cfg: MediaPipeReviewConfig):
        mp, mp_python, vision = _ensure_mediapipe(cfg)
        if not cfg.hand_landmarker_task.exists():
            raise FileNotFoundError(
                f"缺少 hand_landmarker.task: {cfg.hand_landmarker_task}"
            )
        self._mp = mp
        base_options = mp_python.BaseOptions(model_asset_path=str(cfg.hand_landmarker_task))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=cfg.detector_max_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._detector.close()

    def detect(self, image: Image.Image, frame_idx: int) -> List[Proposal]:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.asarray(image))
        result = self._detector.detect(mp_image)
        width, height = image.size
        proposals: List[Proposal] = []
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
            if handedness:
                category = handedness[0]
                handedness_score = _safe_float(category.score)
                if str(category.category_name).lower().startswith("left"):
                    handedness_score *= -1.0
            proposals.append(
                Proposal(
                    frame_idx=frame_idx,
                    det_idx=det_idx,
                    bbox_xyxy=bbox_xyxy,
                    score=float(abs(handedness_score)),
                    handedness_score=handedness_score,
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


def _candidate_card(chain: Dict, frames_relpaths: Dict[int, str]) -> Dict:
    preview_frame = int(chain["preview_frame"])
    return {
        "chain_index": chain["chain_index"],
        "role_hint": chain["role_hint"],
        "score": chain["score"],
        "start_frame": chain["start_frame"],
        "end_frame": chain["end_frame"],
        "num_frames": chain["num_frames"],
        "preview_frame": preview_frame,
        "preview_relpath": frames_relpaths[preview_frame],
    }


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
            per_frame_proposals[packet.frame_idx] = detector.detect(review_image, packet.frame_idx)
    finally:
        detector.close()

    chains = build_candidate_chains(
        per_frame_proposals,
        image_size=image_size,
        max_gap=cfg.track_max_gap,
        min_chain_frames=cfg.min_chain_frames,
    )
    top_k = cfg.top_k_chains
    visible_chains = chains[:top_k]
    for idx, chain in enumerate(chains):
        chain["hidden"] = idx >= top_k
    merge_questions = build_merge_questions(visible_chains)

    chain_lookup: Dict[Tuple[int, int], int] = {}
    for chain in chains:
        for proposal in chain["proposals"]:
            chain_lookup[(int(proposal["frame_idx"]), int(proposal["det_idx"]))] = int(chain["chain_index"])

    proposals_npz_path = _save_proposals_npz(bundle_dir, per_frame_proposals, chain_lookup)
    bundle = {
        "clip_id": clip_id,
        "episode_id": clip.episode.episode_id,
        "episode_name": clip.episode.episode_name,
        "dataset_name": clip.episode.dataset_name,
        "clip_start": clip.clip_start,
        "clip_end": clip.clip_end,
        "dirty_reason": clip.dirty_reason,
        "bad_frames": clip.bad_frames,
        "proposals_npz_relpath": str(proposals_npz_path.relative_to(cfg.artifacts_dir)),
        "frames": [
            {"frame_idx": frame_idx, "relpath": frames_relpaths[frame_idx]}
            for frame_idx in sorted(frames_relpaths.keys())
        ],
        "chains": visible_chains,
        "merge_questions": merge_questions,
        "card_view": [_candidate_card(chain, frames_relpaths) for chain in visible_chains],
    }
    with (bundle_dir / "bundle.json").open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    return {
        "bundle_dir": str(bundle_dir),
        "bundle_relpath": str(bundle_dir.relative_to(cfg.artifacts_dir)),
        "bundle": bundle,
        "chains": chains,
    }
