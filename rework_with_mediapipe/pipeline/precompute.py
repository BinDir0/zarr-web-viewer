from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from app.config import AppConfig
from pipeline.chains import build_candidate_chains, build_merge_questions
from pipeline.frame_sources import make_frame_source
from pipeline.types import ClipRef, Proposal


def _ensure_mediapipe(cfg: AppConfig):
    repo_root = str(cfg.mediapipe_repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import mediapipe as mp  # type: ignore

    return mp


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
    def __init__(self, cfg: AppConfig):
        mp = _ensure_mediapipe(cfg)
        import mediapipe.tasks.python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        model_path = cfg.cache_dir / "hand_landmarker.task"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing hand landmarker task model at {model_path}. "
                "Place a MediaPipe hand_landmarker.task model there before preprocessing."
            )
        self._mp = mp
        self._vision = vision
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=int(cfg.pipeline.get("max_hands", 4)),
            min_hand_detection_confidence=float(cfg.pipeline.get("min_detection_confidence", 0.35)),
            min_hand_presence_confidence=float(cfg.pipeline.get("min_presence_confidence", 0.35)),
            min_tracking_confidence=float(cfg.pipeline.get("min_tracking_confidence", 0.35)),
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
            landmarks_2d = [[float(lm.x), float(lm.y), float(getattr(lm, "visibility", 1.0))] for lm in norm_lms]
            landmarks_3d = [[float(lm.x), float(lm.y), float(lm.z)] for lm in world_lms]
            bbox_xyxy = _compute_bbox(landmarks_2d, width, height)
            palm_normal, wrist_frame = _derive_palm_geometry(landmarks_3d)
            handedness_score = 0.0
            if handedness:
                category = handedness[0]
                handedness_score = float(category.score)
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


def preprocess_clip(clip: ClipRef, clip_id: int, cfg: AppConfig) -> Dict:
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
            review_image = _resize_for_review(packet.image, int(cfg.review.get("preview_width", 640)))
            image_size = review_image.size
            relpath = f"clip_{clip_id:06d}/frames/{packet.frame_idx:06d}.jpg"
            _save_review_frame(review_image, frames_dir / f"{packet.frame_idx:06d}.jpg")
            frames_relpaths[packet.frame_idx] = relpath
            per_frame_proposals[packet.frame_idx] = detector.detect(review_image, packet.frame_idx)
    finally:
        detector.close()

    chains = build_candidate_chains(
        per_frame_proposals,
        image_size=image_size,
        max_gap=int(cfg.pipeline.get("track_max_gap", 2)),
        min_chain_frames=int(cfg.pipeline.get("min_chain_frames", 5)),
    )
    top_k = int(cfg.pipeline.get("top_k_chains", 4))
    visible_chains = chains[:top_k]
    for idx, chain in enumerate(chains):
        chain["hidden"] = idx >= top_k
    merge_questions = build_merge_questions(visible_chains)
    bundle = {
        "clip_id": clip_id,
        "episode_id": clip.episode.episode_id,
        "episode_name": clip.episode.episode_name,
        "dataset_name": clip.episode.dataset_name,
        "clip_start": clip.clip_start,
        "clip_end": clip.clip_end,
        "dirty_reason": clip.dirty_reason,
        "bad_frames": clip.bad_frames,
        "frames": [
            {"frame_idx": frame_idx, "relpath": frames_relpaths[frame_idx]}
            for frame_idx in sorted(frames_relpaths.keys())
        ],
        "chains": visible_chains,
        "merge_questions": merge_questions,
        "card_view": [_candidate_card(chain, frames_relpaths) for chain in visible_chains],
    }
    with open(bundle_dir / "bundle.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    return {
        "bundle_dir": str(bundle_dir),
        "bundle_relpath": str(bundle_dir.relative_to(cfg.app_root)),
        "bundle": bundle,
        "chains": chains,
    }
