from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _path_exists(candidate: Path) -> bool:
    try:
        return candidate.exists()
    except (OSError, PermissionError):
        return False


def _first_existing_path(*raw_candidates: str) -> str:
    candidates = [Path(item) for item in raw_candidates if item]
    for candidate in candidates:
        if _path_exists(candidate):
            return str(candidate)
    return str(candidates[0]) if candidates else ""


def _default_mano_models_root() -> str:
    return _first_existing_path(
        os.environ.get("MANO_MODELS_ROOT", ""),
        "/share_data/guantianrui/manopth/mano/models",
        "/share_data/guantianrui/manopth/mano/model",
        "/root/manopth/mano/models",
        "/root/manopth/mano/model",
    )


def _default_manopth_root() -> str:
    return _first_existing_path(
        os.environ.get("MANOPTH_ROOT", ""),
        "/share_data/guantianrui/manopth",
        "/root/manopth",
    )


def _default_mediapipe_repo_root() -> str:
    return _first_existing_path(
        os.environ.get("MEDIAPIPE_REPO_ROOT", ""),
        "/share_data/guantianrui/mediapipe",
        "/root/mediapipe",
    )


def _default_ultralytics_repo_root() -> str:
    return _first_existing_path(
        os.environ.get("ULTRALYTICS_REPO_ROOT", ""),
        "/share_data/guantianrui/ultralytics",
        "/root/ultralytics",
    )


def _default_yolo_model_path() -> str:
    return _first_existing_path(
        os.environ.get("YOLO_MODEL_PATH", ""),
        "/share_data/guantianrui/HaWoR/weights/external/detector.pt",
        "/share_data/guantianrui/HaWoR/thirdparty/DROID-SLAM/weights/external/detector.pt",
        "/share_data/guantianrui/HaWoR-any4d/thirdparty/DROID-SLAM/weights/external/detector.pt",
        "/share_data/guantianrui/WiLoR/detector.pt",
        "/share_data/guantianrui/YOLO11n-pose-hands/runs/pose/train/weights/best.pt",
        "/share_data/guantianrui/YOLO11n-pose-hands/runs/pose/train/weights/last.pt",
        "/share_data/guantianrui/EgoYOLO/detector.pt",
        "/share_data/guantianrui/EgoYOLO/best.pt",
        "/root/.openclaw/workspace/projects/hawor_original/HaWoR/weights/external/detector.pt",
        "/root/.openclaw/workspace/projects/hawor_original/HaWoR/thirdparty/DROID-SLAM/weights/external/detector.pt",
        "/root/.openclaw/workspace/projects/hawor_original/HaWoR-any4d/thirdparty/DROID-SLAM/weights/external/detector.pt",
        "/root/YOLO11n-pose-hands/runs/pose/train/weights/best.pt",
        "/root/YOLO11n-pose-hands/runs/pose/train/weights/last.pt",
        "/root/ultralytics/detector.pt",
        "/root/ultralytics/best.pt",
        "/root/ultralytics/weights/detector.pt",
        "/root/ultralytics/weights/best.pt",
    )


def _default_hand_landmarker_task() -> str:
    return _first_existing_path(
        os.environ.get("HAND_LANDMARKER_TASK", ""),
        "/share_data/guantianrui/mediapipe/hand_landmarker.task",
        "/root/mediapipe/hand_landmarker.task",
    )


@dataclass(frozen=True)
class MediaPipeReviewConfig:
    raw: Dict[str, Any]
    repo_root: Path

    @property
    def replace_home(self) -> bool:
        return bool(self.raw.get("replace_home", True))

    @property
    def db_path(self) -> Path:
        return Path(str(self.raw["db_path"]))

    @property
    def artifacts_dir(self) -> Path:
        return Path(str(self.raw["artifacts_dir"]))

    @property
    def bundles_dir(self) -> Path:
        return self.artifacts_dir / "bundles"

    @property
    def exports_dir(self) -> Path:
        return self.artifacts_dir / "exports"

    @property
    def cache_dir(self) -> Path:
        return self.artifacts_dir / "cache"

    @property
    def source_annotations_db(self) -> Path:
        return Path(str(self.raw["source_annotations_db"]))

    @property
    def source_config_path(self) -> Path:
        return Path(str(self.raw["source_config_path"]))

    @property
    def factory_base(self) -> Path:
        return Path(str(self.raw["factory_base"]))

    @property
    def factory_start(self) -> int:
        return int(self.raw["factory_start"])

    @property
    def factory_end(self) -> int:
        return int(self.raw["factory_end"])

    @property
    def hand_landmarker_task(self) -> Path:
        return Path(str(self.raw["hand_landmarker_task"]))

    @property
    def prefer_installed_mediapipe(self) -> bool:
        return bool(self.raw.get("prefer_installed_mediapipe", True))

    @property
    def prefer_installed_ultralytics(self) -> bool:
        return bool(self.raw.get("prefer_installed_ultralytics", True))

    @property
    def mediapipe_repo_root(self) -> Path:
        return Path(str(self.raw["mediapipe_repo_root"]))

    @property
    def ultralytics_repo_root(self) -> Path:
        return Path(str(self.raw["ultralytics_repo_root"]))

    @property
    def yolo_model_path(self) -> Path:
        raw_value = str(self.raw.get("yolo_model_path", "") or "")
        return Path(raw_value) if raw_value else Path("/__missing_yolo_model__")

    @property
    def yolo_tracker_config(self) -> str:
        return str(self.raw["yolo_tracker_config"])

    @property
    def yolo_device(self) -> str:
        return str(self.raw.get("yolo_device", ""))

    @property
    def yolo_confidence(self) -> float:
        return float(self.raw["yolo_confidence"])

    @property
    def yolo_iou(self) -> float:
        return float(self.raw["yolo_iou"])

    @property
    def yolo_max_det(self) -> int:
        return int(self.raw["yolo_max_det"])

    @property
    def yolo_imgsz(self) -> int:
        return int(self.raw["yolo_imgsz"])

    @property
    def manopth_root(self) -> Path:
        return Path(str(self.raw["manopth_root"]))

    @property
    def mano_models_root(self) -> Path:
        return Path(str(self.raw["mano_models_root"]))

    @property
    def clip_context_frames(self) -> int:
        return int(self.raw["clip_context_frames"])

    @property
    def clip_merge_gap(self) -> int:
        return int(self.raw["clip_merge_gap"])

    @property
    def max_clip_frames(self) -> int:
        return int(self.raw["max_clip_frames"])

    @property
    def overlap_frames(self) -> int:
        return int(self.raw["overlap_frames"])

    @property
    def preview_width(self) -> int:
        return int(self.raw["preview_width"])

    @property
    def frame_stride_for_cards(self) -> int:
        return int(self.raw["frame_stride_for_cards"])

    @property
    def keyframe_motion_center_ratio(self) -> float:
        return float(self.raw["keyframe_motion_center_ratio"])

    @property
    def keyframe_motion_area_ratio_high(self) -> float:
        return float(self.raw["keyframe_motion_area_ratio_high"])

    @property
    def keyframe_motion_area_ratio_low(self) -> float:
        return float(self.raw["keyframe_motion_area_ratio_low"])

    @property
    def keyframe_motion_rotation_deg(self) -> float:
        return float(self.raw["keyframe_motion_rotation_deg"])

    @property
    def keyframe_extra_random_min_frames(self) -> int:
        return int(self.raw["keyframe_extra_random_min_frames"])

    @property
    def recovery_anchor_stride_frames(self) -> int:
        return int(self.raw["recovery_anchor_stride_frames"])

    @property
    def min_export_episode_frames(self) -> int:
        return int(self.raw["min_export_episode_frames"])

    @property
    def detector_max_hands(self) -> int:
        return int(self.raw["detector_max_hands"])

    @property
    def min_detection_confidence(self) -> float:
        return float(self.raw["min_detection_confidence"])

    @property
    def min_presence_confidence(self) -> float:
        return float(self.raw["min_presence_confidence"])

    @property
    def min_tracking_confidence(self) -> float:
        return float(self.raw["min_tracking_confidence"])

    @property
    def track_max_gap(self) -> int:
        return int(self.raw["track_max_gap"])

    @property
    def min_chain_frames(self) -> int:
        return int(self.raw["min_chain_frames"])

    @property
    def top_k_chains(self) -> int:
        return int(self.raw["top_k_chains"])

    @property
    def max_review_tracks(self) -> int:
        return int(self.raw["max_review_tracks"])

    @property
    def fit_device(self) -> str:
        return str(self.raw["fit_device"])

    @property
    def fit_stage1_steps(self) -> int:
        return int(self.raw["fit_stage1_steps"])

    @property
    def fit_stage2_steps(self) -> int:
        return int(self.raw["fit_stage2_steps"])

    @property
    def fit_stage3_steps(self) -> int:
        return int(self.raw["fit_stage3_steps"])

    @property
    def fit_learning_rate(self) -> float:
        return float(self.raw["fit_learning_rate"])

    @property
    def fit_reprojection_weight(self) -> float:
        return float(self.raw["fit_reprojection_weight"])

    @property
    def fit_relative_3d_weight(self) -> float:
        return float(self.raw["fit_relative_3d_weight"])

    @property
    def fit_temporal_weight(self) -> float:
        return float(self.raw["fit_temporal_weight"])

    @property
    def fit_shape_weight(self) -> float:
        return float(self.raw["fit_shape_weight"])

    @property
    def fit_pose_prior_weight(self) -> float:
        return float(self.raw["fit_pose_prior_weight"])

    @property
    def mano_use_pca(self) -> bool:
        return bool(self.raw["mano_use_pca"])

    @property
    def mano_ncomps(self) -> int:
        return int(self.raw["mano_ncomps"])

    @property
    def mano_center_idx(self) -> int:
        return int(self.raw["mano_center_idx"])

    @property
    def mano_flat_hand_mean(self) -> bool:
        return bool(self.raw["mano_flat_hand_mean"])

    @property
    def max_median_reproj_error_px(self) -> float:
        return float(self.raw["max_median_reproj_error_px"])

    @property
    def max_p95_reproj_error_px(self) -> float:
        return float(self.raw["max_p95_reproj_error_px"])

    def ensure_runtime_dirs(self) -> None:
        for path in (self.artifacts_dir, self.bundles_dir, self.exports_dir, self.cache_dir, self.db_path.parent):
            path.mkdir(parents=True, exist_ok=True)

    def artifact_abspath(self, relpath: str) -> Path:
        return self.artifacts_dir / relpath


def load_config(config_path: str | None = None) -> MediaPipeReviewConfig:
    repo_root = Path(__file__).resolve().parents[1]
    resolved_path = Path(config_path or os.environ.get("ZARR_VIEWER_CONFIG") or (repo_root / "config.yaml"))
    with resolved_path.open("r", encoding="utf-8") as f:
        root_cfg = yaml.safe_load(f) or {}

    mp_cfg = dict(root_cfg.get("mediapipe_review", {}))
    defaults = {
        "replace_home": True,
        "db_path": str(repo_root / "mediapipe_review.db"),
        "artifacts_dir": str(repo_root / "mediapipe_review_artifacts"),
        "source_annotations_db": str(repo_root / "annotations.db"),
        "source_config_path": str(resolved_path),
        "factory_base": root_cfg.get("factory_base", ""),
        "factory_start": int(root_cfg.get("factory_start", 1)),
        "factory_end": int(root_cfg.get("factory_end", 0)),
        "prefer_installed_mediapipe": True,
        "prefer_installed_ultralytics": True,
        "hand_landmarker_task": _default_hand_landmarker_task(),
        "mediapipe_repo_root": _default_mediapipe_repo_root(),
        "ultralytics_repo_root": _default_ultralytics_repo_root(),
        "yolo_model_path": _default_yolo_model_path(),
        "yolo_tracker_config": "bytetrack.yaml",
        "yolo_device": "",
        "yolo_confidence": 0.05,
        "yolo_iou": 0.60,
        "yolo_max_det": 6,
        "yolo_imgsz": 1280,
        "manopth_root": _default_manopth_root(),
        "mano_models_root": _default_mano_models_root(),
        "clip_context_frames": 60,
        "clip_merge_gap": 15,
        "max_clip_frames": 240,
        "overlap_frames": 30,
        "preview_width": 640,
        "frame_stride_for_cards": 3,
        "keyframe_motion_center_ratio": 0.18,
        "keyframe_motion_area_ratio_high": 1.8,
        "keyframe_motion_area_ratio_low": 0.55,
        "keyframe_motion_rotation_deg": 50.0,
        "keyframe_extra_random_min_frames": 72,
        "recovery_anchor_stride_frames": 10,
        "min_export_episode_frames": 30,
        "detector_max_hands": 6,
        "min_detection_confidence": 0.10,
        "min_presence_confidence": 0.10,
        "min_tracking_confidence": 0.10,
        "track_max_gap": 2,
        "min_chain_frames": 2,
        "top_k_chains": 4,
        "max_review_tracks": 12,
        "fit_device": "cpu",
        "fit_stage1_steps": 80,
        "fit_stage2_steps": 150,
        "fit_stage3_steps": 200,
        "fit_learning_rate": 0.03,
        "fit_reprojection_weight": 5.0,
        "fit_relative_3d_weight": 2.0,
        "fit_temporal_weight": 0.05,
        "fit_shape_weight": 0.001,
        "fit_pose_prior_weight": 0.001,
        "mano_use_pca": True,
        "mano_ncomps": 15,
        "mano_center_idx": 0,
        "mano_flat_hand_mean": True,
        "max_median_reproj_error_px": 18.0,
        "max_p95_reproj_error_px": 40.0,
    }
    raw = _deep_update(defaults, mp_cfg)
    cfg = MediaPipeReviewConfig(raw=raw, repo_root=repo_root)
    cfg.ensure_runtime_dirs()
    return cfg
