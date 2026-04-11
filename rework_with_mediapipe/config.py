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


def _default_mano_models_root() -> str:
    candidates = [
        Path("/root/manopth/mano/model"),
        Path("/root/manopth/mano/models"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


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
    def mediapipe_repo_root(self) -> Path:
        return Path(str(self.raw["mediapipe_repo_root"]))

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
        "hand_landmarker_task": str(Path("/root/mediapipe/hand_landmarker.task")),
        "mediapipe_repo_root": "/root/mediapipe",
        "manopth_root": "/root/manopth",
        "mano_models_root": _default_mano_models_root(),
        "clip_context_frames": 60,
        "clip_merge_gap": 15,
        "max_clip_frames": 240,
        "overlap_frames": 30,
        "preview_width": 640,
        "frame_stride_for_cards": 3,
        "detector_max_hands": 4,
        "min_detection_confidence": 0.35,
        "min_presence_confidence": 0.35,
        "min_tracking_confidence": 0.35,
        "track_max_gap": 2,
        "min_chain_frames": 5,
        "top_k_chains": 4,
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
