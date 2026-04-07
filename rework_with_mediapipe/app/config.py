from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: Dict[str, Any]

    @property
    def server(self) -> Dict[str, Any]:
        return dict(self.raw.get("server", {}))

    @property
    def paths(self) -> Dict[str, Any]:
        return dict(self.raw.get("paths", {}))

    @property
    def review(self) -> Dict[str, Any]:
        return dict(self.raw.get("review", {}))

    @property
    def pipeline(self) -> Dict[str, Any]:
        return dict(self.raw.get("pipeline", {}))

    @property
    def fit(self) -> Dict[str, Any]:
        return dict(self.raw.get("fit", {}))

    @property
    def export(self) -> Dict[str, Any]:
        return dict(self.raw.get("export", {}))

    @property
    def app_root(self) -> Path:
        return Path(self.paths["app_root"])

    @property
    def app_db(self) -> Path:
        return Path(self.paths["app_db"])

    @property
    def bundles_dir(self) -> Path:
        return Path(self.paths["bundles_dir"])

    @property
    def exports_dir(self) -> Path:
        return Path(self.paths["exports_dir"])

    @property
    def cache_dir(self) -> Path:
        return Path(self.paths["cache_dir"])

    @property
    def zarr_viewer_root(self) -> Path:
        return Path(self.paths["zarr_viewer_root"])

    @property
    def zarr_viewer_db(self) -> Path:
        return Path(self.paths["zarr_viewer_db"])

    @property
    def zarr_viewer_config(self) -> Path:
        return Path(self.paths["zarr_viewer_config"])

    @property
    def mediapipe_repo_root(self) -> Path:
        return Path(self.paths["mediapipe_repo_root"])

    @property
    def manopth_root(self) -> Path:
        return Path(self.paths["manopth_root"])

    @property
    def mano_models_root(self) -> Path:
        return Path(self.paths["mano_models_root"])

    def ensure_runtime_dirs(self) -> None:
        for path in (self.app_root, self.bundles_dir, self.exports_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(explicit_path: str | None = None) -> AppConfig:
    config_path = (
        explicit_path
        or os.environ.get("RWM_CONFIG")
        or str(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    )
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config_dir = Path(config_path).resolve().parent
    app_root = config_dir.parent
    defaults = {
        "paths": {
            "app_root": str(app_root),
            "app_db": str(app_root / "review.db"),
            "bundles_dir": str(app_root / "bundles"),
            "exports_dir": str(app_root / "exports"),
            "cache_dir": str(app_root / "cache"),
            "zarr_viewer_root": str(app_root.parent),
            "zarr_viewer_db": str(app_root.parent / "annotations.db"),
            "zarr_viewer_config": str(app_root.parent / "config.yaml"),
            "mediapipe_repo_root": "/root/mediapipe",
            "manopth_root": "/root/manopth",
            "mano_models_root": "/root/manopth/mano/models",
        }
    }
    raw = _deep_update(defaults, raw)
    local_override = Path(config_path).with_name("local.yaml")
    if local_override.exists():
        with open(local_override, "r", encoding="utf-8") as f:
            raw = _deep_update(raw, yaml.safe_load(f) or {})
    cfg = AppConfig(raw=raw)
    cfg.ensure_runtime_dirs()
    return cfg
