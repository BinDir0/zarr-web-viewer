from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


FORMAT_VERSION = "mediapipe_box_wds_v1"


@dataclass(frozen=True)
class PathMap:
    old: str
    new: str


def _json_dumps(payload: Dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_export_items(manifest_path: Path) -> List[Dict[str, Any]]:
    if manifest_path.suffix == ".json":
        payload = _load_json(manifest_path)
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            return [dict(item) for item in payload["items"]]
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        raise ValueError(f"{manifest_path} is JSON, but has no top-level items list")

    items: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_no} is not valid JSONL") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{manifest_path}:{line_no} is not a JSON object")
            items.append(item)
    return items


def _parse_path_maps(raw_values: Sequence[str]) -> List[PathMap]:
    maps: List[PathMap] = []
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"--path-map must be OLD=NEW, got: {raw}")
        old, new = raw.split("=", 1)
        if not old:
            raise ValueError(f"--path-map OLD cannot be empty: {raw}")
        maps.append(PathMap(old=old.rstrip("/"), new=new.rstrip("/")))
    return maps


def _resolve_artifact_path(raw_path: str, manifest_path: Path, path_maps: Sequence[PathMap]) -> Path:
    mapped = str(raw_path)
    for path_map in path_maps:
        if mapped == path_map.old or mapped.startswith(path_map.old + "/"):
            mapped = path_map.new + mapped[len(path_map.old) :]
            break
    path = Path(mapped)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_schema(path: Path) -> Dict[str, Dict[str, Any]]:
    schema: Dict[str, Dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            array = data[key]
            schema[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    return schema


def _side_sparse(data: np.lib.npyio.NpzFile, side: str, frame_start: int, frame_end: int) -> Dict[str, np.ndarray]:
    prefix = f"{side}_"
    frame_indices = np.asarray(data[prefix + "frame_indices"], dtype=np.int32)
    keep = (frame_indices >= int(frame_start)) & (frame_indices <= int(frame_end))
    return {
        f"{prefix}frame_indices": frame_indices[keep].astype(np.int32, copy=False),
        f"{prefix}bbox_xyxy": np.asarray(data[prefix + "bbox_xyxy"], dtype=np.float32)[keep],
        f"{prefix}bbox_xyxy_orig": np.asarray(data[prefix + "bbox_xyxy_orig"], dtype=np.float32)[keep],
        f"{prefix}score": np.asarray(data[prefix + "score"], dtype=np.float32)[keep],
    }


def _make_boxes_npz(fit_npz_path: Path, frame_start: int, frame_end: int) -> bytes:
    all_frames = np.arange(int(frame_start), int(frame_end) + 1, dtype=np.int32)
    arrays: Dict[str, np.ndarray] = {"frame_indices": all_frames}
    with np.load(fit_npz_path, allow_pickle=False) as data:
        for side in ("left", "right"):
            sparse = _side_sparse(data, side, frame_start, frame_end)
            arrays.update(sparse)

            valid = np.zeros((len(all_frames),), dtype=np.bool_)
            dense_box = np.full((len(all_frames), 4), np.nan, dtype=np.float32)
            dense_box_orig = np.full((len(all_frames), 4), np.nan, dtype=np.float32)
            dense_score = np.full((len(all_frames),), np.nan, dtype=np.float32)
            if len(all_frames):
                rel_indices = sparse[f"{side}_frame_indices"] - int(frame_start)
                valid[rel_indices] = True
                dense_box[rel_indices] = sparse[f"{side}_bbox_xyxy"]
                dense_box_orig[rel_indices] = sparse[f"{side}_bbox_xyxy_orig"]
                dense_score[rel_indices] = sparse[f"{side}_score"]
            arrays[f"{side}_valid"] = valid
            arrays[f"{side}_bbox_xyxy_dense"] = dense_box
            arrays[f"{side}_bbox_xyxy_orig_dense"] = dense_box_orig
            arrays[f"{side}_score_dense"] = dense_score

    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _sample_key(item: Dict[str, Any]) -> str:
    clip_id = int(item["clip_id"])
    subepisode_index = int(item.get("subepisode_index", 0))
    return f"clip_{clip_id:06d}_sub_{subepisode_index:06d}"


def _tar_add_bytes(tar: tarfile.TarFile, arcname: str, payload: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(payload))


def _tar_add_file(tar: tarfile.TarFile, arcname: str, path: Path) -> None:
    info = tar.gettarinfo(str(path), arcname)
    info.mtime = 0
    info.mode = 0o644
    with path.open("rb") as f:
        tar.addfile(info, f)


def _artifact_paths(item: Dict[str, Any], manifest_path: Path, path_maps: Sequence[PathMap]) -> Dict[str, Path]:
    required_keys = ("bundle_json", "proposals_npz", "fit_json", "fit_npz")
    missing_keys = [key for key in required_keys if not item.get(key)]
    if missing_keys:
        raise ValueError(f"manifest item clip_id={item.get('clip_id')} is missing paths: {missing_keys}")
    return {
        key: _resolve_artifact_path(str(item[key]), manifest_path=manifest_path, path_maps=path_maps)
        for key in required_keys
    }


def _build_sample_meta(
    item: Dict[str, Any],
    key: str,
    artifact_paths: Dict[str, Path],
    source_manifest: Path,
) -> Dict[str, Any]:
    frame_start = int(item["frame_start"])
    frame_end = int(item["frame_end"])
    return {
        "format_version": FORMAT_VERSION,
        "__key__": key,
        "source_manifest": str(source_manifest),
        "clip_id": int(item["clip_id"]),
        "parent_clip_id": int(item.get("parent_clip_id", item["clip_id"])),
        "review_unit": str(item.get("review_unit", "episode")),
        "subepisode_index": int(item.get("subepisode_index", 0)),
        "episode_id": str(item.get("episode_id", "")),
        "episode_name": str(item.get("episode_name", "")),
        "dataset_name": str(item.get("dataset_name", "")),
        "clip_start": int(item["clip_start"]),
        "clip_end": int(item["clip_end"]),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "num_frames": int(item.get("num_frames", frame_end - frame_start + 1)),
        "status": str(item.get("status", "")),
        "dirty_reason": str(item.get("dirty_reason", "")),
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_sha256": {name: _sha256_file(path) for name, path in artifact_paths.items()},
        "fit_npz_schema": _npz_schema(artifact_paths["fit_npz"]),
        "boxes_npz_schema": {
            "frame_indices": {"shape": [frame_end - frame_start + 1], "dtype": "int32"},
            "left_frame_indices": {"shape": ["N_left"], "dtype": "int32"},
            "left_bbox_xyxy": {"shape": ["N_left", 4], "dtype": "float32"},
            "left_bbox_xyxy_orig": {"shape": ["N_left", 4], "dtype": "float32"},
            "left_score": {"shape": ["N_left"], "dtype": "float32"},
            "left_valid": {"shape": [frame_end - frame_start + 1], "dtype": "bool"},
            "left_bbox_xyxy_dense": {"shape": [frame_end - frame_start + 1, 4], "dtype": "float32"},
            "left_bbox_xyxy_orig_dense": {"shape": [frame_end - frame_start + 1, 4], "dtype": "float32"},
            "left_score_dense": {"shape": [frame_end - frame_start + 1], "dtype": "float32"},
            "right_frame_indices": {"shape": ["N_right"], "dtype": "int32"},
            "right_bbox_xyxy": {"shape": ["N_right", 4], "dtype": "float32"},
            "right_bbox_xyxy_orig": {"shape": ["N_right", 4], "dtype": "float32"},
            "right_score": {"shape": ["N_right"], "dtype": "float32"},
            "right_valid": {"shape": [frame_end - frame_start + 1], "dtype": "bool"},
            "right_bbox_xyxy_dense": {"shape": [frame_end - frame_start + 1, 4], "dtype": "float32"},
            "right_bbox_xyxy_orig_dense": {"shape": [frame_end - frame_start + 1, 4], "dtype": "float32"},
            "right_score_dense": {"shape": [frame_end - frame_start + 1], "dtype": "float32"},
        },
    }


def _iter_manifest_items(manifest_paths: Sequence[Path]) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for manifest_path in manifest_paths:
        for item in _load_export_items(manifest_path):
            yield manifest_path, item


def build_webdataset(
    manifest_paths: Sequence[Path],
    output_dir: Path,
    name_prefix: str,
    shard_size: int,
    path_maps: Sequence[PathMap],
    skip_missing: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.jsonl"

    items = list(_iter_manifest_items(manifest_paths))
    written = 0
    skipped: List[Dict[str, Any]] = []
    shard_paths: List[Path] = []
    tar: tarfile.TarFile | None = None

    try:
        with index_path.open("w", encoding="utf-8") as index_file:
            for item_index, (manifest_path, item) in enumerate(items):
                key = _sample_key(item)
                try:
                    artifacts = _artifact_paths(item, manifest_path, path_maps)
                    missing = [str(path) for path in artifacts.values() if not path.exists()]
                    if missing:
                        raise FileNotFoundError(f"missing artifacts for {key}: {missing}")

                    shard_index = written // int(shard_size)
                    if tar is None or written % int(shard_size) == 0:
                        if tar is not None:
                            tar.close()
                        shard_path = shards_dir / f"{name_prefix}-{shard_index:06d}.tar"
                        shard_paths.append(shard_path)
                        tar = tarfile.open(shard_path, "w", format=tarfile.PAX_FORMAT)

                    assert tar is not None
                    meta = _build_sample_meta(item, key, artifacts, manifest_path)
                    boxes_npz = _make_boxes_npz(
                        artifacts["fit_npz"],
                        frame_start=int(meta["frame_start"]),
                        frame_end=int(meta["frame_end"]),
                    )

                    _tar_add_bytes(tar, f"{key}.json", _json_dumps(meta))
                    _tar_add_bytes(tar, f"{key}.boxes.npz", boxes_npz)
                    _tar_add_file(tar, f"{key}.fit.npz", artifacts["fit_npz"])
                    _tar_add_file(tar, f"{key}.fit.json", artifacts["fit_json"])
                    _tar_add_file(tar, f"{key}.bundle.json", artifacts["bundle_json"])
                    _tar_add_file(tar, f"{key}.proposals.npz", artifacts["proposals_npz"])

                    index_record = {
                        "key": key,
                        "shard": str(shard_paths[-1]),
                        "source_manifest": str(manifest_path),
                        "clip_id": int(item["clip_id"]),
                        "subepisode_index": int(item.get("subepisode_index", 0)),
                        "frame_start": int(meta["frame_start"]),
                        "frame_end": int(meta["frame_end"]),
                    }
                    index_file.write(json.dumps(index_record, ensure_ascii=False, sort_keys=True) + "\n")
                    written += 1
                except Exception as exc:
                    if not skip_missing:
                        raise
                    skipped.append(
                        {
                            "item_index": item_index,
                            "clip_id": item.get("clip_id"),
                            "subepisode_index": item.get("subepisode_index", 0),
                            "reason": str(exc),
                        }
                    )
    finally:
        if tar is not None:
            tar.close()

    dataset_info = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "manifest_paths": [str(path) for path in manifest_paths],
        "output_dir": str(output_dir),
        "shards_dir": str(shards_dir),
        "index_path": str(index_path),
        "name_prefix": name_prefix,
        "shard_size": int(shard_size),
        "num_input_items": len(items),
        "num_samples": written,
        "num_skipped": len(skipped),
        "skipped": skipped[:100],
        "shards": [str(path) for path in shard_paths],
        "sample_members": [
            "<key>.json",
            "<key>.boxes.npz",
            "<key>.fit.npz",
            "<key>.fit.json",
            "<key>.bundle.json",
            "<key>.proposals.npz",
        ],
    }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return dataset_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WebDataset shards from mediapipe review export manifest.")
    parser.add_argument("--manifest", action="append", required=True, help="Export manifest.jsonl or summary.json. Repeatable.")
    parser.add_argument("--output-dir", required=True, help="Output directory for shards, index.jsonl, dataset_info.json.")
    parser.add_argument("--name-prefix", default="mediapipe_boxes", help="Tar shard name prefix.")
    parser.add_argument("--shard-size", type=int, default=1000, help="Samples per tar shard.")
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Rewrite artifact path prefixes, e.g. /old/root=/new/root. Repeatable.",
    )
    parser.add_argument("--skip-missing", action="store_true", help="Skip items with missing artifacts instead of failing.")
    args = parser.parse_args()

    if args.shard_size <= 0:
        raise ValueError("--shard-size must be > 0")

    info = build_webdataset(
        manifest_paths=[Path(path) for path in args.manifest],
        output_dir=Path(args.output_dir),
        name_prefix=str(args.name_prefix),
        shard_size=int(args.shard_size),
        path_maps=_parse_path_maps(args.path_map),
        skip_missing=bool(args.skip_missing),
    )
    print(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
