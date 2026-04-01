import base64
import glob
import hashlib
import io
import json
import os
import pickle
import re
import sqlite3
import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml
from flask import Flask, g, jsonify, redirect, render_template, request
from PIL import Image, ImageDraw

# 加载配置
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)
    SERVER_PORT = int(config.get("server_port", 9470))
    DEBUG_MODE = bool(config.get("debug_mode", True))
    USE_RELOADER = bool(config.get("use_reloader", False))
    FACTORY_BASE = config["factory_base"]
    FACTORY_START = int(config["factory_start"])
    FACTORY_END = int(config["factory_end"])
    legacy_buildai_root = config.get("legacy_buildai_dataset_root", config.get("buildai_dataset_root"))
    LEGACY_BUILDAI_ROOT = Path(legacy_buildai_root) if legacy_buildai_root else None
    LEGACY_BUILDAI_DATASET_NAME = config.get("legacy_buildai_dataset_name", "BuildAI-10k")
    legacy_buildai_annotations_db = config.get("legacy_buildai_annotations_db")
    LEGACY_BUILDAI_ANNOTATIONS_DB = Path(legacy_buildai_annotations_db) if legacy_buildai_annotations_db else None
    source_factory_annotations_db = config.get("source_factory_annotations_db")
    SOURCE_FACTORY_ANNOTATIONS_DB = Path(source_factory_annotations_db) if source_factory_annotations_db else None
    sanity_results_globs = config.get("sanity_results_globs", [])
    if isinstance(sanity_results_globs, str):
        sanity_results_globs = [sanity_results_globs]
    SANITY_RESULTS_GLOBS = [str(x) for x in sanity_results_globs if str(x).strip()]
    SANITY_RESULTS_MAX_FRAMES_PER_EPISODE = int(config.get("sanity_results_max_frames_per_episode", 8))
    SANITY_RESULTS_PROGRESS_INTERVAL_SEC = float(config.get("sanity_results_progress_interval_sec", 1.5))
    SANITY_RESULTS_FRAME_MAX_WIDTH = int(config.get("sanity_results_frame_max_width", 360))
    SANITY_RESULTS_SAMPLE_EVERY_N = max(1, int(config.get("sanity_results_sample_every_n", 30)))
    SANITY_RESULTS_SAMPLE_SEED = str(config.get("sanity_results_sample_seed", "sanity-results-fixed-v1"))
    sanity_results_cache_dir = config.get("sanity_results_cache_dir", "/DATA/guantianrui/zarr-web-viewer-cache")
    SANITY_RESULTS_CACHE_DIR = Path(str(sanity_results_cache_dir))
    FACTORY_EPISODES_CACHE_FILE = Path(
        config.get(
            "factory_episode_cache_file",
            str(Path(__file__).parent / f".factory_episodes_cache_{FACTORY_START}_{FACTORY_END}.json"),
        )
    )
    SQLITE_BUSY_TIMEOUT_MS = int(config.get("sqlite_busy_timeout_ms", 15000))

DB_PATH = Path(__file__).parent / "annotations.db"

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)

# 启用响应压缩
try:
    from flask_compress import Compress
    Compress(app)
    print("✓ Flask-Compress 已启用")
except ImportError:
    pass

# ─── Episode 扫描与缓存 ───────────────────────────────────────────────

_ALL_EPISODES: Optional[List[Dict]] = None
_FACTORY_EPISODE_LOOKUP: Optional[Dict[str, Dict]] = None
_FACTORY_INDEXES: Dict[int, dict] = {}  # fid -> parsed index JSON, loaded on demand
_SANITY_RESULTS_EPISODES: Optional[List[Dict]] = None
_SANITY_RESULTS_INDEX: Optional[Dict[str, Any]] = None
_SANITY_RESULTS_SIGNATURE: Optional[tuple] = None
EPISODE_CACHE_VERSION = 1
SANITY_RESULTS_INDEX_CACHE_VERSION = 2
SANITY_RESULTS_RENDER_CACHE_VERSION = 1
_FACTORY_SCAN_LOCK = Lock()
_SANITY_RESULTS_LOCK = Lock()
_BAD_SOURCE_ROWS_CACHE: Dict[str, Dict] = {}
_BAD_SOURCE_ROWS_LOCK = Lock()
_TRACKS_PATH_CACHE: Dict[str, Optional[str]] = {}
_TRACKS_PATH_LOCK = Lock()
_SANITY_RESULTS_CACHE_ROOT: Optional[Path] = None
_SANITY_RESULTS_CACHE_ROOT_LOCK = Lock()
_RESULTS_TAR_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}
_RESULTS_TAR_INDEX_LOCK = Lock()


def _load_episode_cache(cache_file: Path, cache_key: Dict) -> Optional[List[Dict]]:
    """从磁盘缓存加载 episode 列表。"""
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != EPISODE_CACHE_VERSION:
            return None
        if payload.get("cache_key") != cache_key:
            return None
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            return None
        print(f"✓ 从缓存加载 episodes: {cache_file} ({len(episodes)} 条)")
        return episodes
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"⚠ 读取 episode 缓存失败 ({cache_file}): {e}")
        return None


def _save_episode_cache(cache_file: Path, cache_key: Dict, episodes: List[Dict]) -> None:
    """将 episode 列表写入磁盘缓存。"""
    payload = {
        "version": EPISODE_CACHE_VERSION,
        "cache_key": cache_key,
        "episodes": episodes,
    }
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(payload, f)
        print(f"✓ 已写入 episode 缓存: {cache_file}")
    except OSError as e:
        print(f"⚠ 写入 episode 缓存失败 ({cache_file}): {e}")


def _get_sanity_results_cache_root() -> Path:
    global _SANITY_RESULTS_CACHE_ROOT
    if _SANITY_RESULTS_CACHE_ROOT is not None:
        return _SANITY_RESULTS_CACHE_ROOT

    with _SANITY_RESULTS_CACHE_ROOT_LOCK:
        if _SANITY_RESULTS_CACHE_ROOT is not None:
            return _SANITY_RESULTS_CACHE_ROOT

        candidates = [
            SANITY_RESULTS_CACHE_DIR,
            Path(__file__).parent / ".cache",
            Path("/tmp/zarr-web-viewer-cache"),
        ]
        for candidate in candidates:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                _SANITY_RESULTS_CACHE_ROOT = candidate
                print(f"✓ sanity results 缓存目录: {_SANITY_RESULTS_CACHE_ROOT}")
                return _SANITY_RESULTS_CACHE_ROOT
            except OSError:
                continue

        _SANITY_RESULTS_CACHE_ROOT = Path(__file__).parent
        return _SANITY_RESULTS_CACHE_ROOT


def _get_sanity_results_index_cache_file() -> Path:
    return _get_sanity_results_cache_root() / "sanity_results_index.pkl"


def _get_sanity_results_render_cache_dir() -> Path:
    path = _get_sanity_results_cache_root() / "rendered_results_frames"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_sanity_results_tar_index_cache_dir() -> Path:
    path = _get_sanity_results_cache_root() / "tar_member_offsets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hash_json_payload(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _load_pickle_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError) as e:
        print(f"⚠ 读取缓存失败 ({path}): {e}")
        return None


def _save_pickle_cache(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as e:
        print(f"⚠ 写入缓存失败 ({path}): {e}")


def _expand_result_files(patterns: List[str]) -> List[Path]:
    files = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            p = Path(path)
            if p.is_file():
                files.add(p.resolve())
    return sorted(files)


def _get_paths_signature(paths: List[Path]) -> tuple:
    signature = []
    for path in paths:
        try:
            st = path.stat()
            signature.append((str(path), st.st_mtime_ns, st.st_size))
        except OSError:
            signature.append((str(path), None, None))
    return tuple(signature)


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(0, num_bytes))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _print_sanity_results_progress(
    processed_bytes: int,
    total_bytes: int,
    file_index: int,
    total_files: int,
    total_lines: int,
    total_episodes: int,
    path: Optional[Path] = None,
) -> None:
    pct = 100.0 if total_bytes <= 0 else min(100.0, processed_bytes * 100.0 / total_bytes)
    file_part = f"{file_index}/{total_files}" if total_files > 0 else "0/0"
    path_part = f" | {path.name}" if path is not None else ""
    print(
        "[sanity preload] "
        f"{pct:5.1f}% | files {file_part} | "
        f"bytes {_format_bytes(processed_bytes)}/{_format_bytes(total_bytes)} | "
        f"lines {total_lines:,} | episodes {total_episodes:,}{path_part}",
        flush=True,
    )


def _guess_dataset_name_from_tar_path(tar_path: str) -> str:
    parent = Path(tar_path).parent.name
    if parent.startswith("factory"):
        return parent
    match = re.search(r"/(factory\d{3})/", tar_path)
    if match:
        return match.group(1)
    return "results_jsonl"


def _pick_evenly_spaced_result_frames(entries: List[Dict], max_frames: int) -> List[Dict]:
    if max_frames <= 0 or len(entries) <= max_frames:
        return list(entries)
    if max_frames == 1:
        return [entries[len(entries) // 2]]
    out: List[Dict] = []
    last_idx = len(entries) - 1
    for i in range(max_frames):
        idx = round(i * last_idx / (max_frames - 1))
        out.append(entries[idx])
    return out


def _result_item_is_bad(item: Dict) -> bool:
    try:
        if int(item.get("is_bad", 0)) != 0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        return int(item.get("label", 0)) == 1
    except (TypeError, ValueError):
        return False


def _normalized_boxes_from_result_item(item: Dict) -> List[Dict]:
    detections = item.get("detections")
    if isinstance(detections, list) and detections:
        out = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            box = det.get("box_cxcywh")
            if not isinstance(box, list) or len(box) != 4:
                continue
            try:
                cx, cy, w, h = [float(x) for x in box]
                score = float(det.get("score", 0.0))
                side_label = int(det.get("side_label", 0))
            except (TypeError, ValueError):
                continue
            out.append({"cx": cx, "cy": cy, "w": w, "h": h, "score": score, "side_label": side_label})
        if out:
            return out

    boxes = item.get("boxes")
    scores = item.get("scores")
    side_labels = item.get("side_labels")
    out = []
    if isinstance(boxes, list):
        for idx, box in enumerate(boxes):
            if not isinstance(box, list) or len(box) != 4:
                continue
            try:
                cx, cy, w, h = [float(x) for x in box]
                score = float(scores[idx]) if isinstance(scores, list) and idx < len(scores) else 0.0
                side_label = int(side_labels[idx]) if isinstance(side_labels, list) and idx < len(side_labels) else 0
            except (TypeError, ValueError):
                continue
            out.append({"cx": cx, "cy": cy, "w": w, "h": h, "score": score, "side_label": side_label})
    return out


def _should_keep_result_frame(frame_id: str) -> bool:
    if SANITY_RESULTS_SAMPLE_EVERY_N <= 1:
        return True
    digest = hashlib.sha1(f"{SANITY_RESULTS_SAMPLE_SEED}:{frame_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SANITY_RESULTS_SAMPLE_EVERY_N == 0


def scan_sanity_results_index(force_rescan: bool = False) -> Dict[str, Any]:
    global _SANITY_RESULTS_EPISODES, _SANITY_RESULTS_INDEX, _SANITY_RESULTS_SIGNATURE
    empty = {"frames": [], "episodes": [], "episodes_by_id": {}}
    if not SANITY_RESULTS_GLOBS:
        return empty

    files = _expand_result_files(SANITY_RESULTS_GLOBS)
    signature = _get_paths_signature(files)
    if _SANITY_RESULTS_INDEX is not None and _SANITY_RESULTS_SIGNATURE == signature and not force_rescan:
        return _SANITY_RESULTS_INDEX

    with _SANITY_RESULTS_LOCK:
        if _SANITY_RESULTS_INDEX is not None and _SANITY_RESULTS_SIGNATURE == signature and not force_rescan:
            return _SANITY_RESULTS_INDEX

        if not files:
            _SANITY_RESULTS_EPISODES = []
            _SANITY_RESULTS_INDEX = empty
            _SANITY_RESULTS_SIGNATURE = signature
            print("⚠ 未匹配到 sanity results.jsonl 文件，回退旧 sanity 数据源")
            return empty

        cache_file = _get_sanity_results_index_cache_file()
        if not force_rescan:
            cached = _load_pickle_cache(cache_file)
            if isinstance(cached, dict):
                if (
                    cached.get("version") == SANITY_RESULTS_INDEX_CACHE_VERSION
                    and cached.get("signature") == signature
                    and isinstance(cached.get("index"), dict)
                ):
                    _SANITY_RESULTS_INDEX = cached["index"]
                    _SANITY_RESULTS_EPISODES = _SANITY_RESULTS_INDEX.get("episodes", [])
                    _SANITY_RESULTS_SIGNATURE = signature
                    print(
                        "✓ 从磁盘缓存加载 sanity results 索引: "
                        f"{len(_SANITY_RESULTS_INDEX.get('frames', []))} sampled 帧, "
                        f"{len(_SANITY_RESULTS_EPISODES)} 个 episodes"
                    )
                    return _SANITY_RESULTS_INDEX

        episodes_by_video: Dict[str, Dict[str, Any]] = {}
        frames: List[Dict[str, Any]] = []
        total_candidate_frames = 0
        total_lines = 0
        total_bytes = sum(path.stat().st_size for path in files if path.exists())
        processed_bytes = 0
        last_progress_time = time.time()
        print(
            f"开始预热 sanity results.jsonl: {len(files)} 个文件, 总大小 {_format_bytes(total_bytes)}",
            flush=True,
        )
        for file_idx, path in enumerate(files, start=1):
            try:
                with path.open("rb") as f:
                    for raw_line in f:
                        processed_bytes += len(raw_line)
                        total_lines += 1
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if time.time() - last_progress_time >= SANITY_RESULTS_PROGRESS_INTERVAL_SEC:
                            _print_sanity_results_progress(
                                processed_bytes=processed_bytes,
                                total_bytes=total_bytes,
                                file_index=file_idx,
                                total_files=len(files),
                                total_lines=total_lines,
                                total_episodes=len(episodes_by_video),
                                path=path,
                            )
                            last_progress_time = time.time()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        video_key = str(item.get("video_key", "")).strip()
                        member_name = str(item.get("member_name", "")).strip()
                        tar_path = str(item.get("tar_path", "")).strip()
                        if not video_key or not member_name or not tar_path:
                            continue

                        try:
                            frame_index = int(item.get("frame_index", 0))
                        except (TypeError, ValueError):
                            frame_index = 0

                        frame_id = f"{video_key}\u0001{frame_index}"
                        total_candidate_frames += 1
                        if not _should_keep_result_frame(frame_id):
                            continue

                        dataset_name = _guess_dataset_name_from_tar_path(tar_path)
                        ep = episodes_by_video.setdefault(
                            video_key,
                            {
                                "episode_id": video_key,
                                "episode_name": video_key,
                                "dataset_name": dataset_name,
                                "num_frames": 0,
                                "sampled_result_frame_count": 0,
                                "sampled_result_frame_indices": [],
                            },
                        )
                        ep["num_frames"] = max(ep["num_frames"], frame_index + 1)
                        ep["sampled_result_frame_count"] += 1
                        ep["sampled_result_frame_indices"].append(frame_index)

                        frames.append(
                            {
                                "frame_id": frame_id,
                                "episode_id": video_key,
                                "episode_name": video_key,
                                "dataset_name": dataset_name,
                                "frame_index": frame_index,
                                "tar_path": tar_path,
                                "member_name": member_name,
                                "normalized_boxes": _normalized_boxes_from_result_item(item),
                                "prob_is_bad": item.get("prob_is_bad"),
                                "is_bad": _result_item_is_bad(item),
                                "queue_index": len(frames),
                            }
                        )
                _print_sanity_results_progress(
                    processed_bytes=processed_bytes,
                    total_bytes=total_bytes,
                    file_index=file_idx,
                    total_files=len(files),
                    total_lines=total_lines,
                    total_episodes=len(episodes_by_video),
                    path=path,
                )
            except OSError as e:
                print(f"⚠ 读取 results.jsonl 失败 ({path}): {e}")

        episodes: List[Dict[str, Any]] = []
        episodes_by_id: Dict[str, Dict[str, Any]] = {}
        for idx, raw_ep in enumerate(sorted(episodes_by_video.values(), key=lambda x: x["episode_id"])):
            ep = dict(raw_ep)
            ep["sampled_result_frame_indices"] = sorted(set(ep["sampled_result_frame_indices"]))
            ep["episode_index"] = idx
            episodes.append(ep)
            episodes_by_id[ep["episode_id"]] = ep

        for queue_index, frame in enumerate(frames):
            ep = episodes_by_id.get(frame["episode_id"])
            if ep is None:
                continue
            frame["queue_index"] = queue_index
            frame["episode_index"] = ep["episode_index"]
            frame["episode_frame_count"] = ep["sampled_result_frame_count"]

        index = {
            "frames": frames,
            "episodes": episodes,
            "episodes_by_id": episodes_by_id,
            "total_candidate_frames": total_candidate_frames,
            "sampled_frame_count": len(frames),
            "sample_every_n": SANITY_RESULTS_SAMPLE_EVERY_N,
            "sample_seed": SANITY_RESULTS_SAMPLE_SEED,
        }
        _save_pickle_cache(
            cache_file,
            {
                "version": SANITY_RESULTS_INDEX_CACHE_VERSION,
                "signature": signature,
                "index": index,
            },
        )

        _SANITY_RESULTS_INDEX = index
        _SANITY_RESULTS_EPISODES = episodes
        _SANITY_RESULTS_SIGNATURE = signature
        print(
            f"✓ 加载 sanity results.jsonl: {len(files)} 个文件, "
            f"{len(frames)}/{total_candidate_frames} sampled 帧 (1/{SANITY_RESULTS_SAMPLE_EVERY_N}), "
            f"{len(episodes)} 个 episodes"
        )
        return index


def scan_sanity_result_episodes(force_rescan: bool = False) -> List[Dict]:
    return scan_sanity_results_index(force_rescan=force_rescan).get("episodes", [])


def _count_jpg_files(dir_path: Path) -> int:
    """快速统计目录中的 jpg/jpeg 文件数量。"""
    try:
        with os.scandir(dir_path) as it:
            return sum(
                1
                for entry in it
                if entry.is_file()
                and entry.name.lower().endswith((".jpg", ".jpeg"))
            )
    except OSError:
        return 0


_LEGACY_BUILDAI_EPISODE_RE = re.compile(r"^factory(?P<factory>\d{3})_worker(?P<worker>\d{3})_.+_crop\d+$")


def resolve_legacy_buildai_episode(
    episode_id: str,
    episode_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Optional[Dict]:
    """按 episode_id 直接反推旧 BuildAI raw crop 目录，无需全量扫描。"""
    if LEGACY_BUILDAI_ROOT is None:
        return None

    match = _LEGACY_BUILDAI_EPISODE_RE.match(episode_id)
    if match is None:
        return None

    factory_id = match.group("factory")
    worker_id = match.group("worker")
    crop_dir = LEGACY_BUILDAI_ROOT / f"factory_{factory_id}" / f"worker_{worker_id}" / "processed" / episode_id
    extracted_dir = crop_dir / "extracted_images"
    if not extracted_dir.is_dir():
        return None

    return {
        "source_type": "legacy_buildai_raw",
        "episode_id": episode_id,
        "episode_name": episode_name or episode_id,
        "dataset_name": dataset_name or LEGACY_BUILDAI_DATASET_NAME,
        "crop_dir": str(crop_dir),
        "num_frames": -1,  # 延迟到真正加载该 episode 时再统计
    }


def _get_factory_index(fid: int) -> Optional[dict]:
    """延迟加载并缓存单个 factory 的 _video_index.json。"""
    if fid in _FACTORY_INDEXES:
        return _FACTORY_INDEXES[fid]

    factory_dir = os.path.join(FACTORY_BASE, f"factory{fid:03d}")
    index_path = os.path.join(factory_dir, "_video_index.json")

    if not os.path.exists(index_path):
        _FACTORY_INDEXES[fid] = None
        return None

    try:
        with open(index_path, "r") as f:
            index = json.load(f)
        _FACTORY_INDEXES[fid] = index
        return index
    except (json.JSONDecodeError, OSError):
        _FACTORY_INDEXES[fid] = None
        return None


def _get_episode_frames(ep: Dict):
    """从缓存的 factory index 中提取单个 episode 的帧名和 offset。"""
    fid = ep["_fid"]
    video_key = ep["episode_id"]

    index = _get_factory_index(fid)
    if index is None:
        return None, None

    info = index.get("videos", {}).get(video_key)
    if info is None:
        return None, None

    frames = info.get("frames", [])
    if not frames:
        return [], None

    if isinstance(frames[0], dict):
        frame_names = [f["name"] for f in frames]
        frame_offsets = [[f["offset"], f["size"]] for f in frames]
    else:
        frame_names = frames
        frame_offsets = None

    return frame_names, frame_offsets


def scan_factory_episodes(force_rescan: bool = False) -> List[Dict]:
    """扫描 factory range 中的所有视频，返回轻量 episode 列表。

    只存元数据（video_key, shard, num_frames 等），不存帧名/offset。
    帧数据按需通过 _get_episode_frames() 延迟加载。
    """
    global _ALL_EPISODES, _FACTORY_EPISODE_LOOKUP
    if _ALL_EPISODES is not None and not force_rescan:
        return _ALL_EPISODES
    with _FACTORY_SCAN_LOCK:
        if _ALL_EPISODES is not None and not force_rescan:
            return _ALL_EPISODES
        _FACTORY_EPISODE_LOOKUP = None

        cache_key = {
            "type": "factory",
            "factory_base": FACTORY_BASE,
            "factory_start": FACTORY_START,
            "factory_end": FACTORY_END,
        }
        if not force_rescan:
            cached = _load_episode_cache(FACTORY_EPISODES_CACHE_FILE, cache_key)
            if cached is not None:
                _ALL_EPISODES = cached
                return cached

        episodes: List[Dict] = []

        for fid in range(FACTORY_START, FACTORY_END + 1):
            index = _get_factory_index(fid)
            if index is None:
                continue

            factory_dir = os.path.join(FACTORY_BASE, f"factory{fid:03d}")
            videos = index.get("videos", {})

            for video_key, info in sorted(videos.items()):
                frames = info.get("frames", [])
                if not frames:
                    continue

                episodes.append({
                    "episode_id": video_key,
                    "episode_name": info.get("video_name", video_key),
                    "dataset_name": f"factory{fid:03d}",
                    "shard_path": os.path.join(factory_dir, info["shard"]),
                    "seq_folder": os.path.join(factory_dir, "outputs", video_key),
                    "num_frames": len(frames),
                    "_fid": fid,  # for lazy frame loading
                })

        _ALL_EPISODES = episodes
        _save_episode_cache(FACTORY_EPISODES_CACHE_FILE, cache_key, episodes)
        print(f"✓ 扫描完成: factory{FACTORY_START:03d}~factory{FACTORY_END:03d}, 共 {len(episodes)} 个 videos")
        return episodes


def get_factory_episode_lookup() -> Dict[str, Dict]:
    """返回 episode_id -> episode 的惰性索引。"""
    global _FACTORY_EPISODE_LOOKUP
    if _FACTORY_EPISODE_LOOKUP is None:
        _FACTORY_EPISODE_LOOKUP = {e["episode_id"]: e for e in scan_factory_episodes()}
    return _FACTORY_EPISODE_LOOKUP


def hydrate_legacy_buildai_episode(ep: Dict) -> Optional[Dict]:
    """补齐旧 BuildAI raw episode 的实际帧数。"""
    if ep.get("num_frames", -1) >= 0:
        return ep

    crop_path = Path(ep["crop_dir"])
    extracted_dir = crop_path / "extracted_images"
    if not extracted_dir.exists():
        return None

    frame_count = _count_jpg_files(extracted_dir)
    if frame_count <= 0:
        return None

    hydrated = dict(ep)
    hydrated["num_frames"] = frame_count
    return hydrated


# ─── 图像加载与 YOLO Box 渲染 ─────────────────────────────────────────


def find_model_tracks_path(seq_folder: str) -> Optional[Path]:
    """定位 seq/crop 目录下的 model_tracks.npy，并缓存结果。"""
    seq_path = Path(seq_folder)
    cache_key = str(seq_path.resolve())

    with _TRACKS_PATH_LOCK:
        cached = _TRACKS_PATH_CACHE.get(cache_key)
        if cached is not None:
            return Path(cached) if cached else None

    if not seq_path.exists():
        tracks_path = None
    else:
        tracks_dirs = sorted(seq_path.glob("tracks_*"))
        if not tracks_dirs:
            tracks_path = None
        else:
            candidate = tracks_dirs[0] / "model_tracks.npy"
            tracks_path = candidate if candidate.exists() else None

    with _TRACKS_PATH_LOCK:
        _TRACKS_PATH_CACHE[cache_key] = str(tracks_path) if tracks_path is not None else ""

    return tracks_path


def load_track_boxes(seq_folder: str, target_frames: Optional[set] = None) -> Dict[int, list]:
    """加载 model_tracks.npy，返回 per-frame box 查找表。

    Returns:
        {frame_idx: [(x1, y1, x2, y2, conf, voted_handedness), ...]}
    """
    frame_boxes: Dict[int, list] = {}
    tracks_path = find_model_tracks_path(seq_folder)
    if tracks_path is None:
        return frame_boxes

    try:
        tracks_data = np.load(str(tracks_path), allow_pickle=True).item()
        for track_id, detections in tracks_data.items():
            all_h = [det["det_handedness"][0] for det in detections]
            voted_h = 1 if sum(1 for h in all_h if h > 0) > len(all_h) / 2 else 0
            for det in detections:
                f = det["frame"]
                if target_frames is not None and f not in target_frames:
                    continue
                box = det["det_box"][0]
                frame_boxes.setdefault(f, []).append(
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]), voted_h)
                )
    except Exception as e:
        print(f"⚠ 加载 model_tracks 失败 ({tracks_path.parent.parent.name}): {e}")
    return frame_boxes


def pick_frames_with_boxes(num_frames: int, frame_boxes: Dict[int, list], num_picks: int = 2) -> List[int]:
    """在目标位置（1/3、2/3）附近选取有检测框的帧。"""
    if num_frames <= num_picks:
        return list(range(num_frames))

    targets = [num_frames // 3, 2 * num_frames // 3]
    frames_with_boxes = set(frame_boxes.keys())

    if not frames_with_boxes:
        return targets

    picked = []
    for target in targets:
        best = None
        for delta in range(num_frames):
            for candidate in (target + delta, target - delta):
                if 0 <= candidate < num_frames and candidate in frames_with_boxes:
                    best = candidate
                    break
            if best is not None:
                break
        picked.append(best if best is not None else target)

    return picked


def load_episode_frames_from_shard(
    shard_path: str,
    frame_names: List[str],
    frame_offsets: Optional[List[List]],
    frame_indices: List[int],
    frame_boxes: Dict[int, list],
    max_width: int = 640,
) -> List[str]:
    """从 tar shard 加载指定帧并绘制 YOLO 检测框，返回 base64 编码图像列表。"""
    results = []

    # 预读所有需要的帧数据（同一 shard 只开一次文件句柄）
    frame_data = {}
    if frame_offsets is not None:
        try:
            with open(shard_path, "rb") as f:
                for frame_idx in frame_indices:
                    if 0 <= frame_idx < len(frame_names):
                        offset, size = frame_offsets[frame_idx]
                        f.seek(offset)
                        frame_data[frame_idx] = f.read(size)
        except Exception as e:
            print(f"⚠ 读取 shard 失败 ({os.path.basename(shard_path)}): {e}")
            return results
    else:
        # 无 offset 回退：用 tarfile（慢）
        import tarfile
        try:
            with tarfile.open(shard_path, "r") as tar:
                for frame_idx in frame_indices:
                    if 0 <= frame_idx < len(frame_names):
                        member = tar.getmember(frame_names[frame_idx])
                        frame_data[frame_idx] = tar.extractfile(member).read()
        except Exception as e:
            print(f"⚠ 读取 tar 失败 ({os.path.basename(shard_path)}): {e}")
            return results

    for frame_idx in frame_indices:
        if frame_idx not in frame_data:
            continue

        try:
            img = Image.open(io.BytesIO(frame_data[frame_idx])).convert("RGB")

            # 绘制左右手检测框（蓝=左手，红=右手）
            if frame_idx in frame_boxes:
                draw = ImageDraw.Draw(img)
                for x1, y1, x2, y2, conf, handedness in frame_boxes[frame_idx]:
                    if handedness == 0:
                        color, label = "#00BFFF", f"L {conf:.2f}"
                    else:
                        color, label = "#FF4444", f"R {conf:.2f}"
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_OUTLINE_WIDTH)
                    draw.text((x1 + 2, y1 - 16), label, fill=color)

            # 缩放
            w, h = img.size
            if w > max_width:
                new_h = int(h * max_width / w)
                img = img.resize((max_width, new_h), Image.Resampling.BILINEAR)

            # 编码为 WebP base64
            buf = io.BytesIO()
            try:
                img.save(buf, format="WEBP", quality=80, method=4)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                results.append(f"data:image/webp;base64,{b64}")
            except Exception:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=75)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                results.append(f"data:image/jpeg;base64,{b64}")

        except Exception as e:
            print(f"⚠ 解码帧失败 (shard={os.path.basename(shard_path)}, frame={frame_idx}): {e}")

    return results


def _get_results_render_cache_paths(entry: Dict[str, Any], max_width: int) -> Tuple[Path, Path]:
    key = _hash_json_payload(
        {
            "version": SANITY_RESULTS_RENDER_CACHE_VERSION,
            "tar_path": entry.get("tar_path"),
            "member_name": entry.get("member_name"),
            "normalized_boxes": entry.get("normalized_boxes", []),
            "max_width": max_width,
        }
    )
    cache_dir = _get_sanity_results_render_cache_dir()
    return cache_dir / f"{key}.webp", cache_dir / f"{key}.jpg"


def _load_cached_image_data_url(entry: Dict[str, Any], max_width: int) -> Optional[str]:
    webp_path, jpg_path = _get_results_render_cache_paths(entry, max_width)
    for path, mime in ((webp_path, "image/webp"), (jpg_path, "image/jpeg")):
        if not path.exists():
            continue
        try:
            return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"
        except OSError:
            continue
    return None


def _save_cached_rendered_image(entry: Dict[str, Any], max_width: int, encoded_bytes: bytes, use_webp: bool) -> None:
    webp_path, jpg_path = _get_results_render_cache_paths(entry, max_width)
    target_path = webp_path if use_webp else jpg_path
    try:
        target_path.write_bytes(encoded_bytes)
    except OSError:
        pass


def _get_results_tar_index_cache_file(tar_path: str) -> Path:
    return _get_sanity_results_tar_index_cache_dir() / f"{hashlib.sha1(tar_path.encode('utf-8')).hexdigest()}.pkl"


def _build_results_tar_offset_index(tar_path: str) -> Optional[Dict[str, Any]]:
    import tarfile

    try:
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
    except Exception as e:
        print(f"⚠ 构建 tar 索引失败 ({tar_path}): {e}", flush=True)
        return None

    offsets: Dict[str, Tuple[int, int]] = {}
    basename_to_name: Dict[str, str] = {}
    ambiguous_basenames: Set[str] = set()
    names: List[str] = []
    for member in members:
        if not member.isfile():
            continue
        names.append(member.name)
        offsets[member.name] = (int(member.offset_data), int(member.size))
        base = os.path.basename(member.name)
        if base in ambiguous_basenames:
            continue
        if base in basename_to_name:
            basename_to_name.pop(base, None)
            ambiguous_basenames.add(base)
        else:
            basename_to_name[base] = member.name

    return {
        "offsets": offsets,
        "names": names,
        "basename_to_name": basename_to_name,
    }


def _get_results_tar_offset_index(tar_path: str) -> Optional[Dict[str, Any]]:
    resolved = str(Path(tar_path).resolve())
    with _RESULTS_TAR_INDEX_LOCK:
        cached = _RESULTS_TAR_INDEX_CACHE.get(resolved)
        if cached is not None:
            return cached

    cache_file = _get_results_tar_index_cache_file(resolved)
    cached_payload = _load_pickle_cache(cache_file)
    if isinstance(cached_payload, dict) and isinstance(cached_payload.get("index"), dict):
        index = cached_payload["index"]
        with _RESULTS_TAR_INDEX_LOCK:
            _RESULTS_TAR_INDEX_CACHE[resolved] = index
        return index

    index = _build_results_tar_offset_index(resolved)
    if index is None:
        return None
    _save_pickle_cache(cache_file, {"index": index})
    with _RESULTS_TAR_INDEX_LOCK:
        _RESULTS_TAR_INDEX_CACHE[resolved] = index
    return index


def _resolve_results_tar_member(
    tar_index: Dict[str, Any],
    member_name: str,
) -> Optional[Tuple[str, Tuple[int, int]]]:
    offsets = tar_index.get("offsets", {})
    names = tar_index.get("names", [])
    basename_to_name = tar_index.get("basename_to_name", {})

    for candidate in (member_name, f"./{member_name}"):
        offset = offsets.get(candidate)
        if offset is not None:
            return candidate, offset

    base = os.path.basename(member_name)
    canonical = basename_to_name.get(base)
    if canonical:
        offset = offsets.get(canonical)
        if offset is not None:
            return canonical, offset

    for name in names:
        if name.endswith(member_name):
            offset = offsets.get(name)
            if offset is not None:
                return name, offset
    return None


def _read_raw_member_from_tar_file(fp, offset_and_size: Tuple[int, int]) -> bytes:
    offset, size = offset_and_size
    fp.seek(offset)
    return fp.read(size)


def load_episode_frames_from_results(
    frame_entries: List[Dict],
    max_width: int = SANITY_RESULTS_FRAME_MAX_WIDTH,
) -> List[Optional[str]]:
    """从 results.jsonl 提供的 tar_path/member_name 直接加载帧并绘制检测框。"""
    results: List[Optional[str]] = [None] * len(frame_entries)
    misses_by_tar: Dict[str, List[Dict[str, Any]]] = {}

    for idx, entry in enumerate(frame_entries):
        cached = _load_cached_image_data_url(entry, max_width)
        if cached is not None:
            results[idx] = cached
            continue
        misses_by_tar.setdefault(entry["tar_path"], []).append({"idx": idx, "entry": entry})

    raw_frame_data: Dict[int, bytes] = {}
    for tar_path, entries in misses_by_tar.items():
        tar_index = _get_results_tar_offset_index(tar_path)
        if tar_index is None:
            continue
        try:
            with open(tar_path, "rb") as fp:
                for wrapped in entries:
                    entry = wrapped["entry"]
                    resolved = _resolve_results_tar_member(tar_index, entry["member_name"])
                    if resolved is None:
                        print(
                            f"⚠ tar 中未找到成员 ({os.path.basename(tar_path)} :: {entry['member_name']})",
                            flush=True,
                        )
                        continue
                    _, offset_and_size = resolved
                    raw_frame_data[wrapped["idx"]] = _read_raw_member_from_tar_file(fp, offset_and_size)
        except Exception as e:
            print(f"⚠ 读取 results tar 失败 ({tar_path}): {e}", flush=True)

    for idx, entry in enumerate(frame_entries):
        if results[idx] is not None:
            continue
        raw = raw_frame_data.get(idx)
        if raw is None:
            continue
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            draw = ImageDraw.Draw(img)
            w, h = img.size
            for box in entry.get("normalized_boxes", []):
                cx = float(box["cx"]) * w
                cy = float(box["cy"]) * h
                bw = float(box["w"]) * w
                bh = float(box["h"]) * h
                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0
                score = float(box.get("score", 0.0))
                handedness = int(box.get("side_label", 0))
                if handedness == 0:
                    color, label = "#00BFFF", f"L {score:.2f}"
                else:
                    color, label = "#FF4444", f"R {score:.2f}"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_OUTLINE_WIDTH)
                draw.text((x1 + 2, max(0, y1 - 16)), label, fill=color)

            if w > max_width:
                img = img.resize((max_width, int(h * max_width / w)), Image.Resampling.BILINEAR)

            buf = io.BytesIO()
            try:
                img.save(buf, format="WEBP", quality=78, method=4)
                encoded = buf.getvalue()
                _save_cached_rendered_image(entry, max_width, encoded, use_webp=True)
                results[idx] = f"data:image/webp;base64,{base64.b64encode(encoded).decode('utf-8')}"
            except Exception:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=72)
                encoded = buf.getvalue()
                _save_cached_rendered_image(entry, max_width, encoded, use_webp=False)
                results[idx] = f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('utf-8')}"
        except Exception as e:
            print(f"⚠ 解码 results 帧失败 ({entry.get('member_name')}): {e}", flush=True)

    return results


def load_episode_frames_from_raw_buildai(
    crop_dir: str,
    frame_indices: List[int],
    frame_boxes: Optional[Dict[int, list]] = None,
    max_width: int = 640,
) -> List[str]:
    """从旧 BuildAI raw crop 目录读取帧并绘制 box。"""
    crop_path = Path(crop_dir)
    extracted_dir = crop_path / "extracted_images"
    if frame_boxes is None:
        frame_boxes = load_track_boxes(crop_dir, target_frames=set(frame_indices))
    results: List[str] = []
    for frame_idx in frame_indices:
        img_path = extracted_dir / f"{frame_idx:06d}.jpg"
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            if frame_idx in frame_boxes:
                draw = ImageDraw.Draw(img)
                for x1, y1, x2, y2, conf, handedness in frame_boxes[frame_idx]:
                    if handedness == 0:
                        color, label = "#00BFFF", f"L {conf:.2f}"
                    else:
                        color, label = "#FF4444", f"R {conf:.2f}"
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_OUTLINE_WIDTH)
                    draw.text((x1 + 2, y1 - 16), label, fill=color)

            w, h = img.size
            if w > max_width:
                img = img.resize((max_width, int(h * max_width / w)), Image.Resampling.BILINEAR)

            buf = io.BytesIO()
            try:
                img.save(buf, format="WEBP", quality=80, method=4)
                results.append(f"data:image/webp;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}")
            except Exception:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=75)
                results.append(f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}")
        except Exception as e:
            print(f"⚠ 旧 BuildAI 帧加载失败 ({img_path}): {e}")
    return results


# ─── 数据库 ───────────────────────────────────────────────────────────


def get_db_connection() -> sqlite3.Connection:
    conn = getattr(g, "_db_conn", None)
    if conn is None:
        conn = create_sqlite_connection(DB_PATH)
        g._db_conn = conn
    return conn


def create_sqlite_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=max(1.0, SQLITE_BUSY_TIMEOUT_MS / 1000.0))
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        # 对只读/外部数据库，WAL 可能不可用；主流程仍可继续。
        pass
    return conn


def commit_with_retry(conn: sqlite3.Connection, retries: int = 4, base_sleep: float = 0.25) -> None:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            last_error = e
            if "locked" not in str(e).lower() or attempt == retries - 1:
                raise
            time.sleep(base_sleep * (attempt + 1))
    if last_error is not None:
        raise last_error


@app.teardown_appcontext
def close_db_connection(_):
    conn = getattr(g, "_db_conn", None)
    if conn is not None:
        conn.close()


def init_db() -> None:
    """初始化数据库"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = create_sqlite_connection(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL UNIQUE,
                episode_name TEXT,
                dataset_name TEXT,
                episode_index INTEGER,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        for col in ("episode_name", "dataset_name", "episode_index"):
            try:
                conn.execute(f"ALTER TABLE annotations ADD COLUMN {col} TEXT")
            except Exception:
                pass

        conn.execute("""
            CREATE TABLE IF NOT EXISTS view_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                viewed_at TEXT NOT NULL,
                UNIQUE(episode_id, user_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviewing_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                started_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviewed_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                has_annotation BOOLEAN NOT NULL DEFAULT 0,
                reviewed_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS annotation_reasons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sanity_frame_annotations (
                episode_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                dataset_name TEXT,
                episode_name TEXT,
                episode_index INTEGER,
                state TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (episode_id, frame_index)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sanity_reviewed_frames (
                episode_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                dataset_name TEXT,
                episode_name TEXT,
                episode_index INTEGER,
                user_id TEXT NOT NULL,
                reviewed_at TEXT NOT NULL,
                PRIMARY KEY (episode_id, frame_index)
            )
        """)

        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_view_log_episode ON view_log(episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_view_log_user ON view_log(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_reviewing_episode ON reviewing_episodes(episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_reviewed_episode ON reviewed_episodes(episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_annotation_reasons_episode ON annotation_reasons(episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_sanity_frame_annotations_episode ON sanity_frame_annotations(episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_sanity_reviewed_frames_episode ON sanity_reviewed_frames(episode_id)",
        ]:
            conn.execute(idx_sql)

        commit_with_retry(conn)
    finally:
        conn.close()


# ─── 历史「标 X」episode（BAD_FRAMES / BAD_FRAME）与返工标注 ─────────────

REWORK_MAX_FRAMES_PER_EPISODE = 24
BOX_OUTLINE_WIDTH = 10
SANITY_REASON_V1_PREFIX = "SANITY_REASON_V1:"
SANITY_REASON_V2_PREFIX = "SANITY_REASON_V2:"


def parse_legacy_bad_frame_indices(content: str) -> List[int]:
    """解析旧版 sanity 标注中的问题帧索引（相对 episode）。"""
    if not content:
        return []
    if content.startswith("BAD_FRAMES:"):
        bad_str = content[len("BAD_FRAMES:") :]
    elif content.startswith("BAD_FRAME:"):
        bad_str = content[len("BAD_FRAME:") :]
    else:
        return []
    out: List[int] = []
    for x in bad_str.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except ValueError:
            continue
    return out


def normalize_reason_payload(bad_box, not_clear) -> Dict[str, List[int]]:
    def _ints(seq) -> List[int]:
        out: List[int] = []
        if not isinstance(seq, list):
            return out
        for x in seq:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out

    bb = sorted(set(_ints(bad_box)))
    nc = sorted(set(i for i in _ints(not_clear) if i not in bb))
    return {"bad_box": bb, "not_clear": nc}


def normalize_sanity_reason_payload(
    wrong_annotation=None,
    missing_annotation=None,
    bad_box=None,
    not_clear=None,
) -> Dict[str, List[int]]:
    def _ints(seq) -> List[int]:
        out: List[int] = []
        if not isinstance(seq, list):
            return out
        for x in seq:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out

    wa_src = wrong_annotation if wrong_annotation is not None else bad_box
    ma_src = missing_annotation if missing_annotation is not None else not_clear
    wa = sorted(set(_ints(wa_src)))
    ma = sorted(set(i for i in _ints(ma_src) if i not in wa))
    return {"wrong_annotation": wa, "missing_annotation": ma}


def _parse_reason_payload_with_prefix(content: str, prefix: str) -> Optional[Dict]:
    if not content or not content.startswith(prefix):
        return None
    try:
        raw = json.loads(content[len(prefix) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    return normalize_reason_payload(raw.get("bad_box"), raw.get("not_clear"))


def _parse_sanity_reason_payload_with_prefix(content: str, prefix: str) -> Optional[Dict]:
    if not content or not content.startswith(prefix):
        return None
    try:
        raw = json.loads(content[len(prefix) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    return normalize_sanity_reason_payload(
        wrong_annotation=raw.get("wrong_annotation"),
        missing_annotation=raw.get("missing_annotation"),
        bad_box=raw.get("bad_box"),
        not_clear=raw.get("not_clear"),
    )


def parse_rework_v1_payload(content: str) -> Optional[Dict]:
    """解析返工标注 REWORK_V1:{json}，用于回填 UI。"""
    return _parse_reason_payload_with_prefix(content, "REWORK_V1:")


def parse_sanity_reason_payload(content: str) -> Optional[Dict]:
    """解析 sanity check 的错误原因扩展字段。"""
    parsed_v2 = _parse_sanity_reason_payload_with_prefix(content, SANITY_REASON_V2_PREFIX)
    if parsed_v2 is not None:
        return parsed_v2
    return _parse_sanity_reason_payload_with_prefix(content, SANITY_REASON_V1_PREFIX)


def build_annotation_status(content: str, sanity_reason: Optional[Dict] = None) -> Dict:
    rework = parse_rework_v1_payload(content)
    if rework is not None:
        bad_frames = sorted(set(rework["bad_box"] + rework["not_clear"]))
        return {
            "has_annotation": True,
            "mark_type": "bad" if bad_frames else "alright",
            "bad_frames": bad_frames,
            "reasoned_annotation": rework,
        }

    if content.startswith("BAD_FRAMES:") or content.startswith("BAD_FRAME:"):
        bad_frames = sorted(set(parse_legacy_bad_frame_indices(content)))
        if sanity_reason is None:
            reasoned = {"wrong_annotation": list(bad_frames), "missing_annotation": []}
        else:
            wa = [i for i in sanity_reason["wrong_annotation"] if i in bad_frames]
            ma = [i for i in sanity_reason["missing_annotation"] if i in bad_frames and i not in wa]
            assigned = set(wa) | set(ma)
            missing = [i for i in bad_frames if i not in assigned]
            reasoned = {"wrong_annotation": sorted(set(wa + missing)), "missing_annotation": ma}
        return {
            "has_annotation": True,
            "mark_type": "bad" if bad_frames else "alright",
            "bad_frames": bad_frames,
            "reasoned_annotation": reasoned,
        }

    return {
        "has_annotation": True,
        "mark_type": "alright",
        "bad_frames": [],
        "reasoned_annotation": {"wrong_annotation": [], "missing_annotation": []},
    }


def is_factory_dataset_name(dataset_name: Optional[str]) -> bool:
    return isinstance(dataset_name, str) and dataset_name.startswith("factory")


def _fetch_annotation_rows(conn: sqlite3.Connection, where_sql: str) -> List[sqlite3.Row]:
    return conn.execute(
        f"""
        SELECT episode_id, episode_name, dataset_name, episode_index, content
        FROM annotations
        WHERE {where_sql}
        ORDER BY rowid
        """
    ).fetchall()


def load_legacy_buildai_annotation_rows() -> List[Dict]:
    """读取旧 BuildAI flaw 数据库中的 BAD_FRAME(S) 记录。"""
    return load_bad_annotation_rows_from_db(LEGACY_BUILDAI_ANNOTATIONS_DB, "旧 BuildAI flaw")


def _get_bad_rows_cache_signature(db_path: Path) -> Optional[tuple]:
    try:
        st = db_path.stat()
        return (str(db_path.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def load_bad_annotation_rows_from_db(db_path: Optional[Path], label: str) -> List[Dict]:
    """读取外部 source flaw 数据库中的 BAD_FRAME(S) 记录。"""
    if db_path is None:
        return []
    if not db_path.exists():
        print(f"⚠ {label} 数据库不存在: {db_path}")
        return []

    signature = _get_bad_rows_cache_signature(db_path)
    cache_key = str(db_path.resolve())
    if signature is not None:
        with _BAD_SOURCE_ROWS_LOCK:
            cached = _BAD_SOURCE_ROWS_CACHE.get(cache_key)
            if cached and cached.get("signature") == signature:
                return cached["rows"]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = _fetch_annotation_rows(conn, "content LIKE 'BAD_FRAMES:%' OR content LIKE 'BAD_FRAME:%'")
        out = [dict(row) for row in rows]
        if signature is not None:
            with _BAD_SOURCE_ROWS_LOCK:
                _BAD_SOURCE_ROWS_CACHE[cache_key] = {"signature": signature, "rows": out}
        return out
    except Exception as e:
        print(f"⚠ 读取 {label} 数据库失败: {e}")
        return []
    finally:
        conn.close()


def load_rework_source_rows() -> List[Dict]:
    """加载返工来源 flaw rows，优先走进程内缓存。"""
    return (
        load_bad_annotation_rows_from_db(SOURCE_FACTORY_ANNOTATIONS_DB, "100k flaw source")
        + load_legacy_buildai_annotation_rows()
    )


def build_legacy_buildai_lookup(candidate_rows: List[Dict], factory_lookup: Dict[str, Dict]) -> Dict[str, Dict]:
    """只为当前坏条目按需解析旧 BuildAI raw 路径，不做全量扫描。"""
    lookup: Dict[str, Dict] = {}
    seen = set()
    for row in candidate_rows:
        eid = row["episode_id"]
        if eid in seen or eid in factory_lookup:
            continue
        seen.add(eid)
        ep = resolve_legacy_buildai_episode(
            eid,
            episode_name=row.get("episode_name"),
            dataset_name=row.get("dataset_name"),
        )
        if ep is not None:
            lookup[eid] = ep
    return lookup


def is_supported_rework_episode_id(episode_id: str, factory_lookup: Dict[str, Dict]) -> bool:
    """判断 rework 条目是否属于当前支持的 factory 或 legacy BuildAI raw。"""
    if episode_id in factory_lookup:
        return True
    return LEGACY_BUILDAI_ROOT is not None and _LEGACY_BUILDAI_EPISODE_RE.match(episode_id) is not None


def build_rework_queue_rows(
    local_rows: List[sqlite3.Row],
    source_rows: List[Dict],
    factory_lookup: Dict[str, Dict],
) -> List[Dict]:
    """合并 source flaw 数据库与本地 REWORK 结果，生成有效队列。"""
    local_by_id: Dict[str, Dict] = {row["episode_id"]: dict(row) for row in local_rows}
    if source_rows:
        base_rows = source_rows
    else:
        base_rows = [
            dict(row)
            for row in local_rows
            if row["content"].startswith("BAD_FRAMES:") or row["content"].startswith("BAD_FRAME:")
        ]

    merged: List[Dict] = []
    seen = set()

    for row in base_rows:
        eid = row["episode_id"]
        if eid in seen:
            continue
        if not is_supported_rework_episode_id(eid, factory_lookup):
            continue
        local_row = local_by_id.get(eid)
        seen.add(eid)
        if local_row and (
            str(local_row.get("content", "")).startswith("BAD_FRAMES:")
            or str(local_row.get("content", "")).startswith("BAD_FRAME:")
        ):
            merged.append(local_row)
        else:
            merged.append(dict(row))

    return merged


def build_rework_frame_indices(
    num_frames: int,
    legacy_bad: List[int],
    max_frames: int = REWORK_MAX_FRAMES_PER_EPISODE,
) -> List[int]:
    """仅保留历史标为问题的帧，控制上限。"""
    legacy_ok = sorted({i for i in legacy_bad if isinstance(i, int) and 0 <= i < num_frames})
    return legacy_ok[:max_frames]


def _normalize_sanity_frame_state(state: Optional[str]) -> str:
    if state == "bad_box":
        return "wrong_annotation"
    if state == "not_clear":
        return "missing_annotation"
    if state in ("wrong_annotation", "missing_annotation"):
        return str(state)
    return "ok"


def _load_results_frame_review_info(
    conn: sqlite3.Connection,
    frame_batch: List[Dict[str, Any]],
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    episode_ids = sorted({frame["episode_id"] for frame in frame_batch if frame.get("episode_id")})
    if not episode_ids:
        return {}
    placeholders = ",".join(["?"] * len(episode_ids))
    annotation_rows = conn.execute(
        f"""
        SELECT episode_id, frame_index, state
        FROM sanity_frame_annotations
        WHERE episode_id IN ({placeholders})
        """,
        episode_ids,
    ).fetchall()
    reviewed_rows = conn.execute(
        f"""
        SELECT episode_id, frame_index
        FROM sanity_reviewed_frames
        WHERE episode_id IN ({placeholders})
        """,
        episode_ids,
    ).fetchall()
    frame_pairs = {(frame["episode_id"], int(frame["frame_index"])) for frame in frame_batch}
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in reviewed_rows:
        key = (row["episode_id"], int(row["frame_index"]))
        if key in frame_pairs:
            out[key] = {"reviewed": True, "state": "ok"}
    for row in annotation_rows:
        key = (row["episode_id"], int(row["frame_index"]))
        if key in frame_pairs:
            out[key] = {
                "reviewed": True,
                "state": _normalize_sanity_frame_state(row["state"]),
            }
    return out


def _upsert_legacy_safe_annotation(
    cursor: sqlite3.Cursor,
    episode_id: str,
    episode_name: str,
    dataset_name: str,
    episode_index: Optional[int],
    content: str,
    updated_at: str,
) -> bool:
    existing_row = cursor.execute(
        "SELECT dataset_name FROM annotations WHERE episode_id = ?",
        (episode_id,),
    ).fetchone()
    if existing_row is not None and not is_factory_dataset_name(existing_row["dataset_name"]):
        return False

    cursor.execute(
        """
        INSERT INTO annotations (episode_id, episode_name, dataset_name, episode_index, content, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(episode_id) DO UPDATE SET
            episode_name = excluded.episode_name,
            dataset_name = excluded.dataset_name,
            episode_index = excluded.episode_index,
            content = excluded.content,
            updated_at = excluded.updated_at
        """,
        (episode_id, episode_name, dataset_name, episode_index, content, updated_at),
    )
    return True


def _finalize_results_episode_annotations(
    conn: sqlite3.Connection,
    results_index: Dict[str, Any],
    episode_ids: Set[str],
    user_id: str,
) -> List[str]:
    cursor = conn.cursor()
    episodes_by_id = results_index.get("episodes_by_id", {})
    skipped_legacy_conflicts: List[str] = []
    now = datetime.utcnow().isoformat()

    for episode_id in sorted(episode_ids):
        ep = episodes_by_id.get(episode_id)
        if ep is None:
            continue
        dataset_name = ep.get("dataset_name", "")
        if not is_factory_dataset_name(dataset_name):
            continue

        valid_frame_indices = {int(x) for x in ep.get("sampled_result_frame_indices", [])}
        expected_count = len(valid_frame_indices)
        if expected_count <= 0:
            continue
        reviewed_rows = cursor.execute(
            "SELECT frame_index FROM sanity_reviewed_frames WHERE episode_id = ?",
            (episode_id,),
        ).fetchall()
        reviewed_count = sum(1 for row in reviewed_rows if int(row["frame_index"]) in valid_frame_indices)
        if reviewed_count < expected_count:
            continue

        wrong_annotation: List[int] = []
        missing_annotation: List[int] = []
        rows = cursor.execute(
            """
            SELECT frame_index, state
            FROM sanity_frame_annotations
            WHERE episode_id = ?
            ORDER BY frame_index
            """,
            (episode_id,),
        ).fetchall()
        for row in rows:
            frame_index = int(row["frame_index"])
            if frame_index not in valid_frame_indices:
                continue
            state = _normalize_sanity_frame_state(row["state"])
            if state == "wrong_annotation":
                wrong_annotation.append(frame_index)
            elif state == "missing_annotation":
                missing_annotation.append(frame_index)

        bad_frames = sorted(set(wrong_annotation + missing_annotation))
        content = "BAD_FRAMES:" + ",".join(map(str, bad_frames)) if bad_frames else "alright"
        ok = _upsert_legacy_safe_annotation(
            cursor,
            episode_id=episode_id,
            episode_name=ep.get("episode_name", episode_id),
            dataset_name=dataset_name,
            episode_index=ep.get("episode_index"),
            content=content,
            updated_at=now,
        )
        if not ok:
            skipped_legacy_conflicts.append(episode_id)
            continue

        if bad_frames:
            reason_content = SANITY_REASON_V2_PREFIX + json.dumps(
                {"wrong_annotation": wrong_annotation, "missing_annotation": missing_annotation},
                separators=(",", ":"),
            )
            cursor.execute(
                """
                INSERT INTO annotation_reasons (episode_id, content, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    content = excluded.content,
                    updated_at = excluded.updated_at
                """,
                (episode_id, reason_content, now),
            )
        else:
            cursor.execute("DELETE FROM annotation_reasons WHERE episode_id = ?", (episode_id,))

        cursor.execute(
            """
            INSERT OR REPLACE INTO reviewed_episodes (episode_id, user_id, has_annotation, reviewed_at)
            VALUES (?, ?, ?, ?)
            """,
            (episode_id, user_id, bool(bad_frames), now),
        )

    return skipped_legacy_conflicts


# ─── 路由 ─────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("sanity_check.html")


@app.route("/sanity-check")
def sanity_check():
    return render_template("sanity_check.html")


@app.route("/rework")
def rework_check():
    """仅审核历史标为 X（BAD_FRAMES）的 episode，三态帧标注。"""
    return render_template("rework_check.html")


@app.route("/api/total-episodes", methods=["GET"])
def api_total_episodes():
    """获取总 episode 数"""
    results_index = scan_sanity_results_index()
    result_frames = results_index.get("frames", [])
    if result_frames:
        dataset_counts: Dict[str, int] = {}
        for frame in result_frames:
            dataset_name = frame.get("dataset_name", "unknown")
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        return jsonify({
            "success": True,
            "queue_mode": "frame",
            "item_label": "frames",
            "total_items": len(result_frames),
            "total_frames": len(result_frames),
            "total_episodes": len(result_frames),
            "datasets": [{"name": name, "count": count} for name, count in sorted(dataset_counts.items())],
        })

    episodes = scan_factory_episodes()
    dataset_counts: Dict[str, int] = {}
    for ep in episodes:
        dataset_name = ep.get("dataset_name", "unknown")
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
    return jsonify({
        "success": True,
        "queue_mode": "episode",
        "item_label": "episodes",
        "total_items": len(episodes),
        "total_episodes": len(episodes),
        "datasets": [{"name": name, "count": count} for name, count in sorted(dataset_counts.items())],
    })


@app.route("/api/episodes/sequential", methods=["GET"])
def api_episodes_sequential():
    """顺序加载 episodes，返回带 YOLO 框的中间帧图像"""
    timer_start = time.time()
    timers: Dict[str, float] = {}

    limit = request.args.get("limit", default=210, type=int)
    global_offset = request.args.get("offset", default=0, type=int)
    user_id = request.args.get("user_id", default="anonymous", type=str)

    conn = get_db_connection()

    t0 = time.time()
    results_index = scan_sanity_results_index()
    result_frames = results_index.get("frames", [])
    timers["scan_episodes"] = time.time() - t0

    if result_frames:
        total_items = len(result_frames)
        if global_offset >= total_items:
            return jsonify({
                "success": False,
                "message": f"Index {global_offset} 超出范围！总共只有 {total_items} 个 frames",
                "queue_mode": "frame",
                "total_items": total_items,
                "total_episodes": total_items,
            }), 400

        end_idx = min(global_offset + limit, total_items)
        batch_frames = result_frames[global_offset:end_idx]
        next_offset = end_idx if end_idx < total_items else 0

        t1 = time.time()
        review_info_by_frame = _load_results_frame_review_info(conn, batch_frames)
        timers["db_query"] = time.time() - t1

        t2 = time.time()
        rendered_images = load_episode_frames_from_results(batch_frames, max_width=SANITY_RESULTS_FRAME_MAX_WIDTH)
        items: List[Dict[str, Any]] = []
        fail_reasons: List[str] = []
        for frame, image in zip(batch_frames, rendered_images):
            if image is None:
                fail_reasons.append(f"load results frame returned empty for {frame['frame_id']}")
                continue
            review_info = review_info_by_frame.get((frame["episode_id"], int(frame["frame_index"])), {})
            item = {
                "frame_id": frame["frame_id"],
                "episode_id": frame["episode_id"],
                "episode_name": frame.get("episode_name", frame["episode_id"]),
                "dataset_name": frame.get("dataset_name", ""),
                "episode_index": frame.get("episode_index"),
                "frame_index": int(frame["frame_index"]),
                "queue_index": int(frame.get("queue_index", 0)),
                "prob_is_bad": frame.get("prob_is_bad"),
                "image": image,
                "annotation_state": review_info.get("state", "ok"),
                "was_reviewed": bool(review_info.get("reviewed", False)),
            }
            items.append(item)
        timers["load_and_encode"] = time.time() - t2

        total_time = time.time() - timer_start
        timers["total"] = total_time
        timers["other"] = total_time - sum(v for k, v in timers.items() if k not in ("total", "other"))

        print(f"\n{'=' * 60}")
        print(f"[{user_id[:8]}] results frame queue (offset={global_offset}, limit={limit})")
        print(f"{'=' * 60}")
        print(f"  扫描索引:       {timers.get('scan_episodes', 0):.3f}s")
        print(f"  数据库查询:     {timers.get('db_query', 0):.3f}s")
        print(f"  加载+编码图像:  {timers.get('load_and_encode', 0):.3f}s")
        print(f"  成功/失败:      {len(items)}/{len(batch_frames) - len(items)}")
        print(f"  总耗时:         {total_time:.3f}s")
        print(f"{'=' * 60}\n")

        resp: Dict[str, Any] = {
            "success": True,
            "queue_mode": "frame",
            "item_label": "frames",
            "items": items,
            "episodes": items,
            "has_more": next_offset != 0,
            "next_offset": next_offset,
            "total_items": total_items,
            "total_frames": total_items,
            "total_episodes": total_items,
            "collected_count": len(items),
            "performance": timers,
        }
        if not items and fail_reasons:
            resp["debug_fail_reasons"] = fail_reasons[:5]
        return jsonify(resp)

    all_episodes = scan_factory_episodes()

    total_episodes = len(all_episodes)

    if global_offset >= total_episodes:
        return jsonify({
            "success": False,
            "message": f"Index {global_offset} 超出范围！总共只有 {total_episodes} 个 episodes",
            "queue_mode": "episode",
            "total_episodes": total_episodes,
        }), 400

    end_idx = min(global_offset + limit, total_episodes)
    batch = all_episodes[global_offset:end_idx]
    next_offset = end_idx if end_idx < total_episodes else 0

    t1 = time.time()
    episode_ids = [ep["episode_id"] for ep in batch]
    dataset_names = sorted({ep.get("dataset_name") for ep in batch if is_factory_dataset_name(ep.get("dataset_name"))})
    annotation_rows = []
    reason_rows = []
    if episode_ids and dataset_names:
        placeholders = ",".join(["?"] * len(episode_ids))
        dataset_placeholders = ",".join(["?"] * len(dataset_names))
        annotation_rows = conn.execute(
            f"""
            SELECT episode_id, content
            FROM annotations
            WHERE episode_id IN ({placeholders})
              AND dataset_name IN ({dataset_placeholders})
            """,
            episode_ids + dataset_names,
        ).fetchall()
        reason_rows = conn.execute(
            f"SELECT episode_id, content FROM annotation_reasons WHERE episode_id IN ({placeholders})",
            episode_ids,
        ).fetchall()
    timers["db_query"] = time.time() - t1

    sanity_reason_by_id: Dict[str, Dict] = {}
    for row in reason_rows:
        parsed = parse_sanity_reason_payload(row[1])
        if parsed is not None:
            sanity_reason_by_id[row[0]] = parsed

    annotation_status: Dict[str, Dict] = {}
    for row in annotation_rows:
        ep_id, content = row[0], row[1]
        annotation_status[ep_id] = build_annotation_status(content, sanity_reason_by_id.get(ep_id))

    fail_reasons = []

    def process_episode(ep: Dict) -> Optional[Dict]:
        """加载单个 episode 的帧并编码"""
        try:
            sampled_frames = ep.get("sampled_frames")
            if isinstance(sampled_frames, list):
                if not sampled_frames:
                    fail_reasons.append(f"no sampled frames: {ep['episode_id']}")
                    return None

                frame_indices = [int(item["frame_index"]) for item in sampled_frames]
                images = load_episode_frames_from_results(sampled_frames)
                if not images:
                    fail_reasons.append(f"load results frames returned empty for {ep['episode_id']}")
                    return None

                result = {
                    "success": True,
                    "episode_id": ep["episode_id"],
                    "episode_name": ep["episode_name"],
                    "dataset_name": ep["dataset_name"],
                    "episode_index": ep.get("episode_index", 0),
                    "num_frames": ep.get("num_frames", len(frame_indices)),
                    "start_idx": 0,
                    "images": images,
                    "frame_indices": frame_indices,
                }
                if ep["episode_id"] in annotation_status:
                    result["annotation"] = annotation_status[ep["episode_id"]]
                return result

            num_frames = ep["num_frames"]
            if num_frames == 0:
                fail_reasons.append(f"0 frames: {ep['episode_id']}")
                return None

            # 检查 detect_track 是否已完成
            stage1_done = find_model_tracks_path(ep["seq_folder"]) is not None

            if not stage1_done:
                # stage1 未完成，返回占位 episode（无图片，前端显示提示）
                result = {
                    "success": True,
                    "episode_id": ep["episode_id"],
                    "episode_name": ep["episode_name"],
                    "dataset_name": ep["dataset_name"],
                    "episode_index": 0,
                    "num_frames": num_frames,
                    "start_idx": 0,
                    "images": [],
                    "frame_indices": [],
                    "stage1_pending": True,
                }
                if ep["episode_id"] in annotation_status:
                    result["annotation"] = annotation_status[ep["episode_id"]]
                return result

            # 加载 track 数据，选取有检测框的帧
            frame_boxes = load_track_boxes(ep["seq_folder"])
            frame_indices = pick_frames_with_boxes(num_frames, frame_boxes)

            # 延迟加载帧名和 offset
            frame_names, frame_offsets = _get_episode_frames(ep)
            if frame_names is None or len(frame_names) == 0:
                fail_reasons.append(f"no frame data for {ep['episode_id']}")
                return None

            images = load_episode_frames_from_shard(
                ep["shard_path"],
                frame_names,
                frame_offsets,
                frame_indices,
                frame_boxes,
            )
            if not images:
                fail_reasons.append(f"load_episode_frames returned empty for {ep['episode_id']}")
                return None

            result = {
                "success": True,
                "episode_id": ep["episode_id"],
                "episode_name": ep["episode_name"],
                "dataset_name": ep["dataset_name"],
                "episode_index": 0,
                "num_frames": num_frames,
                "start_idx": 0,
                "images": images,
                "frame_indices": frame_indices,
            }

            if ep["episode_id"] in annotation_status:
                result["annotation"] = annotation_status[ep["episode_id"]]

            return result
        except Exception as e:
            fail_reasons.append(f"exception: {e} for {ep['episode_id']}")
            return None

    # 并行加载所有 episodes 的帧
    t2 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_episode, ep) for ep in batch]
        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)
    timers["load_and_encode"] = time.time() - t2

    total_time = time.time() - timer_start
    timers["total"] = total_time
    timers["other"] = total_time - sum(v for k, v in timers.items() if k != "total" and k != "other")

    fail_count = len(batch) - len(results)
    print(f"\n{'=' * 60}")
    print(f"[{user_id[:8]}] 性能报告 (offset={global_offset}, limit={limit})")
    print(f"{'=' * 60}")
    print(f"  扫描episodes:   {timers.get('scan_episodes', 0):.3f}s")
    print(f"  数据库查询:     {timers.get('db_query', 0):.3f}s")
    print(f"  加载+编码图像:  {timers.get('load_and_encode', 0):.3f}s")
    print(f"  成功/失败:      {len(results)}/{fail_count}")
    print(f"  总耗时:         {total_time:.3f}s")
    if fail_reasons:
        print(f"  --- 失败原因（前5条）---")
        for reason in fail_reasons[:5]:
            print(f"    ✗ {reason}")
    print(f"{'=' * 60}\n")

    resp = {
        "success": True,
        "queue_mode": "episode",
        "item_label": "episodes",
        "episodes": results,
        "has_more": next_offset != 0,
        "next_offset": next_offset,
        "total_items": total_episodes,
        "total_episodes": total_episodes,
        "collected_count": len(results),
        "performance": timers,
    }

    if not results and fail_reasons:
        resp["debug_fail_reasons"] = fail_reasons[:5]

    return jsonify(resp)


@app.route("/api/rework/total", methods=["GET"])
def api_rework_total():
    conn = get_db_connection()
    factory_lookup = get_factory_episode_lookup()
    local_rows = _fetch_annotation_rows(
        conn,
        "(content LIKE 'BAD_FRAMES:%' OR content LIKE 'BAD_FRAME:%' OR content LIKE 'REWORK_V1:%')",
    )
    source_rows = load_rework_source_rows()
    rows = build_rework_queue_rows(local_rows, source_rows, factory_lookup)
    return jsonify({"success": True, "total_episodes": len(rows)})


@app.route("/api/rework/episodes", methods=["GET"])
def api_rework_episodes():
    """分页加载仍标记为 BAD_FRAMES/BAD_FRAME 的 episode，兼容 factory 与旧 BuildAI raw 数据源。"""
    timer_start = time.time()
    timers: Dict[str, float] = {}

    limit = request.args.get("limit", default=210, type=int)
    global_offset = request.args.get("offset", default=0, type=int)
    user_id = request.args.get("user_id", default="anonymous", type=str)

    t0 = time.time()
    conn = get_db_connection()
    id_to_ep = get_factory_episode_lookup()
    local_rows = _fetch_annotation_rows(
        conn,
        "(content LIKE 'BAD_FRAMES:%' OR content LIKE 'BAD_FRAME:%' OR content LIKE 'REWORK_V1:%')",
    )
    source_rows = load_rework_source_rows()
    bad_rows = build_rework_queue_rows(local_rows, source_rows, id_to_ep)
    timers["scan_and_db"] = time.time() - t0

    total_bad = len(bad_rows)
    if total_bad == 0:
        return jsonify(
            {
                "success": True,
                "episodes": [],
                "has_more": False,
                "next_offset": 0,
                "total_episodes": 0,
                "collected_count": 0,
                "performance": {**timers, "total": time.time() - timer_start},
            }
        )

    if global_offset >= total_bad:
        return jsonify(
            {
                "success": False,
                "message": f"Index {global_offset} 超出范围！待返工共 {total_bad} 条",
                "total_episodes": total_bad,
            }
        ), 400

    end_idx = min(global_offset + max(1, limit), total_bad)
    batch_rows: List[Dict] = bad_rows[global_offset:end_idx]
    t1 = time.time()
    legacy_buildai_lookup = build_legacy_buildai_lookup(batch_rows, id_to_ep)
    timers["resolve_legacy_batch"] = time.time() - t1
    batch_ids = [row["episode_id"] for row in batch_rows]
    next_offset = end_idx if end_idx < total_bad else 0

    t1 = time.time()
    ann_rows: List[sqlite3.Row] = []
    if batch_ids:
        ph = ",".join(["?"] * len(batch_ids))
        ann_rows = conn.execute(
            f"""
            SELECT episode_id, content
            FROM annotations
            WHERE episode_id IN ({ph})
            """,
            batch_ids,
        ).fetchall()
    timers["db_query"] = time.time() - t1

    rework_prefill: Dict[str, Dict] = {}
    for ep_id, content in ann_rows:
        rw = parse_rework_v1_payload(content)
        if rw:
            rework_prefill[ep_id] = rw

    fail_reasons: List[str] = []

    def process_rework_episode(row: Dict) -> Optional[Dict]:
        try:
            ep_id = row["episode_id"]
            content = row["content"]
            ep = id_to_ep.get(ep_id)
            if ep is None:
                ep = hydrate_legacy_buildai_episode(legacy_buildai_lookup.get(ep_id))
            if ep is None:
                fail_reasons.append(f"未找到可回放数据源: {ep_id}")
                return None

            num_frames = ep["num_frames"]
            if num_frames == 0:
                fail_reasons.append(f"0 frames: {ep_id}")
                return None

            legacy_bad = parse_legacy_bad_frame_indices(content)
            frame_indices = build_rework_frame_indices(num_frames, legacy_bad)
            if not frame_indices:
                fail_reasons.append(f"no valid bad frames for {ep_id}")
                return None

            result: Dict = {
                "success": True,
                "episode_id": ep["episode_id"],
                "episode_name": row.get("episode_name") or ep["episode_name"],
                "dataset_name": row.get("dataset_name") or ep["dataset_name"],
                "episode_index": row.get("episode_index", ep.get("episode_index", 0)),
                "num_frames": num_frames,
                "start_idx": ep.get("start_idx", 0),
                "legacy_bad_frames": legacy_bad,
                "rework_annotation": rework_prefill.get(ep_id),
            }

            if ep.get("source_type") == "legacy_buildai_raw":
                frame_boxes = load_track_boxes(ep["crop_dir"], target_frames=set(frame_indices))
                images = load_episode_frames_from_raw_buildai(ep["crop_dir"], frame_indices, frame_boxes=frame_boxes)
                if not images:
                    fail_reasons.append(f"legacy raw frames empty for {ep_id}")
                    return None
                result["images"] = images
                result["frame_indices"] = frame_indices
                return result

            stage1_done = find_model_tracks_path(ep["seq_folder"]) is not None
            if not stage1_done:
                result["images"] = []
                result["frame_indices"] = []
                result["stage1_pending"] = True
                return result

            frame_names, frame_offsets = _get_episode_frames(ep)
            if frame_names is None or len(frame_names) == 0:
                fail_reasons.append(f"no frame data for {ep_id}")
                return None

            frame_boxes = load_track_boxes(ep["seq_folder"], target_frames=set(frame_indices))
            images = load_episode_frames_from_shard(
                ep["shard_path"],
                frame_names,
                frame_offsets,
                frame_indices,
                frame_boxes,
            )
            if not images:
                fail_reasons.append(f"load_episode_frames returned empty for {ep_id}")
                return None

            result["images"] = images
            result["frame_indices"] = frame_indices
            return result
        except Exception as e:
            fail_reasons.append(f"exception: {e} for {row['episode_id']}")
            return None

    t2 = time.time()
    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_rework_episode, row) for row in batch_rows]
        for future in futures:
            r = future.result()
            if r is not None:
                results.append(r)
    timers["load_and_encode"] = time.time() - t2

    total_time = time.time() - timer_start
    timers["total"] = total_time
    timers["other"] = total_time - sum(v for k, v in timers.items() if k not in ("total", "other"))

    print(f"\n{'=' * 60}")
    print(f"[rework/{user_id[:8]}] offset={global_offset}, limit={limit}, 待返工总数={total_bad}")
    print(
        f"  本批请求 id 数: {len(batch_ids)}, "
        f"factory 命中: {sum(1 for i in batch_ids if i in id_to_ep)}, "
        f"legacy raw 命中: {sum(1 for i in batch_ids if i in legacy_buildai_lookup)}, "
        f"成功加载: {len(results)}"
    )
    print(
        "  耗时: "
        f"scan_and_db={timers.get('scan_and_db', 0):.2f}s, "
        f"resolve_legacy_batch={timers.get('resolve_legacy_batch', 0):.2f}s, "
        f"db_query={timers.get('db_query', 0):.2f}s, "
        f"load_and_encode={timers.get('load_and_encode', 0):.2f}s, "
        f"total={timers.get('total', 0):.2f}s"
    )
    print(f"{'=' * 60}\n")

    resp = {
        "success": True,
        "episodes": results,
        "has_more": next_offset != 0,
        "next_offset": next_offset,
        "total_episodes": total_bad,
        "collected_count": len(results),
        "performance": timers,
    }
    if not results and fail_reasons:
        resp["debug_fail_reasons"] = fail_reasons[:5]
    return jsonify(resp)


@app.route("/api/rework/submit", methods=["POST"])
def api_rework_submit():
    """保存返工三态：REWORK_V1: {\"bad_box\":[], \"not_clear\":[]}"""
    data = request.get_json(silent=True) or {}
    items = data.get("episodes", [])
    if not isinstance(items, list):
        return jsonify({"success": False, "message": "episodes 须为数组"}), 400

    conn = get_db_connection()
    now = datetime.utcnow().isoformat()

    for item in items:
        eid = item.get("episode_id")
        if not eid:
            continue
        episode_name = item.get("episode_name", "")
        dataset_name = item.get("dataset_name", "")
        episode_index = item.get("episode_index", None)
        reasoned = normalize_reason_payload(item.get("bad_box"), item.get("not_clear"))
        bb = reasoned["bad_box"]
        nc = reasoned["not_clear"]
        payload = json.dumps({"bad_box": bb, "not_clear": nc}, separators=(",", ":"))
        content = "REWORK_V1:" + payload

        conn.execute(
            """
            INSERT INTO annotations (episode_id, episode_name, dataset_name, episode_index, content, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(episode_id) DO UPDATE SET
                episode_name = excluded.episode_name,
                dataset_name = excluded.dataset_name,
                episode_index = excluded.episode_index,
                content = excluded.content,
                updated_at = excluded.updated_at
            """,
            (eid, episode_name, dataset_name, episode_index, content, now),
        )
    commit_with_retry(conn)
    return jsonify({"success": True, "saved": len(items)})


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """获取查看统计信息"""
    conn = get_db_connection()
    total_views = conn.execute("SELECT COUNT(*) FROM view_log").fetchone()[0]
    total_users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM view_log").fetchone()[0]
    user_stats = conn.execute(
        "SELECT user_id, COUNT(*) as view_count FROM view_log GROUP BY user_id ORDER BY view_count DESC"
    ).fetchall()
    total_viewed = conn.execute("SELECT COUNT(DISTINCT episode_id) FROM view_log").fetchone()[0]

    return jsonify({
        "total_views": total_views,
        "total_users": total_users,
        "total_viewed_episodes": total_viewed,
        "user_stats": [{"user_id": row[0], "view_count": row[1]} for row in user_stats],
    })


@app.route("/api/annotation/<path:episode_id>", methods=["GET"])
def api_get_annotation(episode_id: str):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT content, updated_at FROM annotations WHERE episode_id = ?", (episode_id,)
    ).fetchone()

    if row is None:
        return jsonify({"episode_id": episode_id, "content": "", "updated_at": None})

    return jsonify({"episode_id": episode_id, "content": row["content"], "updated_at": row["updated_at"]})


@app.route("/api/annotation/<path:episode_id>", methods=["POST"])
def api_set_annotation(episode_id: str):
    data = request.get_json(silent=True) or {}
    content = data.get("content", "")
    episode_name = data.get("episode_name", "")
    dataset_name = data.get("dataset_name", "")
    episode_index = data.get("episode_index", None)

    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO annotations (episode_id, episode_name, dataset_name, episode_index, content, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(episode_id) DO UPDATE SET
            episode_name = excluded.episode_name,
            dataset_name = excluded.dataset_name,
            episode_index = excluded.episode_index,
            content = excluded.content,
            updated_at = excluded.updated_at
        """,
        (episode_id, episode_name, dataset_name, episode_index, content, datetime.utcnow().isoformat()),
    )
    commit_with_retry(conn)
    return jsonify({"ok": True})


@app.route("/api/sanity-check/start-review", methods=["POST"])
def api_start_review():
    try:
        data = request.json
        episode_ids = data.get("episode_ids", [])
        user_id = data.get("user_id", "anonymous")
        session_id = data.get("session_id", str(datetime.utcnow().timestamp()))

        conn = get_db_connection()
        for episode_id in episode_ids:
            conn.execute(
                "INSERT OR IGNORE INTO reviewing_episodes (episode_id, user_id, session_id, started_at) VALUES (?, ?, ?, ?)",
                (episode_id, user_id, session_id, datetime.utcnow().isoformat()),
            )
        commit_with_retry(conn)

        return jsonify({"success": True, "message": f"已标记 {len(episode_ids)} 个 episodes 为审核中", "session_id": session_id})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/sanity-check/transfer-review", methods=["POST"])
def api_transfer_review():
    try:
        data = request.json
        old_session_id = data.get("old_session_id")
        new_session_id = data.get("new_session_id")
        if not old_session_id or not new_session_id:
            return jsonify({"success": False, "message": "缺少session_id"}), 400

        conn = get_db_connection()
        conn.execute("UPDATE reviewing_episodes SET session_id = ? WHERE session_id = ?", (new_session_id, old_session_id))
        commit_with_retry(conn)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/sanity-check/cancel-review", methods=["POST"])
def api_cancel_review():
    try:
        data = request.json
        session_id = data.get("session_id")
        user_id = data.get("user_id")

        conn = get_db_connection()
        if session_id:
            conn.execute("DELETE FROM reviewing_episodes WHERE session_id = ?", (session_id,))
        elif user_id:
            conn.execute("DELETE FROM reviewing_episodes WHERE user_id = ?", (user_id,))
        commit_with_retry(conn)

        return jsonify({"success": True, "message": "已取消审核状态"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/sanity-check/submit", methods=["POST"])
def api_sanity_check_submit():
    """提交 sanity check 标注结果"""
    try:
        data = request.json or {}
        queue_mode = data.get("queue_mode", "episode")
        raw_episodes = data.get("episodes", [])
        reviewed_episodes = data.get("reviewed_episodes", [])
        user_id = data.get("user_id", "anonymous")
        session_id = data.get("session_id")

        conn = get_db_connection()
        cursor = conn.cursor()
        skipped_legacy_conflicts: List[str] = []

        if queue_mode == "frame":
            frame_items = data.get("frames", [])
            if not isinstance(frame_items, list):
                return jsonify({"success": False, "message": "frames 须为数组"}), 400

            results_index = scan_sanity_results_index()
            affected_episode_ids: Set[str] = set()
            saved_count = 0
            bad_count = 0
            now = datetime.utcnow().isoformat()

            for item in frame_items:
                episode_id = item.get("episode_id")
                dataset_name = item.get("dataset_name", "")
                episode_name = item.get("episode_name", episode_id or "")
                episode_index = item.get("episode_index")
                state = str(item.get("state", "ok"))
                if not episode_id or not is_factory_dataset_name(dataset_name):
                    continue
                try:
                    frame_index = int(item.get("frame_index"))
                except (TypeError, ValueError):
                    continue

                existing_row = cursor.execute(
                    "SELECT dataset_name FROM annotations WHERE episode_id = ?",
                    (episode_id,),
                ).fetchone()
                if existing_row is not None and not is_factory_dataset_name(existing_row["dataset_name"]):
                    skipped_legacy_conflicts.append(episode_id)
                    continue

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO sanity_reviewed_frames
                        (episode_id, frame_index, dataset_name, episode_name, episode_index, user_id, reviewed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (episode_id, frame_index, dataset_name, episode_name, episode_index, user_id, now),
                )

                normalized_state = _normalize_sanity_frame_state(state)
                if normalized_state in ("wrong_annotation", "missing_annotation"):
                    cursor.execute(
                        """
                        INSERT INTO sanity_frame_annotations
                        (episode_id, frame_index, dataset_name, episode_name, episode_index, state, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(episode_id, frame_index) DO UPDATE SET
                            dataset_name = excluded.dataset_name,
                            episode_name = excluded.episode_name,
                            episode_index = excluded.episode_index,
                            state = excluded.state,
                            updated_at = excluded.updated_at
                        """,
                        (episode_id, frame_index, dataset_name, episode_name, episode_index, normalized_state, now),
                    )
                    bad_count += 1
                else:
                    cursor.execute(
                        "DELETE FROM sanity_frame_annotations WHERE episode_id = ? AND frame_index = ?",
                        (episode_id, frame_index),
                    )

                saved_count += 1
                affected_episode_ids.add(episode_id)

            skipped_legacy_conflicts.extend(
                _finalize_results_episode_annotations(conn, results_index, affected_episode_ids, user_id)
            )

            if session_id:
                cursor.execute("DELETE FROM reviewing_episodes WHERE session_id = ?", (session_id,))
            else:
                cursor.execute("DELETE FROM reviewing_episodes WHERE user_id = ?", (user_id,))

            commit_with_retry(conn)

            return jsonify({
                "success": True,
                "message": f"已提交 {saved_count} 个 frames ({saved_count - bad_count} 正常, {bad_count} 有问题)",
                "total_count": saved_count,
                "alright_count": saved_count - bad_count,
                "bad_count": bad_count,
                "completed_episode_count": len(affected_episode_ids),
                "skipped_legacy_conflicts": sorted(set(skipped_legacy_conflicts))[:20],
            })

        normalized_episodes = []
        if isinstance(raw_episodes, list) and raw_episodes:
            for episode_info in raw_episodes:
                episode_id = episode_info.get("episode_id")
                if not episode_id:
                    continue
                reasoned = normalize_sanity_reason_payload(
                    wrong_annotation=episode_info.get("wrong_annotation"),
                    missing_annotation=episode_info.get("missing_annotation"),
                    bad_box=episode_info.get("bad_box"),
                    not_clear=episode_info.get("not_clear"),
                )
                bad_frames = sorted(set(reasoned["wrong_annotation"] + reasoned["missing_annotation"]))
                normalized_episodes.append({
                    "episode_id": episode_id,
                    "dataset": episode_info.get("dataset_name", episode_info.get("dataset", "")),
                    "episode_name": episode_info.get("episode_name", ""),
                    "episode_index": episode_info.get("episode_index"),
                    "bad_frames": bad_frames,
                    "reasoned_annotation": reasoned,
                })
        else:
            marked_frames = data.get("marked_frames", [])
            marked_by_episode: Dict[str, List[int]] = {}
            for frame in marked_frames:
                episode_id = frame.get("episode_id")
                if not episode_id:
                    continue
                try:
                    frame_index = int(frame.get("frame_index"))
                except (TypeError, ValueError):
                    continue
                marked_by_episode.setdefault(episode_id, []).append(frame_index)

            for episode_info in reviewed_episodes:
                episode_id = episode_info.get("episode_id")
                if not episode_id:
                    continue
                bad_frames = sorted(set(marked_by_episode.get(episode_id, [])))
                normalized_episodes.append({
                    "episode_id": episode_id,
                    "dataset": episode_info.get("dataset", ""),
                    "episode_name": episode_info.get("episode_name", ""),
                    "episode_index": episode_info.get("episode_index"),
                    "bad_frames": bad_frames,
                    "reasoned_annotation": {"wrong_annotation": list(bad_frames), "missing_annotation": []},
                })

        if not isinstance(reviewed_episodes, list) or not reviewed_episodes:
            reviewed_episodes = [
                {
                    "episode_id": item["episode_id"],
                    "dataset": item["dataset"],
                    "episode_name": item["episode_name"],
                    "episode_index": item["episode_index"],
                    "has_annotation": bool(item["bad_frames"]),
                }
                for item in normalized_episodes
            ]

        for episode_info in normalized_episodes:
            episode_id = episode_info.get("episode_id")
            dataset = episode_info.get("dataset")
            episode_name = episode_info.get("episode_name")
            episode_index = episode_info.get("episode_index")
            bad_frames = episode_info.get("bad_frames", [])
            reasoned = episode_info.get("reasoned_annotation", {"wrong_annotation": [], "missing_annotation": []})

            if not is_factory_dataset_name(dataset):
                continue

            if not _upsert_legacy_safe_annotation(
                cursor,
                episode_id=episode_id,
                episode_name=episode_name,
                dataset_name=dataset,
                episode_index=episode_index,
                content=("BAD_FRAMES:" + ",".join(map(str, bad_frames))) if bad_frames else "alright",
                updated_at=datetime.utcnow().isoformat(),
            ):
                skipped_legacy_conflicts.append(episode_id)
                continue

            if bad_frames:
                reason_content = SANITY_REASON_V2_PREFIX + json.dumps(reasoned, separators=(",", ":"))
                cursor.execute(
                    """
                    INSERT INTO annotation_reasons (episode_id, content, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(episode_id) DO UPDATE SET
                        content = excluded.content,
                        updated_at = excluded.updated_at
                    """,
                    (episode_id, reason_content, datetime.utcnow().isoformat()),
                )
            else:
                cursor.execute("DELETE FROM annotation_reasons WHERE episode_id = ?", (episode_id,))

        for episode_info in reviewed_episodes:
            episode_id = episode_info.get("episode_id")
            has_annotation = episode_info.get("has_annotation", False)
            dataset = episode_info.get("dataset", "")
            if episode_id in skipped_legacy_conflicts or not is_factory_dataset_name(dataset):
                continue
            cursor.execute(
                "INSERT OR REPLACE INTO reviewed_episodes (episode_id, user_id, has_annotation, reviewed_at) VALUES (?, ?, ?, ?)",
                (episode_id, user_id, has_annotation, datetime.utcnow().isoformat()),
            )

        effective_reviewed_episodes = [
            ep for ep in reviewed_episodes
            if is_factory_dataset_name(ep.get("dataset", "")) and ep.get("episode_id") not in skipped_legacy_conflicts
        ]

        if session_id:
            cursor.execute("DELETE FROM reviewing_episodes WHERE session_id = ?", (session_id,))
        else:
            cursor.execute("DELETE FROM reviewing_episodes WHERE user_id = ?", (user_id,))

        commit_with_retry(conn)

        alright_count = len([ep for ep in effective_reviewed_episodes if not ep.get("has_annotation")])
        bad_count = len([ep for ep in effective_reviewed_episodes if ep.get("has_annotation")])

        return jsonify({
            "success": True,
            "message": f"已提交 {len(effective_reviewed_episodes)} 个标注 ({alright_count} 正常, {bad_count} 有问题)",
            "total_count": len(effective_reviewed_episodes),
            "alright_count": alright_count,
            "bad_count": bad_count,
            "skipped_legacy_conflicts": skipped_legacy_conflicts[:20],
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


def main():
    init_db()
    should_preload = (not USE_RELOADER) or (os.environ.get("WERKZEUG_RUN_MAIN") == "true")
    if should_preload:
        # sanity-check 若配置了 results.jsonl，则优先预热该缓存；否则走原 factory 缓存。
        if scan_sanity_results_index().get("frames"):
            print("✓ 已预热 sanity results.jsonl 缓存")
        else:
            scan_factory_episodes()
    else:
        print("跳过 reloader 父进程中的 factory 预热，等待实际服务进程启动")
    print(f"启动服务器在端口 {SERVER_PORT}")
    print(f"访问: http://localhost:{SERVER_PORT}")
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=DEBUG_MODE, use_reloader=USE_RELOADER, threaded=True)


if __name__ == "__main__":
    main()
