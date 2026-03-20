import base64
import io
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
    LEGACY_BUILDAI_EPISODE_LIST_FILE = config.get(
        "legacy_buildai_episode_list_file",
        config.get("episode_list_file"),
    )
    LEGACY_BUILDAI_DATASET_NAME = config.get("legacy_buildai_dataset_name", "BuildAI-10k")
    legacy_buildai_annotations_db = config.get("legacy_buildai_annotations_db")
    LEGACY_BUILDAI_ANNOTATIONS_DB = Path(legacy_buildai_annotations_db) if legacy_buildai_annotations_db else None
    source_factory_annotations_db = config.get("source_factory_annotations_db")
    SOURCE_FACTORY_ANNOTATIONS_DB = Path(source_factory_annotations_db) if source_factory_annotations_db else None
    FACTORY_EPISODES_CACHE_FILE = Path(
        config.get(
            "factory_episode_cache_file",
            str(Path(__file__).parent / f".factory_episodes_cache_{FACTORY_START}_{FACTORY_END}.json"),
        )
    )
    default_legacy_cache = (
        LEGACY_BUILDAI_ROOT / "_vla_episodes_cache.json"
        if LEGACY_BUILDAI_ROOT is not None
        else Path(__file__).parent / ".legacy_buildai_episodes_cache.json"
    )
    LEGACY_BUILDAI_EPISODES_CACHE_FILE = Path(
        config.get("legacy_buildai_episode_cache_file", str(default_legacy_cache))
    )

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
_FACTORY_INDEXES: Dict[int, dict] = {}  # fid -> parsed index JSON, loaded on demand
_LEGACY_BUILDAI_EPISODES: Optional[List[Dict]] = None
EPISODE_CACHE_VERSION = 1


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
    global _ALL_EPISODES
    if _ALL_EPISODES is not None and not force_rescan:
        return _ALL_EPISODES

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


def scan_legacy_buildai_episodes(force_rescan: bool = False) -> List[Dict]:
    """扫描旧版 BuildAI raw 数据集，仅供 rework 队列回放历史 flaw 数据。"""
    global _LEGACY_BUILDAI_EPISODES

    if _LEGACY_BUILDAI_EPISODES is not None and not force_rescan:
        return _LEGACY_BUILDAI_EPISODES

    cache_key = {
        "type": "legacy_buildai_raw",
        "legacy_buildai_root": str(LEGACY_BUILDAI_ROOT) if LEGACY_BUILDAI_ROOT is not None else "",
        "legacy_buildai_episode_list_file": LEGACY_BUILDAI_EPISODE_LIST_FILE or "",
        "legacy_buildai_dataset_name": LEGACY_BUILDAI_DATASET_NAME,
    }
    if not force_rescan:
        cached = _load_episode_cache(LEGACY_BUILDAI_EPISODES_CACHE_FILE, cache_key)
        if cached is not None:
            _LEGACY_BUILDAI_EPISODES = cached
            return cached

    episodes: List[Dict] = []

    if LEGACY_BUILDAI_EPISODE_LIST_FILE and os.path.exists(LEGACY_BUILDAI_EPISODE_LIST_FILE):
        print(f"📋 从旧 BuildAI 列表文件加载: {LEGACY_BUILDAI_EPISODE_LIST_FILE}")
        with open(LEGACY_BUILDAI_EPISODE_LIST_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                crop_dir = line[:-4] if line.endswith(".mp4") else line
                crop_path = Path(crop_dir)
                episodes.append({
                    "source_type": "legacy_buildai_raw",
                    "episode_id": crop_path.name,
                    "episode_name": crop_path.name,
                    "dataset_name": LEGACY_BUILDAI_DATASET_NAME,
                    "crop_dir": str(crop_path),
                    "num_frames": -1,  # 延迟到实际加载时统计
                })
        _LEGACY_BUILDAI_EPISODES = episodes
        _save_episode_cache(LEGACY_BUILDAI_EPISODES_CACHE_FILE, cache_key, episodes)
        print(f"✓ 旧 BuildAI 列表加载完成: 共 {len(episodes)} 个 episodes")
        return episodes

    if LEGACY_BUILDAI_ROOT is None:
        _LEGACY_BUILDAI_EPISODES = episodes
        return episodes

    if not LEGACY_BUILDAI_ROOT.exists():
        print(f"⚠ 旧 BuildAI raw 数据集路径不存在: {LEGACY_BUILDAI_ROOT}")
        _LEGACY_BUILDAI_EPISODES = episodes
        return episodes

    print(f"🔍 扫描旧 BuildAI raw 数据集: {LEGACY_BUILDAI_ROOT}")
    for extracted_dir in sorted(LEGACY_BUILDAI_ROOT.glob("*/*/processed/*/extracted_images")):
        crop_dir = extracted_dir.parent
        frame_count = len(list(extracted_dir.glob("*.jpg")))
        if frame_count == 0:
            continue
        episodes.append({
            "source_type": "legacy_buildai_raw",
            "episode_id": crop_dir.name,
            "episode_name": crop_dir.name,
            "dataset_name": LEGACY_BUILDAI_DATASET_NAME,
            "crop_dir": str(crop_dir),
            "num_frames": frame_count,
        })

    _LEGACY_BUILDAI_EPISODES = episodes
    _save_episode_cache(LEGACY_BUILDAI_EPISODES_CACHE_FILE, cache_key, episodes)
    print(f"✓ 旧 BuildAI 扫描完成: 共 {len(episodes)} 个 episodes")
    return episodes


def hydrate_legacy_buildai_episode(ep: Dict) -> Optional[Dict]:
    """补齐旧 BuildAI raw episode 的实际帧数。"""
    if ep.get("num_frames", -1) >= 0:
        return ep

    crop_path = Path(ep["crop_dir"])
    extracted_dir = crop_path / "extracted_images"
    if not extracted_dir.exists():
        return None

    frame_count = len(list(extracted_dir.glob("*.jpg")))
    if frame_count <= 0:
        return None

    hydrated = dict(ep)
    hydrated["num_frames"] = frame_count
    return hydrated


# ─── 图像加载与 YOLO Box 渲染 ─────────────────────────────────────────


def load_track_boxes(seq_folder: str) -> Dict[int, list]:
    """加载 model_tracks.npy，返回 per-frame box 查找表。

    Returns:
        {frame_idx: [(x1, y1, x2, y2, conf, voted_handedness), ...]}
    """
    seq_path = Path(seq_folder)
    frame_boxes: Dict[int, list] = {}

    if not seq_path.exists():
        return frame_boxes

    tracks_dirs = sorted(seq_path.glob("tracks_*"))
    if not tracks_dirs:
        return frame_boxes
    tracks_path = tracks_dirs[0] / "model_tracks.npy"
    if not tracks_path.exists():
        return frame_boxes
    try:
        tracks_data = np.load(str(tracks_path), allow_pickle=True).item()
        for track_id, detections in tracks_data.items():
            all_h = [det["det_handedness"][0] for det in detections]
            voted_h = 1 if sum(1 for h in all_h if h > 0) > len(all_h) / 2 else 0
            for det in detections:
                f = det["frame"]
                box = det["det_box"][0]
                frame_boxes.setdefault(f, []).append(
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]), voted_h)
                )
    except Exception as e:
        print(f"⚠ 加载 model_tracks 失败 ({seq_path.name}): {e}")
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
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=10)
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


def load_episode_frames_from_raw_buildai(
    crop_dir: str,
    frame_indices: List[int],
    max_width: int = 640,
) -> List[str]:
    """从旧 BuildAI raw crop 目录读取帧并绘制 box。"""
    crop_path = Path(crop_dir)
    extracted_dir = crop_path / "extracted_images"
    frame_boxes = load_track_boxes(crop_dir)
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
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
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
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        g._db_conn = conn
    return conn


@app.teardown_appcontext
def close_db_connection(_):
    conn = getattr(g, "_db_conn", None)
    if conn is not None:
        conn.close()


def init_db() -> None:
    """初始化数据库"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
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

        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_view_log_episode ON view_log(episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_view_log_user ON view_log(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_reviewing_episode ON reviewing_episodes(episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_reviewed_episode ON reviewed_episodes(episode_id)",
        ]:
            conn.execute(idx_sql)

        conn.commit()
    finally:
        conn.close()


# ─── 历史「标 X」episode（BAD_FRAMES / BAD_FRAME）与返工标注 ─────────────

REWORK_MAX_FRAMES_PER_EPISODE = 24


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


def parse_rework_v1_payload(content: str) -> Optional[Dict]:
    """解析返工标注 REWORK_V1:{json}，用于回填 UI。"""
    if not content or not content.startswith("REWORK_V1:"):
        return None
    try:
        raw = json.loads(content[len("REWORK_V1:") :])
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
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

    return {"bad_box": _ints(raw.get("bad_box")), "not_clear": _ints(raw.get("not_clear"))}


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


def load_bad_annotation_rows_from_db(db_path: Optional[Path], label: str) -> List[Dict]:
    """读取外部 source flaw 数据库中的 BAD_FRAME(S) 记录。"""
    if db_path is None:
        return []
    if not db_path.exists():
        print(f"⚠ {label} 数据库不存在: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = _fetch_annotation_rows(conn, "content LIKE 'BAD_FRAMES:%' OR content LIKE 'BAD_FRAME:%'")
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"⚠ 读取 {label} 数据库失败: {e}")
        return []
    finally:
        conn.close()


def build_rework_queue_rows(
    local_rows: List[sqlite3.Row],
    source_rows: List[Dict],
    factory_lookup: Dict[str, Dict],
    legacy_buildai_lookup: Dict[str, Dict],
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
        if eid not in factory_lookup and eid not in legacy_buildai_lookup:
            continue
        local_row = local_by_id.get(eid)
        if local_row and str(local_row.get("content", "")).startswith("REWORK_V1:"):
            continue
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
    frame_boxes: Dict[int, list],
    legacy_bad: List[int],
    max_frames: int = REWORK_MAX_FRAMES_PER_EPISODE,
) -> List[int]:
    """合并：历史标错的帧 + 与现逻辑一致的抽样帧，控制上限。"""
    picks = pick_frames_with_boxes(num_frames, frame_boxes)
    legacy_ok = sorted({i for i in legacy_bad if isinstance(i, int) and 0 <= i < num_frames})
    merged = sorted(set(legacy_ok) | set(picks))
    if len(merged) <= max_frames:
        return merged
    out: List[int] = []
    for i in legacy_ok:
        if len(out) >= max_frames:
            break
        out.append(i)
    for p in picks:
        if len(out) >= max_frames:
            break
        if p not in out:
            out.append(p)
    return sorted(out)[:max_frames]


# ─── 路由 ─────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return redirect("/rework")


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
    episodes = scan_factory_episodes()
    return jsonify({
        "success": True,
        "total_episodes": len(episodes),
        "datasets": [{"name": f"factory{FACTORY_START:03d}-{FACTORY_END:03d}", "count": len(episodes)}],
    })


@app.route("/api/episodes/sequential", methods=["GET"])
def api_episodes_sequential():
    """顺序加载 episodes，返回带 YOLO 框的中间帧图像"""
    timer_start = time.time()
    timers: Dict[str, float] = {}

    limit = request.args.get("limit", default=210, type=int)
    global_offset = request.args.get("offset", default=0, type=int)
    user_id = request.args.get("user_id", default="anonymous", type=str)

    # 扫描 episodes
    t0 = time.time()
    all_episodes = scan_factory_episodes()
    timers["scan_episodes"] = time.time() - t0

    total_episodes = len(all_episodes)

    if global_offset >= total_episodes:
        return jsonify({
            "success": False,
            "message": f"Index {global_offset} 超出范围！总共只有 {total_episodes} 个 episodes",
            "total_episodes": total_episodes,
        }), 400

    # 切片
    end_idx = min(global_offset + limit, total_episodes)
    batch = all_episodes[global_offset:end_idx]
    next_offset = end_idx if end_idx < total_episodes else 0

    # 批量查询标注状态
    t1 = time.time()
    conn = get_db_connection()
    episode_ids = [ep["episode_id"] for ep in batch]
    annotation_rows = []
    if episode_ids:
        placeholders = ",".join(["?"] * len(episode_ids))
        annotation_rows = conn.execute(
            f"SELECT episode_id, content FROM annotations WHERE episode_id IN ({placeholders})",
            episode_ids,
        ).fetchall()
    timers["db_query"] = time.time() - t1

    annotation_status: Dict[str, Dict] = {}
    for row in annotation_rows:
        ep_id, content = row[0], row[1]
        if content.startswith("BAD_FRAMES:") or content.startswith("BAD_FRAME:"):
            bad_str = content.replace("BAD_FRAMES:", "").replace("BAD_FRAME:", "")
            bad_frames = [int(x) for x in bad_str.split(",") if x.strip()]
            annotation_status[ep_id] = {"has_annotation": True, "mark_type": "bad", "bad_frames": bad_frames}
        else:
            annotation_status[ep_id] = {"has_annotation": True, "mark_type": "alright", "bad_frames": []}

    fail_reasons = []

    def process_episode(ep: Dict) -> Optional[Dict]:
        """加载单个 episode 的帧并编码"""
        try:
            num_frames = ep["num_frames"]
            if num_frames == 0:
                fail_reasons.append(f"0 frames: {ep['episode_id']}")
                return None

            # 检查 detect_track 是否已完成
            seq_path = Path(ep["seq_folder"])
            stage1_done = (
                seq_path.exists()
                and any(seq_path.glob("tracks_*/model_tracks.npy"))
            )

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
        "episodes": results,
        "has_more": next_offset != 0,
        "next_offset": next_offset,
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
    factory_lookup = {e["episode_id"]: e for e in scan_factory_episodes()}
    legacy_buildai_lookup = {e["episode_id"]: e for e in scan_legacy_buildai_episodes()}
    local_rows = _fetch_annotation_rows(
        conn,
        "(content LIKE 'BAD_FRAMES:%' OR content LIKE 'BAD_FRAME:%' OR content LIKE 'REWORK_V1:%')",
    )
    source_rows = (
        load_bad_annotation_rows_from_db(SOURCE_FACTORY_ANNOTATIONS_DB, "100k flaw source")
        + load_legacy_buildai_annotation_rows()
    )
    rows = build_rework_queue_rows(local_rows, source_rows, factory_lookup, legacy_buildai_lookup)
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
    id_to_ep = {e["episode_id"]: e for e in scan_factory_episodes()}
    legacy_buildai_lookup = {e["episode_id"]: e for e in scan_legacy_buildai_episodes()}
    local_rows = _fetch_annotation_rows(
        conn,
        "(content LIKE 'BAD_FRAMES:%' OR content LIKE 'BAD_FRAME:%' OR content LIKE 'REWORK_V1:%')",
    )
    source_rows = (
        load_bad_annotation_rows_from_db(SOURCE_FACTORY_ANNOTATIONS_DB, "100k flaw source")
        + load_legacy_buildai_annotation_rows()
    )
    bad_rows = build_rework_queue_rows(local_rows, source_rows, id_to_ep, legacy_buildai_lookup)
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

            result: Dict = {
                "success": True,
                "episode_id": ep["episode_id"],
                "episode_name": row.get("episode_name") or ep["episode_name"],
                "dataset_name": row.get("dataset_name") or ep["dataset_name"],
                "episode_index": row.get("episode_index", ep.get("episode_index", 0)),
                "num_frames": num_frames,
                "start_idx": ep.get("start_idx", 0),
                "legacy_bad_frames": parse_legacy_bad_frame_indices(content),
                "rework_annotation": rework_prefill.get(ep_id),
            }

            if ep.get("source_type") == "legacy_buildai_raw":
                frame_boxes = load_track_boxes(ep["crop_dir"])
                legacy_bad = parse_legacy_bad_frame_indices(content)
                frame_indices = build_rework_frame_indices(num_frames, frame_boxes, legacy_bad)
                images = load_episode_frames_from_raw_buildai(ep["crop_dir"], frame_indices)
                if not images:
                    fail_reasons.append(f"legacy raw frames empty for {ep_id}")
                    return None
                result["images"] = images
                result["frame_indices"] = frame_indices
                return result

            seq_path = Path(ep["seq_folder"])
            stage1_done = seq_path.exists() and any(seq_path.glob("tracks_*/model_tracks.npy"))
            if not stage1_done:
                result["images"] = []
                result["frame_indices"] = []
                result["stage1_pending"] = True
                return result

            frame_boxes = load_track_boxes(ep["seq_folder"])
            legacy_bad = parse_legacy_bad_frame_indices(content)
            frame_indices = build_rework_frame_indices(num_frames, frame_boxes, legacy_bad)

            frame_names, frame_offsets = _get_episode_frames(ep)
            if frame_names is None or len(frame_names) == 0:
                fail_reasons.append(f"no frame data for {ep_id}")
                return None

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
        bad_box = item.get("bad_box") or []
        not_clear = item.get("not_clear") or []
        if not isinstance(bad_box, list):
            bad_box = []
        if not isinstance(not_clear, list):
            not_clear = []

        def _norm_ints(seq) -> List[int]:
            o: List[int] = []
            for x in seq:
                try:
                    o.append(int(x))
                except (TypeError, ValueError):
                    continue
            return o

        bb = sorted(set(_norm_ints(bad_box)))
        nc = sorted(set(_norm_ints(not_clear)))
        nc = [i for i in nc if i not in bb]
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
    conn.commit()
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
    conn.commit()
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
        conn.commit()

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
        conn.commit()
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
        conn.commit()

        return jsonify({"success": True, "message": "已取消审核状态"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/sanity-check/submit", methods=["POST"])
def api_sanity_check_submit():
    """提交 sanity check 标注结果"""
    try:
        data = request.json
        marked_frames = data.get("marked_frames", [])
        reviewed_episodes = data.get("reviewed_episodes", [])
        user_id = data.get("user_id", "anonymous")
        session_id = data.get("session_id")

        conn = get_db_connection()
        cursor = conn.cursor()

        for episode_info in reviewed_episodes:
            episode_id = episode_info.get("episode_id")
            dataset = episode_info.get("dataset")
            episode_name = episode_info.get("episode_name")
            episode_index = episode_info.get("episode_index")

            bad_frames = [f.get("frame_index") for f in marked_frames if f.get("episode_id") == episode_id]

            if bad_frames:
                content = "BAD_FRAMES:" + ",".join(map(str, bad_frames))
            else:
                content = "alright"

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
                (episode_id, episode_name, dataset, episode_index, content, datetime.utcnow().isoformat()),
            )

        for episode_info in reviewed_episodes:
            episode_id = episode_info.get("episode_id")
            has_annotation = episode_info.get("has_annotation", False)
            cursor.execute(
                "INSERT OR REPLACE INTO reviewed_episodes (episode_id, user_id, has_annotation, reviewed_at) VALUES (?, ?, ?, ?)",
                (episode_id, user_id, has_annotation, datetime.utcnow().isoformat()),
            )

        if session_id:
            cursor.execute("DELETE FROM reviewing_episodes WHERE session_id = ?", (session_id,))
        else:
            cursor.execute("DELETE FROM reviewing_episodes WHERE user_id = ?", (user_id,))

        conn.commit()

        alright_count = len([ep for ep in reviewed_episodes if not ep.get("has_annotation")])
        bad_count = len([ep for ep in reviewed_episodes if ep.get("has_annotation")])

        return jsonify({
            "success": True,
            "message": f"已提交 {len(reviewed_episodes)} 个标注 ({alright_count} 正常, {bad_count} 有问题)",
            "total_count": len(reviewed_episodes),
            "alright_count": alright_count,
            "bad_count": bad_count,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


def main():
    init_db()
    should_preload = (not USE_RELOADER) or (os.environ.get("WERKZEUG_RUN_MAIN") == "true")
    if should_preload:
        # 启动时预扫描 episodes；若启用 Werkzeug reloader，仅在实际服务进程中执行一次。
        scan_factory_episodes()
    else:
        print("跳过 reloader 父进程中的预扫描，等待实际服务进程启动")
    print(f"启动服务器在端口 {SERVER_PORT}")
    print(f"访问: http://localhost:{SERVER_PORT}")
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=DEBUG_MODE, use_reloader=USE_RELOADER, threaded=True)


if __name__ == "__main__":
    main()
