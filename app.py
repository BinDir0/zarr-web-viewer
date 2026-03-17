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
from flask import Flask, g, jsonify, render_template, request
from PIL import Image, ImageDraw

# 加载配置
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)
    SERVER_PORT = int(config.get("server_port", 9470))
    DEBUG_MODE = bool(config.get("debug_mode", True))
    FACTORY_BASE = config["factory_base"]
    FACTORY_START = int(config["factory_start"])
    FACTORY_END = int(config["factory_end"])

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
    print(f"✓ 扫描完成: factory{FACTORY_START:03d}~factory{FACTORY_END:03d}, 共 {len(episodes)} 个 videos")
    return episodes


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


# ─── 路由 ─────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("sanity_check.html")


@app.route("/sanity-check")
def sanity_check():
    return render_template("sanity_check.html")


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
    # 启动时预扫描 episodes
    scan_factory_episodes()
    print(f"启动服务器在端口 {SERVER_PORT}")
    print(f"访问: http://localhost:{SERVER_PORT}")
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=DEBUG_MODE, threaded=True)


if __name__ == "__main__":
    main()
