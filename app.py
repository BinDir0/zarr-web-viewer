import base64
import io
import os
import sqlite3
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
import zarr
from flask import Flask, abort, g, jsonify, render_template, request, send_file
from PIL import Image

# 翻译相关
import http.client
import json as json_module

# MANO 相关导入
try:
    import torch
    # 尝试导入 MANO
    MANO_AVAILABLE = False
    MANO_LAYER = None
    try:
        # 尝试从常见路径导入 manopth
        mano_paths = [
            '/home/guantianrui/manopth',
            '/home/guantianrui',
        ]
        for path in mano_paths:
            if path not in sys.path and os.path.exists(path):
                sys.path.insert(0, path)
        
        from manopth.manolayer import ManoLayer
        MANO_AVAILABLE = True
        print("✓ MANO 模型已成功加载")
    except ImportError:
        print("⚠️  MANO 未安装，将使用简化的手部渲染")
        MANO_AVAILABLE = False
except ImportError:
    print("⚠️  PyTorch 未安装，将使用简化的手部渲染")
    MANO_AVAILABLE = False

# 创建线程池用于并行处理图像
IMAGE_PROCESSING_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# 加载配置
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)
    SERVER_PORT = int(config.get("server_port", 8300))
    DEBUG_MODE = bool(config.get("debug_mode", True))
    
    # 支持单个或多个数据集
    ZARR_DATASETS = []
    
    # 向后兼容：如果使用旧的单数据集配置
    if "zarr_dataset_path" in config:
        ZARR_DATASETS.append({
            "name": "默认数据集",
            "path": Path(config["zarr_dataset_path"]),
            "enabled": True,
        })
    
    # 新的多数据集配置
    if "zarr_datasets" in config and isinstance(config["zarr_datasets"], list):
        for ds in config["zarr_datasets"]:
            if ds.get("enabled", True):  # 只加载启用的数据集
                ZARR_DATASETS.append({
                    "name": ds.get("name", "未命名数据集"),
                    "path": Path(ds["path"]),
                    "enabled": True,
                })
    
    if not ZARR_DATASETS:
        print("⚠️  警告: 未配置任何数据集，请在 config.yaml 中配置 zarr_datasets")
        ZARR_DATASETS = []

DB_PATH = Path(__file__).parent / "annotations.db"

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)


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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL UNIQUE,
                episode_name TEXT,
                dataset_name TEXT,
                episode_index INTEGER,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        
        # 添加新字段（如果是旧数据库升级）
        try:
            conn.execute("ALTER TABLE annotations ADD COLUMN episode_name TEXT")
        except:
            pass
        try:
            conn.execute("ALTER TABLE annotations ADD COLUMN dataset_name TEXT")
        except:
            pass
        try:
            conn.execute("ALTER TABLE annotations ADD COLUMN episode_index INTEGER")
        except:
            pass
        try:
            conn.execute("ALTER TABLE annotations ADD COLUMN issues TEXT")  # JSON 格式存储问题列表
        except:
            pass
        try:
            conn.execute("ALTER TABLE annotations ADD COLUMN additional_notes TEXT")  # 额外说明
        except:
            pass
        try:
            conn.execute("ALTER TABLE annotations ADD COLUMN is_valid INTEGER DEFAULT 1")  # 1=有效, 0=有问题
        except:
            pass
        try:
            conn.execute("ALTER TABLE annotations ADD COLUMN reviewer_name TEXT")  # 审核人姓名
        except:
            pass
        
        # 查看记录表：记录谁查看了哪个episode
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS view_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                viewed_at TEXT NOT NULL,
                UNIQUE(episode_id, user_id)
            )
            """
        )
        
        # 创建索引加速查询
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_view_log_episode 
            ON view_log(episode_id)
            """
        )
        
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_view_log_user 
            ON view_log(user_id)
            """
        )
        
        conn.commit()
    finally:
        conn.close()


# 全局缓存：避免重复加载数据集
_DATASETS_CACHE = None

def load_zarr_datasets():
    """加载所有启用的zarr数据集（带缓存）"""
    global _DATASETS_CACHE
    
    # 如果已经加载过，直接返回缓存
    if _DATASETS_CACHE is not None:
        return _DATASETS_CACHE
    
    datasets = []
    for ds_config in ZARR_DATASETS:
        path = ds_config["path"]
        name = ds_config["name"]
        
        if not path.exists():
            print(f"⚠️  数据集 '{name}' 路径不存在: {path}")
            continue
        
        try:
            zarr_root = zarr.open(str(path), mode="r")
            datasets.append({
                "name": name,
                "path": str(path),
                "zarr_root": zarr_root,
            })
            print(f"✓ 成功加载数据集: {name} ({path})")
        except Exception as e:
            print(f"✗ 加载数据集 '{name}' 失败: {e}")
    
    # 缓存结果
    _DATASETS_CACHE = datasets
    return datasets


def get_viewed_episodes(user_id: Optional[str] = None) -> set:
    """获取已被查看的episode ID集合"""
    conn = get_db_connection()
    
    if user_id:
        # 只获取该用户查看过的
        rows = conn.execute(
            "SELECT DISTINCT episode_id FROM view_log WHERE user_id = ?",
            (user_id,)
        ).fetchall()
    else:
        # 获取所有被查看过的
        rows = conn.execute(
            "SELECT DISTINCT episode_id FROM view_log"
        ).fetchall()
    
    return {row[0] for row in rows}


def mark_episode_viewed(episode_id: str, user_id: str) -> None:
    """标记episode已被查看"""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO view_log (episode_id, user_id, viewed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(episode_id, user_id) DO UPDATE SET viewed_at = excluded.viewed_at
            """,
            (episode_id, user_id, datetime.utcnow().isoformat())
        )
        conn.commit()
    except Exception as e:
        print(f"标记查看失败: {e}")


# 全局缓存：存储所有 bad episodes
_BAD_EPISODES_CACHE = None

def load_bad_episodes_cache() -> dict:
    """一次性加载所有 bad episodes（启动时调用）
    
    Returns:
        dict: {episode_id: "BAD_FRAMES:N"}
    """
    global _BAD_EPISODES_CACHE
    
    if _BAD_EPISODES_CACHE is not None:
        return _BAD_EPISODES_CACHE
    
    sanity_check_db = Path("/home/guantianrui/zarr-viewer/annotations.db")
    if not sanity_check_db.exists():
        print("⚠️ Sanity check 数据库不存在，跳过自动过滤")
        _BAD_EPISODES_CACHE = {}
        return _BAD_EPISODES_CACHE
    
    try:
        conn = sqlite3.connect(sanity_check_db)
        # 只读取 bad episodes（不是 alright 的）
        rows = conn.execute(
            "SELECT episode_id, content FROM annotations WHERE content != 'alright'"
        ).fetchall()
        conn.close()
        
        _BAD_EPISODES_CACHE = {row[0]: row[1] for row in rows}
        bad_count = len(_BAD_EPISODES_CACHE)
        print(f"✓ 已加载 Sanity Check 结果: {bad_count} 个有问题的 episodes 将被自动过滤")
        
        return _BAD_EPISODES_CACHE
    except Exception as e:
        print(f"⚠️ 加载 sanity check 数据失败: {e}")
        _BAD_EPISODES_CACHE = {}
        return _BAD_EPISODES_CACHE


def check_sanity_check_status(episode_id: str) -> Optional[str]:
    """检查 episode 在 sanity check 数据库中的状态（使用缓存）
    
    Args:
        episode_id: Episode ID (如 "TACO::0")
    
    Returns:
        None: 未找到或状态为 alright
        "BAD_FRAMES:N": 有 N 个坏帧
    """
    bad_episodes = load_bad_episodes_cache()
    return bad_episodes.get(episode_id, None)


_MARKED_BAD_EPISODES = set()  # 记录已经标记过的，避免重复标记

def mark_sanity_check_failed_batch(bad_episodes_list: list):
    """批量标记 bad episodes（在收集完所有需要标记的后调用）"""
    if not bad_episodes_list:
        return
    
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        
        for episode_id, sanity_status, episode_name, dataset_name, episode_index in bad_episodes_list:
            if episode_id in _MARKED_BAD_EPISODES:
                continue  # 已经标记过，跳过
            
            issues = ["sanity_check_failed"]
            additional_notes = f"Sanity check 标记为: {sanity_status}"
            
            conn.execute(
                """
                INSERT INTO annotations (
                    episode_id, episode_name, dataset_name, episode_index, 
                    content, issues, additional_notes, is_valid, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET 
                    issues = excluded.issues,
                    additional_notes = excluded.additional_notes,
                    is_valid = excluded.is_valid,
                    updated_at = excluded.updated_at
                """,
                (episode_id, episode_name, dataset_name, episode_index, 
                 f"自动过滤: {sanity_status}", 
                 json_module.dumps(issues, ensure_ascii=False), 
                 additional_notes, 
                 0,  # is_valid = 0
                 datetime.utcnow().isoformat()),
            )
            _MARKED_BAD_EPISODES.add(episode_id)
        
        conn.commit()
        print(f"✗ 已自动过滤 {len(bad_episodes_list)} 个有问题的 episodes")
    except Exception as e:
        print(f"批量标记 sanity check 失败时出错: {e}")
    finally:
        if conn:
            conn.close()


def get_random_episodes(total_limit: int = 50, user_id: Optional[str] = None, exclude_viewed_by_others: bool = True) -> List[Dict]:
    """随机获取未被查看的episodes（从所有数据集中混合抽取）
    
    Args:
        total_limit: 总共返回多少个 episodes（从所有数据集混合）
        user_id: 当前用户ID
        exclude_viewed_by_others: 是否排除其他人已查看的episodes
    """
    import random
    
    # 获取已查看的episodes
    if exclude_viewed_by_others:
        viewed_episodes = get_viewed_episodes()  # 所有人查看过的
    elif user_id:
        viewed_episodes = get_viewed_episodes(user_id)  # 只该用户查看过的
    else:
        viewed_episodes = set()
    
    # 第一步：收集所有数据集的可用episodes信息（不读取数据，只记录索引）
    all_available_episodes = []  # 存储 (dataset_info, episode_index) 元组
    sanity_check_filtered_count = 0  # 统计被 sanity check 过滤的数量
    bad_episodes_to_mark = []  # 收集需要标记的 bad episodes
    datasets = load_zarr_datasets()
    
    for dataset in datasets:
        dataset_name = dataset["name"]
        zarr_root = dataset["zarr_root"]
        
        try:
            # 从 meta/episode_name 和 meta/episode_ends 读取 episode 信息
            if "meta" not in zarr_root:
                print(f"数据集 '{dataset_name}' 未找到 meta 组")
                continue
            
            meta = zarr_root["meta"]
            
            # 兼容 episode_name 或 episode_names（复数）
            episode_name_key = None
            if "episode_names" in meta:
                episode_name_key = "episode_names"
            elif "episode_name" in meta:
                episode_name_key = "episode_name"
            
            if episode_name_key is None or "episode_ends" not in meta:
                print(f"数据集 '{dataset_name}' 未找到 episode_name(s) 或 episode_ends")
                print(f"  可用的 meta 键: {list(meta.keys())}")
                continue
            
            # 获取总数
            total_episodes = len(meta[episode_name_key])
            
            # 收集该数据集的所有可用episode索引
            for i in range(total_episodes):
                episode_id = f"{dataset_name}::{i}"
                
                # 检查 sanity check 状态（使用缓存，非常快）
                sanity_status = check_sanity_check_status(episode_id)
                if sanity_status:
                    # 有问题，收集待标记，稍后批量标记
                    episode_name_str = ""
                    try:
                        episode_name = meta[episode_name_key][i]
                        episode_name_str = episode_name if isinstance(episode_name, str) else episode_name.decode('utf-8')
                    except:
                        pass
                    
                    bad_episodes_to_mark.append((episode_id, sanity_status, episode_name_str, dataset_name, i))
                    sanity_check_filtered_count += 1
                    continue  # 跳过这个 episode
                
                if episode_id not in viewed_episodes:
                    # 记录：(数据集信息, episode索引, meta引用)
                    all_available_episodes.append({
                        'dataset_name': dataset_name,
                        'dataset_path': dataset['path'],
                        'zarr_root': zarr_root,
                        'meta': meta,
                        'episode_name_key': episode_name_key,
                        'episode_index': i,
                        'total_in_dataset': total_episodes,
                    })
        
        except Exception as e:
            print(f"获取数据集 '{dataset_name}' 的episodes失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 第二步：如果没有未查看的，就从所有中选择
    if not all_available_episodes:
        print("所有episodes都已被查看，从全部数据集中随机选择")
        for dataset in datasets:
            dataset_name = dataset["name"]
            zarr_root = dataset["zarr_root"]
            
            try:
                if "meta" not in zarr_root:
                    continue
                
                meta = zarr_root["meta"]
                episode_name_key = None
                if "episode_names" in meta:
                    episode_name_key = "episode_names"
                elif "episode_name" in meta:
                    episode_name_key = "episode_name"
                
                if episode_name_key is None:
                    continue
                
                total_episodes = len(meta[episode_name_key])
                
                for i in range(total_episodes):
                    episode_id = f"{dataset_name}::{i}"
                    
                    # 检查 sanity check 状态
                    sanity_status = check_sanity_check_status(episode_id)
                    if sanity_status:
                        # 有问题，跳过（已经在第一步标记过了，这里只需跳过）
                        continue
                    
                    all_available_episodes.append({
                        'dataset_name': dataset_name,
                        'dataset_path': dataset['path'],
                        'zarr_root': zarr_root,
                        'meta': meta,
                        'episode_name_key': episode_name_key,
                        'episode_index': i,
                        'total_in_dataset': total_episodes,
                    })
            except Exception as e:
                print(f"重新扫描数据集 '{dataset_name}' 失败: {e}")
    
    # 第三步：从所有可用episodes中随机采样
    sample_size = min(total_limit, len(all_available_episodes))
    if sample_size == 0:
        return []
    
    sampled_episodes_info = random.sample(all_available_episodes, sample_size)
    
    # 第四步：读取采样的episodes的实际数据
    all_episodes = []
    for ep_info in sampled_episodes_info:
        try:
            i = ep_info['episode_index']
            dataset_name = ep_info['dataset_name']
            meta = ep_info['meta']
            episode_name_key = ep_info['episode_name_key']
            
            episode_name = meta[episode_name_key][i]
            ep_end_idx = meta["episode_ends"][i]
            
            # 计算起始索引
            if i == 0:
                ep_start_idx = 0
            else:
                ep_start_idx = int(meta["episode_ends"][i - 1])
            
            num_frames = int(ep_end_idx - ep_start_idx)
            
            episode_name_str = episode_name if isinstance(episode_name, str) else episode_name.decode('utf-8')
            
            # 创建唯一的 episode ID
            episode_id = f"{dataset_name}::{i}"
            
            all_episodes.append({
                "id": episode_id,
                "name": episode_name_str,
                "dataset": dataset_name,
                "dataset_index": i,
                "start_idx": int(ep_start_idx),
                "end_idx": int(ep_end_idx),
                "num_frames": num_frames,
                "total_in_dataset": ep_info['total_in_dataset'],
            })
        except Exception as e:
            print(f"读取episode数据失败: {e}")
    
    # 批量标记 bad episodes（如果有的话）
    if bad_episodes_to_mark:
        mark_sanity_check_failed_batch(bad_episodes_to_mark)
        print(f"🔍 Sanity Check: 共过滤了 {sanity_check_filtered_count} 个有问题的 episodes")
    
    return all_episodes


def get_episode_data(episode_id: str) -> Optional[Dict]:
    """获取单个episode的数据"""
    try:
        # 解析 episode_id: dataset_name::episode_index
        if "::" not in episode_id:
            print(f"无效的 episode_id 格式: {episode_id}")
            return None
        
        dataset_name, episode_index = episode_id.split("::", 1)
        
        # 找到对应的数据集
        datasets = load_zarr_datasets()
        target_dataset = None
        for ds in datasets:
            if ds["name"] == dataset_name:
                target_dataset = ds
                break
        
        if target_dataset is None:
            print(f"未找到数据集: {dataset_name}")
            return None
        
        zarr_root = target_dataset["zarr_root"]
        
        # 直接从 meta 中读取 episode 信息
        if "meta" not in zarr_root:
            print(f"数据集 '{dataset_name}' 未找到 meta 组")
            return None
        
        meta = zarr_root["meta"]
        
        # 获取 episode_name 字段名
        episode_name_key = None
        if "episode_names" in meta:
            episode_name_key = "episode_names"
        elif "episode_name" in meta:
            episode_name_key = "episode_name"
        
        if episode_name_key is None or "episode_ends" not in meta:
            print(f"数据集 '{dataset_name}' 未找到必要的 meta 字段")
            return None
        
        # 解析 episode_index
        try:
            episode_idx = int(episode_index)
        except (ValueError, TypeError):
            print(f"无效的 episode_index: {episode_index}")
            return None
        
        # 检查索引是否有效
        total_episodes = len(meta[episode_name_key])
        if episode_idx < 0 or episode_idx >= total_episodes:
            print(f"episode_index {episode_idx} 超出范围 [0, {total_episodes})")
            return None
        
        # 读取该 episode 的信息
        episode_name = meta[episode_name_key][episode_idx]
        end_idx = int(meta["episode_ends"][episode_idx])
        
        # 计算起始索引
        if episode_idx == 0:
            start_idx = 0
        else:
            start_idx = int(meta["episode_ends"][episode_idx - 1])
        
        episode_name_str = episode_name if isinstance(episode_name, str) else episode_name.decode('utf-8')
        
        # 获取 data 组
        if "data" not in zarr_root:
            print("未找到 data 组")
            return None
        
        data = zarr_root["data"]
        
        # 计算帧数
        num_frames = end_idx - start_idx
        
        # 提取该 episode 的数据切片
        result = {
            "id": episode_id,
            "name": episode_name_str,
            "dataset": dataset_name,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_frames": num_frames,
        }
        
        # 读取所有图像帧（用于生成视频）
        if "image" in data:
            result["images"] = data["image"][start_idx:end_idx]
        else:
            result["images"] = None
        
        # 读取深度数据的形状（用于确定目标分辨率，不读取实际数据）
        if "depth" in data:
            result["depth_shape"] = data["depth"].shape
        else:
            result["depth_shape"] = None
        
        # 读取 instruction 数据
        if "instruction" in data:
            result["instructions"] = data["instruction"][start_idx:end_idx]
        if "instruction_num" in data:
            result["instruction_num"] = data["instruction_num"][start_idx:end_idx]
        
        # 读取所有帧的 action/state 数据
        if "state" in data:
            state_group = data["state"]
            result["state"] = {}
            for key in state_group.keys():
                result["state"][key] = state_group[key][start_idx:end_idx]
        
        if "action" in data:
            action_group = data["action"]
            result["action"] = {}
            for key in action_group.keys():
                result["action"][key] = action_group[key][start_idx:end_idx]
        
        # 读取手的可见性
        if "presence" in data:
            result["presence"] = data["presence"][start_idx:end_idx]
        
        # 读取相机参数
        if "intrinsic" in data:
            result["intrinsic"] = data["intrinsic"][start_idx:end_idx]
        if "extrinsic" in data:
            result["extrinsic"] = data["extrinsic"][start_idx:end_idx]
        
        return result
        
    except Exception as e:
        print(f"获取episode数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_middle_frame_indices(start_idx: int, end_idx: int, num_frames: int = 2):
    """计算需要读取的中间帧索引
    
    Args:
        start_idx: episode 起始索引
        end_idx: episode 结束索引
        num_frames: 需要的帧数
    
    Returns:
        需要读取的帧索引列表（绝对索引）
    """
    total_frames = end_idx - start_idx
    if total_frames <= 0:
        return []
    
    if total_frames <= num_frames:
        return list(range(start_idx, end_idx))
    
    # 均匀分布在中间区域
    middle_start = start_idx + total_frames // 4
    middle_end = start_idx + 3 * total_frames // 4
    indices = np.linspace(middle_start, middle_end, num_frames, dtype=int).tolist()
    
    return indices


# ========== MANO 和渲染相关的辅助函数 ==========

def rot6_to_rotmat(r6: np.ndarray) -> np.ndarray:
    """将6D旋转表示转换为3x3旋转矩阵"""
    a1 = r6[:3]
    a2 = r6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)
    return R.astype(np.float32)


def rotmat_to_axisangle(R: np.ndarray) -> np.ndarray:
    """将旋转矩阵转换为轴角表示"""
    R = R.astype(np.float32)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros((3,), dtype=np.float32)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return (axis * theta).astype(np.float32)


def project_points(pts_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """将3D点投影到2D图像平面"""
    zs = pts_cam[:, 2] + 1e-8
    us = fx * (pts_cam[:, 0] / zs) + cx
    vs = fy * (pts_cam[:, 1] / zs) + cy
    return np.stack([us, vs], axis=1)


def parse_intrinsics(intrinsics):
    """解析相机内参"""
    if intrinsics is None:
        return None
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        return intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    if intrinsics.shape == (9,):
        intrinsics = intrinsics.reshape(3, 3)
        return intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    if intrinsics.shape == (4,):
        return intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    return None


def generate_mano_mesh(mano_layer, pose_params, global_r_aa, trans, shape_params=None):
    """使用MANO模型生成手部网格和关节
    
    Args:
        mano_layer: MANO layer instance
        pose_params: 手指姿态参数 (15 or 45 维)
        global_r_aa: 全局旋转（轴角表示）
        trans: 平移向量
        shape_params: 手部形状参数（10维，可选）
    
    Returns:
        verts_np: 顶点坐标 (778, 3)
        joints_np: 关节坐标 (21, 3)
    """
    if not MANO_AVAILABLE or mano_layer is None:
        return None, None
    
    # 构造完整的姿态参数
    if pose_params.shape[-1] == 15:
        theta = np.concatenate([global_r_aa.reshape(1, 3), pose_params.reshape(1, 15)], axis=1)
    else:
        theta = np.concatenate([global_r_aa.reshape(1, 3), pose_params.reshape(1, 45)], axis=1)
    theta_t = torch.from_numpy(theta).float()
    
    # 形状参数
    if shape_params is not None:
        beta_t = torch.from_numpy(shape_params.reshape(1, 10)).float()
    else:
        beta_t = torch.zeros((1, 10), dtype=torch.float32)
    
    # MANO 推理
    with torch.no_grad():
        verts, joints = mano_layer(theta_t, beta_t)
        # 转换单位并添加平移
        verts_np = verts.detach().cpu().numpy()[0] / 1000.0 + trans.reshape(1, 3)
        joints_np = joints.detach().cpu().numpy()[0] / 1000.0 + trans.reshape(1, 3)
    
    return verts_np.astype(np.float32), joints_np.astype(np.float32)


def cleanup_video_cache(cache_dir: str, max_size_gb: float = 10.0, max_age_days: int = 7):
    """清理视频缓存目录
    
    策略：
    1. 删除超过 max_age_days 天的文件
    2. 如果总大小超过 max_size_gb，删除最旧的文件直到低于限制
    
    Args:
        cache_dir: 缓存目录路径
        max_size_gb: 最大缓存大小（GB）
        max_age_days: 文件最大保留天数
    """
    if not os.path.exists(cache_dir):
        return
    
    import time
    
    current_time = time.time()
    max_age_seconds = max_age_days * 86400  # 转换为秒
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024  # 转换为字节
    
    # 收集所有缓存文件信息
    files_info = []
    total_size = 0
    
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        if os.path.isfile(filepath):
            try:
                stat = os.stat(filepath)
                file_age = current_time - stat.st_mtime
                file_size = stat.st_size
                files_info.append({
                    'path': filepath,
                    'age': file_age,
                    'size': file_size,
                    'mtime': stat.st_mtime
                })
                total_size += file_size
            except Exception as e:
                print(f"⚠️ 无法读取文件信息: {filepath}, {e}")
    
    deleted_count = 0
    freed_size = 0
    
    # 策略1: 删除过期文件
    for file_info in files_info[:]:
        if file_info['age'] > max_age_seconds:
            try:
                os.remove(file_info['path'])
                deleted_count += 1
                freed_size += file_info['size']
                total_size -= file_info['size']
                files_info.remove(file_info)
                print(f"🗑️ 删除过期缓存: {os.path.basename(file_info['path'])} (保留了 {file_info['age']/86400:.1f} 天)")
            except Exception as e:
                print(f"⚠️ 删除文件失败: {file_info['path']}, {e}")
    
    # 策略2: 如果总大小超限，删除最旧的文件
    if total_size > max_size_bytes:
        # 按修改时间排序（最旧的在前）
        files_info.sort(key=lambda x: x['mtime'])
        
        for file_info in files_info:
            if total_size <= max_size_bytes * 0.8:  # 清理到 80% 以下
                break
            try:
                os.remove(file_info['path'])
                deleted_count += 1
                freed_size += file_info['size']
                total_size -= file_info['size']
                print(f"🗑️ 删除旧缓存（空间不足）: {os.path.basename(file_info['path'])}")
            except Exception as e:
                print(f"⚠️ 删除文件失败: {file_info['path']}, {e}")
    
    if deleted_count > 0:
        print(f"✓ 缓存清理完成: 删除 {deleted_count} 个文件，释放 {freed_size/(1024*1024):.2f} MB")
        print(f"  当前缓存大小: {total_size/(1024*1024):.2f} MB / {max_size_gb*1024:.2f} MB")


def convert_video_to_h264(input_path: str, output_path: str, quality_mode: str = 'fast'):
    """使用 ffmpeg 将视频转换为 H.264 编码
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        quality_mode: 编码模式 'fast' | 'balanced' | 'quality'
    """
    import subprocess
    
    try:
        # 所有模式都使用 -g 3（每 3 帧一个关键帧），兼顾 seek 速度和文件大小
        # 视频很短（~5s），文件增大有限
        if quality_mode == 'fast':
            # 快速模式：原始视频使用
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '25',
                '-g', '3', '-keyint_min', '3',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
        elif quality_mode == 'quality':
            # 质量模式：渲染视频使用
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',     # 更好的质量
                '-crf', '20',            # 高质量
                '-g', '3', '-keyint_min', '3',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
        else:  # balanced
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-g', '3', '-keyint_min', '3',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60秒超时
        )
        
        if result.returncode != 0:
            print(f"FFmpeg 转换失败: {result.stderr}")
            return False
        
        return True
    except subprocess.TimeoutExpired:
        print("FFmpeg 转换超时")
        return False
    except Exception as e:
        print(f"FFmpeg 转换错误: {e}")
        return False


def resize_frame_to_target(frame, target_width=None, target_height=None, max_width=512):
    """Resize 视频帧到目标分辨率（参考 zarr-viewer 的逻辑）
    
    Args:
        frame: 输入帧 (numpy array)
        target_width: 目标宽度（来自 depth）
        target_height: 目标高度（来自 depth）
        max_width: 最大宽度限制
    
    Returns:
        调整大小后的帧
    """
    current_height, current_width = frame.shape[:2]
    
    # 决定目标尺寸
    if target_width and target_height:
        # 如果指定了目标宽高（来自depth），使用这个分辨率
        if target_width > max_width:
            scale = max_width / target_width
            resize_width = max_width
            resize_height = int(target_height * scale)
        else:
            resize_width = target_width
            resize_height = target_height
    else:
        # 否则保持原始宽高比
        if current_width > max_width:
            resize_width = max_width
            resize_height = int(current_height * (max_width / current_width))
        else:
            resize_width = current_width
            resize_height = current_height
    
    # 只在需要时才缩放
    if (resize_width, resize_height) != (current_width, current_height):
        resized = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        return resized
    
    return frame


def create_video_from_frames(frames, output_path: str, fps: int = 30, 
                            target_width=None, target_height=None, max_width: int = 512):
    """从帧序列创建视频
    
    Args:
        frames: 图像帧数组列表或numpy数组
        output_path: 输出视频路径
        fps: 帧率
        target_width: 目标宽度（来自 depth）
        target_height: 目标高度（来自 depth）
        max_width: 最大宽度限制
    """
    # 处理 numpy 数组或列表
    if isinstance(frames, np.ndarray):
        if frames.shape[0] == 0:
            raise ValueError("No frames to create video")
    elif not frames or len(frames) == 0:
        raise ValueError("No frames to create video")
    
    # 获取第一帧并 resize
    first_frame = frames[0]
    if first_frame.dtype == np.float32 or first_frame.dtype == np.float64:
        first_frame = (first_frame * 255).astype(np.uint8)
    elif first_frame.dtype != np.uint8:
        first_frame = first_frame.astype(np.uint8)
    
    # Resize 第一帧以获取目标尺寸
    first_frame = resize_frame_to_target(first_frame, target_width, target_height, max_width)
    height, width = first_frame.shape[:2]
    
    print(f"视频分辨率: {width}x{height}")
    
    # 尝试多个编码器以确保兼容性（按优先级排序）
    codecs_to_try = [
        ('mp4v', 'MPEG-4'),           # 最兼容，几乎所有系统都支持
        ('XVID', 'Xvid'),             # 备选 MPEG-4
        ('X264', 'H.264'),            # H.264 软件编码
        ('avc1', 'H.264 AVC'),        # H.264 硬件加速
        ('H264', 'H.264'),            # H.264 通用
    ]
    
    out = None
    used_codec = None
    for codec, name in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                used_codec = f"{codec} ({name})"
                print(f"✓ 使用视频编码器: {used_codec}")
                break
            else:
                out.release()
                out = None
        except Exception as e:
            continue
    
    if out is None or not out.isOpened():
        raise RuntimeError(f"无法创建视频写入器，尝试了所有编码器: {output_path}")
    
    try:
        # 确定帧数
        num_frames = frames.shape[0] if isinstance(frames, np.ndarray) else len(frames)
        
        for i in range(num_frames):
            frame = frames[i]
            
            # 确保数据类型正确
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            elif frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Resize 帧到目标分辨率
            frame = resize_frame_to_target(frame, target_width, target_height, max_width)
            
            # 转换为BGR（OpenCV格式）
            if len(frame.shape) == 2:
                # 灰度图转BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 3:
                # RGB转BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.shape[2] == 4:
                # RGBA转BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            out.write(frame)
    finally:
        out.release()
    
    # 记录视频生成成功
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        codec_info = f" [{used_codec}]" if used_codec else ""
        print(f"✓ 视频生成成功: {output_path} ({file_size:.2f} MB, {num_frames} 帧){codec_info}")


def render_hand_on_frame(frame, mano_params=None, wrist_params=None, extrinsic=None, 
                        intrinsic=None, presence=None, shape_params=None, mano_layers=None,
                        fingertips=None):
    """在帧上渲染手部动作（参考 visualize_episode_video.py 的实现）
    
    Args:
        frame: 原始图像帧（RGB 格式）
        mano_params: MANO参数 (dict with 'left' and 'right')
        wrist_params: 手腕参数 (dict with translations and rotations)
        extrinsic: 相机外参 (4x4矩阵)
        intrinsic: 相机内参 (3x3矩阵或4维向量)
        presence: 手的可见性 (0=无, 1=左手, 2=右手, 3=双手)
        shape_params: 形状参数 (dict with 'left' and 'right', 可选)
        mano_layers: MANO模型层 (dict with 'left' and 'right', 可选)
        fingertips: 指尖位置 (dict with 'left' and 'right', 每个是15维的3D坐标)
    
    Returns:
        渲染后的图像帧（RGB 格式，用于视频生成）
    """
    # 确保图像格式正确
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        frame = (frame * 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    # 转换为BGR格式（OpenCV绘图函数需要 BGR）
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
    H, W = frame.shape[:2]
    
    # 检查手的可见性
    L_present = True
    R_present = True
    if presence is not None:
        L_present = presence in [1, 3]
        R_present = presence in [2, 3]
    
    # 解析相机内参
    fx, fy, cx, cy = 384.0, 384.0, 192.0, 192.0
    if intrinsic is not None:
        parsed = parse_intrinsics(intrinsic)
        if parsed:
            fx, fy, cx, cy = parsed
    
    # 解析相机外参
    if extrinsic is not None:
        if extrinsic.shape == (16,):
            extr_mat = extrinsic.reshape(4, 4)
        else:
            extr_mat = extrinsic
        R_wc = extr_mat[:3, :3]
        t_wc = extr_mat[:3, 3]
    else:
        R_wc = np.eye(3, dtype=np.float32)
        t_wc = np.zeros(3, dtype=np.float32)
    
    # ========== 预计算：检测手部标注是否超出画面范围 ==========
    all_projected_points = []
    cached_mano_data = {}  # 缓存MANO计算结果以避免重复计算
    
    # 如果有MANO模型，收集所有需要渲染的点
    if MANO_AVAILABLE and mano_layers is not None and mano_params is not None and wrist_params is not None:
        # 计算左手的投影点
        if L_present and 'left' in mano_params and 'left_translation' in wrist_params:
            try:
                L_pose = mano_params['left']
                Lt3 = wrist_params['left_translation']
                L_rot6 = wrist_params['left_rotation']
                L_rotmat = rot6_to_rotmat(L_rot6)
                L_aa = rotmat_to_axisangle(L_rotmat)
                L_shape = shape_params.get('left') if shape_params else None
                
                L_verts_world, L_joints_world = generate_mano_mesh(
                    mano_layers['left'], L_pose, L_aa, Lt3, L_shape
                )
                
                if L_verts_world is not None:
                    # 缓存计算结果
                    cached_mano_data['left'] = {
                        'verts_world': L_verts_world,
                        'joints_world': L_joints_world
                    }
                    
                    verts_cam = (R_wc @ L_verts_world.T).T + t_wc
                    uv_verts = project_points(verts_cam, fx, fy, cx, cy)
                    # 只收集有效深度的点
                    valid_mask = verts_cam[:, 2] > 1e-6
                    all_projected_points.extend(uv_verts[valid_mask].tolist())
            except Exception as e:
                pass  # 忽略错误，继续处理
        
        # 计算右手的投影点
        if R_present and 'right' in mano_params and 'right_translation' in wrist_params:
            try:
                R_pose = mano_params['right']
                Rt3 = wrist_params['right_translation']
                R_rot6 = wrist_params['right_rotation']
                R_rotmat = rot6_to_rotmat(R_rot6)
                R_aa = rotmat_to_axisangle(R_rotmat)
                R_shape = shape_params.get('right') if shape_params else None
                
                R_verts_world, R_joints_world = generate_mano_mesh(
                    mano_layers['right'], R_pose, R_aa, Rt3, R_shape
                )
                
                if R_verts_world is not None:
                    # 缓存计算结果
                    cached_mano_data['right'] = {
                        'verts_world': R_verts_world,
                        'joints_world': R_joints_world
                    }
                    
                    verts_cam = (R_wc @ R_verts_world.T).T + t_wc
                    uv_verts = project_points(verts_cam, fx, fy, cx, cy)
                    valid_mask = verts_cam[:, 2] > 1e-6
                    all_projected_points.extend(uv_verts[valid_mask].tolist())
            except Exception as e:
                pass
    
    # 计算是否需要缩放画面
    scale_factor = 1.0
    offset_x = 0
    offset_y = 0
    new_W = W
    new_H = H
    img_overlay = frame.copy()
    
    if len(all_projected_points) > 0:
        pts = np.array(all_projected_points)
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        
        # 检查是否超出边界（留10像素边距）
        margin = 10
        out_of_bounds = (min_x < margin or min_y < margin or 
                        max_x > W - margin or max_y > H - margin)
        
        if out_of_bounds:
            # 计算需要的画布尺寸
            required_w = max(max_x - min_x + 2 * margin, W)
            required_h = max(max_y - min_y + 2 * margin, H)
            
            # 计算缩放比例（保持原始画面宽高比）
            scale_w = W / required_w
            scale_h = H / required_h
            scale_factor = min(scale_w, scale_h, 0.85)  # 最多缩小到85%
            
            # 计算新的画布尺寸和偏移
            scaled_w = int(W * scale_factor)
            scaled_h = int(H * scale_factor)
            new_W = W
            new_H = H
            offset_x = (new_W - scaled_w) // 2
            offset_y = (new_H - scaled_h) // 2
            
            # 缩放原始图像并放在新画布中央（使用深灰色背景以突出显示缩放效果）
            scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            # 创建深灰色背景 (40, 40, 40)
            img_overlay = np.full((new_H, new_W, 3), 40, dtype=np.uint8)
            img_overlay[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w] = scaled_frame
            
            # 调整相机内参以匹配缩放
            fx = fx * scale_factor
            fy = fy * scale_factor
            cx = cx * scale_factor + offset_x
            cy = cy * scale_factor + offset_y
            
            # 更新画面尺寸
            H, W = new_H, new_W
    
    # 如果有MANO模型且有必要参数，使用完整渲染
    if MANO_AVAILABLE and mano_layers is not None and mano_params is not None and wrist_params is not None:
        # 定义MANO关节树（手指连接关系）
        joint_tree = [
            [(0, 1), (1, 2), (2, 3), (3, 4)],      # 大拇指
            [(0, 5), (5, 6), (6, 7), (7, 8)],      # 食指
            [(0, 9), (9, 10), (10, 11), (11, 12)], # 中指
            [(0, 13), (13, 14), (14, 15), (15, 16)], # 无名指
            [(0, 17), (17, 18), (18, 19), (19, 20)]  # 小指
        ]
        
        # 创建半透明层用于点云混合
        overlay_alpha = img_overlay.copy()
        
        # 渲染左手
        if L_present and 'left' in mano_params and 'left_translation' in wrist_params:
            # 使用缓存的MANO数据（如果有），否则重新计算
            if 'left' in cached_mano_data:
                L_verts_world = cached_mano_data['left']['verts_world']
                L_joints_world = cached_mano_data['left']['joints_world']
            else:
                # 提取参数
                L_pose = mano_params['left']
                Lt3 = wrist_params['left_translation']
                L_rot6 = wrist_params['left_rotation']
                
                # 转换旋转
                L_rotmat = rot6_to_rotmat(L_rot6)
                L_aa = rotmat_to_axisangle(L_rotmat)
                
                # 获取形状参数
                L_shape = shape_params.get('left') if shape_params else None
                
                # 生成MANO网格和关节
                L_verts_world, L_joints_world = generate_mano_mesh(
                    mano_layers['left'], L_pose, L_aa, Lt3, L_shape
                )
            
            if L_verts_world is not None and L_joints_world is not None:
                # 转换到相机坐标系
                verts_cam = (R_wc @ L_verts_world.T).T + t_wc
                joints_cam = (R_wc @ L_joints_world.T).T + t_wc
                
                # 投影到图像平面
                uv_verts = project_points(verts_cam, fx, fy, cx, cy)
                uv_joints = project_points(joints_cam, fx, fy, cx, cy).astype(np.int32)
                
                # 绘制点云（小点，半透明）- 青色
                mask_verts = (verts_cam[:, 2] > 1e-6) & \
                            (uv_verts[:, 0] >= 0) & (uv_verts[:, 0] < W) & \
                            (uv_verts[:, 1] >= 0) & (uv_verts[:, 1] < H)
                for pt in uv_verts[mask_verts].astype(np.int32):
                    cv2.circle(overlay_alpha, tuple(pt), 1, (255, 255, 0), -1)
                
                # 应用半透明混合
                cv2.addWeighted(overlay_alpha, 0.6, img_overlay, 0.4, 0, img_overlay)
                
                # 绘制关节连线（骨架）- 亮蓝色
                mask_joints = (joints_cam[:, 2] > 1e-6) & \
                             (uv_joints[:, 0] >= 0) & (uv_joints[:, 0] < W) & \
                             (uv_joints[:, 1] >= 0) & (uv_joints[:, 1] < H)
                for finger_chain in joint_tree:
                    for (j1, j2) in finger_chain:
                        if mask_joints[j1] and mask_joints[j2]:
                            cv2.line(img_overlay, tuple(uv_joints[j1]), tuple(uv_joints[j2]), 
                                   (255, 100, 0), 2)
                
                # 绘制关节点 - 深蓝色
                for i in range(21):
                    if mask_joints[i]:
                        cv2.circle(img_overlay, tuple(uv_joints[i]), 2, (255, 0, 0), -1)
                        cv2.circle(img_overlay, tuple(uv_joints[i]), 3, (255, 255, 255), 1)
        
        # 渲染右手
        if R_present and 'right' in mano_params and 'right_translation' in wrist_params:
            # 使用缓存的MANO数据（如果有），否则重新计算
            if 'right' in cached_mano_data:
                R_verts_world = cached_mano_data['right']['verts_world']
                R_joints_world = cached_mano_data['right']['joints_world']
            else:
                # 提取参数
                R_pose = mano_params['right']
                Rt3 = wrist_params['right_translation']
                R_rot6 = wrist_params['right_rotation']
                
                # 转换旋转
                R_rotmat = rot6_to_rotmat(R_rot6)
                R_aa = rotmat_to_axisangle(R_rotmat)
                
                # 获取形状参数
                R_shape = shape_params.get('right') if shape_params else None
                
                # 生成MANO网格和关节
                R_verts_world, R_joints_world = generate_mano_mesh(
                    mano_layers['right'], R_pose, R_aa, Rt3, R_shape
                )
            
            if R_verts_world is not None and R_joints_world is not None:
                # 转换到相机坐标系
                verts_cam = (R_wc @ R_verts_world.T).T + t_wc
                joints_cam = (R_wc @ R_joints_world.T).T + t_wc
                
                # 投影到图像平面
                uv_verts = project_points(verts_cam, fx, fy, cx, cy)
                uv_joints = project_points(joints_cam, fx, fy, cx, cy).astype(np.int32)
                
                # 绘制点云（小点，半透明）- 黄色
                mask_verts = (verts_cam[:, 2] > 1e-6) & \
                            (uv_verts[:, 0] >= 0) & (uv_verts[:, 0] < W) & \
                            (uv_verts[:, 1] >= 0) & (uv_verts[:, 1] < H)
                overlay_alpha2 = img_overlay.copy()
                for pt in uv_verts[mask_verts].astype(np.int32):
                    cv2.circle(overlay_alpha2, tuple(pt), 1, (0, 255, 255), -1)
                
                # 应用半透明混合
                cv2.addWeighted(overlay_alpha2, 0.6, img_overlay, 0.4, 0, img_overlay)
                
                # 绘制关节连线（骨架）- 亮红色
                mask_joints = (joints_cam[:, 2] > 1e-6) & \
                             (uv_joints[:, 0] >= 0) & (uv_joints[:, 0] < W) & \
                             (uv_joints[:, 1] >= 0) & (uv_joints[:, 1] < H)
                for finger_chain in joint_tree:
                    for (j1, j2) in finger_chain:
                        if mask_joints[j1] and mask_joints[j2]:
                            cv2.line(img_overlay, tuple(uv_joints[j1]), tuple(uv_joints[j2]), 
                                   (0, 100, 255), 2)
                
                # 绘制关节点 - 深红色
                for i in range(21):
                    if mask_joints[i]:
                        cv2.circle(img_overlay, tuple(uv_joints[i]), 2, (0, 0, 255), -1)
                        cv2.circle(img_overlay, tuple(uv_joints[i]), 3, (255, 255, 255), 1)
    
    # === 额外渲染：数据集中的 fingertips（无论是否有MANO） ===
    if fingertips is not None:
        finger_names = ['拇指', '食指', '中指', '无名指', '小指']
        
        # 渲染左手指尖（使用不同的颜色和样式以区分）
        if L_present and 'left' in fingertips:
            left_tips = np.array(fingertips['left']).reshape(5, 3)
            tips_cam = (R_wc @ left_tips.T).T + t_wc
            uv_tips = project_points(tips_cam, fx, fy, cx, cy).astype(np.int32)
            mask_tips = (tips_cam[:, 2] > 1e-6) & \
                       (uv_tips[:, 0] >= 0) & (uv_tips[:, 0] < W) & \
                       (uv_tips[:, 1] >= 0) & (uv_tips[:, 1] < H)
            
            # 绘制指尖点 - 使用特殊样式（带边框的圆）
            for i in range(5):
                if mask_tips[i]:
                    pt = tuple(uv_tips[i])
                    # 缩小点的大小：外圈从7改为3，内圈从4改为2
                    cv2.circle(img_overlay, pt, 3, (0, 255, 0), 1)  # 外圈：绿色，线宽1
                    cv2.circle(img_overlay, pt, 2, (255, 255, 0), -1)  # 内圈：青色填充
                    # 标签：显示指尖名称（字体稍小）
                    cv2.putText(img_overlay, f"L{i}", (pt[0]+5, pt[1]-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # 渲染右手指尖
        if R_present and 'right' in fingertips:
            right_tips = np.array(fingertips['right']).reshape(5, 3)
            tips_cam = (R_wc @ right_tips.T).T + t_wc
            uv_tips = project_points(tips_cam, fx, fy, cx, cy).astype(np.int32)
            mask_tips = (tips_cam[:, 2] > 1e-6) & \
                       (uv_tips[:, 0] >= 0) & (uv_tips[:, 0] < W) & \
                       (uv_tips[:, 1] >= 0) & (uv_tips[:, 1] < H)
            
            # 绘制指尖点 - 使用特殊样式（带边框的圆）
            for i in range(5):
                if mask_tips[i]:
                    pt = tuple(uv_tips[i])
                    # 缩小点的大小：外圈从7改为3，内圈从4改为2
                    cv2.circle(img_overlay, pt, 3, (255, 0, 255), 1)  # 外圈：洋红色，线宽1
                    cv2.circle(img_overlay, pt, 2, (0, 255, 255), -1)  # 内圈：黄色填充
                    # 标签：显示指尖名称（字体稍小）
                    cv2.putText(img_overlay, f"R{i}", (pt[0]+5, pt[1]-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
    
    else:
        # Fallback: 简化渲染（只显示手腕位置）
        if wrist_params is not None and intrinsic is not None:
            # 绘制左手腕
            if L_present and 'left_translation' in wrist_params:
                wrist_pos = wrist_params['left_translation']
                if wrist_pos[2] > 0:
                    # 世界坐标转相机坐标
                    wrist_cam = R_wc @ wrist_pos + t_wc
                    if wrist_cam[2] > 0:
                        x = int(fx * wrist_cam[0] / wrist_cam[2] + cx)
                        y = int(fy * wrist_cam[1] / wrist_cam[2] + cy)
                        if 0 <= x < W and 0 <= y < H:
                            cv2.circle(img_overlay, (x, y), 8, (255, 255, 0), -1)
                            cv2.putText(img_overlay, "L", (x+10, y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # 绘制右手腕
            if R_present and 'right_translation' in wrist_params:
                wrist_pos = wrist_params['right_translation']
                if wrist_pos[2] > 0:
                    # 世界坐标转相机坐标
                    wrist_cam = R_wc @ wrist_pos + t_wc
                    if wrist_cam[2] > 0:
                        x = int(fx * wrist_cam[0] / wrist_cam[2] + cx)
                        y = int(fy * wrist_cam[1] / wrist_cam[2] + cy)
                        if 0 <= x < W and 0 <= y < H:
                            cv2.circle(img_overlay, (x, y), 8, (0, 255, 255), -1)
                            cv2.putText(img_overlay, "R", (x+10, y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # 转回 RGB 格式（用于视频生成）
    img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
    return img_overlay


def array_to_base64_image(arr, target_width: int = None, target_height: int = None, max_width: int = 640) -> str:
    """将numpy数组转换为base64编码的图片（优化版，可指定目标分辨率）
    
    Args:
        arr: 图像数组
        target_width: 目标宽度（如果提供，调整到此分辨率）
        target_height: 目标高度（如果提供，调整到此分辨率）
        max_width: 最大宽度限制（用于缩放）
    """
    # 确保数据类型正确
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    
    # 创建PIL图像
    if len(arr.shape) == 2:
        # 灰度图
        img = Image.fromarray(arr, mode="L")
    elif len(arr.shape) == 3 and arr.shape[2] == 3:
        # RGB图
        img = Image.fromarray(arr, mode="RGB")
    elif len(arr.shape) == 3 and arr.shape[2] == 4:
        # RGBA图
        img = Image.fromarray(arr, mode="RGBA")
    else:
        raise ValueError(f"不支持的图像形状: {arr.shape}")
    
    # 决定目标尺寸
    if target_width and target_height:
        # 如果指定了目标宽高（来自depth），使用这个分辨率
        resize_width = target_width
        resize_height = target_height
        
        # 但仍然限制最大宽度
        if resize_width > max_width:
            scale = max_width / resize_width
            resize_width = max_width
            resize_height = int(resize_height * scale)
    else:
        # 否则保持原始宽高比
        original_width, original_height = img.size
        
        if original_width > max_width:
            resize_width = max_width
            resize_height = int(original_height * (max_width / original_width))
        else:
            resize_width = original_width
            resize_height = original_height
    
    # 缩放图像（使用更快的BILINEAR算法代替LANCZOS）
    current_width, current_height = img.size
    if (resize_width, resize_height) != (current_width, current_height):
        img = img.resize((resize_width, resize_height), Image.Resampling.BILINEAR)
    
    # 转换为JPEG以减小文件大小（降低质量以加快速度）
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=75, optimize=False)  # 降低质量，关闭优化以加速
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    
    return f"data:image/jpeg;base64,{img_base64}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """获取查看统计信息"""
    conn = get_db_connection()
    
    # 总查看记录数
    total_views = conn.execute("SELECT COUNT(*) FROM view_log").fetchone()[0]
    
    # 独立用户数
    total_users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM view_log").fetchone()[0]
    
    # 每个用户的查看数
    user_stats = conn.execute(
        """
        SELECT user_id, COUNT(*) as view_count 
        FROM view_log 
        GROUP BY user_id 
        ORDER BY view_count DESC
        """
    ).fetchall()
    
    # 已查看的 episode 数量
    total_viewed_episodes = conn.execute(
        "SELECT COUNT(DISTINCT episode_id) FROM view_log"
    ).fetchone()[0]
    
    return jsonify({
        "total_views": total_views,
        "total_users": total_users,
        "total_viewed_episodes": total_viewed_episodes,
        "user_stats": [
            {"user_id": row[0], "view_count": row[1]} 
            for row in user_stats
        ]
    })


@app.route("/api/debug/dataset/<dataset_name>", methods=["GET"])
def api_debug_dataset(dataset_name: str):
    """调试：查看数据集结构"""
    datasets = load_zarr_datasets()
    target = None
    for ds in datasets:
        if ds["name"] == dataset_name:
            target = ds
            break
    
    if target is None:
        return jsonify({"error": f"Dataset {dataset_name} not found"}), 404
    
    zarr_root = target["zarr_root"]
    
    result = {
        "dataset_name": dataset_name,
        "root_keys": list(zarr_root.keys()),
    }
    
    # 检查 meta
    if "meta" in zarr_root:
        meta = zarr_root["meta"]
        result["meta_keys"] = {}
        for key in meta.keys():
            item = meta[key]
            if hasattr(item, 'shape'):
                result["meta_keys"][key] = {
                    "type": "array",
                    "shape": item.shape,
                    "dtype": str(item.dtype),
                }
                # 获取前几个样本值
                if len(item) > 0:
                    try:
                        sample = item[:min(3, len(item))]
                        if isinstance(sample[0], bytes):
                            sample = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in sample]
                        result["meta_keys"][key]["sample"] = [str(s) for s in sample]
                    except:
                        pass
            else:
                result["meta_keys"][key] = {"type": "unknown"}
    else:
        result["meta_keys"] = None
    
    # 检查 data
    if "data" in zarr_root:
        data = zarr_root["data"]
        result["data_keys"] = list(data.keys())[:20]  # 只显示前20个
    else:
        result["data_keys"] = None
    
    return jsonify(result)


@app.route("/api/episodes", methods=["GET"])
def api_episodes():
    """随机获取未被查看的episodes"""
    # 从查询参数获取参数
    limit = request.args.get('limit', default=50, type=int)
    user_id = request.args.get('user_id', default=None, type=str)
    exclude_others = request.args.get('exclude_others', default='true', type=str).lower() == 'true'
    
    # 限制最大值，避免一次加载太多
    limit = min(limit, 500)
    
    episodes = get_random_episodes(
        total_limit=limit,
        user_id=user_id,
        exclude_viewed_by_others=exclude_others
    )
    
    # 统计每个数据集的实际 episodes 总数
    datasets_info = []
    all_datasets = load_zarr_datasets()
    
    for ds in all_datasets:
        ds_name = ds["name"]
        zarr_root = ds["zarr_root"]
        
        total_count = 0
        if "meta" in zarr_root:
            meta = zarr_root["meta"]
            if "episode_names" in meta:
                total_count = len(meta["episode_names"])
            elif "episode_name" in meta:
                total_count = len(meta["episode_name"])
        
        # 统计当前加载的数量
        loaded_count = sum(1 for ep in episodes if ep["dataset"] == ds_name)
        
        # 统计未被查看的数量
        viewed_all = get_viewed_episodes()
        unviewed_count = 0
        for i in range(total_count):
            episode_id = f"{ds_name}::{i}"
            if episode_id not in viewed_all:
                unviewed_count += 1
        
        datasets_info.append({
            "name": ds_name,
            "path": str(ds["path"]),
            "total_episodes": total_count,
            "loaded_episodes": loaded_count,
            "unviewed_episodes": unviewed_count,
        })
    
    return jsonify({
        "episodes": episodes,
        "datasets": datasets_info,
        "total_episodes": len(episodes),
        "limit": limit,
        "user_id": user_id,
        "exclude_others": exclude_others,
    })


@app.route("/api/episode/<path:episode_id>", methods=["GET"])
def api_episode(episode_id: str):
    """获取单个episode的详细信息"""
    # 获取用户ID并标记为已查看
    user_id = request.args.get('user_id', default=None, type=str)
    if user_id:
        mark_episode_viewed(episode_id, user_id)
    
    episode_data = get_episode_data(episode_id)
    
    if episode_data is None:
        abort(404, "Episode not found")
    
    # 获取指令信息
    instructions = []
    if "instructions" in episode_data and "instruction_num" in episode_data:
        num_inst = int(episode_data["instruction_num"][0])
        inst_data = episode_data["instructions"][0]
        for i in range(num_inst):
            inst = inst_data[i]
            if isinstance(inst, bytes):
                inst = inst.decode('utf-8')
            instructions.append(inst)
    
    return jsonify({
        "episode_id": episode_id,
        "episode_name": episode_data.get("name", ""),
        "dataset_name": episode_data.get("dataset", ""),
        "num_frames": episode_data["num_frames"],
        "instructions": instructions,
    })


@app.route("/api/episode/<path:episode_id>/video/original", methods=["GET"])
def api_episode_video_original(episode_id: str):
    """生成并返回原始视频"""
    episode_data = get_episode_data(episode_id)
    
    if episode_data is None:
        abort(404, "Episode not found")
    
    images = episode_data.get("images")
    if images is None or len(images) == 0:
        abort(404, "No images found for this episode")
    
    # 获取 depth 的形状作为目标分辨率
    target_width, target_height = None, None
    if "depth_shape" in episode_data and episode_data["depth_shape"] is not None:
        depth_shape = episode_data["depth_shape"]
        if len(depth_shape) >= 3:
            # depth shape: (frames, H, W)
            target_height = depth_shape[1]
            target_width = depth_shape[2]
            print(f"使用 depth 分辨率: {target_width}x{target_height}")
    
    # 使用固定路径的缓存文件，避免临时文件问题
    cache_dir = "/DATA/guantianrui/tmp/zarr_video_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # 清理过期和过大的缓存
    cleanup_video_cache(cache_dir, max_size_gb=10.0, max_age_days=7)
    
    # 使用 episode_id 的 hash 作为文件名
    import hashlib
    video_hash = hashlib.md5(f"{episode_id}_original_30fps".encode()).hexdigest()
    video_path = os.path.join(cache_dir, f"{video_hash}.mp4")
    
    # 如果缓存文件不存在，则生成
    if not os.path.exists(video_path):
        try:
            print(f"开始生成原始视频: {episode_id} ({len(images)} 帧)")
            
            # 生成原始视频（使用 depth 的分辨率）
            temp_video = video_path + ".tmp.mp4"
            fps = 30  # 原始数据集为 30 FPS
            create_video_from_frames(images, temp_video, fps=fps, 
                                   target_width=target_width, target_height=target_height, max_width=512)
            
            # 使用 ffmpeg 转换为 H.264（快速模式）
            print(f"转换视频为 H.264 格式...")
            if convert_video_to_h264(temp_video, video_path, quality_mode='fast'):
                print(f"✓ 视频转换成功")
                os.unlink(temp_video)  # 删除临时文件
            else:
                print(f"⚠️  FFmpeg 转换失败，使用原始 mp4v 格式")
                os.rename(temp_video, video_path)
                
        except Exception as e:
            print(f"生成原始视频失败: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(video_path):
                os.unlink(video_path)
            if os.path.exists(temp_video):
                os.unlink(temp_video)
            abort(500, f"Failed to generate video: {str(e)}")
    else:
        print(f"使用缓存的原始视频: {video_path}")
    
    # 返回视频文件（添加缓存头）
    print(f"→ 发送原始视频: {video_path}")
    response = send_file(
        video_path,
        mimetype='video/mp4',
        as_attachment=False,
        conditional=True,  # 支持范围请求
        download_name=f"{episode_data.get('name', 'episode')}_original.mp4"
    )
    # 添加缓存头：缓存1小时
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response


@app.route("/api/episode/<path:episode_id>/frame/original", methods=["GET"])
def api_episode_frame_original(episode_id: str):
    """返回单帧原图（等距抽帧，默认 6 帧）。"""
    episode_data = get_episode_data(episode_id)
    if episode_data is None:
        abort(404, "Episode not found")

    images = episode_data.get("images")
    if images is None or len(images) == 0:
        abort(404, "No images found for this episode")

    i = request.args.get("i", default=0, type=int)
    n = request.args.get("n", default=6, type=int)
    if n < 1:
        n = 1
    i = max(0, min(i, n - 1))

    total = len(images)
    if total == 1:
        idx = 0
    else:
        idx = int(round(i * (total - 1) / (n - 1))) if n > 1 else total // 2

    frame = images[idx]
    # images 通常是 BGR (OpenCV)；转成 RGB 方便 PIL 编码
    if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame

    img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/episode/<path:episode_id>/frame/rendered", methods=["GET"])
def api_episode_frame_rendered(episode_id: str):
    """返回单帧叠加动作投影图（等距抽帧，默认 6 帧）。

    复用现有渲染逻辑：render_hand_on_frame。
    """
    episode_data = get_episode_data(episode_id)
    if episode_data is None:
        abort(404, "Episode not found")

    images = episode_data.get("images")
    if images is None or len(images) == 0:
        abort(404, "No images found for this episode")

    i = request.args.get("i", default=0, type=int)
    n = request.args.get("n", default=6, type=int)
    if n < 1:
        n = 1
    i = max(0, min(i, n - 1))

    total = len(images)
    if total == 1:
        idx = 0
    else:
        idx = int(round(i * (total - 1) / (n - 1))) if n > 1 else total // 2

    frame = images[idx]

    # 尝试读取同帧对应的 mano/wrist/extrinsic/intrinsic（如果存在）
    mano_params = None
    wrist_params = None
    extrinsic = None
    intrinsic = None

    if "mano_params" in episode_data and episode_data["mano_params"] is not None:
        try:
            mano_params = episode_data["mano_params"][idx]
        except Exception:
            mano_params = None
    if "wrist" in episode_data and episode_data["wrist"] is not None:
        try:
            wrist_params = episode_data["wrist"][idx]
        except Exception:
            wrist_params = None
    if "extrinsic" in episode_data and episode_data["extrinsic"] is not None:
        try:
            extrinsic = episode_data["extrinsic"][idx]
        except Exception:
            extrinsic = None
    if "intrinsic" in episode_data and episode_data["intrinsic"] is not None:
        try:
            intrinsic = episode_data["intrinsic"][idx]
        except Exception:
            intrinsic = None

    rendered = render_hand_on_frame(
        frame,
        mano_params=mano_params,
        wrist_params=wrist_params,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        alpha=0.55,  # 叠加半透明
    )

    # render_hand_on_frame 返回 BGR
    if isinstance(rendered, np.ndarray) and rendered.ndim == 3 and rendered.shape[2] == 3:
        rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
    else:
        rendered_rgb = rendered

    img = Image.fromarray(rendered_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/episode/<path:episode_id>/video/rendered", methods=["GET"])
def api_episode_video_rendered(episode_id: str):
    """生成并返回带动作投影的渲染视频"""
    episode_data = get_episode_data(episode_id)
    if episode_data is None:
        abort(404, "Episode not found")

    images = episode_data.get("images")
    if images is None or len(images) == 0:
        abort(404, "No images found for this episode")

    is_egodex = episode_id.startswith("EgoDex")
    if is_egodex:
        print(f"检测到 EgoDex 数据集，将跳过 MANO 渲染，仅渲染 fingertips")

    # 使用固定路径的缓存文件
    cache_dir = "/DATA/guantianrui/tmp/zarr_video_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # 清理过期和过大的缓存
    cleanup_video_cache(cache_dir, max_size_gb=10.0, max_age_days=7)

    import hashlib
    video_hash = hashlib.md5(f"{episode_id}_rendered_30fps".encode()).hexdigest()
    video_path = os.path.join(cache_dir, f"{video_hash}.mp4")

    # 如果缓存文件存在，直接返回
    if os.path.exists(video_path):
        print(f"使用缓存的渲染视频: {video_path}")
        response = send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            conditional=True,
            download_name=f"{episode_data.get('name', 'episode')}_rendered.mp4"
        )
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response

    # 准备渲染数据
    state_data = episode_data.get("state", {})
    action_data = episode_data.get("action", {})
    presence_data = episode_data.get("presence")
    intrinsic_data = episode_data.get("intrinsic")
    extrinsic_data = episode_data.get("extrinsic")

    # 初始化MANO模型（如果可用）
    mano_layers = None
    if MANO_AVAILABLE:
        try:
            mano_root = os.environ.get('MANO_ROOT', '/home/guantianrui/manopth/mano/models')
            if os.path.exists(mano_root):
                from manopth.manolayer import ManoLayer
                mano_layers = {
                    'left': ManoLayer(mano_root=mano_root, use_pca=True, ncomps=45,
                                     flat_hand_mean=True, side='left', center_idx=0),
                    'right': ManoLayer(mano_root=mano_root, use_pca=True, ncomps=45,
                                      flat_hand_mean=True, side='right', center_idx=0)
                }
                print("✓ MANO 模型已初始化用于视频渲染")
        except Exception as e:
            print(f"⚠️  MANO 初始化失败，使用简化渲染: {e}")
            mano_layers = None

    # 生成渲染视频
    try:
        print(f"开始生成渲染视频: {episode_id} ({len(images)} 帧)")

        from concurrent.futures import ThreadPoolExecutor
        import time

        num_frames = len(images)
        print(f"准备并行渲染 {num_frames} 帧...")
        start_time = time.time()

        def render_single_frame(i):
            """渲染单帧的辅助函数"""
            frame = images[i]

            mano_params = None
            if "mano" in state_data and not is_egodex:
                mano = state_data["mano"][i]
                pose_dim = mano.shape[0] // 2
                mano_params = {
                    "left": mano[:pose_dim],
                    "right": mano[pose_dim:],
                }

            wrist_params = None
            if "wrist" in state_data and not is_egodex:
                wrist = state_data["wrist"][i]
                wrist_params = {
                    "left_translation": wrist[:3],
                    "right_translation": wrist[3:6],
                    "left_rotation": wrist[6:12],
                    "right_rotation": wrist[12:18],
                }

            shape_params = None
            if "shape" in state_data and not is_egodex:
                shape = state_data["shape"][i]
                shape_params = {
                    "left": shape[:10],
                    "right": shape[10:20],
                }

            presence = None
            if presence_data is not None:
                presence = int(presence_data[i])

            intrinsic = None
            if intrinsic_data is not None:
                intrinsic = intrinsic_data[i]

            extrinsic = None
            if extrinsic_data is not None:
                extrinsic = extrinsic_data[i]

            fingertips = None
            if "fingertips" in state_data:
                tips = state_data["fingertips"][i]
                fingertips = {
                    "left": tips[:15],
                    "right": tips[15:30],
                }

            return render_hand_on_frame(
                frame,
                mano_params=mano_params,
                wrist_params=wrist_params,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                presence=presence,
                shape_params=shape_params,
                mano_layers=mano_layers,
                fingertips=fingertips
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            rendered_frames = list(executor.map(render_single_frame, range(num_frames)))

        render_time = time.time() - start_time
        print(f"✓ 并行渲染完成，耗时 {render_time:.2f}秒 ({num_frames/render_time:.1f} fps)")

        temp_video = video_path + ".tmp.mp4"
        fps = 30

        target_width, target_height = None, None
        if "depth_shape" in episode_data and episode_data["depth_shape"] is not None:
            depth_shape = episode_data["depth_shape"]
            if len(depth_shape) >= 3:
                target_height = depth_shape[1]
                target_width = depth_shape[2]

        create_video_from_frames(rendered_frames, temp_video, fps=fps,
                               target_width=target_width, target_height=target_height, max_width=512)

        print(f"转换渲染视频为 H.264 格式...")
        if convert_video_to_h264(temp_video, video_path, quality_mode='fast'):
            print(f"✓ 渲染视频转换成功")
            os.unlink(temp_video)
        else:
            print(f"⚠️  FFmpeg 转换失败，使用原始 mp4v 格式")
            os.rename(temp_video, video_path)

        print(f"→ 发送渲染视频: {video_path}")
        response = send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            conditional=True,
            download_name=f"{episode_data.get('name', 'episode')}_rendered.mp4"
        )
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
    except Exception as e:
        print(f"生成渲染视频失败: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(video_path):
            os.unlink(video_path)
        abort(500, f"Failed to generate rendered video: {str(e)}")


@app.route("/api/annotation/<path:episode_id>", methods=["GET"])
def api_get_annotation(episode_id: str):
    """获取episode的标注（包含结构化问题列表）"""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT content, updated_at, issues, additional_notes, is_valid, reviewer_name FROM annotations WHERE episode_id = ?",
        (episode_id,),
    ).fetchone()

    if row is None:
        return jsonify({
            "episode_id": episode_id,
            "content": "",
            "updated_at": None,
            "issues": [],
            "additional_notes": "",
            "is_valid": 1,
            "reviewer_name": "",
        })
    
    # 解析 JSON 格式的 issues
    issues = []
    if row["issues"]:
        try:
            issues = json_module.loads(row["issues"])
        except:
            pass
    
    return jsonify({
        "episode_id": episode_id,
        "content": row["content"] or "",
        "updated_at": row["updated_at"],
        "issues": issues,
        "additional_notes": row["additional_notes"] or "",
        "is_valid": row["is_valid"] if row["is_valid"] is not None else 1,
        "reviewer_name": row["reviewer_name"] or "",
    })


def translate_with_gemini(text: str, api_key="sk-wcBPMUCtgVMe1RY5oNfvVi1do7vXqAQ44cEsQZhkVjv3kJ3g", timeout=30):
    """使用 Gemini API 翻译文本（英文 -> 中文）"""
    if not text:
        return None
    
    conn = None
    try:
        # 创建连接
        conn = http.client.HTTPSConnection("api3.xhub.chat", timeout=timeout)
        
        # 构建请求 payload
        payload = json_module.dumps({
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": f"Please translate the following English text to Chinese. Only return the Chinese translation without any explanations or additional text:\n\n{text}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.8,
                "topP": 0.95,
                "topK": 20,
                # "maxOutputTokens": 512,
                "responseMimeType": "text/plain"
            }
        })
        
        headers = {'Content-Type': 'application/json'}
        
        # 发送请求
        conn.request("POST", f"/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}", payload, headers)
        res = conn.getresponse()
        data = res.read()
        response_text = data.decode("utf-8")
        
        conn.close()
        conn = None
        
        # 解析响应
        response_json = json_module.loads(response_text)
        
        # 提取翻译结果（跳过思维链）
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            candidate = response_json["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    # 跳过 CoT 思考过程（thought: true 的 part）
                    if part.get("thought", False):
                        continue
                    
                    if "text" in part:
                        return part["text"].strip()
        
        return None
        
    except Exception as e:
        if conn:
            try:
                conn.close()
            except:
                pass
        print(f"翻译失败: {e}")
        return None


def translate_batch_with_gemini(texts: list, api_key="sk-wcBPMUCtgVMe1RY5oNfvVi1do7vXqAQ44cEsQZhkVjv3kJ3g", timeout=60):
    """批量翻译多条文本（英文 -> 中文）"""
    if not texts or len(texts) == 0:
        return None
    
    conn = None
    try:
        # 构建批量翻译的 prompt
        numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
        prompt = f"Please translate the following English sentences to Chinese. Keep the same numbering format (1., 2., 3., etc.). Only return the Chinese translations with numbers, no explanations:\n\n{numbered_texts}"
        
        # 创建连接
        conn = http.client.HTTPSConnection("api3.xhub.chat", timeout=timeout)
        
        # 构建请求 payload
        payload = json_module.dumps({
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.8,
                "topP": 0.95,
                "topK": 20,
                "responseMimeType": "text/plain"
            }
        })
        
        headers = {'Content-Type': 'application/json'}
        
        # 发送请求
        conn.request("POST", f"/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}", payload, headers)
        res = conn.getresponse()
        data = res.read()
        response_text = data.decode("utf-8")
        
        conn.close()
        conn = None
        
        # 解析响应
        response_json = json_module.loads(response_text)
        
        # 提取翻译结果（跳过思维链）
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            candidate = response_json["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    # 跳过 CoT 思考过程
                    if part.get("thought", False):
                        continue
                    
                    if "text" in part:
                        full_text = part["text"].strip()
                        
                        # 解析编号的翻译结果
                        lines = full_text.split('\n')
                        translations = []
                        
                        for line in lines:
                            line = line.strip()
                            # 匹配 "1. xxx" 或 "1.xxx" 格式
                            if line and (line[0].isdigit() or (len(line) > 1 and line[0:2].replace('.', '').isdigit())):
                                # 去掉编号，只保留翻译内容
                                parts = line.split('.', 1)
                                if len(parts) == 2:
                                    translations.append(parts[1].strip())
                        
                        if len(translations) == len(texts):
                            return translations
                        else:
                            print(f"翻译数量不匹配: 期望 {len(texts)}, 得到 {len(translations)}")
                            # 如果数量不匹配，返回原始文本
                            return None
        
        return None
        
    except Exception as e:
        if conn:
            try:
                conn.close()
            except:
                pass
        print(f"批量翻译失败: {e}")
        return None


@app.route("/api/translate", methods=["POST"])
def api_translate():
    """翻译文本（英文 -> 中文），支持单条或批量"""
    data = request.get_json(silent=True) or {}
    
    # 支持单条文本
    if "text" in data:
        text = data.get("text", "")
        if not text:
            return jsonify({"success": False, "error": "文本为空"}), 400
        
        try:
            translated = translate_with_gemini(text)
            if translated:
                return jsonify({"success": True, "translated": translated})
            else:
                return jsonify({"success": False, "error": "翻译失败"}), 500
        except Exception as e:
            print(f"翻译失败: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    # 支持批量文本
    elif "texts" in data:
        texts = data.get("texts", [])
        if not texts or len(texts) == 0:
            return jsonify({"success": False, "error": "文本为空"}), 400
        
        try:
            translations = translate_batch_with_gemini(texts)
            if translations and len(translations) == len(texts):
                return jsonify({"success": True, "translations": translations})
            else:
                return jsonify({"success": False, "error": "翻译失败或数量不匹配"}), 500
        except Exception as e:
            print(f"批量翻译失败: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    else:
        return jsonify({"success": False, "error": "缺少 text 或 texts 参数"}), 400


@app.route("/api/instruction/<path:episode_id>", methods=["POST"])
def api_save_instruction_translation(episode_id: str):
    """保存 instruction 的中文翻译"""
    data = request.get_json(silent=True) or {}
    translations = data.get("translations", [])  # [{index: 0, translation: "中文"}]
    
    if not translations:
        return jsonify({"success": False, "error": "翻译内容为空"}), 400
    
    conn = get_db_connection()
    
    # 创建翻译表（如果不存在）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS instruction_translations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL,
            instruction_index INTEGER NOT NULL,
            original_text TEXT,
            translated_text TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_manually_edited INTEGER DEFAULT 0,
            UNIQUE(episode_id, instruction_index)
        )
    """)
    
    # 添加新字段（如果是旧数据库升级）
    try:
        conn.execute("ALTER TABLE instruction_translations ADD COLUMN is_manually_edited INTEGER DEFAULT 0")
    except:
        pass
    
    # 保存每条翻译
    for trans in translations:
        idx = trans.get("index")
        original = trans.get("original", "")
        translation = trans.get("translation", "")
        is_edited = trans.get("is_edited", False)  # 是否被用户手动修改
        
        if translation:
            conn.execute("""
                INSERT INTO instruction_translations 
                (episode_id, instruction_index, original_text, translated_text, updated_at, is_manually_edited)
                VALUES (?, ?, ?, ?, datetime('now'), ?)
                ON CONFLICT(episode_id, instruction_index) DO UPDATE SET 
                    translated_text = excluded.translated_text,
                    updated_at = excluded.updated_at,
                    is_manually_edited = excluded.is_manually_edited
            """, (episode_id, idx, original, translation, 1 if is_edited else 0))
    
    conn.commit()
    conn.close()
    
    return jsonify({"success": True})


@app.route("/api/instruction/<path:episode_id>", methods=["GET"])
def api_get_instruction_translations(episode_id: str):
    """获取 instruction 的中文翻译"""
    conn = get_db_connection()
    
    # 确保表存在
    conn.execute("""
        CREATE TABLE IF NOT EXISTS instruction_translations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL,
            instruction_index INTEGER NOT NULL,
            original_text TEXT,
            translated_text TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_manually_edited INTEGER DEFAULT 0,
            UNIQUE(episode_id, instruction_index)
        )
    """)
    
    # 添加新字段（如果是旧数据库升级）
    try:
        conn.execute("ALTER TABLE instruction_translations ADD COLUMN is_manually_edited INTEGER DEFAULT 0")
    except:
        pass
    
    rows = conn.execute(
        "SELECT instruction_index, translated_text, COALESCE(is_manually_edited, 0) FROM instruction_translations WHERE episode_id = ?",
        (episode_id,)
    ).fetchall()
    
    conn.close()
    
    # 返回翻译文本和是否被编辑的标记
    translations = {}
    for row in rows:
        translations[row[0]] = {
            "text": row[1],
            "is_edited": bool(row[2])
        }
    
    return jsonify({"success": True, "translations": translations})


@app.route("/api/annotation/<path:episode_id>", methods=["POST"])
def api_set_annotation(episode_id: str):
    """保存episode的标注（支持结构化问题列表）"""
    data = request.get_json(silent=True) or {}
    content = data.get("content", "")  # 保留旧的文本内容（向后兼容）
    episode_name = data.get("episode_name", "")
    dataset_name = data.get("dataset_name", "")
    episode_index = data.get("episode_index", None)
    
    # 新的结构化字段
    issues = data.get("issues", [])  # 问题列表，如 ["动作标注不佳", "缺手"]
    additional_notes = data.get("additional_notes", "")  # 额外说明
    is_valid = 1 if len(issues) == 0 else 0  # 没有问题则标记为有效
    reviewer_name = data.get("reviewer_name", "")  # 审核人姓名

    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO annotations (
            episode_id, episode_name, dataset_name, episode_index,
            content, issues, additional_notes, is_valid, reviewer_name, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(episode_id) DO UPDATE SET
            episode_name = excluded.episode_name,
            dataset_name = excluded.dataset_name,
            episode_index = excluded.episode_index,
            content = excluded.content,
            issues = excluded.issues,
            additional_notes = excluded.additional_notes,
            is_valid = excluded.is_valid,
            reviewer_name = excluded.reviewer_name,
            updated_at = excluded.updated_at
        """,
        (episode_id, episode_name, dataset_name, episode_index,
         content, json_module.dumps(issues, ensure_ascii=False), additional_notes, is_valid,
         reviewer_name, datetime.utcnow().isoformat()),
    )
    conn.commit()
    
    return jsonify({"ok": True, "is_valid": is_valid})


@app.route("/admin")
def admin_page():
    """管理后台页面"""
    return render_template("admin.html")


@app.route("/api/user/stats", methods=["GET"])
def api_user_stats():
    """获取当前用户的审核统计（今日 & 总计，按 UTC+8 计算）"""
    reviewer_name = request.args.get('reviewer_name', default=None, type=str)
    
    if not reviewer_name:
        return jsonify({"success": False, "error": "缺少 reviewer_name 参数"}), 400
    
    conn = get_db_connection()
    
    # 计算 UTC+8 的今日日期
    from datetime import timedelta
    utc_now = datetime.utcnow()
    utc8_now = utc_now + timedelta(hours=8)
    today_utc8 = utc8_now.strftime("%Y-%m-%d")
    
    # 获取该用户的所有标注记录
    rows = conn.execute(
        """
        SELECT updated_at 
        FROM annotations 
        WHERE reviewer_name = ?
        ORDER BY updated_at DESC
        """,
        (reviewer_name,)
    ).fetchall()
    
    # 统计总数和今日数
    total_count = len(rows)
    today_count = 0
    
    for row in rows:
        # 将 UTC 时间转换为 UTC+8
        try:
            utc_time = datetime.fromisoformat(row["updated_at"].replace('Z', '+00:00'))
            utc8_time = utc_time + timedelta(hours=8)
            date_utc8 = utc8_time.strftime("%Y-%m-%d")
            
            if date_utc8 == today_utc8:
                today_count += 1
        except Exception as e:
            print(f"解析时间失败: {e}")
            continue
    
    return jsonify({
        "success": True,
        "reviewer_name": reviewer_name,
        "total_count": total_count,
        "today_count": today_count,
        "today_date": today_utc8,
    })


@app.route("/api/admin/stats")
def api_admin_stats():
    """管理后台统计数据"""
    conn = get_db_connection()

    # 总标注数（只统计有审核人的）
    total_annotations = conn.execute("SELECT COUNT(*) FROM annotations WHERE reviewer_name IS NOT NULL AND reviewer_name != ''").fetchone()[0]

    # 总审核人数（去重，排除空值）
    total_reviewers = conn.execute(
        "SELECT COUNT(DISTINCT reviewer_name) FROM annotations WHERE reviewer_name IS NOT NULL AND reviewer_name != ''"
    ).fetchone()[0]

    # 今日标注数（UTC+8）
    from datetime import timezone, timedelta
    utc8 = timezone(timedelta(hours=8))
    today = datetime.now(utc8).strftime("%Y-%m-%d")
    today_annotations = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE substr(updated_at, 1, 10) = ? AND reviewer_name IS NOT NULL AND reviewer_name != ''",
        (today,),
    ).fetchone()[0]

    # 每个审核人的统计
    reviewer_rows = conn.execute(
        """
        SELECT
            reviewer_name,
            COUNT(*) as total_count,
            SUM(CASE WHEN substr(updated_at, 1, 10) = ? THEN 1 ELSE 0 END) as today_count,
            MAX(updated_at) as last_annotation_time
        FROM annotations
        WHERE reviewer_name IS NOT NULL AND reviewer_name != ''
        GROUP BY reviewer_name
        ORDER BY total_count DESC
        """,
        (today,),
    ).fetchall()

    reviewers = []
    for row in reviewer_rows:
        reviewers.append({
            "name": row["reviewer_name"],
            "total_count": row["total_count"],
            "today_count": row["today_count"],
            "last_annotation_time": row["last_annotation_time"],
        })

    # 最近 7 天每日统计
    daily_rows = conn.execute(
        """
        SELECT
            substr(updated_at, 1, 10) as date,
            reviewer_name,
            COUNT(*) as cnt
        FROM annotations
        WHERE reviewer_name IS NOT NULL AND reviewer_name != ''
          AND updated_at >= date('now', '-7 days')
        GROUP BY date, reviewer_name
        ORDER BY date DESC
        """
    ).fetchall()

    daily_map = {}
    for row in daily_rows:
        d = row["date"]
        if d not in daily_map:
            daily_map[d] = {}
        daily_map[d][row["reviewer_name"]] = row["cnt"]

    daily_stats = [{"date": d, "reviewers": r} for d, r in sorted(daily_map.items(), reverse=True)]

    return jsonify({
        "total_annotations": total_annotations,
        "total_reviewers": total_reviewers,
        "today_annotations": today_annotations,
        "reviewers": reviewers,
        "daily_stats": daily_stats,
    })


def main():
    init_db()
    load_bad_episodes_cache()  # 预加载 sanity check 结果
    print(f"启动服务器在端口 {SERVER_PORT}")
    print(f"访问: http://localhost:{SERVER_PORT}")
    print(f"管理后台: http://localhost:{SERVER_PORT}/admin")
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=DEBUG_MODE)


if __name__ == "__main__":
    main()

