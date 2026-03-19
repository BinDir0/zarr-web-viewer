#!/usr/bin/env python3
"""
构建 YOLO 手部检测+质量分类器训练数据集。

从外包标注 (annotations.db) + model_tracks.npy 构建 YOLO 格式数据集：
  - good_left_hand (class 0)
  - good_right_hand (class 1)
  - bad_hand (class 2)

对 bad 标注的帧，用 IoU 对比自动区分"帧不好"和"bbox 不好"。
"""
import argparse
import io
import json
import os
import random
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image


# ─── 数据加载 ─────────────────────────────────────────────────────────


def load_annotations(db_path: str) -> Dict[str, str]:
    """读取 annotations.db，返回 {episode_id: content}。"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT episode_id, content FROM annotations").fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def load_factory_index(factory_dir: str) -> Optional[dict]:
    """加载 _video_index.json。"""
    index_path = os.path.join(factory_dir, "_video_index.json")
    if not os.path.exists(index_path):
        return None
    try:
        with open(index_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_tracks(seq_folder: str) -> Tuple[Optional[dict], Dict[int, int]]:
    """加载 model_tracks.npy，返回 (raw_tracks, {track_id: voted_handedness})。"""
    seq_path = Path(seq_folder)
    if not seq_path.exists():
        return None, {}

    tracks_dirs = sorted(seq_path.glob("tracks_*"))
    if not tracks_dirs:
        return None, {}

    tracks_path = tracks_dirs[0] / "model_tracks.npy"
    if not tracks_path.exists():
        return None, {}

    try:
        tracks_data = np.load(str(tracks_path), allow_pickle=True).item()
    except Exception:
        return None, {}

    # Majority voting per track
    voted = {}
    for track_id, detections in tracks_data.items():
        all_h = [det["det_handedness"][0] for det in detections]
        voted[track_id] = 1 if sum(1 for h in all_h if h > 0) > len(all_h) / 2 else 0

    return tracks_data, voted


def extract_frames_from_shard(
    shard_path: str,
    frame_offsets: List[List],
    frame_indices: List[int],
    num_frames: int,
) -> Dict[int, bytes]:
    """从 shard 中 seek+read 提取指定帧的 JPEG bytes。"""
    frame_data = {}
    if not os.path.exists(shard_path):
        return frame_data

    try:
        with open(shard_path, "rb") as f:
            for idx in frame_indices:
                if 0 <= idx < num_frames and idx < len(frame_offsets):
                    offset, size = frame_offsets[idx]
                    f.seek(offset)
                    frame_data[idx] = f.read(size)
    except Exception as e:
        print(f"  ⚠ 读取 shard 失败 ({os.path.basename(shard_path)}): {e}")

    return frame_data


# ─── 标签逻辑 ─────────────────────────────────────────────────────────


def get_bad_track_ids(tracks_data: dict, bad_frames: List[int]) -> Set[int]:
    """找到在 bad_frames 上有检测的 track_id。"""
    bad_set = set(bad_frames)
    bad_tracks = set()
    for track_id, detections in tracks_data.items():
        for det in detections:
            if det["frame"] in bad_set:
                bad_tracks.add(track_id)
                break
    return bad_tracks


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个 [x1,y1,x2,y2] 框的 IoU。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def redetect_frames(detector, frame_images: Dict[int, bytes], conf_thresh: float = 0.35) -> Dict[int, list]:
    """用 YOLO 重新检测帧，返回 {frame_idx: [(x1,y1,x2,y2,conf,cls), ...]}。"""
    results = {}
    if not frame_images:
        return results

    # 批量准备 PIL images
    pil_images = {}
    for idx, img_bytes in frame_images.items():
        try:
            pil_images[idx] = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            continue

    if not pil_images:
        return results

    # 批量推理
    indices = list(pil_images.keys())
    images = [pil_images[i] for i in indices]

    try:
        preds = detector(images, conf=conf_thresh, verbose=False)
        for idx, pred in zip(indices, preds):
            boxes = pred.boxes
            detections = []
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().tolist()
                conf = float(boxes.conf[j].cpu())
                cls = int(boxes.cls[j].cpu())
                detections.append((x1, y1, x2, y2, conf, cls))
            results[idx] = detections
    except Exception as e:
        print(f"  ⚠ YOLO 重新检测失败: {e}")

    return results


def classify_bad_track_frames(
    tracks_data: dict,
    bad_track_ids: Set[int],
    voted_handedness: Dict[int, int],
    redetected: Dict[int, list],
    iou_high: float = 0.7,
    iou_low: float = 0.3,
) -> Dict[int, Tuple[str, Optional[List[float]], Optional[int]]]:
    """对 bad track 的帧分类。

    Returns:
        {frame_idx: (label, bbox_xyxy, handedness)}
        label: "bad_hand" | "good_hand" | "exclude"
    """
    result = {}

    for track_id in bad_track_ids:
        detections = tracks_data[track_id]
        h = voted_handedness.get(track_id, 0)

        for det in detections:
            fidx = det["frame"]
            track_box = det["det_box"][0][:4].tolist()  # [x1,y1,x2,y2]

            redet_list = redetected.get(fidx, [])

            if not redet_list:
                # YOLO 重新检测无结果 → bbox 是 ByteTrack 补的 → 排除
                result[fidx] = ("exclude", None, None)
                continue

            # 找与 track_box IoU 最高的重新检测框
            best_iou = 0.0
            best_redet = None
            for rd in redet_list:
                iou = compute_iou(track_box, list(rd[:4]))
                if iou > best_iou:
                    best_iou = iou
                    best_redet = rd

            if best_iou >= iou_high:
                # bbox 一致 → 帧本身不好 → bad_hand
                result[fidx] = ("bad_hand", track_box, h)
            elif best_iou < iou_low:
                # bbox 不一致 → 框偏了，帧可能好 → 用新 bbox 标 good
                if best_redet is not None:
                    new_box = list(best_redet[:4])
                    new_h = int(best_redet[5])  # YOLO class
                    result[fidx] = ("good_hand", new_box, new_h)
                else:
                    result[fidx] = ("exclude", None, None)
            else:
                # 不确定 → 排除
                result[fidx] = ("exclude", None, None)

    return result


def sample_no_detection_frames(
    num_frames: int, tracked_frames: Set[int], max_samples: int = 5
) -> List[int]:
    """从没有 track 检测的帧中随机采样。"""
    untracked = [i for i in range(num_frames) if i not in tracked_frames]
    if not untracked:
        return []
    return random.sample(untracked, min(max_samples, len(untracked)))


# ─── 输出 ─────────────────────────────────────────────────────────────


def save_yolo_label(label_path: str, boxes: List[Tuple[int, float, float, float, float]]):
    """写 YOLO 格式 label txt。boxes: [(class_id, cx, cy, w, h), ...]，已归一化。"""
    with open(label_path, "w") as f:
        for cls, cx, cy, w, h in boxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def xyxy_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """[x1,y1,x2,y2] → (cx, cy, w, h) 归一化到 0-1。"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return cx, cy, w, h


# ─── Episode 处理 ─────────────────────────────────────────────────────


def process_alright_episode(
    factory_name: str,
    video_key: str,
    shard_path: str,
    frame_offsets: List[List],
    num_frames: int,
    tracks_data: dict,
    voted_handedness: Dict[int, int],
    output_dir: Path,
    split: str,
    no_det_per_ep: int,
) -> List[dict]:
    """处理 alright episode，所有 tracked frames → good_left/right_hand。"""
    manifest = []

    # 收集所有 tracked frames 及其 bbox
    # {frame_idx: [(track_id, bbox, handedness), ...]}
    frame_info: Dict[int, list] = {}
    tracked_frames: Set[int] = set()
    for track_id, detections in tracks_data.items():
        h = voted_handedness.get(track_id, 0)
        for det in detections:
            fidx = det["frame"]
            bbox = det["det_box"][0][:4].tolist()
            conf = float(det["det_box"][0][4])
            frame_info.setdefault(fidx, []).append((track_id, bbox, h, conf))
            tracked_frames.add(fidx)

    # 采样 no_detection 帧
    no_det_frames = sample_no_detection_frames(num_frames, tracked_frames, no_det_per_ep)

    # 需要提取的所有帧
    all_indices = sorted(set(frame_info.keys()) | set(no_det_frames))
    if not all_indices:
        return manifest

    # 提取帧
    frame_data = extract_frames_from_shard(shard_path, frame_offsets, all_indices, num_frames)

    for fidx in all_indices:
        if fidx not in frame_data:
            continue

        img_bytes = frame_data[fidx]
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img_w, img_h = img.size
        except Exception:
            continue

        fname = f"{factory_name}_{video_key}_f{fidx:04d}.jpg"
        img_path = output_dir / "images" / split / fname
        label_path = output_dir / "labels" / split / (fname.replace(".jpg", ".txt"))

        # 保存图片
        img_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        if fidx in frame_info:
            # Tracked frame → good_left/right_hand
            yolo_boxes = []
            for track_id, bbox, h, conf in frame_info[fidx]:
                cls = 0 if h == 0 else 1  # 0=good_left, 1=good_right
                cx, cy, w, bh = xyxy_to_yolo(bbox, img_w, img_h)
                yolo_boxes.append((cls, cx, cy, w, bh))

            save_yolo_label(str(label_path), yolo_boxes)

            for track_id, bbox, h, conf in frame_info[fidx]:
                manifest.append({
                    "image": f"images/{split}/{fname}",
                    "label": "good_left_hand" if h == 0 else "good_right_hand",
                    "episode_id": video_key,
                    "factory": factory_name,
                    "frame_idx": fidx,
                    "track_id": int(track_id),
                    "handedness": h,
                    "bbox": bbox,
                    "confidence": conf,
                    "source": "alright",
                })
        else:
            # No detection frame → 背景图（空 label）
            save_yolo_label(str(label_path), [])
            manifest.append({
                "image": f"images/{split}/{fname}",
                "label": "background",
                "episode_id": video_key,
                "factory": factory_name,
                "frame_idx": fidx,
                "track_id": None,
                "handedness": None,
                "bbox": None,
                "confidence": None,
                "source": "no_detection",
            })

    return manifest


def process_bad_episode(
    factory_name: str,
    video_key: str,
    shard_path: str,
    frame_offsets: List[List],
    num_frames: int,
    tracks_data: dict,
    voted_handedness: Dict[int, int],
    bad_frames: List[int],
    detector,
    output_dir: Path,
    split: str,
    no_det_per_ep: int,
    iou_high: float,
    iou_low: float,
) -> List[dict]:
    """处理 BAD_FRAMES episode，用 IoU 区分帧不好 vs bbox 不好。"""
    manifest = []

    bad_track_ids = get_bad_track_ids(tracks_data, bad_frames)

    # 如果 bad frames 上没有任何 track → 保守：所有 track 标为 bad
    if not bad_track_ids:
        bad_track_ids = set(tracks_data.keys())

    good_track_ids = set(tracks_data.keys()) - bad_track_ids

    # ── 处理 good tracks（同 alright）──
    good_frame_info: Dict[int, list] = {}
    all_tracked: Set[int] = set()
    for track_id, detections in tracks_data.items():
        h = voted_handedness.get(track_id, 0)
        for det in detections:
            fidx = det["frame"]
            all_tracked.add(fidx)
            if track_id in good_track_ids:
                bbox = det["det_box"][0][:4].tolist()
                conf = float(det["det_box"][0][4])
                good_frame_info.setdefault(fidx, []).append((track_id, bbox, h, conf))

    # ── 收集 bad track 的帧 ──
    bad_track_frames: Set[int] = set()
    for track_id in bad_track_ids:
        for det in tracks_data[track_id]:
            bad_track_frames.add(det["frame"])

    # 采样 no_detection 帧
    no_det_frames = sample_no_detection_frames(num_frames, all_tracked, no_det_per_ep)

    # 提取所有需要的帧
    all_indices = sorted(set(good_frame_info.keys()) | bad_track_frames | set(no_det_frames))
    if not all_indices:
        return manifest

    frame_data = extract_frames_from_shard(shard_path, frame_offsets, all_indices, num_frames)

    # ── 对 bad track 帧重新检测 ──
    bad_frame_data = {i: frame_data[i] for i in bad_track_frames if i in frame_data}
    redetected = redetect_frames(detector, bad_frame_data) if detector and bad_frame_data else {}

    classified = classify_bad_track_frames(
        tracks_data, bad_track_ids, voted_handedness, redetected, iou_high, iou_low
    )

    # ── 写出所有帧 ──
    for fidx in all_indices:
        if fidx not in frame_data:
            continue

        img_bytes = frame_data[fidx]
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img_w, img_h = img.size
        except Exception:
            continue

        fname = f"{factory_name}_{video_key}_f{fidx:04d}.jpg"
        img_path = output_dir / "images" / split / fname
        label_path = output_dir / "labels" / split / (fname.replace(".jpg", ".txt"))
        img_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        with open(img_path, "wb") as f:
            f.write(img_bytes)

        yolo_boxes = []

        # Good track boxes on this frame
        if fidx in good_frame_info:
            for track_id, bbox, h, conf in good_frame_info[fidx]:
                cls = 0 if h == 0 else 1
                cx, cy, w, bh = xyxy_to_yolo(bbox, img_w, img_h)
                yolo_boxes.append((cls, cx, cy, w, bh))
                manifest.append({
                    "image": f"images/{split}/{fname}",
                    "label": "good_left_hand" if h == 0 else "good_right_hand",
                    "episode_id": video_key,
                    "factory": factory_name,
                    "frame_idx": fidx,
                    "track_id": int(track_id),
                    "handedness": h,
                    "bbox": bbox,
                    "confidence": conf,
                    "source": "good_track_in_bad_episode",
                })

        # Bad track classification result
        if fidx in classified:
            label_str, bbox, h = classified[fidx]
            if label_str == "bad_hand" and bbox is not None:
                cx, cy, w, bh = xyxy_to_yolo(bbox, img_w, img_h)
                yolo_boxes.append((2, cx, cy, w, bh))  # class 2 = bad_hand
                manifest.append({
                    "image": f"images/{split}/{fname}",
                    "label": "bad_hand",
                    "episode_id": video_key,
                    "factory": factory_name,
                    "frame_idx": fidx,
                    "track_id": None,
                    "handedness": h,
                    "bbox": bbox,
                    "confidence": None,
                    "source": "bad_annotation_high_iou",
                })
            elif label_str == "good_hand" and bbox is not None:
                cls = 0 if h == 0 else 1
                cx, cy, w, bh = xyxy_to_yolo(bbox, img_w, img_h)
                yolo_boxes.append((cls, cx, cy, w, bh))
                manifest.append({
                    "image": f"images/{split}/{fname}",
                    "label": "good_left_hand" if h == 0 else "good_right_hand",
                    "episode_id": video_key,
                    "factory": factory_name,
                    "frame_idx": fidx,
                    "track_id": None,
                    "handedness": h,
                    "bbox": bbox,
                    "confidence": None,
                    "source": "bad_annotation_low_iou_redetected",
                })
            # else: exclude → no box for this track on this frame

        # No detection background
        if fidx in set(no_det_frames) and fidx not in good_frame_info and fidx not in classified:
            manifest.append({
                "image": f"images/{split}/{fname}",
                "label": "background",
                "episode_id": video_key,
                "factory": factory_name,
                "frame_idx": fidx,
                "track_id": None,
                "handedness": None,
                "bbox": None,
                "confidence": None,
                "source": "no_detection",
            })

        save_yolo_label(str(label_path), yolo_boxes)

    return manifest


# ─── 主流程 ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="构建 YOLO 手部检测+质量分类器训练数据集")
    parser.add_argument("--db", required=True, help="annotations.db 路径")
    parser.add_argument("--factory-base", required=True, help="Factory 数据根目录")
    parser.add_argument("--factory-range", default="1-70", help="Factory 范围，如 1-70")
    parser.add_argument("--val-factory-start", type=int, default=61, help="Val 集起始 factory ID")
    parser.add_argument("--detector", default=None, help="YOLO detector.pt 路径（用于 IoU 区分）")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--no-detection-per-episode", type=int, default=5, help="每 episode 采样的无检测帧数")
    parser.add_argument("--iou-high", type=float, default=0.7, help="IoU 高阈值")
    parser.add_argument("--iou-low", type=float, default=0.3, help="IoU 低阈值")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析 factory 范围
    fstart, fend = map(int, args.factory_range.split("-"))

    # 加载标注
    print(f"📋 加载标注: {args.db}")
    annotations = load_annotations(args.db)
    alright_count = sum(1 for v in annotations.values() if v == "alright")
    bad_count = sum(1 for v in annotations.values() if v.startswith("BAD_FRAME"))
    print(f"  alright: {alright_count}, bad: {bad_count}, total: {len(annotations)}")

    # 加载 YOLO detector（如果提供）
    detector = None
    if args.detector and os.path.exists(args.detector):
        print(f"🔍 加载 YOLO 检测器: {args.detector}")
        try:
            from ultralytics import YOLO
            detector = YOLO(args.detector)
            print("  ✓ 检测器加载成功")
        except Exception as e:
            print(f"  ⚠ 检测器加载失败: {e}")
            print("  → bad episodes 的 IoU 区分将被跳过")
    elif bad_count > 0:
        print("  ⚠ 未提供 --detector，bad episodes 的 IoU 区分将被跳过")

    # 统计
    stats = defaultdict(int)
    all_manifest = []

    # 遍历 factories
    for fid in range(fstart, fend + 1):
        factory_name = f"factory{fid:03d}"
        factory_dir = os.path.join(args.factory_base, factory_name)
        split = "train" if fid < args.val_factory_start else "val"

        index = load_factory_index(factory_dir)
        if index is None:
            continue

        videos = index.get("videos", {})
        factory_count = 0

        for video_key, info in sorted(videos.items()):
            if video_key not in annotations:
                continue

            content = annotations[video_key]
            frames = info.get("frames", [])
            if not frames:
                continue

            # 解析帧信息
            if isinstance(frames[0], dict):
                frame_offsets = [[f["offset"], f["size"]] for f in frames]
            else:
                continue  # 无 offset 信息，跳过

            num_frames = len(frames)
            shard_path = os.path.join(factory_dir, info["shard"])
            seq_folder = os.path.join(factory_dir, "outputs", video_key)

            # 加载 tracks
            tracks_data, voted_h = load_tracks(seq_folder)
            if tracks_data is None:
                stats["skipped_no_tracks"] += 1
                continue

            # 处理
            if content == "alright":
                entries = process_alright_episode(
                    factory_name, video_key, shard_path, frame_offsets,
                    num_frames, tracks_data, voted_h, output_dir, split,
                    args.no_detection_per_episode,
                )
            elif content.startswith("BAD_FRAME"):
                bad_str = content.replace("BAD_FRAMES:", "").replace("BAD_FRAME:", "")
                bad_frame_list = [int(x) for x in bad_str.split(",") if x.strip()]
                entries = process_bad_episode(
                    factory_name, video_key, shard_path, frame_offsets,
                    num_frames, tracks_data, voted_h, bad_frame_list,
                    detector, output_dir, split,
                    args.no_detection_per_episode, args.iou_high, args.iou_low,
                )
            else:
                continue

            all_manifest.extend(entries)
            factory_count += 1

            # 统计
            for e in entries:
                stats[f"{split}_{e['label']}"] += 1

        if factory_count > 0:
            print(f"  ✓ {factory_name} ({split}): {factory_count} episodes 处理完成")

    # 写 dataset.yaml
    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": ["good_left_hand", "good_right_hand", "bad_hand"],
    }
    with open(output_dir / "dataset.yaml", "w") as f:
        import yaml
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    # 写 manifest.json
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(all_manifest, f, indent=2, ensure_ascii=False)

    # 写 stats.json
    stats_out = dict(stats)
    stats_out["total_images"] = len(all_manifest)
    stats_out["total_episodes"] = len(annotations)
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)

    # 打印统计
    print(f"\n{'=' * 60}")
    print(f"数据集构建完成: {output_dir}")
    print(f"{'=' * 60}")
    for k, v in sorted(stats_out.items()):
        print(f"  {k}: {v}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
