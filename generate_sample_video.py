#!/usr/bin/env python3
"""
生成示例视频脚本
用于生成指定 episode 的原始视频和渲染视频
"""

import sys
import os
import requests
import time

# 配置
SERVER_URL = "http://localhost:9472"
EPISODE_NAME = "part5_thread_unthread_bead_necklace_1647"
DATASET_NAME = "EgoDex"
OUTPUT_DIR = "/home/guantianrui/sample_videos"

def find_episode_id():
    """查找 episode ID - 直接从数据集搜索"""
    print(f"正在搜索 episode: {EPISODE_NAME} (数据集: {DATASET_NAME})")
    
    # 直接从数据集中搜索
    import zarr
    
    zarr_path = "/share_data/guantianrui/datasets/EgoDex/egodex_train_filtered_v2.zarr"
    try:
        zarr_root = zarr.open(zarr_path, mode='r')
        meta = zarr_root['meta']
        episode_names = meta['episode_names'][:]
        
        # 搜索匹配的 episode
        for i, name in enumerate(episode_names):
            name_str = name.decode('utf-8') if isinstance(name, bytes) else str(name)
            if EPISODE_NAME in name_str:
                episode_id = f"{DATASET_NAME}::{i}"
                ep_end = int(meta['episode_ends'][i])
                ep_start = 0 if i == 0 else int(meta['episode_ends'][i-1])
                num_frames = ep_end - ep_start
                
                print(f"找到 Episode:")
                print(f"  ID: {episode_id}")
                print(f"  索引: {i}")
                print(f"  名称: {name_str}")
                print(f"  数据集: {DATASET_NAME}")
                print(f"  帧数: {num_frames}")
                return episode_id
        
        print(f"未找到匹配的 episode")
        return None
    except Exception as e:
        print(f"搜索失败: {e}")
        return None

def download_video(episode_id, video_type):
    """下载视频"""
    print(f"\n正在下载 {video_type} 视频...")
    
    url = f"{SERVER_URL}/api/episode/{episode_id}/video/{video_type}"
    
    # 发送请求
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        print(f"下载失败: {response.status_code}")
        return False
    
    # 保存视频
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{EPISODE_NAME}_{video_type}.mp4")
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\r  进度: {progress:.1f}%", end='', flush=True)
    
    print(f"\n✓ 视频已保存: {output_path}")
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件大小: {file_size:.2f} MB")
    return True

def main():
    print("=" * 60)
    print("生成示例视频")
    print("=" * 60)
    
    # 1. 查找 episode ID
    episode_id = find_episode_id()
    if not episode_id:
        print("\n✗ 未找到指定的 episode")
        sys.exit(1)
    
    # 2. 下载原始视频
    success = download_video(episode_id, 'original')
    if not success:
        print("\n✗ 下载原始视频失败")
        sys.exit(1)
    
    # 3. 下载渲染视频
    success = download_video(episode_id, 'rendered')
    if not success:
        print("\n✗ 下载渲染视频失败")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ 所有视频生成完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
