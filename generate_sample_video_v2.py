#!/usr/bin/env python3
"""
生成示例视频脚本 v2
直接使用 episode ID 或通过 API 搜索
"""

import sys
import os
import requests
import time

# 配置
SERVER_URL = "http://localhost:9472"
EPISODE_NAME_PATTERN = "part5_thread_unthread_bead_necklace_1647"
DATASET_NAME = "EgoDex"
OUTPUT_DIR = "/home/guantianrui/sample_videos"

def search_episode_by_api():
    """通过 API 调用 episode，让服务器自己去搜索数据集"""
    print(f"通过 debug API 获取数据集信息...")
    
    # 先获取数据集结构
    response = requests.get(f"{SERVER_URL}/api/debug/dataset/{DATASET_NAME}")
    if response.status_code != 200:
        print(f"获取数据集信息失败: {response.status_code}")
        return None
    
    data = response.json()
    print(f"数据集 {DATASET_NAME} 信息:")
    if 'meta_keys' in data and 'episode_names' in data['meta_keys']:
        ep_info = data['meta_keys']['episode_names']
        print(f"  总 episodes: {ep_info.get('shape', [0])[0]}")
        
        # 获取样本
        if 'sample' in ep_info:
            print(f"  前几个 episodes: {ep_info['sample']}")
    
    # 让用户提供 episode 索引
    print(f"\n由于 API 限制，我将尝试常见的索引范围...")
    
    # 尝试搜索前 2000 个 episodes（分批）
    for start_idx in range(0, 2000, 100):
        print(f"  正在测试索引 {start_idx}-{start_idx+100}...", end='', flush=True)
        
        for i in range(start_idx, start_idx + 100):
            episode_id = f"{DATASET_NAME}::{i}"
            
            # 尝试获取 episode 信息
            try:
                response = requests.get(f"{SERVER_URL}/api/episode/{episode_id}?user_id=test_user", timeout=5)
                if response.status_code == 200:
                    ep_data = response.json()
                    if EPISODE_NAME_PATTERN in ep_data['episode_name']:
                        print(f"\n✓ 找到!")
                        print(f"  ID: {episode_id}")
                        print(f"  名称: {ep_data['episode_name']}")
                        print(f"  帧数: {ep_data['num_frames']}")
                        return episode_id
            except:
                continue
        
        print(" 未找到")
    
    return None

def download_video(episode_id, video_type):
    """下载视频"""
    print(f"\n正在生成并下载 {video_type} 视频...")
    print(f"  (首次生成可能需要较长时间，请耐心等待...)")
    
    url = f"{SERVER_URL}/api/episode/{episode_id}/video/{video_type}"
    
    start_time = time.time()
    
    # 发送请求（设置较长的超时时间，因为首次生成视频需要时间）
    try:
        response = requests.get(url, stream=True, timeout=300)
    except requests.Timeout:
        print(f"\n✗ 请求超时（超过 5 分钟）")
        return False
    
    if response.status_code != 200:
        print(f"\n✗ 下载失败: HTTP {response.status_code}")
        return False
    
    # 保存视频
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{EPISODE_NAME_PATTERN}_{video_type}.mp4")
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\r  下载进度: {progress:.1f}%", end='', flush=True)
    
    elapsed = time.time() - start_time
    print(f"\n✓ 视频已保存: {output_path}")
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件大小: {file_size:.2f} MB")
    print(f"  耗时: {elapsed:.1f} 秒")
    return True

def main():
    print("=" * 60)
    print("生成示例视频 (v2)")
    print("=" * 60)
    
    # 方式1: 尝试直接使用已知的 episode ID（如果你知道的话）
    # 你可以在这里直接设置 episode_id
    episode_id = None  # 例如: "EgoDex::1234"
    
    # 方式2: 通过 API 搜索
    if not episode_id:
        episode_id = search_episode_by_api()
    
    if not episode_id:
        print("\n✗ 未找到指定的 episode")
        print("\n提示: 如果你知道 episode 的索引号，可以直接在脚本中设置 episode_id")
        sys.exit(1)
    
    # 下载原始视频
    print("\n" + "-" * 60)
    success = download_video(episode_id, 'original')
    if not success:
        print("\n✗ 下载原始视频失败")
        sys.exit(1)
    
    # 下载渲染视频
    print("\n" + "-" * 60)
    success = download_video(episode_id, 'rendered')
    if not success:
        print("\n✗ 下载渲染视频失败")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ 所有视频生成完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 列出生成的文件
    print("\n生成的文件:")
    for f in os.listdir(OUTPUT_DIR):
        if EPISODE_NAME_PATTERN in f:
            fpath = os.path.join(OUTPUT_DIR, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
