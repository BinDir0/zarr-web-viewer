#!/usr/bin/env python3
"""
生成并合并视频的完整脚本
"""

import sys
import os
import requests
import subprocess
import time

# 配置
SERVER_URL = "http://localhost:9472"
OUTPUT_DIR = "/home/guantianrui/sample_videos"

def find_episode_id(episode_name, dataset_name="EgoDex"):
    """查找 episode ID - 直接从数据集搜索"""
    print(f"正在搜索 episode: {episode_name} (数据集: {dataset_name})")
    
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
            if episode_name in name_str:
                episode_id = f"{dataset_name}::{i}"
                ep_end = int(meta['episode_ends'][i])
                ep_start = 0 if i == 0 else int(meta['episode_ends'][i-1])
                num_frames = ep_end - ep_start
                
                print(f"✓ 找到 Episode:")
                print(f"  ID: {episode_id}")
                print(f"  索引: {i}")
                print(f"  名称: {name_str}")
                print(f"  帧数: {num_frames}")
                return episode_id, name_str
        
        print(f"✗ 未找到匹配的 episode")
        return None, None
    except Exception as e:
        print(f"✗ 搜索失败: {e}")
        return None, None

def download_video(episode_id, video_type, output_filename):
    """下载视频"""
    print(f"\n正在生成并下载 {video_type} 视频...")
    
    url = f"{SERVER_URL}/api/episode/{episode_id}/video/{video_type}"
    
    start_time = time.time()
    
    # 发送请求
    try:
        response = requests.get(url, stream=True, timeout=300)
    except requests.Timeout:
        print(f"✗ 请求超时")
        return False
    
    if response.status_code != 200:
        print(f"✗ 下载失败: HTTP {response.status_code}")
        return False
    
    # 保存视频
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
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
    
    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ 视频已保存: {output_path}")
    print(f"  文件大小: {file_size:.2f} MB | 耗时: {elapsed:.1f}秒")
    return output_path

def merge_videos(original_path, rendered_path, output_path):
    """合并视频（左右并排）"""
    print(f"\n正在合并视频（左右并排）...")
    print(f"  左侧: {os.path.basename(original_path)}")
    print(f"  右侧: {os.path.basename(rendered_path)}")
    
    cmd = [
        'ffmpeg',
        '-i', original_path,
        '-i', rendered_path,
        '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
        '-map', '[v]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-y',
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"✗ 合并失败: {result.stderr}")
            return False
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✓ 合并成功: {output_path}")
            print(f"  文件大小: {file_size:.2f} MB")
            return True
        else:
            print("✗ 输出文件未生成")
            return False
            
    except Exception as e:
        print(f"✗ 合并出错: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python generate_and_merge.py <episode_name>")
        print("示例: python generate_and_merge.py part1_clean_tableware_859")
        sys.exit(1)
    
    episode_name = sys.argv[1]
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "EgoDex"
    
    print("=" * 70)
    print(f"生成并合并视频: {episode_name}")
    print("=" * 70)
    
    # 1. 查找 episode
    episode_id, full_name = find_episode_id(episode_name, dataset_name)
    if not episode_id:
        print("\n✗ 未找到指定的 episode")
        sys.exit(1)
    
    # 使用完整名称作为文件名前缀
    base_name = episode_name
    
    # 2. 下载原始视频
    print("\n" + "-" * 70)
    original_path = download_video(episode_id, 'original', f"{base_name}_original.mp4")
    if not original_path:
        print("\n✗ 下载原始视频失败")
        sys.exit(1)
    
    # 3. 下载渲染视频
    print("\n" + "-" * 70)
    rendered_path = download_video(episode_id, 'rendered', f"{base_name}_rendered.mp4")
    if not rendered_path:
        print("\n✗ 下载渲染视频失败")
        sys.exit(1)
    
    # 4. 合并视频
    print("\n" + "-" * 70)
    combined_path = os.path.join(OUTPUT_DIR, f"{base_name}_combined.mp4")
    success = merge_videos(original_path, rendered_path, combined_path)
    
    if not success:
        print("\n✗ 合并视频失败")
        sys.exit(1)
    
    # 5. 总结
    print("\n" + "=" * 70)
    print("✓ 所有视频生成完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)
    print("\n生成的文件:")
    for filename in [f"{base_name}_original.mp4", f"{base_name}_rendered.mp4", f"{base_name}_combined.mp4"]:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
