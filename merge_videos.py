#!/usr/bin/env python3
"""
合并原始视频和渲染视频（左右并排）
"""

import subprocess
import os
import sys

# 配置
EPISODE_NAME = "part5_thread_unthread_bead_necklace_1647"
VIDEO_DIR = "/home/guantianrui/sample_videos"
ORIGINAL_VIDEO = os.path.join(VIDEO_DIR, f"{EPISODE_NAME}_original.mp4")
RENDERED_VIDEO = os.path.join(VIDEO_DIR, f"{EPISODE_NAME}_rendered.mp4")
OUTPUT_VIDEO = os.path.join(VIDEO_DIR, f"{EPISODE_NAME}_combined.mp4")

def merge_videos_side_by_side():
    """使用 ffmpeg 将两个视频左右并排合并"""
    
    # 检查输入文件是否存在
    if not os.path.exists(ORIGINAL_VIDEO):
        print(f"✗ 原始视频不存在: {ORIGINAL_VIDEO}")
        return False
    
    if not os.path.exists(RENDERED_VIDEO):
        print(f"✗ 渲染视频不存在: {RENDERED_VIDEO}")
        return False
    
    print("=" * 60)
    print("合并视频（左右并排）")
    print("=" * 60)
    print(f"左侧（原始）: {ORIGINAL_VIDEO}")
    print(f"右侧（渲染）: {RENDERED_VIDEO}")
    print(f"输出文件: {OUTPUT_VIDEO}")
    print()
    
    # 使用 ffmpeg 并排合并视频
    # [0:v] 是原始视频，[1:v] 是渲染视频
    # hstack 将两个视频水平堆叠（左右并排）
    cmd = [
        'ffmpeg',
        '-i', ORIGINAL_VIDEO,
        '-i', RENDERED_VIDEO,
        '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
        '-map', '[v]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-y',  # 覆盖输出文件
        OUTPUT_VIDEO
    ]
    
    print("正在合并视频...")
    print(f"命令: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode != 0:
            print(f"✗ 合并失败!")
            print(f"错误信息: {result.stderr}")
            return False
        
        # 检查输出文件
        if os.path.exists(OUTPUT_VIDEO):
            file_size = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
            print("=" * 60)
            print("✓ 合并成功!")
            print(f"输出文件: {OUTPUT_VIDEO}")
            print(f"文件大小: {file_size:.2f} MB")
            print("=" * 60)
            return True
        else:
            print("✗ 输出文件未生成")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ 合并超时（超过 5 分钟）")
        return False
    except Exception as e:
        print(f"✗ 合并出错: {e}")
        return False

def main():
    success = merge_videos_side_by_side()
    
    if success:
        print("\n提示: 你可以使用视频播放器打开合并后的视频:")
        print(f"  {OUTPUT_VIDEO}")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
