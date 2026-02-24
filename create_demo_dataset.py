#!/usr/bin/env python3
"""
创建一个演示用的 zarr 数据集
用于测试 zarr-viewer 工具
"""

import numpy as np
import zarr
from pathlib import Path


def create_demo_dataset(output_path: str = "./demo_dataset.zarr", num_episodes: int = 5):
    """创建一个演示数据集"""
    
    print(f"创建演示数据集: {output_path}")
    print(f"Episodes 数量: {num_episodes}")
    
    # 创建 zarr 根组
    root = zarr.open(output_path, mode="w")
    
    # 创建 data 组
    data_group = root.create_group("data")
    
    for i in range(num_episodes):
        episode_name = f"episode_{i:04d}"
        print(f"  创建 {episode_name}...")
        
        # 创建 episode 组
        episode = data_group.create_group(episode_name)
        
        # 生成随机参数
        num_frames = np.random.randint(50, 150)
        height, width = 480, 640
        
        # 创建图像数据 (生成彩色渐变图像)
        images = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        for t in range(num_frames):
            # 创建时间变化的渐变图像
            x = np.linspace(0, 255, width).astype(np.uint8)
            y = np.linspace(0, 255, height).astype(np.uint8)
            
            # 随时间变化的颜色
            phase = (t / num_frames) * 2 * np.pi
            r = (np.sin(phase) + 1) / 2
            g = (np.sin(phase + 2*np.pi/3) + 1) / 2
            b = (np.sin(phase + 4*np.pi/3) + 1) / 2
            
            xx, yy = np.meshgrid(x, y)
            images[t, :, :, 0] = (xx * r).astype(np.uint8)
            images[t, :, :, 1] = (yy * g).astype(np.uint8)
            images[t, :, :, 2] = ((xx + yy) / 2 * b).astype(np.uint8)
        
        # 保存图像数据
        episode.create_dataset("images", data=images, chunks=(1, height, width, 3))
        
        # 创建动作数据 (7维: 3D位置 + 4D旋转四元数)
        actions = np.random.randn(num_frames, 7).astype(np.float32)
        # 归一化四元数部分
        actions[:, 3:] /= np.linalg.norm(actions[:, 3:], axis=1, keepdims=True)
        episode.create_dataset("actions", data=actions)
        
        # 添加一些元数据
        episode.attrs["num_frames"] = num_frames
        episode.attrs["task_description"] = f"演示任务 {i+1}"
        episode.attrs["success"] = bool(np.random.rand() > 0.3)
    
    print(f"✓ 演示数据集创建完成: {output_path}")
    print(f"  包含 {num_episodes} 个 episodes")
    print(f"\n使用以下命令查看结构:")
    print(f"  python inspect_zarr.py {output_path}")
    print(f"\n在 config.yaml 中设置:")
    print(f"  zarr_dataset_path: {Path(output_path).resolve()}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = "./demo_dataset.zarr"
    
    if len(sys.argv) > 2:
        num_episodes = int(sys.argv[2])
    else:
        num_episodes = 5
    
    create_demo_dataset(output_path, num_episodes)

