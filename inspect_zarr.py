#!/usr/bin/env python3
"""
检查 zarr 数据集结构的工具脚本
用于了解你的数据集的具体结构，以便调整 app.py 中的数据读取代码
"""

import sys
from pathlib import Path

import zarr
import numpy as np


def print_tree(group, prefix="", max_depth=5, current_depth=0):
    """递归打印 zarr 组的树状结构"""
    if current_depth >= max_depth:
        print(f"{prefix}... (达到最大深度)")
        return
    
    try:
        items = list(group.items())
    except:
        print(f"{prefix}(无法列出内容)")
        return
    
    for i, (key, value) in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = prefix + ("    " if is_last else "│   ")
        
        if isinstance(value, zarr.Group):
            print(f"{prefix}{current_prefix}{key}/ (Group)")
            print_tree(value, next_prefix, max_depth, current_depth + 1)
        elif isinstance(value, zarr.Array):
            shape = value.shape
            dtype = value.dtype
            print(f"{prefix}{current_prefix}{key} (Array: shape={shape}, dtype={dtype})")


def inspect_zarr_dataset(zarr_path: str):
    """检查 zarr 数据集"""
    path = Path(zarr_path)
    
    if not path.exists():
        print(f"❌ 错误: 路径不存在: {zarr_path}")
        return
    
    print(f"📁 检查 zarr 数据集: {zarr_path}\n")
    
    try:
        root = zarr.open(str(path), mode="r")
    except Exception as e:
        print(f"❌ 无法打开 zarr 数据集: {e}")
        return
    
    print("=" * 60)
    print("数据集结构:")
    print("=" * 60)
    print_tree(root)
    
    print("\n" + "=" * 60)
    print("数据集根目录包含的键:")
    print("=" * 60)
    try:
        keys = list(root.keys())
        for key in keys:
            item = root[key]
            if isinstance(item, zarr.Group):
                print(f"  - {key}/ (Group)")
            elif isinstance(item, zarr.Array):
                print(f"  - {key} (Array: {item.shape}, {item.dtype})")
    except Exception as e:
        print(f"错误: {e}")
    
    # 尝试找到 episode
    print("\n" + "=" * 60)
    print("查找 episodes:")
    print("=" * 60)
    
    episodes_found = False
    
    # 检查常见的结构
    possible_paths = [
        "data",
        "episodes",
        "",  # 根目录
    ]
    
    for base_path in possible_paths:
        try:
            if base_path == "":
                group = root
            else:
                if base_path not in root:
                    continue
                group = root[base_path]
            
            if isinstance(group, zarr.Group):
                episode_keys = [k for k in group.keys() if "episode" in k.lower() or k.startswith("ep")]
                if episode_keys:
                    episodes_found = True
                    print(f"\n在 '{base_path or '根目录'}' 找到 {len(episode_keys)} 个 episodes:")
                    for ep_key in episode_keys[:5]:  # 只显示前5个
                        print(f"  - {ep_key}")
                        ep = group[ep_key]
                        if isinstance(ep, zarr.Group):
                            ep_keys = list(ep.keys())
                            print(f"    包含: {', '.join(ep_keys)}")
                            
                            # 检查图像数据
                            if "observations" in ep:
                                obs = ep["observations"]
                                if isinstance(obs, zarr.Group) and "images" in obs:
                                    images = obs["images"]
                                    if isinstance(images, zarr.Array):
                                        print(f"    图像形状: {images.shape}, dtype: {images.dtype}")
                            elif "images" in ep:
                                images = ep["images"]
                                if isinstance(images, zarr.Array):
                                    print(f"    图像形状: {images.shape}, dtype: {images.dtype}")
                            
                            # 检查动作数据
                            if "actions" in ep:
                                actions = ep["actions"]
                                if isinstance(actions, zarr.Array):
                                    print(f"    动作形状: {actions.shape}, dtype: {actions.dtype}")
                    
                    if len(episode_keys) > 5:
                        print(f"  ... 还有 {len(episode_keys) - 5} 个 episodes")
                    break
        except Exception as e:
            print(f"检查 '{base_path}' 时出错: {e}")
    
    if not episodes_found:
        print("⚠️  未找到 episodes，请检查数据集结构")
        print("提示: 你可能需要修改 app.py 中的 get_episode_list() 和 get_episode_data() 函数")
    
    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python inspect_zarr.py <zarr_dataset_path>")
        print("示例: python inspect_zarr.py /path/to/dataset.zarr")
        sys.exit(1)
    
    zarr_path = sys.argv[1]
    inspect_zarr_dataset(zarr_path)

