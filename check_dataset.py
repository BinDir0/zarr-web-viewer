#!/usr/bin/env python3
"""快速检查数据集结构"""
import sys
import zarr

path = "/share_data/guantianrui/datasets/EgoDex/egodex_train_filtered_v2.zarr"
if len(sys.argv) > 1:
    path = sys.argv[1]

print(f"检查数据集: {path}\n")

try:
    root = zarr.open(path, mode='r')
    
    print("=" * 60)
    print("根目录的键:")
    print("=" * 60)
    for key in root.keys():
        item = root[key]
        if isinstance(item, zarr.Group):
            print(f"  - {key}/ (Group)")
        else:
            print(f"  - {key}: shape={item.shape}, dtype={item.dtype}")
    
    print("\n" + "=" * 60)
    if "meta" in root:
        print("meta/ 组的内容:")
        print("=" * 60)
        meta = root["meta"]
        for key in meta.keys():
            item = meta[key]
            if hasattr(item, 'shape'):
                print(f"  - {key}: shape={item.shape}, dtype={item.dtype}")
                # 显示前几个值
                if len(item) > 0:
                    sample = item[:min(3, len(item))]
                    print(f"    示例值: {sample}")
            else:
                print(f"  - {key}")
    else:
        print("⚠️  未找到 meta 组")
    
    print("\n" + "=" * 60)
    if "data" in root:
        print("data/ 组的内容:")
        print("=" * 60)
        data = root["data"]
        for key in list(data.keys())[:15]:
            item = data[key]
            if isinstance(item, zarr.Group):
                print(f"  - {key}/ (Group)")
                # 显示子键
                subkeys = list(item.keys())[:5]
                for subkey in subkeys:
                    subitem = item[subkey]
                    if hasattr(subitem, 'shape'):
                        print(f"      - {subkey}: shape={subitem.shape}")
            elif hasattr(item, 'shape'):
                print(f"  - {key}: shape={item.shape}, dtype={item.dtype}")
    else:
        print("⚠️  未找到 data 组")
        
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

