# 视频缓存管理与 EgoDex 数据集处理

## 更新时间
2026-02-01

## 1. EgoDex 数据集 MANO 跳过策略

### 问题
EgoDex 数据集的 MANO 参数数据质量不佳，导致渲染的手部姿态异常。

### 解决方案
针对 EgoDex 数据集，跳过 MANO 网格渲染，仅渲染 fingertips 点：

```python
# 检测是否是 EgoDex 数据集
is_egodex = "egodex" in episode_id.lower()

# 跳过 MANO 相关参数
if not is_egodex:
    # 读取 mano, wrist, shape 参数
    # 初始化 MANO layers
    # 渲染 MANO 网格
```

### 影响范围
- `api_episode_video_rendered`: 渲染视频生成函数
- 只影响 episode_id 中包含 "egodex" (不区分大小写) 的数据集
- 其他数据集保持原有的完整 MANO 渲染

### 识别逻辑
```python
is_egodex = "egodex" in episode_id.lower()
```

示例：
- `EgoDex::episode_001` → 跳过 MANO
- `egodex_v2::test` → 跳过 MANO  
- `Epic::42285` → 正常渲染 MANO
- `TACO::001` → 正常渲染 MANO

---

## 2. 自动视频缓存清理

### 问题
视频缓存存储在 `/DATA/guantianrui/tmp/zarr_video_cache`，长期运行可能占满磁盘空间。

### 解决方案
实现了 `cleanup_video_cache()` 函数，采用双重清理策略：

#### 策略 1: 基于文件年龄
- 自动删除超过 **7 天**的缓存文件
- 避免陈旧缓存长期占用空间

#### 策略 2: 基于总大小限制
- 缓存目录总大小上限: **10 GB**
- 超过限制时，删除最旧的文件
- 清理到总大小的 **80%** 以下（即 8 GB）

### 函数签名
```python
def cleanup_video_cache(cache_dir: str, max_size_gb: float = 10.0, max_age_days: int = 7):
    """清理视频缓存目录
    
    Args:
        cache_dir: 缓存目录路径
        max_size_gb: 最大缓存大小（GB），默认 10 GB
        max_age_days: 文件最大保留天数，默认 7 天
    """
```

### 调用时机
在每次生成视频前自动调用（无需手动干预）：
- `api_episode_video_original`: 生成原始视频前
- `api_episode_video_rendered`: 生成渲染视频前

### 清理逻辑
```python
# 创建缓存目录
cache_dir = "/DATA/guantianrui/tmp/zarr_video_cache"
os.makedirs(cache_dir, exist_ok=True)

# 自动清理
cleanup_video_cache(cache_dir, max_size_gb=10.0, max_age_days=7)
```

### 日志输出示例
```
🗑️ 删除过期缓存: abc123.mp4 (保留了 8.3 天)
🗑️ 删除旧缓存（空间不足）: def456.mp4
✓ 缓存清理完成: 删除 5 个文件，释放 1234.56 MB
  当前缓存大小: 7890.12 MB / 10240.00 MB
```

### 配置调整
如需调整清理策略，修改调用参数：
```python
# 更大的缓存空间（20 GB）和更长的保留时间（14 天）
cleanup_video_cache(cache_dir, max_size_gb=20.0, max_age_days=14)

# 更小的缓存空间（5 GB）和更短的保留时间（3 天）
cleanup_video_cache(cache_dir, max_size_gb=5.0, max_age_days=3)
```

---

## 3. 使用说明

### 正常使用
无需任何额外操作，系统会自动：
1. 检测 EgoDex 数据集并跳过 MANO 渲染
2. 在生成视频前自动清理过期和超大缓存

### 手动清理缓存
如需立即清空所有缓存：
```bash
bash clear_video_cache.sh
# 或
rm -rf /DATA/guantianrui/tmp/zarr_video_cache/*
```

### 查看缓存状态
```bash
# 查看缓存大小
du -sh /DATA/guantianrui/tmp/zarr_video_cache

# 查看缓存文件数量
ls /DATA/guantianrui/tmp/zarr_video_cache | wc -l

# 查看文件详情（按修改时间排序）
ls -lht /DATA/guantianrui/tmp/zarr_video_cache
```

---

## 4. 技术细节

### 文件结构
```
/DATA/guantianrui/tmp/zarr_video_cache/
├── abc123def456...mp4  # MD5(episode_id + "_original")
├── 789ghi012jkl...mp4  # MD5(episode_id + "_rendered")
└── ...
```

### 缓存命中
- 缓存文件存在 → 直接返回（毫秒级）
- 缓存文件不存在 → 生成视频 → 保存缓存 → 返回

### 清理频率
- 每次视频请求时触发一次检查
- 清理操作本身非常快速（<100ms）
- 不影响视频生成和返回速度

### 线程安全
当前实现为单进程，不考虑多进程并发问题。如部署多个 Flask worker，可能需要：
- 使用文件锁
- 或使用外部缓存系统（Redis + 文件存储）

---

## 5. 监控建议

### 定期检查
```bash
# 每天查看缓存状态
watch -n 86400 'du -sh /DATA/guantianrui/tmp/zarr_video_cache'

# 查看最近的清理日志
grep "缓存清理" /path/to/flask.log
```

### 告警阈值
- 缓存大小持续接近 10 GB → 可能需要增加限制或缩短保留时间
- 频繁触发空间清理 → 考虑增加 max_size_gb
- 大量过期文件 → 可能服务长期未使用，可手动清空

---

## 6. 故障排查

### 视频加载失败
1. 检查缓存目录权限: `ls -ld /DATA/guantianrui/tmp/zarr_video_cache`
2. 检查磁盘空间: `df -h /tmp`
3. 手动清理缓存: `bash clear_video_cache.sh`
4. 重启服务: `pkill -f flask` 然后重启

### 缓存未清理
1. 检查日志中是否有清理输出
2. 确认 `cleanup_video_cache()` 被正确调用
3. 检查文件权限和所有者

### EgoDex 仍显示异常 MANO
1. 确认 episode_id 包含 "egodex" 字符串
2. 清空缓存: `rm -rf /DATA/guantianrui/tmp/zarr_video_cache/*`
3. 检查日志: 应看到 "检测到 EgoDex 数据集，将跳过 MANO 渲染"
4. 重新加载页面

