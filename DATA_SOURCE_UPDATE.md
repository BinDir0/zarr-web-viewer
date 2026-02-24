# 数据来源更新说明

## 变更日期
2026-02-01

## 重要变更

### 从 action 改为 state

**之前的数据来源**：
- ❌ `data/action/mano` - MANO 姿态参数
- ❌ `data/action/wrist` - 手腕位置和旋转
- ❌ `data/action/fingertips` - 指尖位置

**现在的数据来源**：
- ✅ `data/state/mano` - MANO 姿态参数
- ✅ `data/state/wrist` - 手腕位置和旋转
- ✅ `data/state/fingertips` - 指尖位置

## 影响范围

### 代码变更
- ✅ `app.py` - 已更新，从 state 读取数据
- ✅ 自动 fallback 机制已移除
- ✅ 只从 state 读取，不再尝试 action

### 文档更新
- ✅ `FINGERTIPS_RENDERING.md` - 已更新数据路径
- ✅ `CHANGELOG.md` - 已添加说明
- ✅ `TEST_GUIDE.md` - 已更新数据要求
- ✅ `QUICK_START_MANO.md` - 已更新必需字段

## 数据集要求更新

### 必需字段（新）
```
data/
├── image (N, H, W, 3)          # RGB 图像
├── state/
│   ├── mano (N, 90)            # MANO 姿态参数 ⭐
│   └── wrist (N, 18)           # 手腕位置和旋转 ⭐
├── intrinsic (N, 3, 3)         # 相机内参
└── extrinsic (N, 4, 4)         # 相机外参
```

### 可选字段（新）
```
data/
└── state/
    ├── fingertips (N, 30)      # 指尖位置 ⭐ 新增
    └── shape (N, 20)           # 手部形状参数
```

## 迁移指南

### 如果你的数据在 action 中

**选项 1：重新组织数据**（推荐）
```python
# 将 action 中的数据复制到 state
zarr_root['data/state/mano'] = zarr_root['data/action/mano'][:]
zarr_root['data/state/wrist'] = zarr_root['data/action/wrist'][:]
zarr_root['data/state/fingertips'] = zarr_root['data/action/fingertips'][:]
```

**选项 2：创建软链接**（如果数据格式相同）
```python
# 在 zarr 中创建引用
# 注意：需要根据你的 zarr 版本选择合适的方法
```

**选项 3：使用旧版本代码**
- 回退到之前的版本
- 或者自行修改代码以支持 action

### 验证数据

使用 `inspect_zarr.py` 检查数据结构：
```bash
python inspect_zarr.py /path/to/your/dataset.zarr
```

确保输出中包含：
```
data/
├── state/
│   ├── mano ✓
│   ├── wrist ✓
│   └── fingertips ✓ (可选)
```

## 为什么做这个变更？

### 1. 数据语义更清晰
- **state** = 当前状态（观测值）
- **action** = 动作指令（控制量）
- 手部姿态是"状态"而非"动作"

### 2. 与标准格式对齐
- 遵循 robotics 领域的命名惯例
- state-action 分离更清晰

### 3. 简化代码逻辑
- 移除了 fallback 机制
- 减少了条件判断
- 代码更简洁可维护

## 兼容性说明

### ⚠️ 不兼容变更
此更新**不向后兼容**。如果你的数据集使用的是：
- `data/action/mano`
- `data/action/wrist`
- `data/action/fingertips`

你需要：
1. 重新组织数据到 `state/` 下
2. 或使用旧版本代码

### 受影响的功能
- ✅ 视频生成功能 - 需要 state 数据
- ✅ MANO 渲染 - 需要 state/mano
- ✅ Fingertips 渲染 - 需要 state/fingertips
- ❌ API 响应 - 不受影响（只返回元信息）
- ❌ Episode 列表 - 不受影响

## 测试清单

更新数据源后，请验证：

- [ ] 服务器正常启动
- [ ] Episode 列表正常显示
- [ ] 选择 episode 显示详情
- [ ] 原始视频可以播放
- [ ] 渲染视频可以播放
- [ ] MANO 手部正常显示
- [ ] Fingertips 指尖正常显示（如果有数据）
- [ ] 手部位置准确
- [ ] 无错误日志

## 示例代码

### 数据迁移脚本

```python
import zarr

# 打开数据集
zarr_path = "/path/to/your/dataset.zarr"
root = zarr.open(zarr_path, mode='a')

# 确保 state 组存在
if 'state' not in root['data']:
    root['data'].create_group('state')

# 复制数据
if 'mano' in root['data/action']:
    root['data/state/mano'] = root['data/action/mano'][:]
    print("✓ 复制 mano")

if 'wrist' in root['data/action']:
    root['data/state/wrist'] = root['data/action/wrist'][:]
    print("✓ 复制 wrist")

if 'fingertips' in root['data/action']:
    root['data/state/fingertips'] = root['data/action/fingertips'][:]
    print("✓ 复制 fingertips")

print("数据迁移完成！")
```

### 验证数据脚本

```python
import zarr

zarr_path = "/path/to/your/dataset.zarr"
root = zarr.open(zarr_path, mode='r')

# 检查必需字段
required = ['state/mano', 'state/wrist']
for field in required:
    path = f'data/{field}'
    if path in root:
        print(f"✓ {field}: {root[path].shape}")
    else:
        print(f"✗ {field}: 缺失！")

# 检查可选字段
optional = ['state/fingertips', 'state/shape']
for field in optional:
    path = f'data/{field}'
    if path in root:
        print(f"✓ {field}: {root[path].shape}")
    else:
        print(f"○ {field}: 不存在（可选）")
```

## 常见问题

### Q1: 为什么不保持向后兼容？
A: 为了代码简洁和语义清晰。fallback 机制会增加复杂度。

### Q2: 如果我只有 action 数据怎么办？
A: 运行数据迁移脚本，或手动复制数据到 state 组。

### Q3: 这会影响已有的视频吗？
A: 不会。已生成的视频文件不受影响。

### Q4: 性能会有变化吗？
A: 没有变化。只是读取路径不同。

## 技术细节

### 数据格式对比

**MANO 参数**：
```
之前: data/action/mano (N, 90)
现在: data/state/mano (N, 90)
格式: [左手45维, 右手45维]
```

**手腕数据**：
```
之前: data/action/wrist (N, 18)
现在: data/state/wrist (N, 18)
格式: [L_trans(3), R_trans(3), L_rot6(6), R_rot6(6)]
```

**指尖位置**：
```
之前: data/action/fingertips (N, 30)
现在: data/state/fingertips (N, 30)
格式: [左手5指×3, 右手5指×3]
```

### 代码变更对比

**之前**：
```python
# 优先 state，fallback 到 action
if "mano" in state_data:
    mano = state_data["mano"][i]
elif "mano" in action_data:
    mano = action_data["mano"][i]
```

**现在**：
```python
# 只从 state 读取
if "mano" in state_data:
    mano = state_data["mano"][i]
```

## 获取帮助

如果遇到问题：
1. 检查数据集结构：`python inspect_zarr.py dataset.zarr`
2. 查看服务器日志
3. 运行数据迁移脚本
4. 查看相关文档

## 相关文档

- [FINGERTIPS_RENDERING.md](FINGERTIPS_RENDERING.md) - Fingertips 渲染说明
- [TEST_GUIDE.md](TEST_GUIDE.md) - 测试指南
- [CHANGELOG.md](CHANGELOG.md) - 变更日志
- [QUICK_START_MANO.md](QUICK_START_MANO.md) - 快速开始

---

**重要**：请在更新代码后检查你的数据集格式，确保数据在正确的位置。

