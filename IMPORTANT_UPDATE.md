# ⚠️ 重要更新 - 数据来源变更

## 立即行动

### 变更内容
**数据读取位置已改变**：
- ❌ 不再从 `action` 读取
- ✅ 改为从 `state` 读取

### 受影响的数据
```
之前 (action) → 现在 (state)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
data/action/mano       → data/state/mano ⭐
data/action/wrist      → data/state/wrist ⭐
data/action/fingertips → data/state/fingertips ⭐
```

## 快速修复

### 方案 1: 数据迁移（推荐）

运行以下 Python 代码：
```python
import zarr

# 打开数据集
root = zarr.open("/path/to/your/dataset.zarr", mode='a')

# 确保 state 组存在
if 'state' not in root['data']:
    root['data'].create_group('state')

# 复制数据
root['data/state/mano'] = root['data/action/mano'][:]
root['data/state/wrist'] = root['data/action/wrist'][:]
root['data/state/fingertips'] = root['data/action/fingertips'][:]

print("✓ 数据迁移完成！")
```

### 方案 2: 验证现有数据

如果你的数据已经在 `state` 中，只需验证：
```bash
python inspect_zarr.py /path/to/your/dataset.zarr
```

确保看到：
```
✓ data/state/mano
✓ data/state/wrist
✓ data/state/fingertips (可选)
```

## 测试

```bash
# 1. 启动服务
cd /home/guantianrui/zarr-viewer-stage2
python app.py

# 2. 访问 http://localhost:8300
# 3. 选择一个 episode
# 4. 查看渲染视频
# 5. 确认手部正常显示
```

## 为什么改变？

1. **语义清晰**：state（状态）vs action（动作）
2. **代码简洁**：移除了复杂的 fallback 逻辑
3. **标准对齐**：符合 robotics 领域惯例

## 需要帮助？

详细文档：
- [DATA_SOURCE_UPDATE.md](DATA_SOURCE_UPDATE.md) - 完整迁移指南
- [FINGERTIPS_RENDERING.md](FINGERTIPS_RENDERING.md) - 渲染说明
- [CHANGELOG.md](CHANGELOG.md) - 变更记录

---

**重要**：此更新不向后兼容！使用前请确保数据在 `state/` 下。

