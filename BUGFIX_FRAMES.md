# Bug 修复 - frames.length 错误

## 问题描述

**错误信息**：
```
TypeError: Cannot read properties of undefined (reading 'length')
at loadEpisodeDetail (app.js:173:43)
```

## 原因

后端 API (`/api/episode/<episode_id>`) 的返回数据结构已改变：

**之前**：
```json
{
  "episode_id": "...",
  "episode_name": "...",
  "num_frames": 100,
  "frames": [...],  // ← 包含图片数据
  "hand_actions": [...],
  "instructions": [...]
}
```

**现在**：
```json
{
  "episode_id": "...",
  "episode_name": "...",
  "num_frames": 100,
  "instructions": [...]
  // ← 不再包含 frames 字段
}
```

**原因**：改为视频渲染后，不再返回单独的图片帧数据。

## 修复

### 前端修改

**之前的代码**：
```javascript
let infoHTML = `
    <span>📁 数据集: ${data.dataset_name}</span>
    <span>📊 总帧数: ${data.num_frames}</span>
    <span>🖼️ 显示帧数: ${data.frames.length}</span>  // ← 错误！
`;
```

**修复后的代码**：
```javascript
let infoHTML = `
    <span>📁 数据集: ${data.dataset_name}</span>
    <span>📊 总帧数: ${data.num_frames}</span>
    // ← 移除了 frames.length
`;
```

## 影响范围

- ✅ 已修复：`static/app.js` 第173行
- ✅ 不影响视频播放功能
- ✅ 不影响其他功能

## 测试

修复后，测试以下功能：
1. ✅ 选择 episode
2. ✅ 显示 episode 详情
3. ✅ 显示指令（如果有）
4. ✅ 播放原始视频
5. ✅ 播放渲染视频

## 相关变更

这个问题是由于以下变更引起的：
- 从静态图片展示改为视频播放
- API 不再返回 base64 编码的图片数据
- 视频通过单独的 API 端点提供

## 预防措施

将来修改 API 时：
1. 同步更新前端代码
2. 检查所有使用该 API 的地方
3. 添加错误处理和默认值

## 示例：更健壮的代码

```javascript
// 更安全的写法
let infoHTML = `
    <span>📁 数据集: ${data.dataset_name || 'Unknown'}</span>
    <span>📊 总帧数: ${data.num_frames || 0}</span>
`;

// 如果将来需要显示帧数
if (data.frames && data.frames.length > 0) {
    infoHTML += `<span>🖼️ 显示帧数: ${data.frames.length}</span>`;
}
```

---

**状态**：✅ 已修复  
**日期**：2026-02-01

