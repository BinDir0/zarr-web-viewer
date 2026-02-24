# 更新摘要 - Zarr Dataset Viewer Stage 2

## 概述

本次更新将原有的静态图片展示方式改为完整的视频播放方式，并恢复了 instruction 的显示功能。

## 主要变更

### ✅ 1. 恢复 Instruction 显示
- **位置**: Episode 详情页面顶部
- **样式**: 蓝色背景框，带编号的列表
- **数据来源**: `data/instruction` 和 `data/instruction_num` 字段

### ✅ 2. 视频播放功能
替换了原来的静态图片展示，现在显示两个视频：

#### 原始视频
- **内容**: 直接从 episode 的图像帧序列生成
- **位置**: 左侧视频播放器
- **API**: `/api/episode/<episode_id>/video/original`

#### 手部动作渲染视频
- **内容**: 在原始视频上叠加手部动作可视化
- **位置**: 右侧视频播放器
- **API**: `/api/episode/<episode_id>/video/rendered`
- **可视化元素**:
  - 🔵 左手指尖（L0-L4）
  - 🔴 右手指尖（R0-R4）
  - 🟡 左手腕（L_wrist）
  - 🟢 右手腕（R_wrist）

### ✅ 3. 移除手部动作文字可视化
- 删除了原来的"手部动作标注"区域
- 删除了详细的 MANO 参数、手腕位姿、指尖位置等文字展示
- 这些信息现在直接在视频中可视化显示

## 文件修改清单

### 后端文件

#### `app.py`
- **新增导入**: `cv2`, `tempfile`
- **新增函数**:
  - `create_video_from_frames()`: 视频生成
  - `render_hand_on_frame()`: 手部渲染
- **新增 API 端点**:
  - `/api/episode/<episode_id>/video/original`
  - `/api/episode/<episode_id>/video/rendered`
- **修改函数**:
  - `get_episode_data()`: 读取完整 episode 数据而非仅中间帧
  - `api_episode()`: 简化返回数据，只返回元信息

#### `requirements.txt`
- **新增依赖**: `opencv-python>=4.8.0`

### 前端文件

#### `static/app.js`
- **新增函数**:
  - `renderVideos()`: 渲染视频播放器
- **删除函数**:
  - `renderFrames()`: 原图片渲染函数
  - `renderHandActions()`: 手部动作文字渲染函数
- **修改函数**:
  - `loadEpisodeDetail()`: 调用 `renderVideos()` 而非 `renderFrames()`
  - 恢复 `displayInstructions()` 的调用

#### `templates/index.html`
- **修改标题**: "中间帧图像" → "Episode 视频"
- **删除区域**: "动作标注显示区域"
- **保留区域**: "视频显示区域"、"标注编辑区域"

#### `static/styles.css`
- 无需修改（现有样式已适配）

## 技术实现细节

### 视频生成流程

```
1. 前端请求视频 URL
   ↓
2. 后端接收请求，调用 get_episode_data()
   ↓
3. 读取完整 episode 的所有帧数据
   ↓
4. 对于渲染视频：逐帧渲染手部动作
   ↓
5. 使用 OpenCV 生成 MP4 视频文件
   ↓
6. 返回视频文件流给前端
   ↓
7. 前端视频播放器播放
```

### 手部渲染算法

```python
对于每一帧:
    1. 读取指尖 3D 坐标 (左手15维 + 右手15维)
    2. 读取手腕 3D 坐标 (左手3维 + 右手3维)
    3. 读取相机内参矩阵 (3x3)
    4. 读取手的可见性 (0-3)
    
    5. 使用相机内参将 3D 坐标投影到 2D:
       x_2d = fx * x_3d / z_3d + cx
       y_2d = fy * y_3d / z_3d + cy
    
    6. 在图像上绘制:
       - 左手指尖: 蓝色圆点 + 标签
       - 右手指尖: 红色圆点 + 标签
       - 左手腕: 黄色圆点 + 标签
       - 右手腕: 青色圆点 + 标签
    
    7. 根据 presence 字段决定是否显示左/右手
```

### 视频参数

- **编码器**: mp4v (H.264)
- **帧率**: 30 FPS
- **分辨率**: 保持原始图像分辨率
- **格式**: MP4
- **临时文件**: 使用 `tempfile.NamedTemporaryFile` 自动清理

## 数据流对比

### 之前（静态图片）
```
前端 → API → 读取2帧 → 转换为 base64 → JSON 返回 → 前端显示
```

### 现在（视频）
```
前端 → API → 读取所有帧 → 生成视频文件 → 文件流返回 → 前端播放
```

## 性能影响

### 优点
- ✅ 用户体验更好：可以看到完整的动作序列
- ✅ 手部动作可视化更直观
- ✅ 支持视频控制（暂停、快进、慢放等）

### 缺点
- ⚠️ 首次加载时间较长（需要生成视频）
- ⚠️ 服务器 CPU 占用增加（视频编码）
- ⚠️ 网络带宽占用增加（视频文件比图片大）

### 优化建议
1. **添加缓存机制**: 缓存已生成的视频
2. **降低帧率**: 从 30 FPS 降到 15 FPS
3. **降低分辨率**: 在生成前缩放图像
4. **异步生成**: 后台生成，前端轮询
5. **使用更好的编码器**: 如 x264（需要 ffmpeg）

## 兼容性

### 浏览器支持
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

### 数据集要求
- **必需**: `data/image` 字段
- **可选**: `data/action/fingertips`, `data/action/wrist`, `data/intrinsic`, `data/presence`
- **可选**: `data/instruction`, `data/instruction_num`

## 测试清单

- [x] Episode 列表正常加载
- [x] 选择 episode 显示详情
- [x] Instruction 正确显示
- [x] 原始视频可以播放
- [x] 渲染视频可以播放
- [x] 手部标注正确显示
- [x] 标注保存功能正常
- [x] 查看记录功能正常

## 后续改进方向

1. **视频缓存**: 避免重复生成
2. **进度提示**: 显示视频生成进度
3. **下载功能**: 允许下载生成的视频
4. **播放同步**: 两个视频同步播放
5. **帧率控制**: 允许用户调整播放速度
6. **更丰富的渲染**: 添加骨架线、手势识别等
7. **批量生成**: 预生成所有 episodes 的视频

## 回滚方案

如果需要回滚到之前的版本：

1. 恢复 `app.py` 中的 `api_episode()` 函数
2. 恢复 `app.js` 中的 `renderFrames()` 和 `renderHandActions()` 函数
3. 恢复 `index.html` 中的"动作标注显示区域"
4. 移除视频相关的 API 端点和函数

## 联系方式

如有问题，请查看：
- `CHANGELOG.md`: 详细的变更记录
- `TEST_GUIDE.md`: 测试指南和常见问题
- `README.md`: 项目使用说明

