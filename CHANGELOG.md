# 更新日志

## 2026-02-01 - 视频渲染功能更新

### 最新更新 (v2.1.1)

**Bug 修复**：
- 🐛 修复 numpy 数组视频生成错误（`ValueError: The truth value of an array...`）
- 🐛 修复视频生成循环中的 frame 变量未定义问题
- 🐛 添加 chumpy 依赖以支持 MANO 模型
- ✅ 添加视频写入器状态检查

**安装更新**：
```bash
pip install chumpy
# 或
pip install -r requirements.txt
```

### 之前更新 (v2.1)

**⚠️ 重要变更**：数据来源从 action 改为 state
- **之前**：从 `data/action/mano`、`data/action/wrist`、`data/action/fingertips` 读取
- **现在**：从 `data/state/mano`、`data/state/wrist`、`data/state/fingertips` 读取
- **影响**：不向后兼容，需要数据迁移（见 [DATA_SOURCE_UPDATE.md](DATA_SOURCE_UPDATE.md)）
- **原因**：语义更清晰（state=状态，action=动作），代码更简洁

**新增功能**：额外渲染数据集中的 Fingertips
- 除了 MANO 模型生成的手部，现在额外显示数据集中存储的原始指尖位置
- 左手指尖：绿色边框 + 青色内圈
- 右手指尖：洋红色边框 + 黄色内圈
- 用于数据验证和对比分析

**数据来源**：
- MANO 参数：`data/state/mano` 和 `data/state/wrist`
- Fingertips：`data/state/fingertips`

### 主要变更 (v2.0)

1. **恢复 instruction 显示**
   - 重新启用了 instruction 数据的读取和显示
   - 在 episode 详情页面顶部显示任务指令

2. **视频渲染功能**
   - 将静态图片展示改为完整的 episode 视频播放
   - 添加了两个视频：
     - **原始视频**：直接从图像帧生成的原始视频
     - **手部动作渲染视频**：在原始视频上叠加手部动作标注的视频
   
3. **移除手部动作文字可视化**
   - 删除了手部动作的详细文字展示部分
   - 手部动作现在直接在视频中可视化显示

### 技术实现

#### 后端 (app.py)
- 添加了 `cv2` (OpenCV) 依赖用于视频生成
- 新增函数：
  - `create_video_from_frames()`: 从帧序列创建视频
  - `render_hand_on_frame()`: 在帧上渲染手部动作（指尖、手腕位置）
- 新增 API 端点：
  - `/api/episode/<episode_id>/video/original`: 生成并返回原始视频
  - `/api/episode/<episode_id>/video/rendered`: 生成并返回带手部渲染的视频
- 修改了 `get_episode_data()` 函数，现在读取完整 episode 的所有帧数据

#### 前端 (app.js, index.html)
- 新增 `renderVideos()` 函数替代原来的 `renderFrames()` 函数
- 移除了 `renderHandActions()` 函数及相关的手部动作文字显示
- 恢复了 `displayInstructions()` 函数的调用
- 更新了 HTML 模板，移除了"动作标注显示区域"

#### 依赖更新 (requirements.txt)
- 添加了 `opencv-python>=4.8.0`

### 手部动作渲染说明

**完整MANO渲染**（如果安装了MANO模型）：
- **手部点云**：778个顶点的完整手部网格
  - 左手：青色点云（半透明）
  - 右手：黄色点云（半透明）
- **关节骨架**：21个关节点和手指连接线
  - 左手：蓝色关节点，亮蓝色骨架线
  - 右手：红色关节点，亮红色骨架线
- **5条手指链**：大拇指、食指、中指、无名指、小指

**简化渲染**（Fallback，如果没有MANO）：
- **手腕位置**：
  - 左手腕：黄色圆点 + "L" 标签
  - 右手腕：青色圆点 + "R" 标签

渲染方法参考了专业的 `visualizer/visualize_episode_video.py` 实现

### 使用说明

1. 安装新依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 启动服务器：
   ```bash
   python app.py
   ```

3. 在浏览器中访问：
   - 选择一个 episode
   - 查看顶部的任务指令（如果有）
   - 观看两个视频：
     - 左侧：原始视频
     - 右侧：带手部动作渲染的视频

### 注意事项

- 视频生成是实时进行的，首次加载可能需要一些时间（取决于 episode 的帧数）
- 视频使用 mp4v 编码器，帧率为 30 FPS
- 临时视频文件会在发送后自动清理
- 手部渲染只显示可见的手（根据 presence 字段判断）

