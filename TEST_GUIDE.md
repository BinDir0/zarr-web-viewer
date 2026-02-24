# 测试指南

## 快速测试步骤

### 1. 安装依赖

```bash
cd /home/guantianrui/zarr-viewer-stage2
pip install -r requirements.txt
```

### 2. 检查配置

确保 `config.yaml` 中配置了正确的数据集路径：

```yaml
server_port: 8300
debug_mode: true

zarr_datasets:
  - name: "数据集1"
    path: "/path/to/your/zarr/dataset"
    enabled: true
```

### 3. 启动服务器

```bash
python app.py
```

或使用提供的启动脚本：

```bash
bash run.sh
```

### 4. 测试功能

在浏览器中访问 `http://localhost:8300`

#### 测试点 1: Episode 列表加载
- ✓ 左侧边栏应该显示随机加载的 episodes
- ✓ 每个 episode 显示数据集名称、episode 名称、帧数和索引

#### 测试点 2: 选择 Episode
- ✓ 点击左侧的一个 episode
- ✓ 主面板应该显示 episode 详情

#### 测试点 3: Instruction 显示
- ✓ 如果 episode 包含 instruction，应该在顶部显示
- ✓ 指令以列表形式展示，每条指令前有编号

#### 测试点 4: 原始视频播放
- ✓ 左侧视频播放器显示"原始视频"
- ✓ 点击播放按钮，视频应该正常播放
- ✓ 视频内容是 episode 的原始图像帧序列

#### 测试点 5: 手部渲染视频播放
- ✓ 右侧视频播放器显示"手部动作渲染视频"
- ✓ 点击播放按钮，视频应该正常播放
- ✓ 视频中应该显示：
  - 蓝色圆点：左手指尖（标记 L0-L4）
  - 红色圆点：右手指尖（标记 R0-R4）
  - 黄色圆点：左手腕（标记 L_wrist）
  - 青色圆点：右手腕（标记 R_wrist）

#### 测试点 6: 标注功能
- ✓ 在"备注"文本框中输入内容
- ✓ 点击"保存备注"按钮
- ✓ 应该显示"✓ 保存成功"消息
- ✓ 重新选择同一个 episode，备注内容应该被保留

#### 测试点 7: 查看记录
- ✓ 查看过的 episode 会被标记
- ✓ 点击"随机获取新的一批"按钮，应该加载新的未查看 episodes

## 常见问题排查

### 问题 1: 视频无法播放

**可能原因**：
- OpenCV 未正确安装
- 视频编码器不支持

**解决方法**：
```bash
pip uninstall opencv-python
pip install opencv-python==4.8.0.76
```

### 问题 2: 视频加载很慢

**可能原因**：
- Episode 帧数太多
- 服务器性能不足

**解决方法**：
- 这是正常的，视频生成需要时间
- 可以在浏览器开发者工具的网络面板中看到视频加载进度

### 问题 3: 手部渲染不显示

**可能原因**：
- 数据集中缺少 state 数据
- 相机内参数据缺失

**解决方法**：
- 检查数据集是否包含 `state/mano`、`state/wrist`、`state/fingertips`、`intrinsic` 等字段
- 使用 `inspect_zarr.py` 脚本检查数据集结构

### 问题 4: Instruction 不显示

**可能原因**：
- 数据集中没有 instruction 数据

**解决方法**：
- 检查数据集是否包含 `instruction` 和 `instruction_num` 字段
- 使用 `inspect_zarr.py` 脚本检查数据集结构

## 性能优化建议

### 1. 视频缓存
当前实现每次请求都会重新生成视频。如果需要频繁访问同一个 episode，可以考虑：
- 添加视频缓存机制
- 预生成常用 episodes 的视频

### 2. 降低视频质量
如果视频生成太慢，可以修改 `create_video_from_frames()` 函数：
- 降低帧率（从 30 FPS 改为 15 FPS）
- 降低分辨率（在生成前缩放图像）

### 3. 异步生成
可以将视频生成改为异步任务：
- 用户请求时返回"生成中"状态
- 后台生成完成后通知前端
- 前端轮询或使用 WebSocket 获取结果

## API 测试

### 测试 Episode 列表 API

```bash
curl "http://localhost:8300/api/episodes?limit=10&user_id=test_user"
```

### 测试 Episode 详情 API

```bash
curl "http://localhost:8300/api/episode/数据集1::0?user_id=test_user"
```

### 测试原始视频 API

```bash
curl "http://localhost:8300/api/episode/数据集1::0/video/original" -o original.mp4
```

### 测试渲染视频 API

```bash
curl "http://localhost:8300/api/episode/数据集1::0/video/rendered" -o rendered.mp4
```

## 数据集要求

为了正常使用所有功能，数据集应该包含以下字段：

### 必需字段
- `meta/episode_names` 或 `meta/episode_name`: episode 名称列表
- `meta/episode_ends`: episode 结束索引
- `data/image`: 图像数据 (N, H, W, 3)

### 可选字段（用于手部渲染）
- `data/action/fingertips`: 指尖位置 (N, 30) - 左右手各5个指尖的3D坐标
- `data/action/wrist`: 手腕位置和旋转 (N, 18)
- `data/action/mano`: MANO 参数 (N, 90)
- `data/presence`: 手的可见性 (N,) - 0=无手, 1=左手, 2=右手, 3=双手
- `data/intrinsic`: 相机内参 (N, 3, 3)

### 可选字段（用于指令显示）
- `data/instruction`: 指令文本数据
- `data/instruction_num`: 每帧的指令数量

使用 `inspect_zarr.py` 脚本可以查看数据集的完整结构：

```bash
python inspect_zarr.py /path/to/your/dataset.zarr
```

