# 快速开始指南

## 第一步：检查你的 zarr 数据集结构

在配置之前，先使用检查工具了解你的数据集结构：

```bash
cd /home/guantianrui/zarr-viewer
python inspect_zarr.py /path/to/your/dataset.zarr
```

这会显示：
- 数据集的完整结构树
- Episodes 的位置和数量
- 图像和动作数据的形状

**注意**: 该工具已针对以下数据结构进行了优化：
- 所有帧数据连续存储在 `data/` 组中
- Episode 元信息存储在 `meta/episode_name` 和 `meta/episode_ends` 中
- 包含手部 MANO 参数、手腕位姿、指尖位置等

## 第二步：修改配置文件

根据检查结果，编辑 `config.yaml`：

```bash
nano config.yaml
```

修改 `zarr_dataset_path` 为你的实际路径：

```yaml
zarr_dataset_path: /share_data/zhaoguangyu/your-dataset.zarr
server_port: 8300
debug_mode: true
```

## 第三步：（可选）根据数据集结构调整代码

当前代码已经适配了以下数据结构：

```
dataset.zarr/
├── data/
│   ├── action/          # 手部动作（mano, wrist, fingertips, shape）
│   ├── state/           # 当前状态（mano, wrist, fingertips, shape）
│   ├── image            # RGB 图像 (sum_frames, 384, 384, 3)
│   ├── depth            # 深度图
│   ├── instruction      # 任务指令
│   ├── instruction_num  # 可用指令数量
│   └── presence         # 左右手可见性
└── meta/
    ├── episode_name     # Episode 名称
    └── episode_ends     # Episode 结束帧索引
```

如果你的数据集结构不同，可能需要修改 `app.py` 中的：
- `get_episode_list()` - 如果 episode 信息不在 meta/ 中
- `get_episode_data()` - 如果数据字段名称或位置不同

## 第四步：安装依赖

```bash
pip install -r requirements.txt
```

或者使用启动脚本自动安装：

```bash
./run.sh
```

## 第五步：启动服务器

```bash
python app.py
```

或使用启动脚本：

```bash
./run.sh
```

## 第六步：在浏览器中访问

打开浏览器访问：
```
http://localhost:8300
```

或者如果在远程服务器上，使用 SSH 端口转发：
```bash
ssh -L 8300:localhost:8300 user@server
```

然后在本地浏览器访问 `http://localhost:8300`

## 功能说明

### 左侧栏
- 显示所有 episodes
- 点击选择要查看的 episode

### 主面板
- **中间帧图像**: 显示 episode 中间的 2 帧图像
- **手部动作标注**: 显示对应帧的动作向量
- **备注区**: 可以为每个 episode 添加文字备注

### 修改显示的帧数

在 `app.py` 第 241 行左右：

```python
frames = get_middle_frames(episode_data["images"], num_frames=2)
```

修改 `num_frames` 参数即可。

## 常见问题

### Q: 显示"无图像数据"
A: 检查你的数据集中图像数据的路径是否正确，使用 `inspect_zarr.py` 查看结构。

### Q: 显示"无动作数据"
A: 检查你的数据集中是否有 `actions` 字段，路径是否正确。

### Q: 动作数据显示格式不对
A: 在 `static/app.js` 的 `renderActions()` 函数中自定义显示格式。

### Q: 图像加载很慢
A: 图像会转换为 base64 编码传输。对于大图像或高分辨率图像，可能需要：
   - 在服务器端进行图像压缩
   - 减少显示的帧数
   - 使用图像缓存

## 下一步扩展

项目已经搭建好基础框架，你可以根据需求添加：

1. **视频播放**: 在 `app.py` 中添加视频生成和传输功能
2. **多摄像头**: 如果有多个视角的图像，可以扩展显示多个图像
3. **3D 可视化**: 如果有 3D 数据，可以集成 Three.js 或类似库
4. **数据统计**: 添加图表和统计信息
5. **批量导出**: 添加批量导出标注的功能

需要帮助时随时说！

