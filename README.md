# Zarr 数据集查看器

一个简单的 Web 应用，用于查看和标注 zarr 格式的机器人数据集。

## 功能特点

- 🗂️ **支持多数据集**：同时加载和查看多个 zarr 数据集
- 📊 浏览所有 episodes（从 meta/episode_name 读取）
- 📁 按数据集分组显示，支持折叠/展开
- 🖼️ 显示每个 episode 的中间帧图像（默认显示2帧，384x384分辨率）
- 🤖 显示手部动作标注数据：
  - 手腕位姿（平移 + 6D旋转）
  - 指尖 3D 位置（5个手指）
  - MANO 参数（45维，左右手）
  - 手的可见性状态
- 📝 显示任务指令（instruction）
- 💬 为每个 episode 添加备注
- 💾 自动保存备注到本地数据库

## 安装依赖

```bash
cd /home/guantianrui/zarr-viewer
pip install -r requirements.txt
```

## 配置

### 多数据集配置（推荐）

编辑 `config.yaml` 文件，可以配置多个 zarr 数据集：

```yaml
zarr_datasets:
  - name: "数据集1"
    path: /path/to/dataset1.zarr
    enabled: true
  - name: "数据集2"
    path: /path/to/dataset2.zarr
    enabled: true

server_port: 8300
debug_mode: true
```

### 单数据集配置（向后兼容）

也可以使用旧的单数据集格式：

```yaml
zarr_dataset_path: /path/to/your/zarr/dataset
server_port: 8300
debug_mode: true
```

**详细说明请查看 `多数据集配置说明.md`**

## 运行

```bash
python app.py
```

然后在浏览器中访问：`http://localhost:8300`

## 数据集结构说明

本工具支持以下 zarr 数据集结构（连续存储格式）：

```
dataset.zarr/
├── data/
│   ├── action/           # 下一帧本体感知
│   │   ├── mano          # (sum_frames, 90) float32 先左后右，45d mano 参数
│   │   ├── wrist         # (sum_frames, 18) float32 左平移，右平移，左旋转，右旋转
│   │   ├── shape         # (sum_frames, 20) float32 shape parameter for mano
│   │   └── fingertips    # (sum_frames, 30) float32 左右手指尖
│   ├── extrinsic         # (sum_frames, 16) float32 4x4 matrix
│   ├── intrinsic         # (sum_frames, 4)  float32 fx fy cx cy
│   ├── image             # (sum_frames, 384, 384, 3) uint8
│   ├── instruction       # (sum_frames, 5)  string
│   ├── instruction_num   # (sum_frames,)  int8 有几个 instruction 可用
│   ├── state/            # 当前帧本体感知
│   │   ├── mano          # (sum_frames, 90) float32
│   │   ├── wrist         # (sum_frames, 18) float32
│   │   ├── shape         # (sum_frames, 20) float32
│   │   └── fingertips    # (sum_frames, 30) float32
│   ├── depth             # (sum_frames, H, W)  u16 单位为 mm
│   └── presence          # (sum_frames,)    int8 左右手可见情况，1 左 2 右 3 均可见
└── meta/
    ├── episode_name      # (num_episodes,)  string 方便和原始数据集对应
    └── episode_ends      # (num_episodes,)  int64  每个episode的结束帧索引
```

这是一种连续存储格式，所有 episodes 的数据按顺序存储在一起，通过 `meta/episode_ends` 来分割每个 episode。

## 自定义

### 修改显示的帧数

在 `app.py` 中找到这一行并修改 `num_frames` 参数：

```python
frames = get_middle_frames(episode_data["images"], num_frames=2)
```

### 修改动作数据显示方式

编辑 `static/app.js` 中的 `renderActions()` 函数来自定义动作数据的显示格式。

### 添加更多视觉化

你可以在 `templates/index.html` 和 `static/styles.css` 中添加更多的视觉化组件。

## 后续扩展建议

- [ ] 添加视频播放功能
- [ ] 支持多种图像来源（多个摄像头）
- [ ] 添加数据统计和可视化
- [ ] 支持导出标注结果
- [ ] 添加键盘快捷键
- [ ] 支持图像缩放和标注绘制

## 注意事项

1. 确保 zarr 数据集路径正确
2. 数据集需要有读取权限
3. 图像数据会被转换为 base64 编码传输，大图像可能较慢
4. 标注数据保存在 `annotations.db` SQLite 数据库中

## 技术栈

- 后端: Flask + zarr + numpy
- 前端: 原生 JavaScript + CSS
- 数据库: SQLite

