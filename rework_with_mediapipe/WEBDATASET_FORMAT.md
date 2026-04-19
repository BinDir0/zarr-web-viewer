# MediaPipe Review Export 到 WebDataset

这个文档说明如何把 `run_fit --export` 生成的 `manifest.jsonl` / `summary.json` 转成 WebDataset tar shards，以及生成后的 sample 格式。

适用对象：

- 已经完成 `preprocess -> review -> fit -> export`
- 手里有 `manifest.jsonl` 或 `summary.json`
- 想把结果喂给下游训练/数据消费程序

## 1. 构建命令

最直接的用法：

```bash
python3 -m rework_with_mediapipe.jobs.build_webdataset \
  --manifest /path/to/export_YYYYMMDD_HHMMSS/manifest.jsonl \
  --output-dir /path/to/mediapipe_wds
```

如果你手里拿的是 `summary.json`，也可以直接传：

```bash
python3 -m rework_with_mediapipe.jobs.build_webdataset \
  --manifest /path/to/export_YYYYMMDD_HHMMSS/summary.json \
  --output-dir /path/to/mediapipe_wds
```

如果 export 里的 artifact 路径是在另一台机器上生成的，可以用前缀替换：

```bash
python3 -m rework_with_mediapipe.jobs.build_webdataset \
  --manifest /path/to/export_YYYYMMDD_HHMMSS/manifest.jsonl \
  --output-dir /path/to/mediapipe_wds \
  --path-map /home/guantianrui/rework-viewer-v3/zarr-web-viewer=/root/zarr-viewer
```

常用参数：

- `--name-prefix mediapipe_boxes`
  - 控制 shard 文件名，默认就是 `mediapipe_boxes`
- `--shard-size 1000`
  - 每个 tar 里最多多少个 sample
- `--skip-missing`
  - 遇到缺失 artifact 时跳过，而不是直接失败

## 2. 输出目录结构

构建完成后会生成：

```text
mediapipe_wds/
  dataset_info.json
  index.jsonl
  shards/
    mediapipe_boxes-000000.tar
    mediapipe_boxes-000001.tar
    ...
```

其中：

- `dataset_info.json`
  - 数据集级别摘要，包含样本数、shard 列表、跳过的条目等
- `index.jsonl`
  - 每行一个 sample，记录 `key`、来源 manifest、所属 shard、frame range
- `shards/*.tar`
  - 标准 WebDataset shard

## 3. Sample 组织方式

每个 export item 对应一个 WebDataset sample。

当前 export 的粒度是：

- 一个 `clip_id`
- 一个 `subepisode_index`

所以 sample key 规则是：

```text
clip_000001_sub_000000
clip_000001_sub_000001
...
```

每个 sample 会包含这些 member：

```text
<key>.json
<key>.boxes.npz
<key>.fit.npz
<key>.fit.json
<key>.bundle.json
<key>.proposals.npz
```

说明：

- `<key>.json`
  - 当前 sample 的统一元数据
- `<key>.boxes.npz`
  - 为当前 subepisode 单独切出来的框数据，推荐下游优先直接读这个
- `<key>.fit.npz`
  - 原始整段 episode 的 `fit.npz`
- `<key>.fit.json`
  - 原始整段 episode 的 `fit.json`
- `<key>.bundle.json`
  - 原始 preprocess 生成的 bundle 元数据
- `<key>.proposals.npz`
  - 原始 proposal 聚合结果

注意：

- `fit.*` 和 `bundle.*` 仍然是整段 clip 级别 artifact，不会按 subepisode 改写
- `boxes.npz` 才是按当前 sample 的 `frame_start/frame_end` 切出来的子集

## 4. `<key>.json` 字段

`<key>.json` 里包含这些核心字段：

- `format_version`
  - 当前固定为 `mediapipe_box_wds_v1`
- `__key__`
  - WebDataset sample key
- `clip_id`
- `parent_clip_id`
- `review_unit`
- `subepisode_index`
- `episode_id`
- `episode_name`
- `dataset_name`
- `clip_start`
- `clip_end`
- `frame_start`
- `frame_end`
- `num_frames`
- `status`
- `dirty_reason`
- `artifact_paths`
  - 原始 artifact 的绝对路径
- `artifact_sha256`
  - 原始 artifact 的 sha256
- `fit_npz_schema`
  - 原始 `fit.npz` 的 key / shape / dtype
- `boxes_npz_schema`
  - 当前 sample 生成的 `boxes.npz` 的 key / shape / dtype 说明

## 5. `<key>.boxes.npz` 字段

这是下游最建议直接消费的文件。

### 5.1 公共字段

- `frame_indices`
  - 当前 sample 全部帧的绝对 frame index，范围是 `[frame_start, frame_end]`

### 5.2 左手 sparse 字段

- `left_frame_indices`
  - 左手有框的帧索引，绝对 frame index
- `left_bbox_xyxy`
  - 左手框，shape=`[N_left, 4]`
- `left_bbox_xyxy_orig`
  - 左手原图坐标系下的框，shape=`[N_left, 4]`
- `left_score`
  - 左手框分数，shape=`[N_left]`

### 5.3 右手 sparse 字段

- `right_frame_indices`
- `right_bbox_xyxy`
- `right_bbox_xyxy_orig`
- `right_score`

含义和左手完全对称。

### 5.4 左手 dense 字段

- `left_valid`
  - shape=`[T]`，布尔数组；`T = frame_end - frame_start + 1`
- `left_bbox_xyxy_dense`
  - shape=`[T, 4]`，没有框的位置填 `NaN`
- `left_bbox_xyxy_orig_dense`
  - shape=`[T, 4]`，没有框的位置填 `NaN`
- `left_score_dense`
  - shape=`[T]`，没有框的位置填 `NaN`

### 5.5 右手 dense 字段

- `right_valid`
- `right_bbox_xyxy_dense`
- `right_bbox_xyxy_orig_dense`
- `right_score_dense`

含义和左手完全对称。

### 5.6 坐标约定

- `bbox_xyxy`
  - review preview 坐标系
- `bbox_xyxy_orig`
  - 原始图像坐标系

如果你要回到源帧训练或重渲染，优先用 `*_bbox_xyxy_orig`。

## 6. 推荐读取方式

如果你用 Python + `webdataset`：

```python
import io
import json
import numpy as np
import webdataset as wds

dataset = wds.WebDataset("/path/to/mediapipe_wds/shards/mediapipe_boxes-000000.tar")

for sample in dataset:
    meta = json.loads(sample["json"].decode("utf-8"))
    boxes = np.load(io.BytesIO(sample["boxes.npz"]))
    frame_indices = boxes["frame_indices"]
    left_valid = boxes["left_valid"]
    right_valid = boxes["right_valid"]
    break
```

如果你只想先看 tar 里有哪些文件：

```bash
tar -tf /path/to/mediapipe_wds/shards/mediapipe_boxes-000000.tar | head
```

## 7. 设计取舍

当前 builder 的设计是最小可用版本，重点是：

- 不重新解码原视频帧
- 不依赖额外的 `webdataset` Python 包
- 保留原始 artifact，便于回溯
- 同时生成一个下游更好消费的 `boxes.npz`

当前没有内置这些内容：

- 原始 RGB 帧
- review 页面缩略图 jpg
- 按 subepisode 改写后的 `fit.json`

如果后面你希望 WebDataset sample 里直接带 RGB 帧，我建议下一步再单独扩展一个 `frames.tar` / `jpg` 打包版本，不要和这版 box-only 格式混在一起。
