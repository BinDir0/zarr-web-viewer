# YOLO Box Review Pipeline

当前这套 `rework_with_mediapipe` 已改成：

- `preprocess` 用 Ultralytics YOLO + ByteTrack 产出候选框和 track
- 网页只做关键帧左手/右手指派，或标记 `可见但缺框`
- `fit` 不再做 MANO，只做纯框聚合与导出

## 1. 配置

先在 [config.yaml](/root/zarr-viewer/config.yaml) 的 `mediapipe_review` 段里填好：

- `ultralytics_repo_root`: Ultralytics 仓库路径
- `yolo_model_path`: 你的手框模型权重 `.pt`
- `yolo_device`: 例如 `0` / `cuda:0` / 留空

当前默认 tracker 是 `bytetrack.yaml`。

一个可直接用的示例是：

```yaml
mediapipe_review:
  ultralytics_repo_root: /share_data/guantianrui/ultralytics
  yolo_model_path: /share_data/guantianrui/HaWoR/weights/external/detector.pt
  yolo_imgsz: 640
  yolo_device: "0"
```

## 2. Discover + Preprocess

```bash
python3 -m rework_with_mediapipe.jobs.run_preprocess --discover
python3 -m rework_with_mediapipe.jobs.run_preprocess --process --limit 20
```

产物会落到：

- `mediapipe_review.db`
- `mediapipe_review_artifacts/bundles/clip_xxxxxx/`

每条 episode 会生成一个 bundle：

- `bundle.json`
- `proposals.npz`
- `frames/*.jpg`

## 3. 启动网页

沿用原来的 Flask 服务，打开：

- `/mediapipe`

审核员只需要：

- 左键点框: 指成 wearer 左手
- 右键点框: 指成 wearer 右手
- 如果该侧手可见但没框: 勾选缺框
- 确认所有关键帧后提交

## 4. 聚合导出

```bash
python3 -m rework_with_mediapipe.jobs.run_fit --limit 50
python3 -m rework_with_mediapipe.jobs.run_fit --export
```

这里的 `fit` 现在只是名字沿用，实际不会再做 MANO。

每条已审核 episode 会生成：

- `fit.json`
- `fit.npz`

里面保存的是：

- 左右手分段后的 `frame_indices`
- `bbox_xyxy`
- `bbox_xyxy_orig`
- `score`
- `valid_frame_ranges`
- `dropped_frame_ranges`

## 5. 常见问题

- `缺少 yolo_model_path 配置`
  - 说明还没在配置里填权重路径。

- `YOLO tracking 没有返回 track id`
  - 说明当前检测/跟踪配置没正常产出 ByteTrack id，需要检查权重或 tracker。
