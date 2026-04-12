# MediaPipe Review 流程说明

这份文档只说明当前这套 `rework_with_mediapipe` 流水线。

目标是把已有脏标注片段切成 clip，先用 MediaPipe `VIDEO` 模式生成候选 Track，再让外包把这些 Track 归并到 wearer-left / wearer-right，最后管理员跑 MANO 拟合并导出结果。

## 角色划分

管理员负责：

- 准备 `config.yaml`
- 跑 discover
- 跑 preprocess
- 启动 web server
- 定期跑 fit
- 最后跑 export

外包审核员只负责：

- 打开 `/mediapipe`
- 逐条审核 clip
- 把候选 Track 归到 `wearer-left` / `wearer-right`
- 如果某侧可见但没有 proposal 框，标记“该侧可见但缺框”

管理员步骤不在网页里做。网页只给审核员使用。

## 一、依赖与配置

当前配置入口是 [config.yaml](/root/zarr-viewer/config.yaml) 里的 `mediapipe_review` 段。

关键字段：

- `db_path`: MediaPipe review 自己的 sqlite 库
- `artifacts_dir`: 预处理、拟合、导出的产物目录
- `source_annotations_db`: 原始返工/审查标注库
- `hand_landmarker_task`: MediaPipe hand landmarker 模型
- `mediapipe_repo_root`: 可选的 mediapipe 源码目录，只有环境里没有可用 wheel 时才回退使用
- `manopth_root`: `manopth` 代码目录
- `mano_models_root`: MANO 模型目录

建议先在当前环境里检查：

```bash
python3 -m pip show mediapipe
python3 -c "import mediapipe; print(mediapipe.__file__)"
python3 -c "import torch; print(torch.__version__)"
```

## 二、完整流程

### 1. Discover

把原始 `annotations.db` 里需要二次处理的 episode 切成 clip，并写入 `mediapipe_review.db`。

命令：

```bash
python3 -m rework_with_mediapipe.jobs.run_preprocess --discover
```

来源规则在 [discover.py](/root/zarr-viewer/rework_with_mediapipe/discover.py)：

- `REWORK_V1:` 会被当作 `rework_box_issue`
- `BAD_FRAMES:` / `BAD_FRAME:` 且带 `SANITY_REASON_V1/V2/V3:` 的会被当作 `sanity_box_issue`
- 目前只处理 `factory*` 数据

执行结果：

- 向 `clips` 表插入待处理 clip
- 初始状态为 `queued_preprocess`

### 2. Preprocess

对 `queued_preprocess` 的 clip 跑 MediaPipe，生成候选 Track 与网页预览图。

命令：

```bash
python3 -m rework_with_mediapipe.jobs.run_preprocess --process --limit 100
```

说明：

- `--limit` 是这一次最多处理多少个 clip，不是帧数
- 可以多次重复运行，直到队列处理完
- 当前实现优先复用 MediaPipe 自带的 `VIDEO` 模式 tracking；外层只把 MediaPipe 连续输出整理成 review 用的连续片段
- 当前 clip 预处理成功后通常会变成 `ready_for_review`
- 如果候选 Track 过多，clip 会直接进入 `qa_needed_dense_scene`

执行结果会写到：

- `mediapipe_review.db`
- `mediapipe_review_artifacts/bundles/clip_xxxxxx/`

每个 clip 目录下典型产物：

- `bundle.json`
- `proposals.npz`
- `frames/*.jpg`

### 3. 启动审核网页

命令：

```bash
python3 app.py
```

如果 `config.yaml` 里 `mediapipe_review.replace_home: true`，首页会直接跳到 `/mediapipe`。

否则直接访问：

```text
http://<host>:<port>/mediapipe
```

网页说明：

- 仪表盘页：`/mediapipe`
- 审核页：`/mediapipe/clip/<clip_id>`

### 4. 外包审核

外包审核员只需要做下面几步：

1. 打开 `/mediapipe`
2. 点击“开始审核”
3. 观察 clip 播放与候选卡片
4. 把一个或多个 Track 归到左手，或归到右手
5. 不属于 wearer 的手直接忽略
6. 如果某侧手可见但没有 proposal 框，勾“该侧可见但缺框”
7. 提交，自动跳下一条

审核提交后会发生：

- 正常 clip：`clips.status` 变成 `reviewed_ready_for_fit`
- 缺框 clip：`clips.status` 变成 `qa_needed_missing_box`
- `review_payload_json` / `vendor_reviews.review_json` 写入归并结果
- 能进入拟合的 clip 会写入 `fit_jobs.status = queued_fit`

### 5. Fit

管理员对已经 review 完的 clip 跑 MANO 拟合。

命令：

```bash
python3 -m rework_with_mediapipe.jobs.run_fit --limit 100
```

说明：

- `--limit` 是这次最多处理多少个 `queued_fit` 任务
- 可以反复运行

执行结果：

- 成功时，`clips.status` 变成 `fit_ok`
- 如果该 clip 被标记缺框，会停在 `qa_needed_missing_box`
- 如果拟合误差过大，会变成 `fit_failed`

每个 clip 目录下会新增：

- `fit.json`
- `fit.npz`

### 6. Export

当有足够多的 `fit_ok` clip 后，管理员导出 manifest。

命令：

```bash
python3 -m rework_with_mediapipe.jobs.run_fit --export
```

导出结果会写到：

- `mediapipe_review_artifacts/exports/export_YYYYMMDD_HHMMSS/manifest.jsonl`
- `mediapipe_review_artifacts/exports/export_YYYYMMDD_HHMMSS/summary.json`

导出逻辑在 [export.py](/root/zarr-viewer/rework_with_mediapipe/export.py)。

## 三、建议的日常使用方式

### 管理员日常批处理

如果你拿到一批新的源标注，推荐顺序：

```bash
python3 -m rework_with_mediapipe.jobs.run_preprocess --discover
python3 -m rework_with_mediapipe.jobs.run_preprocess --process --limit 500
python3 app.py
```

外包审一段时间后，再跑：

```bash
python3 -m rework_with_mediapipe.jobs.run_fit --limit 200
```

最后在需要出结果时跑：

```bash
python3 -m rework_with_mediapipe.jobs.run_fit --export
```

### 如果预处理队列很长

可以分批跑：

```bash
python3 -m rework_with_mediapipe.jobs.run_preprocess --process --limit 100
python3 -m rework_with_mediapipe.jobs.run_preprocess --process --limit 100
python3 -m rework_with_mediapipe.jobs.run_preprocess --process --limit 100
```

不需要一次性跑完。

## 四、状态说明

`clips.status` 常见取值：

- `queued_preprocess`: 等待预处理
- `preprocessing`: 正在跑 MediaPipe
- `ready_for_review`: 可以给外包审核
- `reviewed_ready_for_fit`: 外包已提交，等待 fit
- `qa_needed_missing_box`: 某侧可见但 proposal 缺框
- `qa_needed_dense_scene`: 候选过多，不适合外包直接审
- `fit_ok`: MANO 拟合通过
- `fit_failed`: 已有 review，但 MANO 拟合失败
- `exported`: 已被导出
- `failed`: 预处理失败

## 五、常见排查

### 1. preprocess 报 mediapipe 导入问题

先确认当前环境：

```bash
python3 -c "import sys, mediapipe; print(sys.executable); print(mediapipe.__file__)"
```

不要用裸 `pip show mediapipe` 判断环境，要用：

```bash
python3 -m pip show mediapipe
```

### 2. preprocess 成功但网页里没有任务

检查是否真的出现了 `ready_for_review`：

```bash
python3 - <<'PY'
import sqlite3
conn = sqlite3.connect("mediapipe_review.db")
for row in conn.execute("select status, count(*) from clips group by status order by status"):
    print(row)
conn.close()
PY
```

### 3. review 完成后 fit 没有东西跑

检查 `fit_jobs`：

```bash
python3 - <<'PY'
import sqlite3
conn = sqlite3.connect("mediapipe_review.db")
for row in conn.execute("select status, count(*) from fit_jobs group by status order by status"):
    print(row)
conn.close()
PY
```

### 4. 想重跑 preprocess / fit

这取决于你想重置哪一层：

- 只想继续跑失败项：直接再次运行对应命令
- 想彻底重做某批 clip：需要先清理 `mediapipe_review.db` 里的对应记录，并删除相应 `artifacts/bundles/clip_xxxxxx`

这类重置动作建议管理员单独执行，不建议交给外包。

## 六、你现在该做什么

你现在已经完成了 preprocess，并且日志显示：

```text
{"discovered": 0, "processed": 10}
```

下一步通常是：

1. 启动网站：`python3 app.py`
2. 让外包进入 `/mediapipe` 开始 review
3. 积累一批 `reviewed_ready_for_fit` 后，管理员运行 `python3 -m rework_with_mediapipe.jobs.run_fit --limit 100`
4. 有足够多 `fit_ok` 后，管理员运行 `python3 -m rework_with_mediapipe.jobs.run_fit --export`

如果只是想先让外包干活，那你现在就可以启动网页了。
