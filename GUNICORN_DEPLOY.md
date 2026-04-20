# Gunicorn 部署说明

这份文档说明如何把当前网站从 `python3 app.py` 切到 `gunicorn`。

目标：

- 提升并发稳定性
- 支持大约 10 名审核员同时使用
- 尽量少改现有代码

## 1. 为什么要切 Gunicorn

当前 `app.py` 里使用的是 Flask 自带开发服务器：

```python
app.run(host="0.0.0.0", port=SERVER_PORT, debug=DEBUG_MODE, use_reloader=USE_RELOADER, threaded=True)
```

这个方式适合开发，不适合多人并发长期使用。

对 10 人并行标注，建议至少切到：

- `gunicorn` 多 worker
- 每个 worker 多线程

## 2. 代码入口

仓库里已经新增：

- [wsgi.py](/root/zarr-viewer/wsgi.py)

它会在 `gunicorn` 启动时执行：

- `init_db()`
- `init_mediapipe_review_runtime()`

然后暴露：

```python
application = app
```

所以 `gunicorn` 可以直接加载：

```text
wsgi:application
```

## 3. 安装 Gunicorn

如果当前环境还没有：

```bash
python3 -m pip install gunicorn
```

建议在你现在实际运行网站的同一个 Python 环境里安装。

## 4. 直接启动

先用下面这条命令验证：

```bash
python3 -m gunicorn \
  -w 4 \
  --threads 8 \
  --timeout 120 \
  --bind 0.0.0.0:9475 \
  wsgi:application
```

参数说明：

- `-w 4`
  - 4 个 worker 进程
- `--threads 8`
  - 每个 worker 8 个线程
- `--timeout 120`
  - 单请求超时 120 秒，避免慢请求太早被杀
- `--bind 0.0.0.0:9475`
  - 监听当前端口

如果机器比较强，也可以先试：

```bash
python3 -m gunicorn \
  -w 6 \
  --threads 8 \
  --timeout 120 \
  --bind 0.0.0.0:9475 \
  wsgi:application
```

## 5. 推荐的初始参数

对“10 个审核员并行”这个量级，我建议先从下面开始：

```bash
python3 -m gunicorn -w 4 --threads 8 --timeout 120 --bind 0.0.0.0:9475 wsgi:application
```

如果还是偶发卡住，再尝试：

```bash
python3 -m gunicorn -w 6 --threads 8 --timeout 120 --bind 0.0.0.0:9475 wsgi:application
```

不要一开始把 worker 开得特别大，否则：

- 每个 worker 都会各自导入应用
- 内存占用会升高
- 初始化开销也会上去

## 6. systemd 常驻运行

仓库里已经放了一份模板：

- [deploy/gunicorn.service.example](/root/zarr-viewer/deploy/gunicorn.service.example)

使用方式：

1. 复制到系统目录

```bash
cp /root/zarr-viewer/deploy/gunicorn.service.example /etc/systemd/system/zarr-viewer.service
```

2. 重新加载 systemd

```bash
systemctl daemon-reload
```

3. 启动服务

```bash
systemctl start zarr-viewer
```

4. 设为开机自启

```bash
systemctl enable zarr-viewer
```

5. 查看状态

```bash
systemctl status zarr-viewer
```

6. 查看日志

```bash
journalctl -u zarr-viewer -f
```

## 7. 切换时的注意事项

切到 `gunicorn` 前：

1. 先停掉旧的 `python3 app.py`
2. 确认端口 `9475` 没被旧进程占着
3. 再启动 `gunicorn`

否则可能会出现：

- 端口冲突
- 你以为已经切过去，其实还是旧服务在跑

## 8. 最小上线步骤

最短路径是：

1. 安装 gunicorn

```bash
python3 -m pip install gunicorn
```

2. 停掉旧服务

3. 直接执行：

```bash
python3 -m gunicorn -w 4 --threads 8 --timeout 120 --bind 0.0.0.0:9475 wsgi:application
```

4. 让 2 到 3 个标注员先试

5. 没问题后再让 10 人同时使用

## 9. 后续建议

如果后面要长期稳定运行，建议再加：

- `nginx` 反向代理
- 请求日志
- 健康检查接口
- 明确的静态文件缓存策略

但这些不是第一步必须。

对你们现在的目标，先把启动方式从 Flask 开发服务器换成 `gunicorn`，收益就已经很明显了。
