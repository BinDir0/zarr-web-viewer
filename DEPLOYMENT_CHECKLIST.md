# 部署检查清单

## 部署前检查

### 1. 依赖安装
```bash
cd /home/guantianrui/zarr-viewer-stage2
pip install -r requirements.txt
```

验证关键依赖：
- [ ] Flask 已安装
- [ ] numpy 已安装
- [ ] zarr 已安装
- [ ] opencv-python 已安装
- [ ] pillow 已安装
- [ ] pyyaml 已安装

### 2. 配置文件检查
- [ ] `config.yaml` 存在
- [ ] 数据集路径正确
- [ ] 端口号设置正确（默认 8300）
- [ ] debug_mode 根据环境设置（生产环境建议 false）

### 3. 数据集验证
```bash
python inspect_zarr.py /path/to/your/dataset.zarr
```

确认数据集包含：
- [ ] `meta/episode_names` 或 `meta/episode_name`
- [ ] `meta/episode_ends`
- [ ] `data/image`
- [ ] （可选）`data/action/fingertips`
- [ ] （可选）`data/action/wrist`
- [ ] （可选）`data/intrinsic`
- [ ] （可选）`data/presence`
- [ ] （可选）`data/instruction`

### 4. 文件完整性检查
- [ ] `app.py` 存在且包含视频生成函数
- [ ] `static/app.js` 存在且包含 `renderVideos()` 函数
- [ ] `static/styles.css` 存在
- [ ] `templates/index.html` 存在
- [ ] `requirements.txt` 包含 opencv-python

### 5. 数据库初始化
启动应用时会自动创建：
- [ ] `annotations.db` 文件会被创建
- [ ] 包含 `annotations` 表
- [ ] 包含 `view_log` 表

## 启动服务

### 方式 1: 直接启动
```bash
python app.py
```

### 方式 2: 使用启动脚本
```bash
bash run.sh
```

### 方式 3: 后台运行
```bash
nohup python app.py > server.log 2>&1 &
```

## 启动后验证

### 1. 服务器启动检查
- [ ] 终端显示 "启动服务器在端口 8300"
- [ ] 终端显示 "访问: http://localhost:8300"
- [ ] 终端显示数据集加载成功信息

### 2. 网页访问检查
访问 `http://localhost:8300`

- [ ] 页面正常加载
- [ ] 标题显示 "Zarr 数据集查看器"
- [ ] 左侧边栏显示 episodes 列表

### 3. 功能测试

#### Episode 列表
- [ ] 左侧显示随机的 episodes
- [ ] 每个 episode 显示数据集名称、名称、帧数
- [ ] 点击 episode 可以选中（背景变色）

#### Episode 详情
- [ ] 点击 episode 后主面板显示详情
- [ ] 顶部显示 episode 名称和基本信息
- [ ] 如果有 instruction，顶部显示指令列表

#### 视频播放
- [ ] 左侧显示"原始视频"播放器
- [ ] 右侧显示"手部动作渲染视频"播放器
- [ ] 点击播放按钮，视频开始播放
- [ ] 视频控制条正常工作（暂停、进度条、音量等）

#### 手部渲染验证
在渲染视频中检查：
- [ ] 可以看到彩色圆点（指尖和手腕）
- [ ] 圆点位置合理（在手的位置）
- [ ] 标签清晰可读（L0-L4, R0-R4, L_wrist, R_wrist）

#### 标注功能
- [ ] 在"备注"文本框中输入内容
- [ ] 点击"保存备注"按钮
- [ ] 显示"✓ 保存成功"消息
- [ ] 切换到其他 episode 再切回来，备注内容保留

#### 查看记录
- [ ] 查看过的 episode 会被记录
- [ ] 点击"随机获取新的一批"可以加载新 episodes

## API 测试

### 测试 Episode 列表
```bash
curl "http://localhost:8300/api/episodes?limit=5" | jq
```
期望：返回 JSON，包含 episodes 数组

### 测试 Episode 详情
```bash
curl "http://localhost:8300/api/episode/数据集名::0" | jq
```
期望：返回 JSON，包含 episode 信息和 instructions

### 测试原始视频
```bash
curl "http://localhost:8300/api/episode/数据集名::0/video/original" -o test_original.mp4
```
期望：下载 MP4 文件，可以正常播放

### 测试渲染视频
```bash
curl "http://localhost:8300/api/episode/数据集名::0/video/rendered" -o test_rendered.mp4
```
期望：下载 MP4 文件，可以正常播放，包含手部标注

## 性能测试

### 1. 视频生成时间
- [ ] 记录首次加载一个 episode 的时间
- [ ] 100 帧 episode 应该在 5-10 秒内生成视频
- [ ] 如果超过 30 秒，考虑优化

### 2. 内存使用
```bash
ps aux | grep python
```
- [ ] 内存使用在合理范围内（< 2GB）
- [ ] 没有明显的内存泄漏

### 3. 并发测试
同时打开多个浏览器标签页：
- [ ] 可以同时查看不同的 episodes
- [ ] 服务器响应正常
- [ ] 没有崩溃或错误

## 常见问题排查

### 问题：视频无法播放
**检查**：
```bash
python -c "import cv2; print(cv2.__version__)"
```
- [ ] OpenCV 版本 >= 4.8.0
- [ ] 浏览器支持 MP4 格式

### 问题：手部渲染不显示
**检查**：
```bash
python inspect_zarr.py /path/to/dataset.zarr
```
- [ ] 数据集包含 `action/fingertips`
- [ ] 数据集包含 `intrinsic`
- [ ] 数据集包含 `presence`

### 问题：Instruction 不显示
**检查**：
- [ ] 数据集包含 `instruction` 字段
- [ ] 数据集包含 `instruction_num` 字段
- [ ] `instruction_num` 值 > 0

### 问题：服务器崩溃
**检查日志**：
```bash
tail -f server.log
```
- [ ] 查看错误堆栈
- [ ] 检查是否是数据集格式问题
- [ ] 检查是否是内存不足

## 生产环境建议

### 1. 配置优化
```yaml
# config.yaml
debug_mode: false  # 关闭调试模式
```

### 2. 使用生产级服务器
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8300 app:app
```

### 3. 添加反向代理（Nginx）
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8300;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. 添加视频缓存
在 Nginx 中配置缓存：
```nginx
location /api/episode/ {
    proxy_cache my_cache;
    proxy_cache_valid 200 1h;
    proxy_pass http://localhost:8300;
}
```

### 5. 监控和日志
- [ ] 设置日志轮转
- [ ] 监控服务器资源使用
- [ ] 设置错误告警

## 安全检查

- [ ] 不要在生产环境使用 `debug_mode: true`
- [ ] 限制文件上传大小
- [ ] 添加访问控制（如需要）
- [ ] 使用 HTTPS（如果是公网访问）
- [ ] 定期备份 `annotations.db`

## 备份建议

### 定期备份
```bash
# 备份数据库
cp annotations.db annotations.db.backup.$(date +%Y%m%d)

# 备份配置
cp config.yaml config.yaml.backup.$(date +%Y%m%d)
```

### 备份计划
- [ ] 每天备份数据库
- [ ] 每周备份配置文件
- [ ] 保留最近 30 天的备份

## 完成确认

- [ ] 所有依赖已安装
- [ ] 配置文件正确
- [ ] 数据集可访问
- [ ] 服务器正常启动
- [ ] 网页可以访问
- [ ] 所有功能测试通过
- [ ] API 测试通过
- [ ] 性能可接受
- [ ] 日志正常
- [ ] 备份计划已设置

## 部署完成！

如有问题，请查看：
- `TEST_GUIDE.md` - 详细测试指南
- `UPDATE_SUMMARY.md` - 更新摘要
- `CHANGELOG.md` - 变更记录

祝使用愉快！🎉

