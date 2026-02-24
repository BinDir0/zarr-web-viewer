# 快速开始 - MANO手部渲染版本

## 5分钟快速部署

### 步骤 1: 安装依赖
```bash
cd /home/guantianrui/zarr-viewer-stage2
pip install -r requirements.txt
```

### 步骤 2: 安装MANO（可选，推荐）
```bash
# 克隆manopth库
cd /home/guantianrui
git clone https://github.com/hassony2/manopth.git
cd manopth
pip install -e .

# 下载MANO模型（需要注册）
# 访问: https://mano.is.tue.mpg.de/
# 下载后放到: /home/guantianrui/manopth/mano/models/
```

### 步骤 3: 配置数据集
编辑 `config.yaml`：
```yaml
server_port: 8300
debug_mode: true

zarr_datasets:
  - name: "我的数据集"
    path: "/path/to/your/dataset.zarr"
    enabled: true
```

### 步骤 4: 启动服务
```bash
python app.py
```

### 步骤 5: 访问Web界面
浏览器打开：`http://localhost:8300`

## 验证安装

### 检查MANO是否安装
```bash
python -c "from manopth.manolayer import ManoLayer; print('✓ MANO已安装')"
```

成功：看到 `✓ MANO已安装`
失败：看到 ImportError → 将使用简化渲染（仍可正常工作）

### 检查服务器启动
终端应显示：
```
✓ MANO 模型已成功加载  （如果安装了MANO）
✓ 成功加载数据集: 我的数据集
启动服务器在端口 8300
访问: http://localhost:8300
```

## 使用示例

### 示例 1: 基本使用
1. 打开 `http://localhost:8300`
2. 左侧选择一个episode
3. 查看：
   - 顶部：任务指令（如果有）
   - 左侧视频：原始视频
   - 右侧视频：手部渲染视频

### 示例 2: 添加备注
1. 在"备注"文本框中输入内容
2. 点击"保存备注"
3. 备注会被保存到数据库

### 示例 3: API调用
```bash
# 下载原始视频
curl "http://localhost:8300/api/episode/我的数据集::0/video/original" -o original.mp4

# 下载渲染视频
curl "http://localhost:8300/api/episode/我的数据集::0/video/rendered" -o rendered.mp4
```

## 渲染效果说明

### 完整MANO渲染（推荐）
如果安装了MANO，你会看到：
- ✓ 完整的手部点云（778个顶点）
- ✓ 21个关节点和连接线
- ✓ 清晰的手指骨架
- ✓ 左手蓝/青色，右手红/黄色

### 简化渲染（Fallback）
如果没有MANO，你会看到：
- ✓ 手腕位置标记
- ⚠️ 没有详细的手指信息

## 常见问题

### Q1: 视频加载很慢？
A: 这是正常的，视频是实时生成的。首次加载需要：
- 读取数据
- MANO推理
- 渲染每一帧
- 生成视频文件

建议：等待几秒钟，视频会自动开始播放

### Q2: 视频中只显示手腕？
A: MANO模型未加载，使用了简化渲染。
解决：安装MANO模型（见步骤2）

### Q3: 手部位置不准确？
A: 可能的原因：
- 相机参数不正确
- 数据集格式问题
- 坐标系不匹配

解决：检查数据集是否包含正确的 `intrinsic` 和 `extrinsic` 数据

### Q4: 如何加速渲染？
A: 几个方法：
1. 使用GPU（如果有CUDA）
2. 降低视频帧率（修改代码中的 `fps=30` → `fps=15`）
3. 减少点云密度
4. 启用视频缓存（需要自行实现）

## 数据集要求清单

### ✓ 必需字段
- [x] `data/image` - RGB图像
- [x] `data/state/mano` - MANO参数
- [x] `data/state/wrist` - 手腕数据
- [x] `data/intrinsic` - 相机内参
- [x] `data/extrinsic` - 相机外参

### ⭐ 可选字段（提升效果）
- [ ] `data/state/fingertips` - 指尖位置（用于额外渲染）
- [ ] `data/state/shape` - 手部形状参数
- [ ] `data/presence` - 手的可见性
- [ ] `data/instruction` - 任务指令

### 检查数据集
```bash
python inspect_zarr.py /path/to/your/dataset.zarr
```

## 性能参考

### 配置：Intel i7, 16GB RAM, 无GPU
- 100帧episode：约 8-12秒
- 200帧episode：约 15-20秒
- 500帧episode：约 35-50秒

### 配置：Intel i7, 16GB RAM, NVIDIA GPU
- 100帧episode：约 3-5秒
- 200帧episode：约 6-8秒
- 500帧episode：约 12-20秒

## 下一步

### 了解更多
- [RENDERING_GUIDE.md](RENDERING_GUIDE.md) - 详细渲染原理
- [FINAL_UPDATE.md](FINAL_UPDATE.md) - 最新更新说明
- [TEST_GUIDE.md](TEST_GUIDE.md) - 完整测试指南

### 高级功能
- 批量生成视频
- 自定义渲染参数
- 添加视频缓存
- 性能优化

### 问题反馈
如有问题，请查看：
1. 服务器日志（终端输出）
2. 浏览器控制台（F12）
3. 相关文档

## 快速命令参考

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务器
python app.py

# 后台运行
nohup python app.py > server.log 2>&1 &

# 停止服务器
pkill -f "python app.py"

# 查看日志
tail -f server.log

# 检查数据集
python inspect_zarr.py /path/to/dataset.zarr

# 测试API
curl http://localhost:8300/api/episodes | jq
```

## 成功标志

看到这些表示成功：
- ✅ 服务器正常启动
- ✅ 网页可以访问
- ✅ 左侧显示episodes列表
- ✅ 点击episode显示详情
- ✅ 视频可以播放
- ✅ 手部动作清晰可见（如果有MANO）

祝使用愉快！🎉

