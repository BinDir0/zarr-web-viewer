# 最终更新 - 完整MANO手部渲染 + Fingertips

## 更新时间
2026-02-01

## 最新功能 (v2.1)

### ✅ 额外渲染 Fingertips
- **新增**：在 MANO 手部渲染基础上，额外显示数据集中的原始指尖位置
- **用途**：数据验证、精度对比、质量检查
- **样式**：
  - 左手指尖：绿色边框 + 青色内圈 (7px/4px)
  - 右手指尖：洋红色边框 + 黄色内圈 (7px/4px)
  - 带标签：L0-L4, R0-R4
- **优势**：
  - 对比 MANO 生成的指尖和数据集存储的指尖
  - 发现数据标注问题
  - 评估模型拟合精度

## 主要改进 (v2.0)

### ✅ 采用专业渲染方法

参考了 `visualizer/visualize_episode_video.py` 的专业实现，大幅提升手部可视化质量。

### 关键改进点

#### 1. 完整的MANO模型支持
- **之前**：只显示指尖和手腕位置（5+2个点）
- **现在**：完整的手部网格（778个顶点）+ 关节骨架（21个关节）

#### 2. 专业的可视化效果
- **之前**：简单的圆点标记
- **现在**：
  - 半透明点云显示完整手部形状
  - 关节骨架显示手指连接关系
  - 清晰的颜色区分（左手蓝/青，右手红/黄）

#### 3. 准确的坐标变换
- **之前**：简单的3D→2D投影
- **现在**：
  - 6D旋转 → 旋转矩阵 → 轴角表示
  - 世界坐标 → 相机坐标 → 图像坐标
  - 正确处理相机外参和内参

#### 4. Fallback机制
- 如果没有安装MANO模型，自动降级到简化渲染
- 确保在任何环境下都能正常工作

## 技术实现

### 新增核心函数

#### 旋转转换
```python
def rot6_to_rotmat(r6)
    # 6D旋转表示 → 3x3旋转矩阵

def rotmat_to_axisangle(R)
    # 旋转矩阵 → 轴角表示（用于MANO）
```

#### 投影函数
```python
def project_points(pts_cam, fx, fy, cx, cy)
    # 3D相机坐标 → 2D图像坐标
    # u = fx * (x / z) + cx
    # v = fy * (y / z) + cy
```

#### MANO网格生成
```python
def generate_mano_mesh(mano_layer, pose, rotation, trans, shape)
    # 输入: MANO参数
    # 输出: 顶点(778,3) + 关节(21,3)
```

#### 完整渲染函数
```python
def render_hand_on_frame(frame, mano_params, wrist_params, 
                         extrinsic, intrinsic, presence, 
                         shape_params, mano_layers)
    # 1. 生成MANO网格
    # 2. 坐标变换
    # 3. 投影到2D
    # 4. 绘制点云、骨架、关节
```

### 渲染流程

```
数据读取
  ↓
旋转转换 (6D → RotMat → AxisAngle)
  ↓
MANO推理 (Pose + Rotation → Vertices + Joints)
  ↓
坐标变换 (World → Camera → Image)
  ↓
渲染绘制
  ├─ 点云（半透明混合）
  ├─ 关节连线（骨架）
  └─ 关节点（标记）
  ↓
视频生成 (30 FPS, MP4)
```

## 视觉效果对比

### 之前（简化版）
```
🔵 L0  🔵 L1  🔵 L2  🔵 L3  🔵 L4   （5个指尖点）
🟡 L_wrist                       （1个手腕点）
```

### 现在（完整MANO）
```
     ⚪⚪⚪⚪⚪
    ⚪⚪⚪⚪⚪⚪⚪     （778个顶点的点云）
   ⚪⚪⚪⚪⚪⚪⚪⚪⚪
    ⚪⚪⚪⚪⚪⚪⚪
     
    🔵─🔵─🔵─🔵─🔵    （21个关节 + 连线）
    🔵─🔵─🔵─🔵─🔵
      🔵─🔵─🔵─🔵
    🔵─🔵─🔵─🔵─🔵
    🔵─🔵─🔵─🔵─🔵
         🔵 (手腕)
```

## 安装要求

### 必需依赖（已有）
- Flask
- NumPy
- OpenCV
- Zarr

### 新增依赖
- **PyTorch** >= 2.0.0 （用于MANO推理）
- **manopth** （MANO模型库）

### 安装步骤

1. **安装基础依赖**：
```bash
cd /home/guantianrui/zarr-viewer-stage2
pip install -r requirements.txt
```

2. **安装MANO模型**：
```bash
# 克隆 manopth
cd /home/guantianrui
git clone https://github.com/hassony2/manopth.git
cd manopth
pip install -e .

# 下载MANO模型文件（需要注册）
# 访问: https://mano.is.tue.mpg.de/
# 将模型文件放到: /home/guantianrui/manopth/mano/models/
```

3. **设置环境变量**（可选）：
```bash
export MANO_ROOT=/home/guantianrui/manopth/mano/models
```

4. **验证安装**：
```bash
python -c "from manopth.manolayer import ManoLayer; print('MANO OK')"
```

## 数据集要求

### 必需字段
- `data/image`: RGB图像
- `data/state/mano` 或 `data/action/mano`: MANO参数 (N, 90)
- `data/state/wrist` 或 `data/action/wrist`: 手腕数据 (N, 18)
- `data/intrinsic`: 相机内参
- `data/extrinsic`: 相机外参

### 可选字段
- `data/state/shape`: 手部形状参数 (N, 20)
- `data/presence`: 手的可见性 (N,)

## 使用方法

### 方法1：Web界面
1. 启动服务器：`python app.py`
2. 访问：`http://localhost:8300`
3. 选择一个episode
4. 观看两个视频：
   - 左侧：原始视频
   - 右侧：手部渲染视频

### 方法2：API调用
```bash
# 获取原始视频
curl "http://localhost:8300/api/episode/数据集::0/video/original" -o original.mp4

# 获取渲染视频（带MANO手部）
curl "http://localhost:8300/api/episode/数据集::0/video/rendered" -o rendered.mp4
```

## 性能特点

### 渲染速度
- **首次加载**：需要初始化MANO模型（约1-2秒）
- **单帧渲染**：约20-50ms（取决于手部数量）
- **完整视频**：取决于帧数，100帧约5-10秒

### 内存使用
- **基础服务**：约500MB
- **加载MANO**：+200MB
- **渲染视频**：+根据帧数和分辨率

### 优化建议
1. 启用GPU加速（如果有CUDA）
2. 缓存生成的视频
3. 降低帧率（30→15 FPS）
4. 使用批量推理（如果需要批量生成）

## 故障排查

### 问题1：MANO未加载
**日志显示**：`⚠️  MANO 未安装，将使用简化的手部渲染`

**解决**：
- 检查manopth是否正确安装
- 检查MANO模型文件是否存在
- 查看详细错误信息

### 问题2：视频中只显示手腕
**原因**：MANO模型未成功加载，使用了Fallback

**解决**：
1. 验证MANO安装：`python -c "from manopth.manolayer import ManoLayer"`
2. 检查MANO_ROOT环境变量
3. 重启服务器

### 问题3：手部位置不准确
**原因**：坐标系或相机参数问题

**解决**：
1. 检查外参是否正确
2. 确认单位（米 vs 毫米）
3. 验证相机内参格式

### 问题4：渲染很慢
**原因**：MANO推理计算量大

**解决**：
1. 使用GPU：`export CUDA_VISIBLE_DEVICES=0`
2. 降低帧率
3. 减少点云密度
4. 启用视频缓存

## 对比参考实现

| 特性 | visualize_episode_video.py | zarr-viewer-stage2 |
|------|---------------------------|-------------------|
| MANO渲染 | ✓ | ✓ |
| 点云显示 | ✓ | ✓ |
| 关节骨架 | ✓ | ✓ |
| 批量推理 | ✓ (32帧/批) | ✗ (逐帧) |
| 相机轨迹 | ✓ | ✗ |
| 预加载优化 | ✓ | ✗ |
| Web界面 | ✗ | ✓ |
| 实时生成 | ✗ | ✓ |
| Fallback模式 | ✗ | ✓ |

## 未来改进方向

### 短期（1-2周）
1. ✅ 采用专业渲染方法
2. ⬜ 添加视频缓存机制
3. ⬜ 优化渲染性能

### 中期（1个月）
1. ⬜ 批量预生成功能
2. ⬜ 添加相机轨迹可视化
3. ⬜ 支持参数调整（点云密度、透明度等）

### 长期（2-3个月）
1. ⬜ 多视角渲染
2. ⬜ 实时手势识别
3. ⬜ 3D交互查看器

## 相关文档

- [RENDERING_GUIDE.md](RENDERING_GUIDE.md) - 详细的渲染指南
- [CHANGELOG.md](CHANGELOG.md) - 完整变更记录
- [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md) - 更新摘要
- [TEST_GUIDE.md](TEST_GUIDE.md) - 测试指南
- [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - 部署清单

## 参考资料

- [MANO官网](https://mano.is.tue.mpg.de/)
- [manopth GitHub](https://github.com/hassony2/manopth)
- [参考实现](../visualizer/visualize_episode_video.py)

## 致谢

感谢 `visualizer/visualize_episode_video.py` 提供的专业实现参考，使本项目的手部渲染质量得到大幅提升！

---

**最后更新**：2026-02-01  
**版本**：2.0 (完整MANO渲染版本)

