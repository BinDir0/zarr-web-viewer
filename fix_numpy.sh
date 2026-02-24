#!/bin/bash

echo "================================"
echo "  NumPy 降级修复脚本"
echo "================================"
echo ""

# 检查当前 numpy 版本
echo "检查当前 NumPy 版本..."
CURRENT_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "当前 NumPy 版本: $CURRENT_VERSION"
else
    echo "❌ 无法获取 NumPy 版本"
    exit 1
fi

# 检查是否需要降级
if [[ "$CURRENT_VERSION" =~ ^2\. ]]; then
    echo "⚠️  检测到 NumPy 2.x，需要降级"
    echo ""
    
    # 降级 numpy
    echo "正在降级 NumPy..."
    pip install 'numpy<2.0.0' --quiet
    
    if [ $? -eq 0 ]; then
        NEW_VERSION=$(python -c "import numpy; print(numpy.__version__)")
        echo "✓ NumPy 已降级到: $NEW_VERSION"
    else
        echo "❌ NumPy 降级失败"
        exit 1
    fi
else
    echo "✓ NumPy 版本正常 ($CURRENT_VERSION < 2.0)"
fi

echo ""
echo "================================"
echo "  验证依赖"
echo "================================"
echo ""

# 验证关键依赖
echo "检查关键依赖..."

# chumpy
python -c "import chumpy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ chumpy: OK"
else
    echo "⚠️  chumpy: 缺失，正在安装..."
    pip install chumpy --quiet
fi

# torch
python -c "import torch" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ torch: OK"
else
    echo "⚠️  torch: 缺失"
fi

# cv2
python -c "import cv2" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ opencv-python: OK"
else
    echo "⚠️  opencv-python: 缺失"
fi

# 测试 MANO
echo ""
echo "测试 MANO 导入..."
python -c "from manopth.manolayer import ManoLayer; print('✓ MANO: OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  MANO: 无法导入（可能缺少模型文件）"
fi

echo ""
echo "================================"
echo "  修复完成！"
echo "================================"
echo ""
echo "下一步："
echo "1. 重启服务器: ./run.sh"
echo "2. 检查日志，确认看到: ✓ MANO 模型已成功加载"
echo "3. 不应该再看到重复的数据集加载"
echo ""

