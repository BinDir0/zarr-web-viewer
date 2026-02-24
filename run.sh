#!/bin/bash

# Zarr 数据集查看器 - 启动脚本

echo "================================"
echo "  Zarr 数据集查看器"
echo "================================"
echo ""

# 检查配置文件
if [ ! -f "config.yaml" ]; then
    echo "❌ 错误: 未找到 config.yaml 文件"
    exit 1
fi

# 检查是否已安装依赖
echo "检查依赖..."
python3 -c "import flask, zarr, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  检测到缺少依赖，正在安装..."
    pip install -r requirements.txt
else
    echo "✓ 依赖已安装"
fi

echo ""
echo "正在启动服务器..."
echo ""

# 运行应用
python3 app.py

