#!/bin/bash
# 清理视频缓存脚本

CACHE_DIR="/DATA/guantianrui/tmp/zarr_video_cache"

if [ -d "$CACHE_DIR" ]; then
    echo "正在清理视频缓存: $CACHE_DIR"
    rm -rf "$CACHE_DIR"/*
    echo "✓ 缓存已清理"
    ls -lh "$CACHE_DIR" 2>/dev/null | wc -l
else
    echo "缓存目录不存在: $CACHE_DIR"
fi

