#!/bin/bash
# 前端服务守护脚本 - 自动重启
# 用法: setsid bash serve_daemon.sh &

cd /root/autodl-tmp/URLLM-project/webapp
PYTHON=/root/miniconda3/envs/urllm/bin/python
LOG=/tmp/webapp_daemon.log

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动前端服务..." >> "$LOG"
    $PYTHON serve.py >> "$LOG" 2>&1
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 服务退出 (code=$EXIT_CODE), 3秒后重启..." >> "$LOG"
    sleep 3
done
