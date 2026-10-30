#!/bin/bash
# 启动 RADAR Web 服务。用法：
#   bash radar_web/start.sh              # 默认 0.0.0.0:8000
#   bash radar_web/start.sh 9000         # 自定义端口
#   bash radar_web/start.sh 0.0.0.0 9000 # 自定义 host + port
set -e

# ---- 定位仓库根（自动探测常见位置）----
for CAND in "/damo-radar" "/root/repo/damo-radar" "$(pwd)" "$(pwd)/damo-radar"; do
    [ -f "$CAND/radar_web/server.py" ] && REPO="$CAND" && break
done
[ -z "${REPO:-}" ] && { echo "!! 未找到 damo-radar 仓库（含 radar_web/server.py）"; exit 1; }

HOST="${1:-0.0.0.0}"
PORT="${2:-8000}"
cd "$REPO"
exec uvicorn radar_web.server:app --host "$HOST" --port "$PORT" --timeout-graceful-shutdown 3
