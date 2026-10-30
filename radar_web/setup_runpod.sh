#!/bin/bash
# RunPod 一键部署脚本：装环境 + 下载权重 + 启动服务
# 用法（Jupyter/SSH 终端里，仓库 clone 到哪都行）:
#   git clone https://github.com/victor-jh-wang/damo-radar.git
#   bash damo-radar/radar_web/setup_runpod.sh
set -e

# ---- 定位仓库根（自动探测常见位置）----
for CAND in "/damo-radar" "/root/repo/damo-radar" "$(pwd)/damo-radar"; do
    [ -f "$CAND/radar_web/server.py" ] && REPO="$CAND" && break
done
[ -z "${REPO:-}" ] && { echo "!! 未找到 damo-radar 仓库（含 radar_web/server.py），请先 git clone"; exit 1; }
echo "==> 仓库位置: $REPO"

# 1. Miniconda（已装则跳过）
if [ ! -d /root/miniconda3 ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/mc.sh
    bash /root/mc.sh -b -p /root/miniconda3
    rm /root/mc.sh
fi
source /root/miniconda3/etc/profile.d/conda.sh
grep -q 'miniconda3/etc/profile.d/conda.sh' ~/.bashrc || echo 'source /root/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc

# 2. conda 环境 + 依赖（已建则跳过创建）
if ! conda activate radar 2>/dev/null; then
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    conda create -n radar python=3.10 -y
    conda activate radar
    grep -q 'conda activate radar' ~/.bashrc || echo 'conda activate radar' >> ~/.bashrc
    pip install -r "$REPO/requirements.txt" -r "$REPO/radar_web/requirements.txt"
fi

# 3. 权重（已下载则跳过）
if [ ! -f "$REPO/ckpt/checkpoint_radar_pretrain.pth" ]; then
    cd "$REPO/download_scripts"
    python download_checkpoints.py
    cd "$REPO"
fi

# 4. 部署完成（服务由用户手动启动）
cd "$REPO"
echo "==> 部署完成。手动启动服务："
echo "    uvicorn radar_web.server:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 3"
