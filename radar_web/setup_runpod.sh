#!/bin/bash
# RunPod 一键部署脚本（从头装环境并启动服务）
# 用法（RunPod 启动命令，单行）:
#   git clone https://github.com/victor-jh-wang/damo-radar.git /root/repo/damo-radar && bash /root/repo/damo-radar/radar_web/setup_runpod.sh
set -e

# 1. Miniconda（已装则跳过）
if [ ! -d /root/miniconda3 ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/mc.sh
    bash /root/mc.sh -b -p /root/miniconda3
    rm /root/mc.sh
fi
source /root/miniconda3/etc/profile.d/conda.sh
grep -q 'miniconda3/etc/profile.d/conda.sh' ~/.bashrc || echo 'source /root/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc

# 2. conda 环境 + 依赖（已建则跳过创建）
conda activate radar 2>/dev/null || {
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    conda create -n radar python=3.10 -y
    conda activate radar
    grep -q 'conda activate radar' ~/.bashrc || echo 'conda activate radar' >> ~/.bashrc
    pip install -r /root/repo/damo-radar/requirements.txt -r /root/repo/damo-radar/radar_web/requirements.txt
}

# 3. 权重（已下载则跳过）
if [ ! -f /root/repo/damo-radar/ckpt/checkpoint_radar_pretrain.pth ]; then
    cd /root/repo/damo-radar/download_scripts
    python download_checkpoints.py
    cd /root/repo/damo-radar
fi

# 4. 启动（前台运行，保持 pod 存活）
echo "==> All ready, starting uvicorn ..."
exec uvicorn radar_web.server:app --host 0.0.0.0 --port 8000
