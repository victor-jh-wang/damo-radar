#!/bin/bash
# RunPod 一键部署脚本（幂等：重复执行会跳过已完成的步骤）
# 用法（作为 Pod 的 Container start command）：
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/victor-jh-wang/damo-radar/main/radar_web/setup_runpod.sh)"
# 或手动：bash radar_web/setup_runpod.sh
set -e

REPO_DIR=/root/repo/damo-radar
LOG=/root/setup.log
exec > >(tee -a "$LOG") 2>&1
echo "===== RADAR RunPod setup 开始: $(date) ====="

# 1. Miniconda（已装则跳过）
if [ ! -d /root/miniconda3 ]; then
    echo "--> [1/5] 安装 Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/mc.sh
    bash /root/mc.sh -b -p /root/miniconda3
    rm -f /root/mc.sh
else
    echo "--> [1/5] Miniconda 已存在，跳过"
fi
source /root/miniconda3/etc/profile.d/conda.sh
grep -q 'etc/profile.d/conda.sh' ~/.bashrc || echo 'source /root/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc

# 2. 代码（已 clone 则 git pull 更新）
if [ ! -d "$REPO_DIR" ]; then
    echo "--> [2/5] 克隆代码..."
    mkdir -p /root/repo
    git clone https://github.com/victor-jh-wang/damo-radar.git "$REPO_DIR"
else
    echo "--> [2/5] 代码已存在，git pull 更新..."
    git -C "$REPO_DIR" config --global --add safe.directory "$REPO_DIR" 2>/dev/null || true
    git -C "$REPO_DIR" pull || echo "   git pull 失败（可能有本地改动），继续用现有代码"
fi
cd "$REPO_DIR"

# 3. conda 环境 + 依赖（环境存在则只补装依赖，很快）
if ! conda env list | grep -q '^radar '; then
    echo "--> [3/5] 创建 conda 环境并安装依赖（约 5-10 分钟）..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    conda create -n radar python=3.10 -y
    grep -q 'conda activate radar' ~/.bashrc || echo 'conda activate radar' >> ~/.bashrc
else
    echo "--> [3/5] conda 环境已存在"
fi
conda activate radar

# 4. 权重（缺文件才下载，约 1.5GB）
if [ ! -f ckpt/checkpoint_radar_pretrain.pth ]; then
    echo "--> [4/5] 下载权重（约 1.5GB）..."
    (cd download_scripts && python download_checkpoints.py)
else
    echo "--> [4/5] 权重已存在，跳过"
fi

# 5. 启动服务（前台运行，Pod 日志可直接看到 uvicorn 输出）
echo "--> [5/5] 启动 uvicorn..."
echo "===== setup 完成，服务启动: $(date) ====="
exec uvicorn radar_web.server:app --host 0.0.0.0 --port 8000
