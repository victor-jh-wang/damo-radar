# RADAR Web 推理服务

零侵入封装官方 `RADAR_inference/inference_demo.py` 的网页推理服务：浏览器拖拽上传 nii.gz → 点击推理 → 实时查看服务器日志 → 展示 168 项器官-疾病概率表格，可下载 CSV。

不修改官方推理代码，`server.py` 直接调用 `initialize()` / `evaluate()`，推理结果通过读取其生成的 CSV 转成 JSON 返回前端。

## 目录结构

```
radar_web/
├── server.py          # FastAPI 服务（模型加载、上传、推理、WebSocket 日志）
├── templates/
│   └── index.html     # 前端页面（拖拽上传、日志滚动、结果表格）
├── requirements.txt   # Web 层依赖（官方项目依赖另行安装）
└── uploads/           # 运行时自动生成：上传文件与推理结果（可定期清理）
```

## 部署步骤（RunPod / 通用 GPU 服务器）

### 方式一：一键脚本（推荐）

Jupyter/SSH 终端里执行两条命令（clone 到哪个目录都行，脚本自动探测）：

```bash
git clone https://github.com/victor-jh-wang/damo-radar.git
bash damo-radar/radar_web/setup_runpod.sh
```

脚本 `radar_web/setup_runpod.sh` 会依次：装 Miniconda → 建 radar 环境 → 装两套依赖 → 下载权重（每步已存在则跳过，可重复执行）→ 前台启动 uvicorn。全程约 10-15 分钟。

> 注意：不要把命令放进 RunPod 的 "Container start command"——该框不按标准 shell 方式执行命令，多行/重定向会失败。请通过 Jupyter Lab 或 SSH 终端操作。

### 方式二：手动逐段执行

```bash
# 1. Miniconda + 写入 .bashrc（以后每个新终端 conda 直接可用并自动激活 radar 环境）
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/mc.sh
bash /root/mc.sh -b -p /root/miniconda3
rm /root/mc.sh
echo 'source /root/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc

# 2. 代码
git clone https://github.com/victor-jh-wang/damo-radar.git /root/repo/damo-radar
cd /root/repo/damo-radar

# 3. conda 环境 + 依赖
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n radar python=3.10 -y
echo 'conda activate radar' >> ~/.bashrc
conda activate radar
pip install -r requirements.txt -r radar_web/requirements.txt

# 4. 权重（约 1.5GB，默认下载到仓库 ckpt/ 目录）
cd download_scripts
python download_checkpoints.py
cd ..

# 5. 启动（必须在仓库根目录）
uvicorn radar_web.server:app --host 0.0.0.0 --port 8000
```

说明：

- 启动后模型自动加载入显存（约 1-2 分钟），**看到 `Model ready.` 才算就绪**
- 国内网络下载权重不通时，第 4 步前执行 `export HF_ENDPOINT=https://hf-mirror.com`
- RunPod 需在 Pod 设置 Expose HTTP ports 中加入 `8000`（改配置会重置 pod，务必先改配置再装环境）
- 日常更新代码只需 `git pull` 后重启 uvicorn，环境不用动

## 本地调试（Mock 模式，无需 GPU / torch）

在自己的 Mac 上不加载真实模型，用假输出模拟完整推理流程，用于调试页面、上传接口和日志流：

```bash
RADAR_WEB_MOCK=1 uvicorn radar_web.server:app --port 8000
```

Mock 模式下：
- 不 import `inference_demo`，不需要 torch / GPU / 权重文件，只装 `radar_web/requirements.txt` 即可运行
- `initialize()` 变为 2 秒假加载；`evaluate()` 会对每个上传文件模拟约 2 秒推理并输出进度日志
- 结果为随机概率的 4 个示例检查项（肝囊肿、肾囊肿、脂肪肝、胆囊结石），仅用于验证前端表格/CSV 下载

## 使用方法

1. 在 3D Slicer 中将院内 DICOM 目标序列（门静脉期薄层轴位）导出为 `.nii.gz`
2. 网页中拖入一个或多个 `.nii.gz` 文件，点击「开始推理」
3. 日志区实时显示服务器推理进度（含 tqdm 进度条）
4. 完成后表格按概率降序展示 168 个检查项，可通过概率阈值过滤、下载 CSV
5. 点击结果标题旁的「查看影像」可在浏览器中直接浏览该 nii.gz（niivue 前端渲染：三平面视图、可拖动切换层面、右键调窗宽窗位）。niivue 库已内置在 `static/` 目录，内网无外网也能用

## 技术说明

- **路径引导**：`server.py` 顶部将 `RADAR_inference/` 加入 `sys.path` 并 `os.chdir` 到该目录，以兼容官方代码中的相对路径 `../ckpt`
- **零侵入**：不改动 `RADAR_inference/` 任何文件；结果通过 `pd.read_csv` 读取 `RADAR_infer_results_{task_id}.csv`
- **串行推理**：全局 GPU 锁保证同一时刻只有一个推理任务占用显存
- **日志流**：推理线程通过重定向 stdout/stderr 捕获 tqdm 进度，WebSocket 轮询推送到浏览器
- **启动即加载**：模型在 FastAPI startup 阶段常驻显存，避免每次请求重复加载权重

## 已知限制

- 单个病例推理需数分钟（滑窗 96×256×384、逐器官二次前向），请耐心等待
- 图像任一维度超过 1000（重采样后）会被官方逻辑跳过，日志中不产生结果
- `uploads/` 目录会随使用积累文件，需要定期手动清理
