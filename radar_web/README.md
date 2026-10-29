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

## 部署步骤（服务器）

### 1. 克隆仓库并下载权重

```bash
git clone https://github.com/victor-jh-wang/damo-radar.git
cd damo-radar
python download_scripts/download_checkpoints.py --local-dir ./ckpt
```

### 2. 创建 conda 环境，安装两套依赖

```bash
conda create -n radar python=3.10 -y
conda activate radar

# 官方推理依赖（仓库根 requirements）
pip install -r requirements.txt

# Web 服务依赖
pip install -r radar_web/requirements.txt
```

### 3. 启动服务（必须在仓库根目录）

```bash
uvicorn radar_web.server:app --host 0.0.0.0 --port 8000
```

启动后模型自动加载入显存（约需 1-2 分钟），加载完成后即可访问 `http://<服务器IP>:8000`。

> 云服务器需在安全组放行 8000 端口。

## 使用方法

1. 在 3D Slicer 中将院内 DICOM 目标序列（门静脉期薄层轴位）导出为 `.nii.gz`
2. 网页中拖入一个或多个 `.nii.gz` 文件，点击「开始推理」
3. 日志区实时显示服务器推理进度（含 tqdm 进度条）
4. 完成后表格按概率降序展示 168 个检查项，可通过概率阈值过滤、下载 CSV

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
