"""
RADAR Web 推理服务（零侵入封装 RADAR_inference/inference_demo.py）

启动方式（在仓库根目录）:
    uvicorn radar_web.server:app --host 0.0.0.0 --port 8000

流程:
    1. 服务启动时调用 initialize() 常驻显存
    2. 前端先连 WebSocket /ws/{task_id}，再 POST /api/infer/{task_id} 上传 nii.gz
    3. 后台线程持 GPU 锁执行 evaluate()，stdout/stderr（含 tqdm 进度）实时写入日志缓冲
    4. WebSocket 轮询推送日志，推理完成后读取 CSV 推送结构化结果
"""
import sys
import os
import io
import json
import asyncio
import threading
import contextlib

# ---- 路径引导（必须在 import inference_demo 之前）----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'RADAR_inference'))
os.chdir(os.path.join(ROOT, 'RADAR_inference'))  # evaluate() 内部使用相对路径 ../ckpt

import pandas as pd
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ---- Mock 模式（本地调试用）----
# 启动方式: RADAR_WEB_MOCK=1 uvicorn radar_web.server:app --port 8000
# 该模式下不加载真实模型、不 import inference_demo（无需 torch/GPU），
# 用假输出模拟完整推理流程，用于在本机调试 radar_web 的页面/接口/日志流。
MOCK_MODE = os.environ.get('RADAR_WEB_MOCK', '') == '1'

if MOCK_MODE:
    import time

    def initialize():
        print('--> [MOCK] initialize() 被调用，模拟加载模型...')
        time.sleep(2)
        print('--> [MOCK] 模型加载完成')
        return 'mock_pad_func', 'mock_model'

    def evaluate(pad_func, model, img_dir, save_dir, save_tag):
        files = [f for f in os.listdir(img_dir) if f.endswith('.nii.gz')]
        fake_items = ['肝_肝囊肿 (Liver_Hepatic cyst)', '肾_囊肿 (Kidney_Cyst)',
                      '肝_脂肪肝 (Liver_Fatty liver)', '胆囊_结石 (Gallbladder_Gallstones)']
        import random
        random.seed(42)
        results = []
        for i, f in enumerate(files):
            print(f'-- [MOCK] 正在推理 {f} ({i+1}/{len(files)}) ...')
            for step in range(5):
                time.sleep(0.3)
                print(f'Infer: {int((step+1)/5*100)}%|{f[:12]}')
            row = {'file_name': f}
            row.update({k: round(random.random(), 3) for k in fake_items})
            results.append(row)
            print(f'-- [MOCK] {f} 完成')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'RADAR_infer_results_{save_tag}.csv')
        pd.DataFrame(results, columns=['file_name'] + fake_items).to_csv(
            save_path, index=False, encoding='utf-8-sig')
        print(f'-- [MOCK] 结果已写入 {save_path}')
else:
    import torch
    from inference_demo import initialize, evaluate

app = FastAPI(title='RADAR Web Inference')

# ---------------- 全局状态 ----------------
MODEL = {}                          # {'pad': ..., 'model': ...}
GPU_LOCK = threading.Lock()         # 串行化推理，防止显存溢出
MODEL_READY = threading.Event()

TASKS_LOCK = threading.Lock()
TASKS = {}                          # task_id -> {'logs': [..], 'done': bool, 'result': ...}


def get_task(task_id):
    with TASKS_LOCK:
        if task_id not in TASKS:
            TASKS[task_id] = {'logs': [], 'done': False, 'result': None}
        return TASKS[task_id]


def append_log(task, text):
    with TASKS_LOCK:
        task['logs'].append(text)


class TaskLogCaptor(io.TextIOBase):
    """捕获 tqdm / print 输出，按行写入任务日志缓冲。"""

    def __init__(self, task):
        self.task = task

    def write(self, s):
        if s and s.strip():
            append_log(self.task, s.rstrip())
        return len(s)

    def flush(self):
        pass


@app.on_event('startup')
def load_model():
    def _load():
        print('--> Loading RADAR model (checkpoint_radar_pretrain.pth) ...')
        pad_func, model = initialize()
        MODEL['pad'] = pad_func
        MODEL['model'] = model
        MODEL_READY.set()
        print('--> Model ready.')

    threading.Thread(target=_load, daemon=True).start()


# ---------------- 页面 ----------------
@app.get('/')
def index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'index.html')
    return HTMLResponse(open(html_path, encoding='utf-8').read())


# ---------------- WebSocket 日志流 ----------------
@app.websocket('/ws/{task_id}')
async def ws_logs(ws: WebSocket, task_id: str):
    await ws.accept()
    task = get_task(task_id)
    sent = 0
    try:
        while True:
            with TASKS_LOCK:
                logs = task['logs']
            while sent < len(logs):
                await ws.send_json({'type': 'log', 'text': logs[sent]})
                sent += 1
            if task['done'] and sent >= len(task['logs']):
                payload = {'type': 'result', 'result': task['result']}
                if task.get('error'):
                    payload['error'] = task['error']
                await ws.send_json(payload)
                break
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass


# ---------------- 上传 + 推理 ----------------
@app.post('/api/infer/{task_id}')
async def api_infer(task_id: str, files: list[UploadFile] = File(...)):
    if not MODEL_READY.is_set():
        return {'ok': False, 'msg': '模型尚未加载完成，请稍后重试'}

    task_dir = os.path.join(ROOT, 'radar_web', 'uploads', task_id)
    img_dir = os.path.join(task_dir, 'images')
    save_dir = os.path.join(task_dir, 'results')
    os.makedirs(img_dir, exist_ok=True)

    for f in files:
        dest = os.path.join(img_dir, os.path.basename(f.filename))
        with open(dest, 'wb') as out:
            out.write(await f.read())

    threading.Thread(target=run_inference, args=(task_id, img_dir, save_dir), daemon=True).start()
    return {'ok': True, 'msg': f'已接收 {len(files)} 个文件，推理已启动'}


def run_inference(task_id, img_dir, save_dir):
    task = get_task(task_id)
    task['done'] = False
    task['result'] = None
    task['error'] = None

    append_log(task, f'--> 任务 {task_id} 开始推理...')
    acquired = GPU_LOCK.acquire(timeout=3600)
    if not acquired:
        append_log(task, '!! 等待 GPU 超时')
        task['error'] = '等待 GPU 锁超时'
        task['done'] = True
        return

    try:
        with contextlib.redirect_stdout(TaskLogCaptor(task)), \
             contextlib.redirect_stderr(TaskLogCaptor(task)):
            evaluate(MODEL['pad'], MODEL['model'], img_dir, save_dir, task_id)
        if not MOCK_MODE:
            torch.cuda.empty_cache()

        csv_path = os.path.join(save_dir, f'RADAR_infer_results_{task_id}.csv')
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df = df.fillna('')
        task['result'] = json.loads(df.to_json(orient='records', force_ascii=False))
        append_log(task, f'--> 推理完成，结果已生成: RADAR_infer_results_{task_id}.csv')
    except Exception as e:
        import traceback
        append_log(task, '!! 推理出错: ' + str(e))
        append_log(task, traceback.format_exc())
        task['error'] = str(e)
    finally:
        GPU_LOCK.release()
        task['done'] = True
