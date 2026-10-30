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
import time
import asyncio
import threading
import contextlib

# ---- 路径引导（必须在 import inference_demo 之前）----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'RADAR_inference'))
os.chdir(os.path.join(ROOT, 'RADAR_inference'))  # evaluate() 内部使用相对路径 ../ckpt

import pandas as pd
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

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
app.mount('/static', StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')), name='static')

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
async def ws_logs(ws: WebSocket, task_id: str, since: int = 0):
    """日志流。since=已收到的日志条数，断线重连时从该位置续传。空闲时发心跳防止代理掐断。"""
    await ws.accept()
    task = get_task(task_id)
    sent = max(0, since)
    last_send = time.time()
    try:
        while True:
            with TASKS_LOCK:
                logs = task['logs']
            while sent < len(logs):
                await ws.send_json({'type': 'log', 'text': logs[sent]})
                sent += 1
                last_send = time.time()
            if task['done'] and sent >= len(task['logs']):
                payload = {'type': 'result', 'result': task['result']}
                if task.get('error'):
                    payload['error'] = task['error']
                await ws.send_json(payload)
                await ws.close()   # 正常关闭（code 1000），避免浏览器报连接丢失
                break
            # 心跳：推理单例耗时数分钟无日志，RunPod 等代理会掐空闲连接
            if time.time() - last_send > 5:
                await ws.send_json({'type': 'ping'})
                last_send = time.time()
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass


# ---------------- 上传 + 推理 ----------------
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB 分块，避免 RunPod HTTPS 代理对大 POST 的超时限制


@app.post('/api/upload/{task_id}/{filename}')
async def api_upload(task_id: str, filename: str, offset: int, request: Request):
    """分块上传（裸二进制请求体）：offset=0 创建/覆盖文件，offset>0 追加。返回服务端当前文件大小供校验。"""
    img_dir = os.path.join(ROOT, 'radar_web', 'uploads', task_id, 'images')
    os.makedirs(img_dir, exist_ok=True)
    dest = os.path.join(img_dir, os.path.basename(filename))
    chunk = await request.body()
    mode = 'ab' if offset > 0 else 'wb'
    with open(dest, mode) as out:
        out.write(chunk)
    return {'ok': True, 'size': os.path.getsize(dest)}


@app.post('/api/infer/{task_id}')
async def api_infer(task_id: str, files: list[UploadFile] = File(None), start: int = 1):
    task_dir = os.path.join(ROOT, 'radar_web', 'uploads', task_id)
    img_dir = os.path.join(task_dir, 'images')
    save_dir = os.path.join(task_dir, 'results')
    os.makedirs(img_dir, exist_ok=True)

    for f in files or []:
        dest = os.path.join(img_dir, os.path.basename(f.filename))
        with open(dest, 'wb') as out:
            out.write(await f.read())

    if not start:
        return {'ok': True, 'msg': f'已上传 {len(files or [])} 个文件'}

    n_imgs = len([f for f in os.listdir(img_dir) if f.endswith(('.nii.gz', '.nii'))])
    if not n_imgs:
        return {'ok': False, 'msg': '任务目录中没有影像文件'}

    if not MODEL_READY.is_set():
        return {'ok': False, 'msg': '模型尚未加载完成，请稍后重试'}
    threading.Thread(target=run_inference, args=(task_id, img_dir, save_dir), daemon=True).start()
    return {'ok': True, 'msg': f'推理已启动（共 {n_imgs} 个影像）'}


# ---------------- 影像预览（前端 niivue 渲染）----------------
@app.get('/api/file/{task_id}/{filename}')
def api_file(task_id: str, filename: str):
    """返回任务目录下的原始 nii.gz，供前端 niivue 渲染。images 里没有则查 results（分割产物）。"""
    fname = os.path.basename(filename)
    for sub in ('images', 'results'):
        path = os.path.join(ROOT, 'radar_web', 'uploads', task_id, sub, fname)
        if os.path.isfile(path):
            # 分割产物不变，允许浏览器缓存，重复查看无需重新下载
            return FileResponse(path, media_type='application/gzip', filename=fname,
                                headers={'Cache-Control': 'private, max-age=86400'})
    return {'ok': False, 'msg': f'文件不存在: {filename}'}


# ---------------- 器官分割（复用主模型 UNet，零侵入）----------------
# 36 器官名（顺序 = 标签-1，与官方 vision_branch.py / inference_demo.py 的 organs 列表一致）
ORGANS = ['肾上腺', '主动脉', '竖脊肌', '脑', '锁骨', '大肠', '十二指肠', '食管', '面部', '股骨',
          '胆囊', '臀肌', '心脏', '髋关节', '肱骨', '髂动脉', '髂静脉', '髂腰肌', '下腔静脉', '肾',
          '肝', '肺', '胰腺', '门静脉', '肺动脉', '肋骨', '骶骨', '肩胛骨', '小肠', '脾',
          '胃', '气管', '膀胱', '颈椎', '腰椎', '胸椎']

SEG_STATE = {}          # key=task/filename -> 'running' | 'done' | ('error', msg)
SEG_STATE_LOCK = threading.Lock()


def _preprocess_for_seg(img_path):
    """复刻 inference_demo.DataFolder.__getitem__ 的预处理（官方逻辑的逐行拷贝，零侵入）。
    额外返回几何信息（resize 比例、crop 起点），用于把分割质心映射回原始文件坐标系。"""
    import numpy as np
    from monai import transforms

    data = {"image": img_path}
    res = transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True)(data)
    image = res["image"]

    affine = res["image_meta_dict"]["affine"]
    spacing = (
        abs(affine[0, 0].item()),
        abs(affine[1, 1].item()),
        abs(affine[2, 2].item())
    )
    _, h, w, d = image.shape

    ref_spacing = (1.0, 1.0, 5.0)
    scale = [spacing[i] / ref_spacing[i] for i in range(3)]
    target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]  # [H', W', D']

    trans = transforms.Compose(
        [
            transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear"),
            transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
        ]
    )
    resized_data = trans(res)

    img = resized_data["image"]   # [C, D', W', H']
    img[img > 400] = 400
    img[img < -300] = -300
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # crop non-zero region in image
    roi_coords = np.nonzero(img[0].cpu().numpy())
    min_dhw = np.min(roi_coords, axis=1)
    max_dhw = np.max(roi_coords, axis=1)

    extend_d = 5
    extend_hw = 20

    min_dhw = np.maximum(
        min_dhw - np.array([extend_d, extend_hw, extend_hw]),
        np.array([0, 0, 0]),
    )
    max_dhw = np.minimum(
        max_dhw + np.array([extend_d, extend_hw, extend_hw]),
        np.array(img.shape[1:]),
    )

    cropped_image = img[
        :,
        min_dhw[0]: max_dhw[0],
        min_dhw[1]: max_dhw[1],
        min_dhw[2]: max_dhw[2]
    ]

    pad_func = transforms.SpatialPadd(
        keys=["image"],
        spatial_size=(96, 256, 384),
        mode='constant',
        constant_values=0,
        method="end"
    )
    padded = pad_func({"image": cropped_image})["image"].as_tensor()

    # 几何映射信息：seg 体素 (d,w,h) -> resize 坐标 = +min_dhw -> 原始体素 = /scale（pad 只在末端无偏移）
    geo = {
        'min_dhw': [int(x) for x in min_dhw],
        'scale': [float(x) for x in scale],
    }
    return padded, geo


def _segment_one(task_id, img_path, save_dir):
    """对单个 nii.gz 跑 37 类器官分割，输出 seg_ct / seg_mask 两个 nii.gz（同一预处理坐标系）。"""
    import shutil
    fname = os.path.basename(img_path)
    ct_name = f'seg_ct_{fname}'
    mask_name = f'seg_mask_{fname}'
    task = get_task(task_id)

    if MOCK_MODE:
        # mock：直接拷贝原文件充当两个产物，仅用于本地调试 UI
        time.sleep(2)
        shutil.copyfile(img_path, os.path.join(save_dir, ct_name))
        shutil.copyfile(img_path, os.path.join(save_dir, mask_name))
        # 假 meta：前几个器官 + 中心坐标，供前端调试图例/定位
        meta = {'organs': {n: {'label': i + 1, 'count': 100,
                               'centroid_seg': [10, 50, 100],
                               'centroid_orig': [100, 50, 10]}
                           for i, n in enumerate(ORGANS[:6])}}
        with open(os.path.join(save_dir, f'seg_meta_{fname}'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False)
        append_log(task, f'--> [MOCK] 分割完成: {mask_name}')
        return

    # 真实模式依赖放在 mock 分支之后，保证 Mac 调试环境无需 torch
    import numpy as np
    import torch
    import torch.nn.functional as F
    import SimpleITK as sitk
    from monai.data.utils import dense_patch_slices
    from inference_demo import _get_scan_interval

    append_log(task, f'--> 器官分割开始: {fname}')
    image, geo = _preprocess_for_seg(img_path)   # [1, D, W, H]，与官方推理完全同款 + 几何信息
    image = image[None].cuda()                   # [1, 1, D, W, H]

    roi_size = (96, 256, 384)
    overlap = 0.25
    image_size = list(image.shape[2:])
    num_spatial_dims = len(image.shape) - 2
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)

    # 显存优化：不聚合 37 通道概率体（约3GB 会 OOM），逐窗 argmax 后直接写标签体（约20MB）
    stitched = torch.zeros(tuple(image_size), device='cuda', dtype=torch.uint8)
    model = MODEL['model']
    key = f'{task_id}/{fname}'
    with torch.no_grad():
        for wi, sl in enumerate(slices):
            win_idx = tuple([slice(None), slice(None)] + list(sl))
            patch = image[win_idx]
            seg = model.visual_encoder(patch, None)[0]        # [1, 37, d, h, w] softmax 概率
            seg = F.interpolate(seg, size=patch.shape[2:], mode='trilinear')
            stitched[win_idx[2:]] = seg.argmax(1)[0]
            if (wi + 1) % 5 == 0 or wi + 1 == 1 or wi + 1 == num_win:
                pct = int((wi + 1) / num_win * 100)
                append_log(task, f'Seg: {pct}% ({wi + 1}/{num_win} windows)')
                with SEG_STATE_LOCK:
                    st = SEG_STATE.get(key)
                    # 兼容 'running' 字符串和 ('running', pct) 元组两种形态
                    if st == 'running' or (isinstance(st, tuple) and st and st[0] == 'running'):
                        SEG_STATE[key] = ('running', pct)
    torch.cuda.empty_cache()

    # 保存：seg_ct 与 seg_mask 同 shape/spacing，niivue 中天然对齐。
    # 注意轴序：预处理后数组为 [D, W, H]，D 是 5mm 层厚方向。
    # SimpleITK GetImageFromArray 把数组最后一维作为 x 轴、第一维作为 z 轴，
    # 因此直接写入 [D, W, H] 并设 spacing (1,1,5)：x=H(1mm), y=W(1mm), z=D(5mm)，比例才正确。
    # （此前错误地 transpose 成 [H,W,D] 再设 (1,1,5)，导致 5mm 轴被标 1mm、1mm 轴被标 5mm，画面比例扭曲）
    mask_arr = stitched.cpu().numpy()                                     # [D, W, H]
    ct_arr = image[0, 0].float().cpu().numpy()                            # [D, W, H]
    for arr, name in ((mask_arr, mask_name), (ct_arr, ct_name)):
        im = sitk.GetImageFromArray(arr)
        im.SetSpacing((1.0, 1.0, 5.0))   # 预处理目标 spacing：面内 1mm，层间 5mm
        sitk.WriteImage(im, os.path.join(save_dir, name))

    # 器官质心（供前端点击疾病项时自动定位视图）
    # centroid_seg: 分割坐标系 [d, w, h]；centroid_orig: 原始文件体素坐标 [h, w, d]（逆 resize/crop 映射）
    organs_info = {}
    nz = np.nonzero(mask_arr)
    labs_flat = mask_arr[nz]
    sx, sy, sz = geo['scale']
    for lab in np.unique(labs_flat):
        if lab == 0:
            continue
        sel = labs_flat == lab
        cnt = int(sel.sum())
        if cnt < 10:            # 忽略噪点
            continue
        cd, cw, ch = (float(nz[ax][sel].mean()) for ax in range(3))
        d_r = cd + geo['min_dhw'][0]
        w_r = cw + geo['min_dhw'][1]
        h_r = ch + geo['min_dhw'][2]
        organs_info[ORGANS[int(lab) - 1]] = {
            'label': int(lab),
            'count': cnt,
            'centroid_seg': [round(cd, 1), round(cw, 1), round(ch, 1)],
            'centroid_orig': [round(h_r / sy, 1), round(w_r / sx, 1), round(d_r / sz, 1)],
        }
    with open(os.path.join(save_dir, f'seg_meta_{fname}'), 'w', encoding='utf-8') as f:
        json.dump({'organs': organs_info}, f, ensure_ascii=False)
    append_log(task, f'--> 器官分割完成: {mask_name}（检出 {len(organs_info)} 个器官）')


@app.post('/api/segment/{task_id}/{filename}')
def api_segment_start(task_id: str, filename: str):
    fname = os.path.basename(filename)
    img_path = os.path.join(ROOT, 'radar_web', 'uploads', task_id, 'images', fname)
    if not os.path.isfile(img_path):
        return {'ok': False, 'msg': f'文件不存在: {fname}'}
    save_dir = os.path.join(ROOT, 'radar_web', 'uploads', task_id, 'results')
    os.makedirs(save_dir, exist_ok=True)
    if os.path.isfile(os.path.join(save_dir, f'seg_mask_{fname}')):
        return {'ok': True, 'status': 'done'}          # 已有缓存
    if not MODEL_READY.is_set() and not MOCK_MODE:
        return {'ok': False, 'msg': '模型尚未加载完成，请稍后重试'}

    key = f'{task_id}/{fname}'
    with SEG_STATE_LOCK:
        if SEG_STATE.get(key) == 'running':
            return {'ok': True, 'status': 'running'}
        SEG_STATE[key] = 'running'

    def _run():
        try:
            with GPU_LOCK:                              # 与主推理共用 GPU 串行锁
                _segment_one(task_id, img_path, save_dir)
            with SEG_STATE_LOCK:
                SEG_STATE[key] = 'done'
        except Exception as e:
            import traceback
            traceback.print_exc()
            with SEG_STATE_LOCK:
                SEG_STATE[key] = ('error', str(e))

    threading.Thread(target=_run, daemon=True).start()
    return {'ok': True, 'status': 'running'}


@app.get('/api/segment/{task_id}/{filename}')
def api_segment_status(task_id: str, filename: str, since: int = 0):
    """分割状态轮询。since=已收到的日志条数，返回增量日志供前端显示。"""
    fname = os.path.basename(filename)
    save_dir = os.path.join(ROOT, 'radar_web', 'uploads', task_id, 'results')
    logs = []
    task = TASKS.get(task_id)
    if task is not None:
        with TASKS_LOCK:
            if since < len(task['logs']):
                logs = task['logs'][since:]
    resp = {'ok': True, 'status': 'pending', 'logs': logs}
    if os.path.isfile(os.path.join(save_dir, f'seg_mask_{fname}')):
        resp['status'] = 'done'
        return resp
    key = f'{task_id}/{fname}'
    with SEG_STATE_LOCK:
        st = SEG_STATE.get(key)
    if isinstance(st, tuple) and st and st[0] == 'error':
        return {'ok': False, 'msg': st[1], 'logs': logs}
    if isinstance(st, tuple) and st and st[0] == 'running':
        resp['status'] = 'running'
        resp['progress'] = st[1] if len(st) > 1 else 0
        return resp
    resp['status'] = 'running' if st == 'running' else 'pending'
    return resp


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
