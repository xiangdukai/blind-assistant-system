"""
盲人辅助系统实时演示
使用双目摄像头运行完整感知流程，可视化所有检测结果
操作：
  q   退出
  s   截图保存
  d   切换深度图显示
"""

import sys
import os
import time
import threading
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 加载系统中文字体（macOS）
_CN_FONT_PATHS = [
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/Library/Fonts/Arial Unicode.ttf',
]
_cn_font_cache = {}

def _get_font(size=18):
    if size in _cn_font_cache:
        return _cn_font_cache[size]
    for fp in _CN_FONT_PATHS:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size)
                _cn_font_cache[size] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()
    _cn_font_cache[size] = font
    return font

def put_cn_text(img, text, pos, color=(255, 255, 255), font_size=18, bg=None):
    """在 OpenCV 图像上渲染中文文字（通过 PIL）"""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _get_font(font_size)
    if bg is not None:
        bbox = draw.textbbox(pos, text, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=bg[::-1])
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.camera import StereoBinocularCamera
from core.object_tracker import ObjectTracker
from core.danger_detection import DangerDetector
from modules.stair_detection import StairDetector
from modules.blind_path_detection import BlindPathDetector
from modules.ocr_recognition import OCRRecognizer
from modules.ai_assistant import AIAssistant
from utils.visualization import Visualizer

# ─── 配置 ────────────────────────────────────────────
CAMERA_INDEX  = 0
FPS           = 30
YOLO_MODEL    = 'yolo11n.pt'   # YOLO11 nano：最轻量级
YOLO_CONF     = 0.45
# 标定参数（640×480），实际运行时按真实分辨率缩放
_CAL_W, _CAL_H = 640, 480
_CAL_MATRIX    = {'fx': 416.84, 'fy': 416.47, 'cx': 338.49, 'cy': 230.42}

# ─── 初始化各模块 ─────────────────────────────────────
print("=" * 50)
print("  盲人辅助系统  实时演示")
print("=" * 50)

# 相机（尝试 1080P，相机不支持时自动降级）
cam_cfg = {
    'type': 'stereo', 'camera_index': CAMERA_INDEX,
    'resolution': {'width': 1920, 'height': 1080}, 'fps': FPS,
    'depth_enabled': True
}
camera = StereoBinocularCamera(cam_cfg)
if not camera.open():
    print("摄像头打开失败，退出")
    sys.exit(1)

# 根据摄像头实际分辨率动态确定 IMG_W/H 及相机矩阵
IMG_W = camera._eye_w
IMG_H = camera._eye_h
sx = IMG_W / _CAL_W  # 水平缩放比
sy = IMG_H / _CAL_H  # 垂直缩放比
CAMERA_MATRIX = {
    'fx': _CAL_MATRIX['fx'] * sx,
    'fy': _CAL_MATRIX['fy'] * sy,
    'cx': _CAL_MATRIX['cx'] * sx,
    'cy': _CAL_MATRIX['cy'] * sy,
}
print(f"显示分辨率: {IMG_W}×{IMG_H}，相机矩阵已缩放 ({sx:.2f}x, {sy:.2f}y)")

# YOLO
yolo = None
try:
    from ultralytics import YOLO
    yolo = YOLO(YOLO_MODEL)
    print(f"YOLO 加载成功: {YOLO_MODEL}")
except Exception as e:
    print(f"YOLO 加载失败: {e}，尝试 yolov8n.pt")
    try:
        yolo = YOLO('yolov8n.pt')
        print("YOLO 加载成功: yolov8n.pt（备用）")
    except Exception as e2:
        print(f"YOLO 加载失败: {e2}")

# 跟踪 + 危险检测
tracker = ObjectTracker({
    'max_age': 30, 'min_hits': 2,
    'distance_threshold': 1.5,
    'camera_matrix': CAMERA_MATRIX
})
danger_detector = DangerDetector({
    'safe_distance': 2.0, 'prediction_time': 3.0,
    'danger_levels': {'high': 1.5, 'medium': 3.0},
    'camera_matrix': CAMERA_MATRIX
})

# 楼梯检测
stair_detector = StairDetector({
    'canny_threshold1': 50, 'canny_threshold2': 150,
    'hough_threshold': 60, 'min_line_length': 40, 'max_line_gap': 15
})

# 盲道检测（颜色兜底）
blind_detector = BlindPathDetector({
    'model_path': 'models/pidnet_s_cityscapes.pth', 'model_type': 'pidnet'
})

# OCR（后台线程懒加载，避免阻塞启动）
ocr = None
_ocr_ready = threading.Event()

def _init_ocr():
    global ocr
    print("OCR 初始化中（后台）...")
    try:
        _ocr = OCRRecognizer({'language': 'ch', 'use_angle_cls': True})
        ocr = _ocr
        print("OCR 初始化完成")
    except Exception as e:
        print(f"OCR 初始化失败: {e}")
    finally:
        _ocr_ready.set()

threading.Thread(target=_init_ocr, daemon=True).start()

# AI辅助感知
ai_assistant = AIAssistant({
    'enabled': True,
    'api_url': 'https://api.wkcloud.com.cn/v1/chat/completions',
    'api_key': 'sk-zokjJ30OdSyjdPaSW524a9MKkaHPBiOG2GKbIrBle8lxlb4I',
    'model': 'gpt-4.1-mini',
    'max_tokens': 80,
    'timeout': 15,
    'jpeg_quality': 60,
    'prompt': '你是盲人出行辅助系统的视觉感知模块。用不超过25个汉字的一句话描述图中对盲人行走最关键的信息。优先关注：障碍物（位置和距离）、台阶坡道、人群车辆、地面状况、路口标志。只输出描述，不加任何前缀。'
})

visualizer = Visualizer()

print("\n按 'q' 退出，'s' 截图，'d' 切换深度图\n")

# ─── 自动路由：基于 YOLO 检测结果路由到各功能模块 ──────
#
# 路由逻辑：
#   危险/跟踪  ← YOLO 检测到人/车等危险类别（COCO 80类已覆盖）
#   楼梯检测  ← YOLO 定制模型检测到 "staircase" 类别  ← 当前用预览分类代替
#   盲道检测  ← YOLO 定制模型检测到 "blind_path" 类别  ← 当前用颜色预筛代替
#   斑马线    ← YOLO 定制模型检测到 "crosswalk" 类别   ← 预留接口
#   OCR       ← YOLO 检测到 "stop_sign"/"traffic_light" 或定时
#   AI感知    ← 无直接危险时触发
#
# 注：楼梯/盲道/斑马线不在 COCO 80 类中，需要训练定制 YOLO 模型后替换
#     _yolo_scene 接口为定制模型预留，当前用轻量图像预筛兜底

# ── COCO 类别路由表 ──
_DANGER_CLASSES    = {0, 1, 2, 3, 5, 7}    # person, bicycle, car, motorcycle, bus, truck
_SIGN_CLASSES      = {9, 11}               # traffic_light, stop_sign → 触发 OCR

# ── 各模块最小触发间隔（帧数） ──
_ROUTE_INTERVAL = {'stair': 10, 'blind': 10, 'ocr': 60, 'ai': 60}
_last_run       = {'stair': -999, 'blind': -999, 'ocr': -999, 'ai': -999}

# ── 结果有效期（帧数）：超过此帧数未检测到则清除缓存结果 ──
_RESULT_TTL     = {'stair': 90, 'blind': 60, 'ocr': 150, 'ai': 90}
_last_detected  = {'stair': -999, 'blind': -999, 'ocr': -999, 'ai': -999}

# ── 定制 YOLO 场景检测器（预留接口，训练后在此处加载）──
# 用法：将 yolo_scene 替换为训练好的定制模型，包含类别:
#   0=staircase, 1=blind_path, 2=crosswalk 等
# yolo_scene = YOLO('models/scene_detector.pt')
yolo_scene = None   # 暂未训练，下方用预筛代替

def _run_scene_yolo(color_image):
    """
    用定制 YOLO 场景检测器获取场景类别集合。
    未来训练好模型后只需替换 yolo_scene，此函数不变。
    当前：yolo_scene=None 时返回空集，由预筛逻辑兜底。
    """
    if yolo_scene is None:
        return set()
    results = yolo_scene(color_image, conf=0.4, verbose=False)
    scene_classes = set()
    for r in results:
        for box in r.boxes:
            scene_classes.add(r.names[int(box.cls[0])])
    return scene_classes

def _preflight_scene(color_image):
    """
    定制 YOLO 未就绪时的轻量图像预筛，用于判断是否值得调用专项算法。
    这是临时兜底方案，不替代真正的 YOLO 场景检测。
    """
    h, w = color_image.shape[:2]
    triggers = set()

    # 楼梯预筛：下半区水平边缘明显强于垂直边缘
    gray = cv2.cvtColor(color_image[h // 3: h * 2 // 3, :], cv2.COLOR_BGR2GRAY)
    hy = float(np.mean(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3).__abs__() > 30))
    hx = float(np.mean(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3).__abs__() > 30))
    if hy > 0.07 and hy > hx * 1.6:
        triggers.add('staircase')

    # 盲道预筛已禁用：颜色方法室内误报率极高，需要定制 YOLO 模型才能可靠检测
    # 待训练 models/scene_detector.pt 后可在 _run_scene_yolo 中启用

    return triggers

def _scene_route(yolo_dets, color_image, frame_count):
    """
    场景路由：根据 YOLO 检测结果决定本帧激活哪些功能模块。
    """
    active = set()
    det_classes = {d['class_id'] for d in yolo_dets}

    # ① 危险预测：YOLO 检测到行人/车辆类别
    if det_classes & _DANGER_CLASSES:
        active.add('danger')

    # ② OCR：YOLO 检测到交通标志/信号灯，或定时触发
    if (det_classes & _SIGN_CLASSES) or (frame_count - _last_run['ocr'] >= _ROUTE_INTERVAL['ocr']):
        active.add('ocr')

    # ③ 楼梯/盲道：优先用定制 YOLO 场景检测器，无模型时用图像预筛兜底
    if frame_count - _last_run['stair'] >= _ROUTE_INTERVAL['stair']:
        _last_run['stair'] = frame_count
        scene_classes = _run_scene_yolo(color_image)
        if not scene_classes:                    # 定制模型未就绪，用预筛
            scene_classes = _preflight_scene(color_image)
        if 'staircase' in scene_classes:
            active.add('stair')
        if 'blind_path' in scene_classes:
            active.add('blind')
        if 'crosswalk' in scene_classes:
            active.add('ocr')                    # 斑马线 → 触发 OCR 识别路口信息

    # ④ 盲道独立间隔：仅当定制 YOLO 模型就绪时才触发
    if 'blind' not in active and yolo_scene is not None and frame_count - _last_run['blind'] >= _ROUTE_INTERVAL['blind']:
        _last_run['blind'] = frame_count
        scene_classes = _run_scene_yolo(color_image)
        if 'blind_path' in scene_classes:
            active.add('blind')

    # ⑤ AI感知：无直接危险且间隔足够
    if 'danger' not in active and frame_count - _last_run['ai'] >= _ROUTE_INTERVAL['ai']:
        active.add('ai')

    return active

# ─── 主循环 ───────────────────────────────────────────
frame_count   = 0
show_depth    = True
fps_timer     = time.time()

# 缓存慢速检测结果（避免每帧重算）
last_stair    = None
last_blind    = None
last_ocr      = None
last_dangers  = []
last_tracked  = []
last_ai_text  = None

screenshot_dir = os.path.join(os.path.dirname(__file__), 'screenshots')

while True:
    ret, color_image, depth_map = camera.read()
    if not ret or color_image is None:
        print("读帧失败，重试...")
        continue

    frame_count += 1

    # ── 1. YOLO 检测（每帧）──────────────────────────
    raw_detections = []
    if yolo is not None:
        results = yolo(color_image, conf=YOLO_CONF, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': int(box.cls[0]),
                    'class_name': r.names[int(box.cls[0])],
                    'confidence': float(box.conf[0])
                })

    # ── 2. 场景路由：决定本帧激活哪些模块 ────────────
    active_modules = _scene_route(raw_detections, color_image, frame_count)

    # ── 3. 3D 跟踪 + 危险预测（仅当危险类别被检测到）──
    if 'danger' in active_modules and depth_map is not None:
        last_tracked = tracker.update(raw_detections, depth_map)
        last_dangers = danger_detector.update(last_tracked)
    elif depth_map is not None:
        # 无危险类别时也更新跟踪（用于显示），但不产生警报
        last_tracked = tracker.update(raw_detections, depth_map)
        last_dangers = []
    else:
        last_tracked = []
        last_dangers = []

    # ── 4. 楼梯检测（场景路由触发）─────────────────
    if 'stair' in active_modules:
        result = stair_detector.detect(color_image)
        if result:
            last_stair = result
            _last_detected['stair'] = frame_count

    # ── 5. 盲道检测（场景路由触发）─────────────────
    if 'blind' in active_modules:
        _last_run['blind'] = frame_count
        result = blind_detector.detect(color_image)
        if result:
            last_blind = result
            _last_detected['blind'] = frame_count

    # ── 6. OCR（场景路由触发：交通标志/信号灯 或 定时）──
    if 'ocr' in active_modules and ocr is not None:
        _last_run['ocr'] = frame_count
        result = ocr.recognize(color_image)
        if result:
            last_ocr = result
            _last_detected['ocr'] = frame_count
            print(f"[OCR] {last_ocr['full_text']}")

    # ── 7. AI辅助感知（场景路由触发）───────────────
    if 'ai' in active_modules:
        _last_run['ai'] = frame_count
        ai_assistant.analyze_async(color_image)

    # 获取AI分析结果（非阻塞）
    ai_result = ai_assistant.get_result()
    if ai_result:
        last_ai_text = ai_result
        _last_detected['ai'] = frame_count
        print(f"[AI] {ai_result}")

    # ── 结果过期清除：超过 TTL 帧数未检测到则清空缓存 ──
    if frame_count - _last_detected['stair'] > _RESULT_TTL['stair']:
        last_stair = None
    if frame_count - _last_detected['blind'] > _RESULT_TTL['blind']:
        last_blind = None
    if frame_count - _last_detected['ocr'] > _RESULT_TTL['ocr']:
        last_ocr = None
    if frame_count - _last_detected['ai'] > _RESULT_TTL['ai']:
        last_ai_text = None

    # ── 可视化 ────────────────────────────────────────
    vis = color_image.copy()

    # 绘制 YOLO 检测框（绿色细框，所有原始检测）
    for det in raw_detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 1)
        cv2.putText(vis, f"{det['class_name']} {det['confidence']:.2f}",
                    (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

    # 绘制跟踪 ID + 3D 距离（蓝色）
    for det in last_tracked:
        x1, y1, x2, y2 = det['bbox']
        tid = det.get('track_id', '?')
        pos = det.get('position_3d')
        dist_str = f"{pos[2]:.1f}m" if pos is not None else ""
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(vis, f"T{tid} {dist_str}",
                    (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)

    # 绘制危险警告（红色横幅）
    for i, d in enumerate(last_dangers):
        lvl   = d['danger_level']
        ttc   = d['time_to_collision']
        direc = d['direction']
        color = (0, 0, 255) if lvl == 'high' else (0, 100, 255) if lvl == 'medium' else (0, 200, 255)
        lvl_cn = {'high': '高危', 'medium': '中危', 'low': '低危'}.get(lvl, lvl)
        dir_cn = {'left': '左方', 'right': '右方', 'front': '正前方',
                  'front_left': '左前方', 'front_right': '右前方'}.get(direc, direc)
        dist   = d.get('distance', 0)
        ttc_str = f"{dist:.1f}m" if ttc < 0.15 else f"{ttc:.1f}s"
        msg   = f"[{lvl_cn}] {dir_cn} {ttc_str}"
        vis = put_cn_text(vis, msg, (5, 5 + i*28), color=color, font_size=20, bg=(0, 0, 0))

    # 绘制楼梯检测结果（绿色线 + 中文文字）
    if last_stair:
        vis = stair_detector.visualize(vis, last_stair)
        dir_cn = '上楼' if last_stair['direction'] == 'up' else '下楼'
        stair_msg = f"楼梯: {dir_cn}，{last_stair['num_steps']} 步"
        vis = put_cn_text(vis, stair_msg, (10, 35), color=(0, 255, 0), font_size=20, bg=(0, 0, 0))

    # 绘制盲道 mask（半透明绿色覆盖）
    if last_blind and last_blind.get('mask') is not None:
        mask = last_blind['mask']
        overlay = vis.copy()
        overlay[mask > 0] = (0, 255, 128)
        vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)
        guidance = last_blind.get('guidance', '')
        vis = put_cn_text(vis, f"盲道: {guidance}", (10, IMG_H - 30),
                          color=(0, 255, 128), font_size=20, bg=(0, 0, 0))

    # 绘制 OCR 结果
    if last_ocr:
        text = last_ocr['full_text'][:40]
        vis = put_cn_text(vis, f"OCR: {text}", (5, IMG_H - 52),
                          color=(255, 255, 0), font_size=18, bg=(0, 0, 0))

    # 绘制 AI 辅助感知结果（顶部，青色）
    if last_ai_text:
        banner_y = 5 + len(last_dangers) * 28
        vis = put_cn_text(vis, f"AI: {last_ai_text}", (5, banner_y),
                          color=(0, 255, 255), font_size=18, bg=(20, 20, 20))

    # FPS
    if frame_count % 30 == 0:
        fps_val = 30.0 / max(time.time() - fps_timer, 1e-6)
        fps_timer = time.time()
        fps_display = fps_val
    else:
        fps_display = getattr(fps_display if 'fps_display' in dir() else None, '__float__', lambda: 0.0)()
    try:
        cv2.putText(vis, f"FPS:{fps_display:.1f}", (IMG_W - 90, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    except Exception:
        pass

    # 深度图（右侧拼接）
    if show_depth and depth_map is not None:
        depth_vis = visualizer.draw_depth_map(depth_map)
        if depth_vis.shape[0] == vis.shape[0]:
            display = np.hstack([vis, depth_vis])
        else:
            display = vis
    else:
        display = vis

    cv2.imshow('Blind Assistant - Live Demo', display)

    # ── 终端实时信息 ──────────────────────────────────
    if frame_count % 30 == 0:
        n_obj = len(raw_detections)
        n_trk = len(last_tracked)
        n_dng = len(last_dangers)
        stair_str = f"{last_stair['direction']} {last_stair['num_steps']}步" if last_stair else "无"
        blind_str = last_blind.get('guidance', '无') if last_blind else "无"
        print(f"帧{frame_count:5d} | YOLO:{n_obj:2d}目标 | 跟踪:{n_trk:2d} | 危险:{n_dng} | "
              f"楼梯:{stair_str} | 盲道:{blind_str}")

    # ── 按键 ──────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        show_depth = not show_depth
        print(f"深度图显示: {'开' if show_depth else '关'}")
    elif key == ord('s'):
        os.makedirs(screenshot_dir, exist_ok=True)
        fname = os.path.join(screenshot_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(fname, display)
        print(f"截图已保存: {fname}")

camera.release()
cv2.destroyAllWindows()
print("演示结束")
