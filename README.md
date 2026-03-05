# 盲人辅助系统 (Blind Assistant System)

基于双目立体相机与 YOLO 模型的智能盲人出行辅助系统

## 项目简介

本系统通过双目立体相机实时感知环境，使用深度学习和计算机视觉技术为视障人士提供全方位出行辅助。系统安装在智能帽上，通过语音播报和震动手环为用户提供实时反馈。

## 核心功能

### 1. YOLO 场景路由（核心架构）
- 每帧运行 YOLO11n 目标检测，根据检测类别决定激活哪些功能模块
- 检测到行人/车辆 → 触发危险预测；检测到信号灯/标志 → 触发 OCR
- 楼梯/盲道/斑马线通过定制 YOLO 模型路由（接口已预留）
- 各模块结果设有 TTL 超时机制，场景消失后自动退出

### 2. 动态危险预测
- YOLO11n 检测行人、自行车、汽车、摩托车、公交、货车
- 3D 多目标跟踪（匈牙利算法 + 卡尔曼滤波）
- 仅对速度 ≥ 0.15 m/s 且朝向摄像头运动的目标预测碰撞时间（TTC）
- 静止物体不触发警报，消除误报

### 3. 楼梯检测
- Canny 边缘检测 + 霍夫变换识别台阶线
- 自动判断上楼/下楼方向，统计台阶数量
- 由水平边缘强度预筛触发，无楼梯时不运行

### 4. AI 辅助感知
- 接入 Claude 视觉模型，每 60 帧对场景进行一次自然语言描述
- 识别障碍物、路况、潜在风险，输出适合视障人士的简洁提示

### 5. 文字识别（OCR）
- PaddleOCR（PP-OCRv4）后台懒加载，不阻塞系统启动
- 由 YOLO 检测到交通标志/信号灯时触发，或每 60 帧定时触发
- 识别门牌号、店铺名称、路牌等文字信息

### 6. 盲道检测
- 接口已实现，颜色预筛已禁用（室内误报率高）
- 需要训练定制 YOLO 模型（`models/scene_detector.pt`）后启用

### 7. 斑马线穿越辅助
- 检测斑马线、信号灯状态、车辆动态
- 评估通行安全性，提供方向引导

### 8. 衣物检索
- 点云处理 + YOLO 目标检测
- 颜色识别和语义地图构建，引导用户找到指定衣物

## 技术栈

- **目标检测**: YOLO11n (Ultralytics)
- **深度学习框架**: PyTorch
- **文字识别**: PaddleOCR (PP-OCRv4)
- **AI 视觉感知**: Claude claude-sonnet-4-6（视觉多模态）
- **计算机视觉**: OpenCV
- **目标跟踪**: 匈牙利算法 + 卡尔曼滤波（3D 空间跟踪）
- **双目深度估计**: SGBM 立体匹配
- **开发语言**: Python 3.10

## 项目结构

```
system/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python 依赖包
├── live_demo.py                 # 实时演示主入口（双目摄像头）
├── focus_adjust.py              # 双目焦距调节辅助工具
├── config/
│   └── config.yaml             # 配置文件
├── models/                      # 模型文件目录
│   └── scene_detector.pt       # 定制场景检测模型（待训练）
├── src/
│   ├── main.py                 # 主程序入口
│   ├── core/                   # 核心算法模块
│   │   ├── danger_detection.py     # 动态危险预测（TTC计算）
│   │   ├── object_tracker.py       # 3D 多目标跟踪
│   │   └── kalman_filter.py        # 卡尔曼滤波器
│   ├── modules/                # 场景检测模块
│   │   ├── ai_assistant.py         # Claude AI 视觉感知
│   │   ├── stair_detection.py      # 楼梯检测
│   │   ├── blind_path_detection.py # 盲道检测
│   │   ├── crosswalk_detection.py  # 斑马线检测
│   │   ├── ocr_recognition.py      # 文字识别
│   │   ├── clothing_search.py      # 衣物检索
│   │   └── navigation.py           # 定位导航
│   ├── feedback/               # 反馈模块
│   │   ├── voice_output.py         # 语音播报
│   │   └── vibration_control.py    # 震动控制
│   └── utils/                  # 工具模块
│       ├── camera.py               # 双目相机接口（SGBM深度估计）
│       └── visualization.py        # 可视化工具
└── tests/                       # 测试代码
```

## 安装说明

### 环境要求

- Python 3.10
- Anaconda 或 Miniconda

### 1. 克隆项目

```bash
git clone <repository-url>
cd system
```

### 2. 创建 Conda 虚拟环境

```bash
conda create -n dc python=3.10
conda activate dc
```

### 3. 安装依赖

```bash
pip install -r requirements.txt

# PaddleOCR（CPU版）
pip install paddlepaddle paddlex
```

### 4. 配置 AI 感知模块（可选）

在 `config/config.yaml` 中填入 Anthropic API Key：
```yaml
ai_assistant:
  api_key: "your-anthropic-api-key"
```

## 使用方法

### 实时演示（主要入口）

```bash
conda activate dc
cd system
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True /path/to/envs/dc/bin/python3.10 -u live_demo.py
```

演示界面按键：
- `q` — 退出
- `s` — 截图保存
- `d` — 切换深度图显示

### 焦距调节

```bash
python focus_adjust.py
```

同时显示左右摄像头画面，用于手动调节镜头焦距。

### 配置说明

[config/config.yaml](config/config.yaml) 主要参数：
- `camera.device_index` — 摄像头设备号
- `yolo.model_path` — YOLO 模型路径
- `tracker.distance_threshold` — 3D 跟踪距离阈值（米）
- `danger_detection.safe_distance` — 危险预警距离（米）

## 硬件

- **相机**: 双目立体摄像头（3840×1080，每眼 1920×1080）
- **震动手环**: 蓝牙/串口连接（可选）
- **骨传导耳机**: 用于语音播报
- **计算设备**: MacBook / 配备 NPU 的嵌入式设备

## 开发路线

- [x] 项目框架搭建
- [x] 双目相机接入与 SGBM 深度估计
- [x] YOLO11n 目标检测
- [x] 3D 多目标跟踪（卡尔曼滤波 + 匈牙利算法）
- [x] 动态危险预测（TTC，仅追踪目标触发）
- [x] 楼梯检测（边缘预筛 + 霍夫变换）
- [x] 文字识别（PaddleOCR 后台懒加载）
- [x] AI 辅助感知（Claude 视觉模型）
- [x] YOLO 场景路由 + 模块 TTL 超时机制
- [x] 实时演示集成（live_demo.py）
- [ ] 盲道检测（需训练定制 YOLO 模型）
- [ ] 斑马线检测完善
- [ ] 衣物检索完善
- [ ] 定位导航实现
- [ ] 语音/震动反馈接入
- [ ] 嵌入式设备部署

