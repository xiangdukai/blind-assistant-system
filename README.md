# 盲人辅助系统 (Blind Assistant System)

基于深度相机与YOLO模型的智能盲人出行辅助系统

## 项目简介

本系统通过深度相机实时感知环境，使用深度学习和计算机视觉技术为视障人士提供全方位出行辅助。系统安装在智能帽上，通过语音播报和震动手环为用户提供实时反馈。

## 核心功能

### 1. 动态危险预测 (核心创新)
- 使用YOLOv8进行目标检测
- DeepSORT多目标跟踪
- 卡尔曼滤波预测运动轨迹
- 提前预警潜在碰撞风险

### 2. 楼梯检测
- Canny边缘检测 + 霍夫变换
- 自动识别上楼/下楼方向
- 统计台阶数量

### 3. 斑马线穿越辅助
- 检测斑马线、信号灯、车辆
- 评估通行安全性
- 方向引导

### 4. 文字识别
- PaddleOCR文字检测与识别
- 帮助识别门牌号、店铺名称等
- 语音播报识别内容

### 5. 衣物检索
- 点云处理 + YOLO目标检测
- 颜色识别和语义地图构建
- 引导用户找到指定衣物

### 6. 盲道检测
- 语义分割识别盲道
- 实时引导保持在盲道上

### 7. 北斗定位导航
- GPS定位
- 调用地图API进行导航

## 技术栈

- **深度学习框架**: PyTorch
- **目标检测**: YOLOv8 (Ultralytics)
- **语义分割**: PIDNet (盲道检测)
- **文字识别**: PaddleOCR
- **计算机视觉**: OpenCV, Open3D
- **目标跟踪**: DeepSORT, 卡尔曼滤波
- **语音合成**: pyttsx3 / OpenAI TTS
- **开发语言**: Python 3.10+

## 项目结构

```
system/
├── README.md                   # 项目说明文档
├── requirements.txt            # Python依赖包
├── config/
│   └── config.yaml            # 配置文件
├── src/
│   ├── main.py                # 主程序入口
│   ├── core/                  # 核心算法模块
│   │   ├── danger_detection.py    # 动态危险预测
│   │   ├── object_tracker.py      # 多目标跟踪
│   │   └── kalman_filter.py       # 卡尔曼滤波器
│   ├── modules/               # 场景检测模块
│   │   ├── stair_detection.py     # 楼梯检测
│   │   ├── crosswalk_detection.py # 斑马线检测
│   │   ├── ocr_recognition.py     # 文字识别
│   │   ├── clothing_search.py     # 衣物检索
│   │   ├── blind_path_detection.py # 盲道检测
│   │   └── navigation.py          # 定位导航
│   ├── feedback/              # 反馈模块
│   │   ├── voice_output.py        # 语音播报
│   │   └── vibration_control.py   # 震动控制
│   └── utils/                 # 工具模块
│       ├── camera.py              # 相机接口
│       └── visualization.py       # 可视化工具
├── models/                    # 预训练模型存放
└── tests/                     # 测试代码
    └── test_images/           # 测试图片
```

## 安装说明

### 环境要求

- Python 3.10 或更高版本
- Anaconda 或 Miniconda
- CUDA 11.8+ (可选，用于GPU加速)

### 1. 克隆项目

```bash
git clone <repository-url>
cd system
```

### 2. 创建Conda虚拟环境

```bash
# 创建名为 BAS 的虚拟环境，Python 3.10
conda create -n BAS python=3.10

# 激活虚拟环境
conda activate BAS
```

### 3. 安装PyTorch

```bash
# CPU版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# GPU版本 (推荐，需要NVIDIA显卡)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 4. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 5. 下载模型

首次运行时，YOLOv8和PaddleOCR模型会自动下载到 `models/` 目录

PIDNet模型需要手动下载:
```bash
# 下载PIDNet预训练模型
mkdir -p models
# 从GitHub下载: https://github.com/XuJiacong/PIDNet
```

## 使用方法

### 基本运行

```bash
# 激活虚拟环境
conda activate BAS

# 运行系统
python src/main.py
```

### 测试单个模块

```bash
# 激活虚拟环境
conda activate BAS

# 测试楼梯检测
python -m pytest tests/test_stair.py

# 测试斑马线检测
python -m pytest tests/test_crosswalk.py
```

### 配置说明

系统配置文件位于 [config/config.yaml](config/config.yaml)，可以修改:
- 相机类型和分辨率
- YOLO模型路径
- 各功能模块的启用/禁用
- 语音和震动反馈参数

## 硬件要求

- **深度相机**: Intel RealSense D435/D455 (推荐) 或普通摄像头
- **震动手环**: 蓝牙/串口连接 (可选)
- **骨传导耳机**: 用于语音播报
- **计算设备**: 树莓派4B或更高配置

## 开发路线

- [x] 项目框架搭建
- [ ] 楼梯检测实现
- [ ] 文字识别实现
- [ ] 斑马线检测实现
- [ ] 动态危险预测实现
- [ ] 衣物检索实现
- [ ] 盲道检测实现
- [ ] 定位导航实现
- [ ] 反馈模块实现
- [ ] 系统集成测试

