"""
盲人辅助系统 - 主程序
整合所有模块，实现完整的辅助功能
"""

import cv2
import yaml
import time
import argparse
from pathlib import Path

# 导入核心模块
from core.danger_detection import DangerDetector
from core.object_tracker import ObjectTracker

# 导入场景检测模块
from modules.stair_detection import StairDetector
from modules.crosswalk_detection import CrosswalkDetector
from modules.ocr_recognition import OCRRecognizer
from modules.clothing_search import ClothingSearcher
from modules.blind_path_detection import BlindPathDetector
from modules.navigation import Navigator
from modules.ai_assistant import AIAssistant

# 导入反馈模块
from feedback.voice_output import VoiceOutput, DangerVoiceGenerator
from feedback.vibration_control import VibrationController

# 导入工具模块
from utils.camera import create_camera
from utils.visualization import Visualizer


class BlindAssistantSystem:
    """盲人辅助系统主类"""

    def __init__(self, config_path: str):
        """
        初始化系统

        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化相机
        self.camera = create_camera(self.config['camera'])

        # 加载YOLO模型
        self.yolo_model = None
        self.yolo_config = self.config.get('yolo', {})
        self._load_yolo()

        # 初始化核心模块
        self.danger_detector = DangerDetector(self.config['danger_detection'])
        self.tracker = ObjectTracker(self.config['tracker'])

        # 初始化场景检测模块
        self.stair_detector = StairDetector(self.config['stair_detection'])
        self.crosswalk_detector = CrosswalkDetector(self.config['crosswalk_detection'])
        self.ocr_recognizer = OCRRecognizer(self.config['ocr'])
        self.clothing_searcher = ClothingSearcher(self.config['clothing_search'])
        self.blind_path_detector = BlindPathDetector(self.config['blind_path_detection'])
        self.navigator = Navigator(self.config['navigation'])
        self.ai_assistant = AIAssistant(self.config.get('ai_assistant', {}))

        # 初始化反馈模块
        self.voice_output = VoiceOutput(self.config['voice'])
        self.vibration_controller = VibrationController(self.config['vibration'])

        # 初始化可视化工具（用于调试）
        self.visualizer = Visualizer()

        # 系统状态
        self.running = False
        self.test_mode = self.config.get('test_mode', {}).get('enabled', False)

        print("盲人辅助系统初始化完成")

    def _load_yolo(self):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            model_path = self.yolo_config.get('model_path', 'yolov8n.pt')
            self.yolo_model = YOLO(model_path)
            print(f"YOLO模型加载成功: {model_path}")
        except ImportError:
            print("警告: ultralytics未安装，YOLO检测不可用 (pip install ultralytics)")
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")

    def start(self):
        """启动系统"""
        print("启动盲人辅助系统...")

        # 打开相机
        if not self.camera.open():
            print("相机打开失败,系统退出")
            return

        self.running = True
        self.voice_output.speak("系统已启动")

        try:
            self.main_loop()
        except KeyboardInterrupt:
            print("\n系统被用户中断")
        finally:
            self.shutdown()

    def main_loop(self):
        """主循环"""
        frame_count = 0
        fps_time = time.time()

        while self.running:
            # 读取图像 (color: BGR, depth: float32 单位米)
            ret, color_image, depth_map = self.camera.read()

            if not ret:
                print("读取图像失败")
                break

            frame_count += 1

            # 步骤1: 目标检测 (YOLO)
            detections = self._yolo_detect(color_image)

            # 步骤2: 多目标跟踪（3D空间跟踪，使用卡尔曼滤波）
            tracked_detections = self.tracker.update(detections, depth_map)

            # 步骤3: 动态危险预测 (最高优先级，每帧运行)
            dangers = []
            if self.config['danger_detection']['enabled']:
                dangers = self.danger_detector.update(tracked_detections)
                self._handle_dangers(dangers)

            # 步骤4: 场景检测 (按优先级调度)
            if frame_count % 5 == 0:
                if self.config['stair_detection']['enabled']:
                    stair_result = self.stair_detector.detect(color_image)
                    self._handle_stair(stair_result)

            if frame_count % 5 == 0:
                if self.config['crosswalk_detection']['enabled']:
                    crosswalk_result = self.crosswalk_detector.detect(color_image, depth_map)
                    self._handle_crosswalk(crosswalk_result)

            if frame_count % 30 == 0:
                if self.config['ocr']['enabled']:
                    ocr_result = self.ocr_recognizer.recognize(color_image, depth_map)
                    self._handle_ocr(ocr_result)

            if frame_count % 10 == 0:
                if self.config.get('blind_path_detection', {}).get('enabled', False):
                    blind_path_result = self.blind_path_detector.detect(color_image)
                    self._handle_blind_path(blind_path_result)

            # 步骤5: AI辅助感知 (非紧急复杂场景，每60帧触发一次异步调用)
            if frame_count % 60 == 0 and len(dangers) == 0:
                self.ai_assistant.analyze_async(color_image)

            # 播报AI分析结果
            ai_result = self.ai_assistant.get_result()
            if ai_result:
                self.voice_output.speak(ai_result)

            # 可视化 (仅测试模式)
            if self.test_mode:
                vis_image = self._visualize(color_image, depth_map,
                                            tracked_detections, dangers)
                cv2.imshow('Blind Assistant System', vis_image)

                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 计算FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()
                print(f"FPS: {fps:.2f}")

    def _yolo_detect(self, image):
        """
        YOLO目标检测

        Args:
            image: BGR彩色图像

        Returns:
            检测结果列表，每项包含 bbox/class_id/class_name/confidence
        """
        if self.yolo_model is None:
            return []

        conf = self.yolo_config.get('confidence_threshold', 0.5)
        device = self.yolo_config.get('device', 'cpu')

        results = self.yolo_model(image, conf=conf, device=device, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.yolo_model.names[class_id]

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                })

        return detections

    def _handle_dangers(self, dangers: list):
        """
        处理危险预警

        Args:
            dangers: 危险列表
        """
        for danger in dangers:
            # 语音播报
            warning_text = DangerVoiceGenerator.generate_danger_warning(danger)
            self.voice_output.speak_immediately(warning_text)

            # 震动反馈
            self.vibration_controller.vibrate_danger(danger)

    def _handle_stair(self, result: dict):
        """
        处理楼梯检测结果

        Args:
            result: 检测结果
        """
        if result is None or not result['detected']:
            return

        # 语音播报
        guidance_text = DangerVoiceGenerator.generate_stair_guidance(result)
        self.voice_output.speak(guidance_text)

        # 震动引导：上楼左手环，下楼右手环
        direction = 'left' if result['direction'] == 'up' else 'right'
        self.vibration_controller.vibrate_direction(direction)

    def _handle_crosswalk(self, result: dict):
        """
        处理斑马线检测结果

        Args:
            result: 检测结果
        """
        if result is None or not result['detected']:
            return

        # 语音播报
        guidance_text = DangerVoiceGenerator.generate_crosswalk_guidance(result)
        self.voice_output.speak(guidance_text)

        # 绿灯可以通行时双手环震动提示
        if result.get('is_safe'):
            self.vibration_controller.vibrate_direction('straight')

    def _handle_ocr(self, result: dict):
        """
        处理文字识别结果

        Args:
            result: 识别结果
        """
        if result is None or not result['detected']:
            return

        # 如果识别到文字，播报
        if result['num_texts'] > 0:
            text_to_speak = f"识别到文字: {result['full_text']}"
            self.voice_output.speak(text_to_speak)

    def _handle_blind_path(self, result: dict):
        """
        处理盲道检测结果

        Args:
            result: 检测结果
        """
        if result is None or not result['detected']:
            return

        guidance = result.get('guidance', 'straight')
        if guidance != 'straight':
            self.vibration_controller.vibrate_direction(guidance)

    def _visualize(self, color_image, depth_map, detections, dangers):
        """
        可视化检测结果

        Args:
            color_image: 彩色图像
            depth_map: 深度图
            detections: 检测结果
            dangers: 危险列表

        Returns:
            可视化图像
        """
        vis_image = color_image.copy()

        # 绘制检测框
        if len(detections) > 0:
            vis_image = self.visualizer.draw_detections(vis_image, detections)

        # 绘制危险警告
        if len(dangers) > 0:
            vis_image = self.visualizer.draw_dangers(vis_image, dangers)

        # 拼接深度图
        if depth_map is not None:
            depth_colormap = self.visualizer.draw_depth_map(depth_map)
            # 确保高度一致
            if depth_colormap.shape[0] == vis_image.shape[0]:
                vis_image = cv2.hconcat([vis_image, depth_colormap])

        return vis_image

    def shutdown(self):
        """关闭系统"""
        print("关闭系统...")

        self.running = False

        # 释放相机
        self.camera.release()

        # 关闭反馈模块
        self.voice_output.shutdown()
        self.vibration_controller.shutdown()

        # 关闭窗口
        cv2.destroyAllWindows()

        print("系统已关闭")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='盲人辅助系统')
    parser.add_argument('--config', type=str,
                       default='config/config.yaml',
                       help='配置文件路径')
    args = parser.parse_args()

    # 创建系统实例
    system = BlindAssistantSystem(args.config)

    # 启动系统
    system.start()


if __name__ == "__main__":
    main()
