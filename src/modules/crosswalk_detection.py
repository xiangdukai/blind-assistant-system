"""
斑马线检测模块
使用YOLO检测斑马线、信号灯、车辆，并进行安全评估
"""

import cv2
import numpy as np
from typing import Optional, Dict, List
from sklearn.cluster import DBSCAN


class CrosswalkDetector:
    """斑马线检测器"""

    def __init__(self, config: dict):
        """
        初始化斑马线检测器

        Args:
            config: 配置字典
        """
        self.model_path = config.get('model_path', 'models/crosswalk.pt')
        self.model = None  # TODO: 加载YOLO模型

    def detect(self, image: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        检测斑马线并评估通行安全性

        Args:
            image: 输入图像
            depth_map: 深度图(可选)

        Returns:
            检测结果字典
        """
        # 步骤1: YOLO检测斑马线、信号灯、车辆
        detections = self._yolo_detect(image)

        # 提取各类目标
        crosswalk_boxes = [d for d in detections if d['class'] == 'crosswalk']
        traffic_light_boxes = [d for d in detections if d['class'] == 'traffic_light']
        vehicle_boxes = [d for d in detections if d['class'] == 'car']

        if len(crosswalk_boxes) == 0:
            return None

        # 步骤2: 提取斑马线方向
        direction = self._extract_direction(image, crosswalk_boxes[0]['bbox'])

        # 步骤3: 识别信号灯状态
        traffic_light_state = self._recognize_traffic_light(image, traffic_light_boxes)

        # 步骤4: 安全评估
        is_safe = self._evaluate_safety(
            traffic_light_state,
            vehicle_boxes,
            crosswalk_boxes[0],
            depth_map
        )

        return {
            'detected': True,
            'crosswalk_bbox': crosswalk_boxes[0]['bbox'],
            'direction': direction,  # 斑马线行进方向角度
            'traffic_light': traffic_light_state,  # 'red', 'green', 'yellow', 'unknown'
            'is_safe': is_safe,
            'num_vehicles': len(vehicle_boxes)
        }

    def _yolo_detect(self, image: np.ndarray) -> List[Dict]:
        """
        使用YOLO进行目标检测

        Args:
            image: 输入图像

        Returns:
            检测结果列表
        """
        # TODO: 实现YOLO检测
        # results = self.model(image)
        # return results

        # 占位符返回
        return []

    def _extract_direction(self, image: np.ndarray, bbox: tuple) -> float:
        """
        提取斑马线方向

        Args:
            image: 输入图像
            bbox: 斑马线边界框 (x1, y1, x2, y2)

        Returns:
            方向角度(度)
        """
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        # 边缘检测
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

        if lines is None:
            return 0.0

        # 计算主要方向
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
                angles.append(angle)

        if len(angles) == 0:
            return 0.0

        # 使用DBSCAN聚类找到主要方向
        angles_array = np.array(angles).reshape(-1, 1)
        clustering = DBSCAN(eps=10, min_samples=2).fit(angles_array)

        # 返回最大簇的中心角度
        labels = clustering.labels_
        if len(labels) > 0:
            main_cluster = max(set(labels), key=list(labels).count)
            main_angles = angles_array[labels == main_cluster]
            return float(np.mean(main_angles))

        return 0.0

    def _recognize_traffic_light(self, image: np.ndarray,
                                traffic_light_boxes: List[Dict]) -> str:
        """
        识别信号灯状态

        Args:
            image: 输入图像
            traffic_light_boxes: 信号灯检测框列表

        Returns:
            信号灯状态: 'red', 'green', 'yellow', 'unknown'
        """
        if len(traffic_light_boxes) == 0:
            return 'unknown'

        # 取第一个信号灯
        bbox = traffic_light_boxes[0]['bbox']
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 定义颜色范围
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([40, 255, 255])

        # 计算各颜色的像素数
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)

        # 返回像素数最多的颜色
        max_pixels = max(red_pixels, green_pixels, yellow_pixels)
        if max_pixels == red_pixels:
            return 'red'
        elif max_pixels == green_pixels:
            return 'green'
        elif max_pixels == yellow_pixels:
            return 'yellow'
        else:
            return 'unknown'

    def _evaluate_safety(self, traffic_light: str, vehicle_boxes: List[Dict],
                        crosswalk: Dict, depth_map: Optional[np.ndarray]) -> bool:
        """
        评估通行安全性

        Args:
            traffic_light: 信号灯状态
            vehicle_boxes: 车辆检测框列表
            crosswalk: 斑马线检测结果
            depth_map: 深度图

        Returns:
            是否安全通行
        """
        # 红灯不安全
        if traffic_light == 'red':
            return False

        # 有车辆接近不安全
        if len(vehicle_boxes) > 0 and depth_map is not None:
            # TODO: 检查车辆距离和速度
            pass

        # 绿灯且无车辆威胁
        if traffic_light == 'green':
            return True

        # 其他情况谨慎判断
        return False


if __name__ == "__main__":
    # 测试代码
    config = {
        'model_path': 'models/crosswalk.pt'
    }

    detector = CrosswalkDetector(config)
    print("斑马线检测模块初始化成功")
