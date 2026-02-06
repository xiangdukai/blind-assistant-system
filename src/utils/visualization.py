"""
可视化工具模块
用于调试和演示时显示检测结果
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


class Visualizer:
    """可视化工具"""

    def __init__(self):
        """初始化可视化工具"""
        # 颜色定义 (BGR格式)
        self.colors = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'purple': (255, 0, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }

        # 危险等级颜色
        self.danger_colors = {
            'high': self.colors['red'],
            'medium': self.colors['orange'],
            'low': self.colors['yellow']
        }

    def draw_detections(self, image: np.ndarray,
                       detections: List[Dict]) -> np.ndarray:
        """
        绘制检测框

        Args:
            image: 输入图像
            detections: 检测结果列表

        Returns:
            绘制后的图像
        """
        vis_image = image.copy()

        for det in detections:
            bbox = det['bbox']
            class_id = det.get('class_id', 0)
            confidence = det.get('confidence', 1.0)
            track_id = det.get('track_id', None)

            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2),
                         self.colors['green'], 2)

            # 添加标签
            label = f"ID:{track_id} {confidence:.2f}" if track_id else f"{confidence:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       self.colors['green'], 1)

        return vis_image

    def draw_dangers(self, image: np.ndarray,
                    dangers: List[Dict]) -> np.ndarray:
        """
        绘制危险目标

        Args:
            image: 输入图像
            dangers: 危险列表

        Returns:
            绘制后的图像
        """
        vis_image = image.copy()

        for danger in dangers:
            track_id = danger['track_id']
            level = danger['danger_level']
            time_to_collision = danger['time_to_collision']
            direction = danger['direction']

            # 使用危险等级颜色
            color = self.danger_colors.get(level, self.colors['yellow'])

            # 绘制警告信息
            text = f"危险! ID:{track_id} {level.upper()} {time_to_collision:.1f}s {direction}"
            self._draw_text_with_background(
                vis_image, text, (10, 30 + len(dangers) * 30),
                color, self.colors['black']
            )

        return vis_image

    def draw_trajectory(self, image: np.ndarray,
                       trajectory: List[np.ndarray],
                       color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        绘制运动轨迹

        Args:
            image: 输入图像
            trajectory: 轨迹点列表
            color: 颜色

        Returns:
            绘制后的图像
        """
        if color is None:
            color = self.colors['blue']

        vis_image = image.copy()

        if len(trajectory) < 2:
            return vis_image

        # 绘制轨迹线
        points = np.array(trajectory, dtype=np.int32)
        for i in range(len(points) - 1):
            cv2.line(vis_image, tuple(points[i][:2]),
                    tuple(points[i+1][:2]), color, 2)

        # 绘制轨迹点
        for point in points:
            cv2.circle(vis_image, tuple(point[:2]), 3, color, -1)

        return vis_image

    def draw_depth_map(self, depth_map: np.ndarray,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        可视化深度图

        Args:
            depth_map: 深度图
            colormap: OpenCV颜色映射

        Returns:
            彩色深度图
        """
        # 归一化深度图
        depth_normalized = cv2.normalize(depth_map, None, 0, 255,
                                        cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 应用颜色映射
        depth_colormap = cv2.applyColorMap(depth_normalized, colormap)

        return depth_colormap

    def draw_safe_zone(self, image: np.ndarray,
                      center: Tuple[int, int],
                      radius: int) -> np.ndarray:
        """
        绘制安全区域

        Args:
            image: 输入图像
            center: 圆心
            radius: 半径

        Returns:
            绘制后的图像
        """
        vis_image = image.copy()

        # 绘制半透明圆形
        overlay = vis_image.copy()
        cv2.circle(overlay, center, radius, self.colors['green'], -1)
        cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)

        # 绘制圆形边界
        cv2.circle(vis_image, center, radius, self.colors['green'], 2)

        return vis_image

    def draw_grid(self, image: np.ndarray,
                 grid_size: int = 50) -> np.ndarray:
        """
        绘制网格

        Args:
            image: 输入图像
            grid_size: 网格大小

        Returns:
            绘制后的图像
        """
        vis_image = image.copy()
        h, w = image.shape[:2]

        # 绘制垂直线
        for x in range(0, w, grid_size):
            cv2.line(vis_image, (x, 0), (x, h),
                    self.colors['white'], 1)

        # 绘制水平线
        for y in range(0, h, grid_size):
            cv2.line(vis_image, (0, y), (w, y),
                    self.colors['white'], 1)

        return vis_image

    def _draw_text_with_background(self, image: np.ndarray,
                                   text: str, position: Tuple[int, int],
                                   text_color: Tuple[int, int, int],
                                   bg_color: Tuple[int, int, int],
                                   font_scale: float = 0.6,
                                   thickness: int = 2):
        """
        绘制带背景的文字

        Args:
            image: 输入图像
            text: 文字内容
            position: 位置
            text_color: 文字颜色
            bg_color: 背景颜色
            font_scale: 字体大小
            thickness: 粗细
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 获取文字尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # 绘制背景矩形
        x, y = position
        cv2.rectangle(image,
                     (x - 5, y - text_height - 5),
                     (x + text_width + 5, y + baseline + 5),
                     bg_color, -1)

        # 绘制文字
        cv2.putText(image, text, position, font,
                   font_scale, text_color, thickness)

    def create_dashboard(self, image: np.ndarray,
                        stats: Dict) -> np.ndarray:
        """
        创建信息面板

        Args:
            image: 输入图像
            stats: 统计信息字典

        Returns:
            添加面板后的图像
        """
        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        # 创建侧边栏
        panel_width = 250
        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

        # 绘制标题
        y_offset = 30
        cv2.putText(panel, "System Status", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   self.colors['green'], 2)

        # 绘制统计信息
        y_offset += 40
        for key, value in stats.items():
            text = f"{key}: {value}"
            cv2.putText(panel, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       self.colors['white'], 1)
            y_offset += 25

        # 拼接面板
        result = np.hstack([vis_image, panel])

        return result


if __name__ == "__main__":
    # 测试代码
    visualizer = Visualizer()
    print("可视化工具初始化成功")

    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # 测试绘制检测框
    detections = [
        {'bbox': [100, 100, 200, 200], 'class_id': 0, 'confidence': 0.95, 'track_id': 1}
    ]
    result = visualizer.draw_detections(test_image, detections)

    print("可视化工具测试完成")
