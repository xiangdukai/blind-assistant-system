"""
楼梯检测模块
使用Canny边缘检测和霍夫变换检测楼梯
"""

import cv2
import numpy as np
from typing import Optional, Dict
from sklearn.cluster import DBSCAN


class StairDetector:
    """楼梯检测器"""

    def __init__(self, config: dict):
        """
        初始化楼梯检测器

        Args:
            config: 配置字典
        """
        # Canny边缘检测参数
        self.canny_threshold1 = config.get('canny_threshold1', 50)
        self.canny_threshold2 = config.get('canny_threshold2', 150)

        # 霍夫变换参数
        self.hough_threshold = config.get('hough_threshold', 100)
        self.min_line_length = config.get('min_line_length', 50)
        self.max_line_gap = config.get('max_line_gap', 10)

        # 水平线判断阈值(角度)
        self.horizontal_angle_threshold = 20

    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        检测图像中的楼梯

        Args:
            image: 输入图像(BGR格式)

        Returns:
            检测结果字典，包含方向、台阶数等信息，如果未检测到则返回None
        """
        # 步骤1: 预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 步骤2: Canny边缘检测
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)

        # 步骤3: 形态学闭运算
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 步骤4: 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            closed,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None or len(lines) == 0:
            return None

        # 步骤5: 筛选水平线
        horizontal_lines = self._filter_horizontal_lines(lines)

        if len(horizontal_lines) < 3:  # 至少需要3条水平线才能判断为楼梯
            return None

        # 步骤6: 聚类统计台阶数
        num_steps = self._count_steps(horizontal_lines)

        # 步骤7: 判断方向(上楼/下楼)
        direction = self._determine_direction(horizontal_lines, image.shape[0])

        return {
            'detected': True,
            'direction': direction,  # 'up' or 'down'
            'num_steps': num_steps,
            'lines': horizontal_lines
        }

    def _filter_horizontal_lines(self, lines: np.ndarray) -> np.ndarray:
        """
        筛选水平线段

        Args:
            lines: 检测到的所有线段

        Returns:
            水平线段数组
        """
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算线段角度
            if x2 - x1 == 0:
                continue
            angle = np.abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))

            # 筛选接近水平的线段
            if angle < self.horizontal_angle_threshold:
                horizontal_lines.append(line[0])

        return np.array(horizontal_lines)

    def _count_steps(self, lines: np.ndarray) -> int:
        """
        使用DBSCAN聚类统计台阶数

        Args:
            lines: 水平线段数组

        Returns:
            台阶数量
        """
        if len(lines) == 0:
            return 0

        # 提取线段的y坐标中点
        y_coords = []
        for line in lines:
            y_mid = (line[1] + line[3]) / 2
            y_coords.append([y_mid])

        # DBSCAN聚类
        clustering = DBSCAN(eps=20, min_samples=1).fit(y_coords)
        num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)

        return num_clusters

    def _determine_direction(self, lines: np.ndarray, image_height: int) -> str:
        """
        判断楼梯方向

        Args:
            lines: 水平线段数组
            image_height: 图像高度

        Returns:
            方向: 'up' 或 'down'
        """
        # 计算所有线段的平均y坐标
        y_coords = []
        for line in lines:
            y_mid = (line[1] + line[3]) / 2
            y_coords.append(y_mid)

        avg_y = np.mean(y_coords)

        # 如果线段主要集中在图像下半部分,说明是上楼
        # 如果主要集中在上半部分,说明是下楼
        if avg_y > image_height / 2:
            return 'up'
        else:
            return 'down'

    def visualize(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image: 原始图像
            result: 检测结果

        Returns:
            可视化后的图像
        """
        vis_image = image.copy()

        if result is None or not result['detected']:
            return vis_image

        # 绘制检测到的水平线
        for line in result['lines']:
            x1, y1, x2, y2 = line
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 添加文字信息
        direction_text = "上楼" if result['direction'] == 'up' else "下楼"
        text = f"{direction_text}, 台阶数: {result['num_steps']}"
        cv2.putText(vis_image, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return vis_image


if __name__ == "__main__":
    # 测试代码
    config = {
        'canny_threshold1': 50,
        'canny_threshold2': 150,
        'hough_threshold': 100,
        'min_line_length': 50,
        'max_line_gap': 10
    }

    detector = StairDetector(config)
    print("楼梯检测模块初始化成功")

    # TODO: 加载测试图片进行测试
    # test_image = cv2.imread('test_images/stairs.jpg')
    # result = detector.detect(test_image)
    # print(result)
