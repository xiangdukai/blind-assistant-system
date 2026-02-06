"""
盲道检测模块
使用语义分割检测盲道
"""

import cv2
import numpy as np
from typing import Optional, Dict


class BlindPathDetector:
    """盲道检测器"""

    def __init__(self, config: dict):
        """
        初始化盲道检测器

        Args:
            config: 配置字典
        """
        self.model_path = config.get('model_path', 'models/pp_liteseg.pth')
        self.model = None  # TODO: 加载语义分割模型 (PP-LiteSeg或YOLO-Seg)

    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        检测图像中的盲道

        Args:
            image: 输入图像

        Returns:
            检测结果字典
        """
        # 步骤1: 语义分割
        segmentation_map = self._segment(image)

        if segmentation_map is None:
            return None

        # 步骤2: 提取盲道区域
        blind_path_mask = self._extract_blind_path(segmentation_map)

        if blind_path_mask is None or np.sum(blind_path_mask) == 0:
            return None

        # 步骤3: 计算盲道中心线
        center_line = self._compute_center_line(blind_path_mask)

        # 步骤4: 计算偏离程度
        deviation = self._calculate_deviation(center_line, image.shape[1])

        # 步骤5: 确定引导方向
        guidance_direction = self._determine_guidance(deviation)

        return {
            'detected': True,
            'mask': blind_path_mask,
            'center_line': center_line,
            'deviation': deviation,  # 偏离中心的像素值,正数向右,负数向左
            'guidance': guidance_direction  # 'left', 'right', 'straight'
        }

    def _segment(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        进行语义分割

        Args:
            image: 输入图像

        Returns:
            分割结果图
        """
        # TODO: 实现语义分割
        # segmentation_map = self.model(image)
        return None

    def _extract_blind_path(self, segmentation_map: np.ndarray) -> Optional[np.ndarray]:
        """
        从分割图中提取盲道区域

        Args:
            segmentation_map: 分割结果图

        Returns:
            盲道mask
        """
        # TODO: 根据类别ID提取盲道
        # blind_path_mask = (segmentation_map == BLIND_PATH_CLASS_ID).astype(np.uint8) * 255
        return None

    def _compute_center_line(self, mask: np.ndarray) -> np.ndarray:
        """
        计算盲道中心线

        Args:
            mask: 盲道mask

        Returns:
            中心线坐标数组
        """
        # 使用形态学骨架化提取中心线
        skeleton = cv2.ximgproc.thinning(mask)

        # 提取中心线点
        points = np.column_stack(np.where(skeleton > 0))

        if len(points) == 0:
            return np.array([])

        return points

    def _calculate_deviation(self, center_line: np.ndarray, image_width: int) -> float:
        """
        计算用户相对于盲道中心的偏离程度

        Args:
            center_line: 中心线坐标
            image_width: 图像宽度

        Returns:
            偏离值(像素),正数表示向右偏离,负数表示向左偏离
        """
        if len(center_line) == 0:
            return 0.0

        # 取图像下半部分的中心线点
        lower_half_points = center_line[center_line[:, 0] > center_line[:, 0].max() * 0.5]

        if len(lower_half_points) == 0:
            return 0.0

        # 计算平均x坐标
        avg_x = np.mean(lower_half_points[:, 1])

        # 计算偏离
        image_center = image_width / 2
        deviation = avg_x - image_center

        return float(deviation)

    def _determine_guidance(self, deviation: float) -> str:
        """
        根据偏离程度确定引导方向

        Args:
            deviation: 偏离值

        Returns:
            引导方向: 'left', 'right', 'straight'
        """
        threshold = 50  # 偏离阈值(像素)

        if abs(deviation) < threshold:
            return 'straight'
        elif deviation > 0:
            return 'left'  # 向右偏离,需要向左调整
        else:
            return 'right'  # 向左偏离,需要向右调整

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

        # 叠加mask
        mask = result['mask']
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 1] = mask  # 绿色通道
        vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)

        # 绘制中心线
        center_line = result['center_line']
        if len(center_line) > 0:
            for point in center_line:
                cv2.circle(vis_image, (point[1], point[0]), 2, (0, 0, 255), -1)

        # 添加引导信息
        guidance_text = f"引导: {result['guidance']}"
        cv2.putText(vis_image, guidance_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return vis_image


if __name__ == "__main__":
    # 测试代码
    config = {
        'model_path': 'models/pp_liteseg.pth'
    }

    detector = BlindPathDetector(config)
    print("盲道检测模块初始化成功")
