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
        self.model_path = config.get('model_path', 'models/pidnet_s_cityscapes.pth')
        self.model_type = config.get('model_type', 'pidnet')
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载分割模型（支持 YOLO-seg .pt 格式）"""
        import os
        if not os.path.exists(self.model_path):
            print(f"盲道检测模型文件不存在 ({self.model_path})，将使用颜色特征检测")
            return
        if self.model_path.endswith('.pt'):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                print(f"盲道检测YOLO-seg模型加载成功: {self.model_path}")
            except Exception as e:
                print(f"盲道检测模型加载失败: {e}，将使用颜色特征检测")
        else:
            print(f"盲道检测模型格式暂不支持自动加载 ({self.model_path})，将使用颜色特征检测")

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
        if self.model is not None:
            # YOLO-seg 推理：返回第一类的分割 mask
            results = self.model(image, verbose=False)
            if results and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                if len(masks) > 0:
                    combined = np.any(masks > 0.5, axis=0).astype(np.uint8) * 255
                    return combined.astype(np.uint8)
            return None

        # ── 无模型：颜色 + 轮廓形状联合检测 ──────────────────
        h, w = image.shape[:2]

        # 只检测下 2/3 区域（盲道在地面，上方不可能有）
        roi_y = h // 3
        roi = image[roi_y:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 双范围黄色：涵盖标准黄、旧化偏橙、逆光偏暗等情况
        mask_std  = cv2.inRange(hsv, np.array([10, 60, 80]),  np.array([38, 255, 255]))  # 标准黄
        mask_pale = cv2.inRange(hsv, np.array([15, 30, 160]), np.array([35, 100, 255]))  # 浅黄/米色
        mask = cv2.bitwise_or(mask_std, mask_pale)

        # 形态学：开运算去小噪点，闭运算填孔连通
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT,    (21, 7))  # 横向拉伸以连通砖块
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

        # 连通域过滤：去掉小面积噪声，保留像素数 ≥ ROI 面积 0.3% 的区域
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = max(roi.shape[0] * roi.shape[1] * 0.003, 800)
        valid = np.zeros_like(mask)
        for i in range(1, n_lab):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                valid[labels == i] = 255

        # 轮廓验证：盲道区域应近似矩形，过滤圆形/细长噪声
        contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(valid)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw == 0 or rh == 0:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            solidity = area / (rw * rh + 1e-6)
            # 盲道砖：长宽比 1~15，实心度 > 0.3
            if 1.0 <= aspect <= 15 and solidity > 0.3:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)

        if cv2.countNonZero(filtered) < 800:
            return None

        # 还原至原图坐标
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[roi_y:] = filtered
        return full_mask

    def _extract_blind_path(self, segmentation_map: np.ndarray) -> Optional[np.ndarray]:
        """
        从分割图中提取盲道区域

        Args:
            segmentation_map: 分割结果图（已由 _segment 生成 mask）

        Returns:
            盲道mask
        """
        # _segment 已直接返回二值 mask，此处直接透传
        return segmentation_map

    def _compute_center_line(self, mask: np.ndarray) -> np.ndarray:
        """
        计算盲道中心线

        Args:
            mask: 盲道mask

        Returns:
            中心线坐标数组
        """
        # 按行扫描 mask，取每行非零像素的中心 x 坐标，构成中心线
        points = []
        for row in range(mask.shape[0]):
            cols = np.where(mask[row] > 0)[0]
            if len(cols) > 0:
                center_x = int((cols[0] + cols[-1]) / 2)
                points.append([row, center_x])

        if len(points) == 0:
            return np.array([])

        return np.array(points)

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
