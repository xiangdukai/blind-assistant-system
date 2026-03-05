"""
文字识别模块
使用PaddleOCR进行文字检测和识别
"""

import cv2
import numpy as np
from typing import Optional, List, Dict


class OCRRecognizer:
    """文字识别器"""

    def __init__(self, config: dict):
        """
        初始化OCR识别器

        Args:
            config: 配置字典
        """
        self.language = config.get('language', 'ch')
        self.use_angle_cls = config.get('use_angle_cls', True)
        self.ocr = None  # TODO: 初始化PaddleOCR

        # 尝试导入PaddleOCR
        try:
            from paddleocr import PaddleOCR
            import paddleocr as _ppocr
            _ver = tuple(int(x) for x in _ppocr.__version__.split('.')[:2])
            if _ver >= (3, 0):
                # PaddleOCR 3.x：用 PP-OCRv4 轻量 mobile 模型，速度更快
                self.ocr = PaddleOCR(lang=self.language, ocr_version='PP-OCRv4')
            else:
                self.ocr = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=self.language,
                    show_log=False
                )
            self._use_predict_api = _ver >= (3, 0)
        except ImportError:
            print("警告: PaddleOCR未安装,OCR功能将不可用")

    def recognize(self, image: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        识别图像中的文字

        Args:
            image: 输入图像
            depth_map: 深度图(可选)

        Returns:
            识别结果字典
        """
        if self.ocr is None:
            return None

        # 步骤1: PaddleOCR检测和识别
        if getattr(self, '_use_predict_api', False):
            # PaddleOCR 3.x: predict() 返回 list of dict
            raw = self.ocr.predict(image)
            if not raw:
                return None
            results = [[]]
            for item in raw:
                for box, (txt, score) in zip(item.get('rec_boxes', []), item.get('rec_texts_with_scores', [])):
                    results[0].append([box, [txt, score]])
            if not results[0]:
                # 尝试兼容另一种返回格式
                results = [[]]
                for item in raw:
                    boxes = item.get('dt_polys', [])
                    texts = item.get('rec_text', [])
                    scores = item.get('rec_score', [])
                    for box, txt, score in zip(boxes, texts, scores):
                        results[0].append([box, [txt, score]])
        else:
            results = self.ocr.ocr(image, cls=self.use_angle_cls)

        if results is None or len(results) == 0 or results[0] is None:
            return None

        # 步骤2: 解析结果
        text_items = []
        for line in results[0]:
            bbox = line[0]  # 边界框坐标
            text = line[1][0]  # 识别的文字
            confidence = line[1][1]  # 置信度

            # 计算文字位置
            center_x = int(np.mean([p[0] for p in bbox]))
            center_y = int(np.mean([p[1] for p in bbox]))

            # 获取深度信息
            distance = None
            if depth_map is not None:
                distance = depth_map[center_y, center_x]

            # 计算相对方向
            direction = self._calculate_direction(center_x, image.shape[1])

            text_items.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox,
                'center': (center_x, center_y),
                'distance': distance,
                'direction': direction
            })

        if len(text_items) == 0:
            return None

        return {
            'detected': True,
            'num_texts': len(text_items),
            'texts': text_items,
            'full_text': ' '.join([item['text'] for item in text_items])
        }

    def _calculate_direction(self, center_x: int, image_width: int) -> str:
        """
        计算文字相对于用户的方向

        Args:
            center_x: 文字中心x坐标
            image_width: 图像宽度

        Returns:
            方向: 'left', 'center', 'right'
        """
        left_threshold = image_width * 0.33
        right_threshold = image_width * 0.67

        if center_x < left_threshold:
            return 'left'
        elif center_x > right_threshold:
            return 'right'
        else:
            return 'center'

    def find_text(self, image: np.ndarray, target_text: str,
                 depth_map: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        查找特定文字

        Args:
            image: 输入图像
            target_text: 目标文字
            depth_map: 深度图

        Returns:
            找到的文字信息
        """
        result = self.recognize(image, depth_map)

        if result is None:
            return None

        # 模糊匹配目标文字
        for text_item in result['texts']:
            if target_text in text_item['text']:
                return text_item

        return None

    def visualize(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        可视化识别结果

        Args:
            image: 原始图像
            result: 识别结果

        Returns:
            可视化后的图像
        """
        vis_image = image.copy()

        if result is None or not result['detected']:
            return vis_image

        # 绘制所有文字框
        for text_item in result['texts']:
            bbox = text_item['bbox']
            # 转换bbox为整数坐标
            points = np.array(bbox, dtype=np.int32)
            cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)

            # 添加文字标注
            text = text_item['text']
            center = text_item['center']
            cv2.putText(vis_image, text, center,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return vis_image


if __name__ == "__main__":
    # 测试代码
    config = {
        'language': 'ch',
        'use_angle_cls': True
    }

    recognizer = OCRRecognizer(config)
    print("OCR识别模块初始化成功")

    # TODO: 加载测试图片进行测试
    # test_image = cv2.imread('test_images/text.jpg')
    # result = recognizer.recognize(test_image)
    # print(result)
