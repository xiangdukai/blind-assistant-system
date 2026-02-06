"""
多目标跟踪模块
使用DeepSORT算法跟踪检测到的目标
"""

import numpy as np
from typing import List, Dict, Tuple


class Track:
    """单个跟踪目标"""

    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], class_id: int):
        """
        初始化跟踪目标

        Args:
            track_id: 跟踪ID
            bbox: 边界框 (x1, y1, x2, y2)
            class_id: 类别ID
        """
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.age = 0  # 轨迹存活帧数
        self.hits = 1  # 匹配成功次数
        self.time_since_update = 0  # 未更新帧数

    def update(self, bbox: Tuple[int, int, int, int]):
        """
        更新轨迹

        Args:
            bbox: 新的边界框
        """
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0

    def predict(self):
        """使用简单运动模型预测下一帧位置"""
        self.age += 1
        self.time_since_update += 1


class ObjectTracker:
    """多目标跟踪器"""

    def __init__(self, config: dict):
        """
        初始化跟踪器

        Args:
            config: 配置字典
        """
        self.max_age = config.get('max_age', 30)  # 最大未匹配帧数
        self.min_hits = config.get('min_hits', 3)  # 最小匹配次数
        self.iou_threshold = config.get('iou_threshold', 0.3)  # IOU阈值

        self.tracks: List[Track] = []  # 活跃轨迹列表
        self.next_id = 1  # 下一个可用ID

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        更新跟踪器

        Args:
            detections: 检测结果列表，每个元素包含bbox和class_id

        Returns:
            更新后的检测结果，添加了track_id字段
        """
        # 步骤1: 预测所有轨迹的位置
        for track in self.tracks:
            track.predict()

        # 步骤2: 匹配检测结果和轨迹
        matches, unmatched_detections, unmatched_tracks = self._match(detections)

        # 步骤3: 更新匹配的轨迹
        for det_idx, track_idx in matches:
            self.tracks[track_idx].update(detections[det_idx]['bbox'])
            detections[det_idx]['track_id'] = self.tracks[track_idx].track_id

        # 步骤4: 为未匹配的检测创建新轨迹
        for det_idx in unmatched_detections:           
            self._create_track(detections[det_idx])
            detections[det_idx]['track_id'] = self.next_id - 1

        # 步骤5: 删除过期轨迹
        self.tracks = [t for t in self.tracks
                      if t.time_since_update < self.max_age]

        # 只返回确认的轨迹
        confirmed_detections = [
            det for det in detections
            if 'track_id' in det and self._is_confirmed(det['track_id'])
        ]

        return confirmed_detections

    def _match(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        匹配检测结果和轨迹

        Args:
            detections: 检测结果列表

        Returns:
            (匹配对列表, 未匹配检测索引, 未匹配轨迹索引)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # 计算IOU矩阵
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                if det['class_id'] != track.class_id:
                    iou_matrix[d, t] = 0  # 不同类别，IOU设为0，无法匹配
                    continue
                iou_matrix[d, t] = self._calculate_iou(det['bbox'], track.bbox)

        # 简单贪婪匹配（实际应使用匈牙利算法）
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        for _ in range(min(len(detections), len(self.tracks))):
            # 找到最大IOU
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break

            d, t = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matches.append((d, t))

            # 标记为已匹配
            iou_matrix[d, :] = 0
            iou_matrix[:, t] = 0

            if d in unmatched_detections:
                unmatched_detections.remove(d)
            if t in unmatched_tracks:
                unmatched_tracks.remove(t)

        return matches, unmatched_detections, unmatched_tracks

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                      bbox2: Tuple[int, int, int, int]) -> float:
        """
        计算两个边界框的IOU

        Args:
            bbox1, bbox2: 边界框 (x1, y1, x2, y2)

        Returns:
            IOU值
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _create_track(self, detection: Dict):
        """
        创建新轨迹

        Args:
            detection: 检测结果
        """
        track = Track(self.next_id, detection['bbox'], detection['class_id'])
        self.tracks.append(track)
        self.next_id += 1

    def _is_confirmed(self, track_id: int) -> bool:
        """
        检查轨迹是否已确认

        Args:
            track_id: 跟踪ID

        Returns:
            是否已确认
        """
        for track in self.tracks:
            if track.track_id == track_id:
                return track.hits >= self.min_hits
        return False


if __name__ == "__main__":
    # 测试代码
    config = {
        'max_age': 30,
        'min_hits': 2,
        'iou_threshold': 0.2
        
    }
    detections = [
        {"bbox":[10,10,30,30],"class_id":0},
        {"bbox":[40,10,50,20],"class_id":1}
    ]
    tracker = ObjectTracker(config)
    print("目标跟踪模块初始化成功")
    matched_detections = tracker.update(detections)
    print("目标跟踪模块数据更新成功(第一次)")
    for idx,det in enumerate(matched_detections):
        class_id  = det.get("class_id","unknowned")
        track_id = det.get("track_id","unknowned")
        print(f"检测结果:{idx+1},class_id:{class_id},track_id:{track_id}\n")
    detections_1 = [
        {"bbox":[10,20,30,40],"class_id":0},
        {"bbox":[40,15,50,25],"class_id":1},
        {"bbox":[20,10,30,30],"class_id":2}
    ] 
    matched_detections = tracker.update(detections_1)
    print("目标跟踪模块数据更新成功(第二次)")
    for idx,det in enumerate(matched_detections):
        class_id  = det.get("class_id","unknowned")
        track_id = det.get("track_id","unknowned")
        print(f"检测结果:{idx+1},class_id:{class_id},track_id:{track_id}\n")