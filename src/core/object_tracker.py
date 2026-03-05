"""
多目标跟踪模块
使用DeepSORT算法跟踪检测到的目标（3D空间跟踪）
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from .kalman_filter import KalmanFilter


class Track:
    """单个3D跟踪目标"""

    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int],
                 class_id: int, position_3d: np.ndarray, camera_matrix: dict):
        """
        初始化跟踪目标

        Args:
            track_id: 跟踪ID
            bbox: 边界框 (x1, y1, x2, y2)
            class_id: 类别ID
            position_3d: 初始3D位置 [X, Y, Z]
            camera_matrix: 相机内参字典
        """
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.age = 0  # 轨迹存活帧数
        self.hits = 1  # 匹配成功次数
        self.time_since_update = 0  # 未更新帧数

        # 3D卡尔曼滤波器
        self.kf = KalmanFilter(dt=1.0/30.0)  # 假设30fps
        self.kf.state[:3] = position_3d  # 初始化位置
        self.kf.P = np.eye(6) * 10.0  # 初始化协方差

        # 保存相机内参
        self.camera_matrix = camera_matrix

        # 3D位置和速度
        self.position_3d = position_3d
        self.velocity_3d = np.zeros(3)

    def update(self, bbox: Tuple[int, int, int, int], position_3d: np.ndarray):
        """
        更新轨迹

        Args:
            bbox: 新的边界框
            position_3d: 新的3D位置
        """
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0

        # 使用卡尔曼滤波器更新3D位置
        self.kf.update(position_3d)
        self.position_3d = self.kf.get_position()
        self.velocity_3d = self.kf.get_velocity()

    def predict(self):
        """使用卡尔曼滤波器预测下一帧3D位置"""
        self.age += 1
        self.time_since_update += 1

        # 卡尔曼预测
        predicted_pos = self.kf.predict()
        self.position_3d = predicted_pos
        self.velocity_3d = self.kf.get_velocity()

        # 预测2D边界框（将3D位置投影回2D）
        # 注：这里简化处理，实际应该考虑物体尺寸
        self.bbox = self._project_3d_to_2d(self.position_3d)

    def _project_3d_to_2d(self, position_3d: np.ndarray) -> Tuple[int, int, int, int]:
        """
        将3D位置投影到2D边界框（简化版）

        Args:
            position_3d: 3D位置 [X, Y, Z]

        Returns:
            边界框 (x1, y1, x2, y2)
        """
        # 从3D投影到像素坐标
        X, Y, Z = position_3d
        fx = self.camera_matrix['fx']
        fy = self.camera_matrix['fy']
        cx = self.camera_matrix['cx']
        cy = self.camera_matrix['cy']

        if Z <= 0:  # 避免除零
            return self.bbox

        u = int(fx * X / Z + cx)
        v = int(fy * Y / Z + cy)

        # 简化：假设边界框尺寸与上一帧相似
        x1, y1, x2, y2 = self.bbox
        w = x2 - x1
        h = y2 - y1

        return (u - w//2, v - h//2, u + w//2, v + h//2)


class ObjectTracker:
    """多目标跟踪器 - 基于3D空间的DeepSORT实现"""

    def __init__(self, config: dict):
        """
        初始化跟踪器

        Args:
            config: 配置字典
        """
        self.max_age = config.get('max_age', 30)  # 最大未匹配帧数
        self.min_hits = config.get('min_hits', 3)  # 最小匹配次数
        self.distance_threshold = config.get('distance_threshold', 1.0)  # 3D距离阈值(米)

        # 相机内参（从配置中读取）
        self.camera_matrix = config.get('camera_matrix', {
            'fx': 525.0,  # 焦距x
            'fy': 525.0,  # 焦距y
            'cx': 319.5,  # 主点x
            'cy': 239.5   # 主点y
        })

        self.tracks: List[Track] = []  # 活跃轨迹列表
        self.next_id = 1  # 下一个可用ID

    def update(self, detections: List[Dict], depth_map: np.ndarray) -> List[Dict]:
        """
        更新跟踪器

        Args:
            detections: 检测结果列表，每个元素包含bbox和class_id
            depth_map: 深度图

        Returns:
            更新后的检测结果，添加了track_id、position_3d、velocity_3d字段
        """
        # 步骤1: 预测所有轨迹的3D位置
        for track in self.tracks:
            track.predict()

        # 步骤2: 将检测结果转换为3D坐标
        detections_3d = []
        for det in detections:
            position_3d = self._get_3d_position(det['bbox'], depth_map)
            if position_3d is not None:
                det['position_3d'] = position_3d
                detections_3d.append(det)

        # 步骤3: 使用匈牙利算法匹配检测和轨迹（基于3D距离）
        matches, unmatched_detections, unmatched_tracks = self._match_hungarian(detections_3d)

        # 步骤4: 更新匹配的轨迹
        for det_idx, track_idx in matches:
            det = detections_3d[det_idx]
            self.tracks[track_idx].update(det['bbox'], det['position_3d'])
            det['track_id'] = self.tracks[track_idx].track_id
            det['velocity_3d'] = self.tracks[track_idx].velocity_3d

        # 步骤5: 为未匹配的检测创建新轨迹
        for det_idx in unmatched_detections:
            det = detections_3d[det_idx]
            self._create_track(det)
            det['track_id'] = self.next_id - 1
            det['velocity_3d'] = np.zeros(3)  # 新轨迹初始速度为0

        # 步骤6: 删除过期轨迹
        self.tracks = [t for t in self.tracks
                      if t.time_since_update < self.max_age]

        # 只返回确认的轨迹
        confirmed_detections = [
            det for det in detections_3d
            if 'track_id' in det and self._is_confirmed(det['track_id'])
        ]

        return confirmed_detections

    def _get_3d_position(self, bbox: Tuple[int, int, int, int],
                        depth_map: np.ndarray) -> Optional[np.ndarray]:
        """
        将2D检测框转换为3D世界坐标

        Args:
            bbox: 边界框 (x1, y1, x2, y2)
            depth_map: 深度图

        Returns:
            3D坐标 [X, Y, Z] (单位：米) 或 None（如果深度无效）
        """
        # 计算边界框中心点
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 边界检查
        h, w = depth_map.shape[:2]
        if not (0 <= center_x < w and 0 <= center_y < h):
            return None

        # 从深度图获取深度值（单位：米）
        depth = depth_map[center_y, center_x]

        # 深度有效性检查
        if depth <= 0 or depth > 10.0:  # 过滤无效深度
            return None

        # 使用相机内参将像素坐标转换为3D坐标
        fx = self.camera_matrix['fx']
        fy = self.camera_matrix['fy']
        cx = self.camera_matrix['cx']
        cy = self.camera_matrix['cy']

        # 相机坐标系（右手坐标系）
        # X: 右, Y: 下, Z: 前
        X = (center_x - cx) * depth / fx
        Y = (center_y - cy) * depth / fy
        Z = depth

        return np.array([X, Y, Z])

    def _match_hungarian(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        使用匈牙利算法匹配检测结果和轨迹（基于3D欧氏距离）

        Args:
            detections: 包含3D位置的检测结果列表

        Returns:
            (匹配对列表, 未匹配检测索引, 未匹配轨迹索引)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # 计算3D距离代价矩阵
        cost_matrix = np.zeros((len(detections), len(self.tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                # 计算3D欧氏距离
                distance = np.linalg.norm(det['position_3d'] - track.position_3d)
                cost_matrix[d, t] = distance

        # 使用匈牙利算法求解最优分配
        # linear_sum_assignment 返回行索引和列索引
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # 过滤距离过大的匹配
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        for d, t in zip(row_indices, col_indices):
            if cost_matrix[d, t] < self.distance_threshold:
                matches.append((d, t))
                unmatched_detections.remove(d)
                unmatched_tracks.remove(t)

        return matches, unmatched_detections, unmatched_tracks

    def _create_track(self, detection: Dict):
        """
        创建新轨迹

        Args:
            detection: 检测结果（包含bbox、class_id、position_3d）
        """
        track = Track(
            self.next_id,
            detection['bbox'],
            detection['class_id'],
            detection['position_3d'],
            self.camera_matrix
        )
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
        'min_hits': 3,
        'distance_threshold': 1.0,
        'camera_matrix': {
            'fx': 525.0,
            'fy': 525.0,
            'cx': 319.5,
            'cy': 239.5
        }
    }

    tracker = ObjectTracker(config)
    print("3D多目标跟踪模块初始化成功")
    print(f"使用匈牙利算法进行最优匹配，3D距离阈值: {config['distance_threshold']}米")
