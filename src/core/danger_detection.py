"""
动态危险预测模块
实现核心创新算法：预测移动物体的运动轨迹并评估碰撞风险
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import time

class DangerDetector:
    """动态危险预测器"""

    def __init__(self, config: dict):
        """
        初始化危险检测器

        Args:
            config: 配置字典，包含安全距离、预测时间等参数
        """
        self.safe_distance = config.get('safe_distance', 2.0)  # 安全距离(米)
        self.prediction_time = config.get('prediction_time', 3.0)  # 预测时间(秒)
        self.history_frames = config.get('history_frames', 10)  # 历史帧数
        self.ttl_threshold = config.get('ttl_threshold', 90)  # 轨迹TTL阈值（帧），默认90帧=3秒（30fps）
        # 相机内参
        self.camera_intrinsics = np.array(config.get('camera_intrinsics',[
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0], 
            [0.0, 0.0, 1.0] 
        ]))
        self.fps = config.get('fps',30)

        # 危险等级阈值
        self.danger_levels = config.get('danger_levels', {
            'high': 1.5,
            'medium': 3.0
        })

        # 跟踪轨迹字典 {track_id: deque}
        self.trajectories = {}
        # 轨迹TTL字典 {track_id: ttl_count}
        self.trajectory_ttl = {}

    def update(self, detections: List[Dict], depth_map: np.ndarray) -> List[Dict]:
        """
        更新检测结果并进行危险评估

        Args:
            detections: 检测结果列表，每个元素包含bbox, class_id, track_id等
            depth_map: 深度图

        Returns:
            危险目标列表，每个元素包含危险等级、碰撞时间、方向等信息
        """
        
        dangers = []

        for det in detections:
            # 步骤1: 获取3D坐标
            position_3d = self._get_3d_position(det['bbox'], depth_map)

            # 步骤2: 更新轨迹
            track_id = det['track_id']
            self._update_trajectory(track_id, position_3d)

            # 步骤3: 计算速度
            velocity, acceleration = self._calculate_velocity_and_acceleration(track_id)

            # 步骤4: 预测轨迹
            predicted_trajectory = self._predict_trajectory(position_3d, velocity, acceleration)

            # 步骤5: 碰撞检测
            collision_info = self._detect_collision(predicted_trajectory)

            # 步骤6: 危险等级评估
            if collision_info is not None:
                danger_level = self._evaluate_danger_level(
                    collision_info['time_to_collision'],
                    det['class_id'],
                    velocity
                )

                dangers.append({
                    'track_id': track_id,
                    'class_id': det['class_id'],
                    'danger_level': danger_level,
                    'time_to_collision': collision_info['time_to_collision'],
                    'direction': collision_info['direction'],
                    'distance': collision_info['distance']
                })

        #更新ttl时间
        for track_id in list(self.trajectory_ttl.keys()):
            self.trajectory_ttl[track_id] += 1
        #过滤已过期的track_id
        expired_track_ids = [ 
            track_id for track_id,ttl in self.trajectory_ttl.items()
            if ttl>self.ttl_threshold
        ]
        #执行删除操作
        for track_id in expired_track_ids:
            del self.trajectories[track_id]
            del self.trajectory_ttl[track_id]

        return dangers

    def _get_3d_position(self, bbox: Tuple[int, int, int, int],
                        depth_map: np.ndarray) -> np.ndarray:
        """
        将2D检测框转换为3D坐标

        Args:
            bbox: 边界框 (x1, y1, x2, y2)
            depth_map: 深度图（shape: [height, width]，单位：米）

        Returns:
            3D坐标 (x, y, z)
        """
        #安全校验bbox边界，避免越界 ==========
        depth_h, depth_w = depth_map.shape
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1, depth_w - 1))
        y1 = max(0, min(y1, depth_h - 1))
        x2 = max(0, min(x2, depth_w - 1))
        y2 = max(0, min(y2, depth_h - 1))
        
        # 若bbox无效（如x1>=x2或y1>=y2），返回默认3D坐标
        if x1 >= x2 or y1 >= y2:
            return np.array([0.0, 0.0, 5.0])
        
        # 提取检测框内的所有深度像素
        bbox_depth = depth_map[y1:y2+1, x1:x2+1]
        
        # 保留 有效深度范围
        valid_depth_mask = (bbox_depth > 0.1) & (bbox_depth < 10.0) & ~np.isnan(bbox_depth)
        valid_depths = bbox_depth[valid_depth_mask]
        
        #中位数滤波
        if len(valid_depths) > 0:
            depth = np.median(valid_depths)
        else:
            #框内无有效值时，使用默认深度兜底
             depth = 5.0  
            
        # 重新计算有效中心点（裁剪后的bbox）
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        fx = self.camera_intrinsics[0, 0]  # 焦距x（像素）
        fy = self.camera_intrinsics[1, 1]  # 焦距y（像素）
        cx = self.camera_intrinsics[0, 2]  # 主点x（像素，图像中心）
        cy = self.camera_intrinsics[1, 2]  # 主点y（像素，图像中心）
        
        position_x = (center_x - cx) * depth / fx
        position_y = (center_y - cy) * depth / fy

        position_3d = np.array([position_x, position_y, depth])

        return position_3d

    def _update_trajectory(self, track_id: int, position: np.ndarray):
        """
        更新目标轨迹历史

        Args:
            track_id: 跟踪ID
            position: 当前3D位置
        """
        #过滤无效位置
        if np.allclose(position, np.array([0.0, 0.0, 5.0])) or position[2] < 0.1:
            self.trajectory_ttl[track_id] = 0
            return
        
        if track_id not in self.trajectories:
            # 初始化deque，设置maxlen自动截断
            self.trajectories[track_id] = deque(maxlen=self.history_frames)

        # 添加当前位置,时间戳
        timestamp = time.time()  # 记录当前时间戳
        self.trajectories[track_id].append((position, timestamp))  # 存储（位置，时间戳）

        #更新ttl
        self.trajectory_ttl[track_id] = 0

    def _calculate_velocity_and_acceleration(self, track_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        计算目标的瞬时速度和加速度（滑动平均平滑）
        
        Args:
            track_id: 跟踪ID
        
        Returns:
            (速度向量vx,vy,vz, 加速度向量ax,ay,az) 或 (None, None)
        """
        if track_id not in self.trajectories:
            return None, None

        trajectory_data= self.trajectories[track_id]
        positions = [item[0] for item in trajectory_data]
        timestamps = [item[1] for item in trajectory_data]

        # 不足2帧：无速度；不足3帧：无加速度（退化为匀速）
        if len(trajectory_data) < 2:
            return None, None
        if len(trajectory_data) < 3:
            # 计算匀速速度
            velocity = (positions[-1] - positions[0]) / (timestamps[-1]-timestamps[-2])
            return velocity, None
        
        # 计算多组速度（滑动窗口），降低噪声影响
        velocities = []
        for i in range(len(trajectory_data) - 1):
            v = (positions[i+1] - positions[i])/(timestamps[i+1]-timestamps[i])
            velocities.append(v)
        velocities = np.array(velocities)
        smoothed_velocity = np.mean(velocities[-3:], axis=0)  # 取最后3组速度平均
        
        # 计算加速度
        accelerations = []
        for i in range(len(velocities) - 1):
            a = (velocities[i+1] - velocities[i]) /(timestamps[i+1]-timestamps[i])
            accelerations.append(a)
        accelerations = np.array(accelerations)
        smoothed_acceleration = np.mean(accelerations[-2:], axis=0)  # 取最后2组加速度平均
        
        return smoothed_velocity, smoothed_acceleration

    def _predict_trajectory(self, position: np.ndarray,
                      velocity: Optional[np.ndarray],
                      acceleration: Optional[np.ndarray],
                    ) -> List[np.ndarray]:
        """
        预测未来轨迹（恒定加速度模型CA），无加速度时退化为匀速模型CV
        
        公式：x(t) = x0 + v0*t + 0.5*a*t²
        Args:
            position: 当前位置 (x0,y0,z0)
            velocity: 速度向量 (vx,vy,vz)
            acceleration: 加速度向量 (ax,ay,az)
            dt: 时间步长(秒)
        
        Returns:
            预测轨迹点列表
        """
        if velocity is None:
            return [position]

        trajectory = []
        num_steps = int(self.prediction_time*self.fps)

        for i in range(num_steps):
            t = i /self.fps  
            if acceleration is not None:
                # 恒定加速度模型
                predicted_pos = position + velocity * t + 0.5 * acceleration * (t **2)
            else:
                # 退化为匀速模型
                predicted_pos = position + velocity * t
            trajectory.append(predicted_pos)

        return trajectory

    def _detect_collision(self, trajectory: List[np.ndarray]) -> Optional[Dict]:
        """
        检测是否有碰撞风险

        Args:
            trajectory: 预测轨迹

        Returns:
            碰撞信息字典或None
        """
        # 相机原点即盲人位置
        origin = np.array([0, 0, 0])

        for i, pos in enumerate(trajectory):
            distance = np.linalg.norm(pos - origin)

            if distance < self.safe_distance:
                # 计算方向
                direction = self._calculate_direction(pos)

                return {
                    'time_to_collision': i /self.fps,
                    'distance': distance,
                    'direction': direction
                }

        return None

    def _calculate_direction(self, position: np.ndarray) -> str:
        """
        计算目标相对于用户的方向

        Args:
            position: 目标位置

        Returns:
            方向字符串: 'front', 'left', 'right'
        """
        angle = np.arctan2(position[0], position[2])
        angle_deg = np.degrees(angle)

        if abs(angle_deg) < 30:
            return 'front'
        elif angle_deg > 0:
            return 'right'
        else:
            return 'left'

    def _evaluate_danger_level(self, time_to_collision: float,
                              class_id: int, velocity: np.ndarray) -> str:
        """
        评估危险等级

        Args:
            time_to_collision: 碰撞时间
            class_id: 目标类别
            velocity: 速度

        Returns:
            危险等级: 'high', 'medium', 'low'
        """
        if time_to_collision < self.danger_levels['high']:
            return 'high'
        elif time_to_collision < self.danger_levels['medium']:
            return 'medium'
        else:
            return 'low'


if __name__ == "__main__":
    # 测试代码
    config = {
        'safe_distance': 2.0,
        'prediction_time': 3.0,
        'history_frames': 10,
        'ttl_threshold': 5 
    }
    detections_1 = [
        {"bbox":[10,10,30,30],"class_id":0,"track_id":1},
        {"bbox":[40,10,50,20],"class_id":1,"track_id":2}
    ]
    detections_2 = [
        {"bbox":[10,20,30,40],"class_id":0,"track_id":1},
        {"bbox":[40,15,50,25],"class_id":1,"track_id":2},
        {"bbox":[20,10,30,30],"class_id":2,"track_id":3}
    ] 
    def create_fake_depth_map(detections, camera_resolution=(640, 480)):
        width, height = camera_resolution

        depth_map = np.full((height, width), 5.0, dtype=np.float32)  # 维度：[height, width]
        
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            
            # 裁剪坐标到图像范围内，避免索引越界
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # 跳过无效的bbox
            if x1 >= x2 or y1 >= y2:
                continue
            # 为不同目标分配不同深度
            if det["class_id"] == 0:
                target_depth = 1.5  # class_id=0的目标深度设为1.5米
            elif det["class_id"] == 1:
                target_depth = 1.8  # class_id=1的目标深度设为1.8米
            else:
                target_depth = 1.7  # 其他类别默认1.7米
            depth_map[y1:y2+1, x1:x2+1] = target_depth
        return depth_map

    detector = DangerDetector(config)
    print("动态危险预测模块初始化成功")
    dangers = detector.update(detections_1,create_fake_depth_map(detections_1))
    print(f"动态危险预测模块更新成功(第一次)\ntrajectories:{detector.trajectories}")
    print(f"dangers:{dangers}")
    detector.update(detections_2,create_fake_depth_map(detections_2))
    print(f"动态危险预测模块更新成功(第二次)\ntrajectories:{detector.trajectories}")
    print(f"dangers:{dangers}")