"""
动态危险预测模块
实现核心创新算法：预测移动物体的运动轨迹并评估碰撞风险
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


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

        # 相机内参
        self.camera_intrinsics = np.array(config.get('camera_intrinsics',[
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0], 
            [0.0, 0.0, 1.0] 
        ]))

        # 危险等级阈值
        self.danger_levels = config.get('danger_levels', {
            'high': 1.5,
            'medium': 3.0
        })

        # 跟踪轨迹字典 {track_id: trajectory_data}
        self.trajectories = {}

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
            velocity = self._calculate_velocity(track_id)

            # 步骤4: 预测轨迹
            predicted_trajectory = self._predict_trajectory(position_3d, velocity)

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

        return dangers

    def _get_3d_position(self, bbox: Tuple[int, int, int, int],
                        depth_map: np.ndarray) -> np.ndarray:
        """
        将2D检测框转换为3D坐标

        Args:
            bbox: 边界框 (x1, y1, x2, y2)
            depth_map: 深度图

        Returns:
            3D坐标 (x, y, z)
        """
        # 计算边界框中心点
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        # 从深度图获取深度值
        depth = depth_map[center_y, center_x]

        # TODO: 使用相机内参将像素坐标转换为3D坐标
        
        fx = self.camera_intrinsics[0, 0]  # 焦距x（像素）
        fy = self.camera_intrinsics[1, 1]  # 焦距y（像素）
        cx = self.camera_intrinsics[0, 2]  # 主点x（像素，图像中心）
        cy = self.camera_intrinsics[1, 2]  # 主点y（像素，图像中心）
        '''
        # 像素坐标(u,v)与相机坐标(X,Y,Z)的关系：
        # u = fx * (X/Z) + cx  → X = (u - cx) * Z / fx
        # v = fy * (Y/Z) + cy  → Y = (v - cy) * Z / fy
        # Z = 深度值（相机坐标系z轴）
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        '''
        position_x = (center_x - cx) * depth / fx
        position_y = (center_y - cy) * depth / fy
    
        # 这里简化处理，实际需要相机标定参数
        position_3d = np.array([position_x, position_y, depth])

        return position_3d

    def _update_trajectory(self, track_id: int, position: np.ndarray):
        """
        更新目标轨迹历史

        Args:
            track_id: 跟踪ID
            position: 当前3D位置
        """
        if track_id not in self.trajectories:
            self.trajectories[track_id] = []

        # 添加当前位置
        self.trajectories[track_id].append(position)

        # 保持固定长度
        if len(self.trajectories[track_id]) > self.history_frames:
            self.trajectories[track_id].pop(0)

    def _calculate_velocity(self, track_id: int) -> Optional[np.ndarray]:
        """
        计算目标的运动速度

        Args:
            track_id: 跟踪ID

        Returns:
            速度向量 (vx, vy, vz) 或 None
        """
        if track_id not in self.trajectories:
            return None

        trajectory = self.trajectories[track_id]
        if len(trajectory) < 2:
            return None

        # 计算平均速度
        velocity = trajectory[-1] - trajectory[0]
        velocity = velocity / len(trajectory)

        return velocity

    def _predict_trajectory(self, position: np.ndarray,
                          velocity: Optional[np.ndarray],
                          dt: float = 0.1) -> List[np.ndarray]:
        """
        预测未来轨迹（匀速直线运动模型）

        Args:
            position: 当前位置
            velocity: 速度向量
            dt: 时间步长(秒)

        Returns:
            预测轨迹点列表
        """
        if velocity is None:
            return [position]

        trajectory = []
        num_steps = int(self.prediction_time / dt)

        for i in range(num_steps):
            predicted_pos = position + velocity * (i * dt)
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
                    'time_to_collision': i * 0.1,  # 时间步长
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
        'history_frames': 10
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

        depth_map = np.full((height, width), 2.0, dtype=np.float32)  # 维度：[height, width]
        
        for det in detections:
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) // 2  # 中心点x（像素）
            cy = (bbox[1] + bbox[3]) // 2  # 中心点y（像素）
            
            # 为不同目标分配不同深度
            if det["class_id"] == 0:
                target_depth = 1.5  # class_id=0的目标深度设为1.5米
            elif det["class_id"] == 1:
                target_depth = 1.8  # class_id=1的目标深度设为1.8米
            else:
                target_depth = 1.7  # 其他类别默认1.7米
            
            if 0 <= cy < height and 0 <= cx < width:  # 避免像素越界
                depth_map[cy, cx] = target_depth
        
        return depth_map

    detector = DangerDetector(config)
    print("动态危险预测模块初始化成功")
    detector.update(detections_1,create_fake_depth_map(detections_1))
    print(f"动态危险预测模块更新成功(第一次)\n{detector.trajectories}")
    detector.update(detections_2,create_fake_depth_map(detections_2))
    print(f"动态危险预测模块更新成功(第二次)\n{detector.trajectories}")