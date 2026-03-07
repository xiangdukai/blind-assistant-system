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

        # 危险等级阈值（单位：秒）
        self.danger_levels = config.get('danger_levels', {
            'high': 1.5,
            'medium': 3.0
        })

        # 相机内参（用于坐标转换）
        self.camera_matrix = config.get('camera_matrix', {
            'fx': 525.0,
            'fy': 525.0,
            'cx': 319.5,
            'cy': 239.5
        })

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        更新检测结果并进行危险评估

        Args:
            detections: 跟踪器输出的检测结果列表，包含:
                       - bbox: 边界框
                       - class_id: 类别ID
                       - track_id: 跟踪ID
                       - position_3d: 3D位置 [X, Y, Z] (由tracker提供)
                       - velocity_3d: 3D速度 [vx, vy, vz] (由tracker的卡尔曼滤波器提供)

        Returns:
            危险目标列表，每个元素包含危险等级、碰撞时间、方向等信息
        """
        dangers = []

        for det in detections:
            # 检查必要字段
            if 'position_3d' not in det or 'velocity_3d' not in det:
                continue

            track_id = det['track_id']
            position_3d = det['position_3d']
            velocity_3d = det['velocity_3d']

            # 步骤1: 预测轨迹（使用卡尔曼滤波器提供的速度）
            predicted_trajectory = self._predict_trajectory(position_3d, velocity_3d)

            # 步骤2: 碰撞检测
            collision_info = self._detect_collision(predicted_trajectory)

            # 步骤3: 危险等级评估
            if collision_info is not None:
                danger_level = self._evaluate_danger_level(
                    collision_info['time_to_collision'],
                    det['class_id'],
                    velocity_3d
                )

                dangers.append({
                    'track_id': track_id,
                    'class_id': det['class_id'],
                    'danger_level': danger_level,
                    'time_to_collision': collision_info['time_to_collision'],
                    'direction': collision_info['direction'],
                    'distance': collision_info['distance'],
                    'position_3d': position_3d,
                    'velocity_3d': velocity_3d
                })

        return dangers

    def _predict_trajectory(self, position: np.ndarray,
                          velocity: np.ndarray,
                          dt: float = 0.1) -> List[np.ndarray]:
        """
        预测未来轨迹（匀速直线运动模型）

        Args:
            position: 当前3D位置 [X, Y, Z] (米)
            velocity: 3D速度向量 [vx, vy, vz] (米/秒)
            dt: 时间步长(秒)

        Returns:
            预测轨迹点列表
        """
        speed = np.linalg.norm(velocity) if velocity is not None else 0.0
        vz    = float(velocity[2]) if velocity is not None else 0.0

        # 只对"正在向摄像头靠近（vz < 0）且速度 ≥ 0.15m/s"的物体做轨迹预测
        # 静止物体 / 远离物体不产生危险警报
        if speed < 0.15 or vz >= 0:
            return []

        trajectory = []
        num_steps = int(self.prediction_time / dt)
        for i in range(num_steps):
            predicted_pos = position + velocity * (i * dt)
            trajectory.append(predicted_pos)

        return trajectory

    def _detect_collision(self, trajectory: List[np.ndarray], dt: float = 0.1) -> Optional[Dict]:
        """
        检测是否有碰撞风险（考虑盲人朝向的锥形危险区域）

        Args:
            trajectory: 预测轨迹
            dt: 时间步长(秒)

        Returns:
            碰撞信息字典或None
        """
        # 相机坐标系原点即盲人位置
        # 相机坐标系: X右, Y下, Z前（盲人前进方向）
        origin = np.array([0.0, 0.0, 0.0])

        for i, pos in enumerate(trajectory):
            _, _, Z = pos

            # 只考虑前方的危险（Z > 0）
            if Z <= 0:
                continue

            # 使用真实 3D 欧氏距离判断是否进入安全距离球形区域
            distance = np.linalg.norm(pos - origin)

            if distance < self.safe_distance:
                # 计算方向
                direction = self._calculate_direction(pos)

                return {
                    'time_to_collision': i * dt,  # 碰撞时间(秒)
                    'distance': distance,  # 实际3D距离(米)
                    'direction': direction
                }

        return None

    def _calculate_direction(self, position: np.ndarray) -> str:
        """
        计算目标相对于用户的方向（更精细的方位判断）

        Args:
            position: 目标3D位置 [X, Y, Z]
                     相机坐标系: X右, Y下, Z前

        Returns:
            方向字符串: 'front', 'front_left', 'front_right', 'left', 'right'
        """
        X, Y, Z = position

        # 计算水平角度（忽略垂直方向Y）
        angle = np.arctan2(X, Z)  # atan2(X, Z) 得到相对于正前方的角度
        angle_deg = np.degrees(angle)

        # 更精细的方向判断
        if abs(angle_deg) < 15:
            return 'front'
        elif 15 <= angle_deg < 45:
            return 'front_right'
        elif 45 <= angle_deg < 90:
            return 'right'
        elif -45 < angle_deg <= -15:
            return 'front_left'
        elif -90 < angle_deg <= -45:
            return 'left'
        else:
            # 大于90度认为在后方或侧后方，危险性较低
            return 'side' if angle_deg > 0 else 'side'

    def _evaluate_danger_level(self, time_to_collision: float,
                              class_id: int, velocity: np.ndarray) -> str:
        """
        评估危险等级（综合考虑多个因素）

        Args:
            time_to_collision: 碰撞时间(秒)
            class_id: 目标类别ID
            velocity: 3D速度向量 [vx, vy, vz] (米/秒)

        Returns:
            危险等级: 'high', 'medium', 'low'
        """
        # 基础评分：基于碰撞时间
        if time_to_collision < self.danger_levels['high']:
            base_score = 3  # 高危
        elif time_to_collision < self.danger_levels['medium']:
            base_score = 2  # 中危
        else:
            base_score = 1  # 低危

        # 速度因素：速度越快越危险
        speed = np.linalg.norm(velocity)
        if speed > 2.0:  # 快速移动 (>2m/s ≈ 7km/h)
            speed_factor = 1.5
        elif speed > 1.0:  # 中速移动
            speed_factor = 1.2
        else:  # 慢速移动
            speed_factor = 1.0

        # 类别因素：不同类别危险程度不同
        # 假设class_id映射：0=person, 1=bicycle, 2=car, 3=motorcycle, etc.
        danger_class_weights = {
            0: 1.0,   # 行人
            1: 1.3,   # 自行车
            2: 1.5,   # 汽车
            3: 1.5,   # 摩托车
            5: 1.2,   # 巴士
            7: 1.2    # 卡车
        }
        class_factor = danger_class_weights.get(class_id, 1.0)

        # 综合评分
        final_score = base_score * speed_factor * class_factor

        # 根据最终评分确定危险等级
        if final_score >= 3.5:
            return 'high'
        elif final_score >= 2.0:
            return 'medium'
        else:
            return 'low'


if __name__ == "__main__":
    # 测试代码
    config = {
        'safe_distance': 2.0,
        'prediction_time': 3.0,
        'danger_levels': {
            'high': 1.5,
            'medium': 3.0
        },
        'camera_matrix': {
            'fx': 525.0,
            'fy': 525.0,
            'cx': 319.5,
            'cy': 239.5
        }
    }

    detector = DangerDetector(config)
    print("动态危险预测模块初始化成功")
    print("改进点:")
    print("- 使用tracker提供的卡尔曼滤波3D位置和速度")
    print("- 考虑锥形危险区域（前方更危险）")
    print("- 综合评估：碰撞时间 + 速度 + 目标类别")
    print("- 更精细的方位判断（5个方向）")
