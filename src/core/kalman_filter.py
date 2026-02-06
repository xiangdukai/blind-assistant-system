"""
卡尔曼滤波器模块
用于轨迹预测和状态估计
"""

import numpy as np


class KalmanFilter:
    """卡尔曼滤波器"""

    def __init__(self, dt: float = 0.1):
        """
        初始化卡尔曼滤波器

        Args:
            dt: 时间步长(秒)
        """
        self.dt = dt

        # 状态向量: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)

        # 状态转移矩阵
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # 观测矩阵
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # 过程噪声协方差
        self.Q = np.eye(6) * 0.1

        # 观测噪声协方差
        self.R = np.eye(3) * 1.0

        # 状态协方差矩阵
        self.P = np.eye(6) * 10.0

    def predict(self) -> np.ndarray:
        """
        预测下一时刻的状态

        Returns:
            预测的位置 [x, y, z]
        """
        # 状态预测
        self.state = self.F @ self.state

        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q

        # 返回预测的位置
        return self.state[:3]

    def update(self, measurement: np.ndarray):
        """
        使用观测值更新状态

        Args:
            measurement: 观测值 [x, y, z]
        """
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 更新状态
        innovation = measurement - self.H @ self.state
        self.state = self.state + K @ innovation

        # 更新协方差
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

    def get_position(self) -> np.ndarray:
        """
        获取当前位置估计

        Returns:
            位置 [x, y, z]
        """
        return self.state[:3]

    def get_velocity(self) -> np.ndarray:
        """
        获取当前速度估计

        Returns:
            速度 [vx, vy, vz]
        """
        return self.state[3:]

    def predict_trajectory(self, num_steps: int) -> np.ndarray:
        """
        预测未来轨迹

        Args:
            num_steps: 预测步数

        Returns:
            轨迹点数组，形状为 (num_steps, 3)
        """
        trajectory = []
        state = self.state.copy()

        for _ in range(num_steps):
            state = self.F @ state
            trajectory.append(state[:3])

        return np.array(trajectory)


class ExtendedKalmanFilter(KalmanFilter):
    """扩展卡尔曼滤波器 (用于非线性系统)"""

    def __init__(self, dt: float = 0.1):
        """
        初始化扩展卡尔曼滤波器

        Args:
            dt: 时间步长(秒)
        """
        super().__init__(dt)
        # 可以在这里添加非线性模型相关的参数

    def predict_nonlinear(self) -> np.ndarray:
        """
        非线性预测
        TODO: 根据实际需求实现非线性状态转移函数

        Returns:
            预测的位置
        """
        # 这里可以实现考虑加速度、转向等非线性因素的预测
        return self.predict()


if __name__ == "__main__":
    # 测试代码
    kf = KalmanFilter(dt=0.1)

    # 模拟观测序列
    measurements = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 2.1, 3.1]),
        np.array([1.2, 2.2, 3.2])
    ]

    for z in measurements:
        # 预测
        predicted = kf.predict()
        print(f"预测位置: {predicted}")

        # 更新
        kf.update(z)
        print(f"更新后位置: {kf.get_position()}")

    # 预测未来轨迹
    future_trajectory = kf.predict_trajectory(10)
    print(f"未来轨迹: {future_trajectory.shape}")

    print("卡尔曼滤波器模块测试完成")
