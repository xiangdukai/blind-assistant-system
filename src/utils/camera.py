"""
相机接口模块
支持RealSense深度相机和普通摄像头
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class Camera:
    """相机基类"""

    def __init__(self, config: dict):
        """
        初始化相机

        Args:
            config: 配置字典
        """
        self.camera_type = config.get('type', 'webcam')
        self.width = config.get('resolution', {}).get('width', 640)
        self.height = config.get('resolution', {}).get('height', 480)
        self.fps = config.get('fps', 30)
        self.depth_enabled = config.get('depth_enabled', False)

        self.is_opened = False

    def open(self) -> bool:
        """
        打开相机

        Returns:
            是否成功
        """
        raise NotImplementedError

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        读取一帧

        Returns:
            (成功标志, 彩色图像, 深度图像)
        """
        raise NotImplementedError

    def release(self):
        """释放相机资源"""
        raise NotImplementedError

    def get_intrinsics(self) -> Optional[dict]:
        """
        获取相机内参

        Returns:
            相机内参字典
        """
        return None


class WebcamCamera(Camera):
    """普通USB摄像头"""

    def __init__(self, config: dict):
        """
        初始化USB摄像头

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.cap = None

    def open(self) -> bool:
        """打开摄像头"""
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("无法打开摄像头")
            return False

        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.is_opened = True
        print(f"摄像头打开成功: {self.width}x{self.height}@{self.fps}fps")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """读取一帧"""
        if not self.is_opened or self.cap is None:
            return False, None, None

        ret, frame = self.cap.read()

        if not ret:
            return False, None, None

        # 普通摄像头没有深度图
        return True, frame, None

    def release(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("摄像头已释放")


class RealSenseCamera(Camera):
    """Intel RealSense深度相机"""

    def __init__(self, config: dict):
        """
        初始化RealSense相机

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.pipeline = None
        self.align = None

        try:
            import pyrealsense2 as rs
            self.rs = rs
        except ImportError:
            print("警告: pyrealsense2未安装,RealSense相机功能不可用")
            self.rs = None

    def open(self) -> bool:
        """打开RealSense相机"""
        if self.rs is None:
            print("pyrealsense2未安装")
            return False

        try:
            # 创建pipeline
            self.pipeline = self.rs.pipeline()
            config = self.rs.config()

            # 配置彩色流
            config.enable_stream(self.rs.stream.color,
                               self.width, self.height,
                               self.rs.format.bgr8, self.fps)

            # 配置深度流
            if self.depth_enabled:
                config.enable_stream(self.rs.stream.depth,
                                   self.width, self.height,
                                   self.rs.format.z16, self.fps)

            # 启动pipeline
            self.pipeline.start(config)

            # 创建align对象，用于对齐深度图和彩色图
            if self.depth_enabled:
                self.align = self.rs.align(self.rs.stream.color)

            self.is_opened = True
            print(f"RealSense相机打开成功: {self.width}x{self.height}@{self.fps}fps")
            return True

        except Exception as e:
            print(f"RealSense相机打开失败: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """读取一帧"""
        if not self.is_opened or self.pipeline is None:
            return False, None, None

        try:
            # 等待帧
            frames = self.pipeline.wait_for_frames()

            # 对齐深度图和彩色图
            if self.depth_enabled and self.align is not None:
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
            else:
                color_frame = frames.get_color_frame()
                depth_frame = None

            if not color_frame:
                return False, None, None

            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())

            depth_image = None
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())

            return True, color_image, depth_image

        except Exception as e:
            print(f"读取帧失败: {e}")
            return False, None, None

    def release(self):
        """释放资源"""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.is_opened = False
            print("RealSense相机已释放")

    def get_intrinsics(self) -> Optional[dict]:
        """获取相机内参"""
        if not self.is_opened or self.pipeline is None:
            return None

        try:
            profile = self.pipeline.get_active_profile()
            color_stream = profile.get_stream(self.rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            return {
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'cx': intrinsics.ppx,
                'cy': intrinsics.ppy,
                'width': intrinsics.width,
                'height': intrinsics.height
            }
        except Exception as e:
            print(f"获取内参失败: {e}")
            return None


def create_camera(config: dict) -> Camera:
    """
    根据配置创建相机对象

    Args:
        config: 配置字典

    Returns:
        相机对象
    """
    camera_type = config.get('type', 'webcam')

    if camera_type == 'realsense':
        return RealSenseCamera(config)
    elif camera_type == 'webcam':
        return WebcamCamera(config)
    else:
        print(f"未知的相机类型: {camera_type}, 使用默认摄像头")
        return WebcamCamera(config)


if __name__ == "__main__":
    # 测试代码
    config = {
        'type': 'webcam',
        'resolution': {'width': 640, 'height': 480},
        'fps': 30,
        'depth_enabled': False
    }

    camera = create_camera(config)

    if camera.open():
        print("相机测试...")

        for i in range(10):
            ret, color, depth = camera.read()
            if ret:
                print(f"帧 {i+1}: 彩色图 {color.shape}")
                if depth is not None:
                    print(f"         深度图 {depth.shape}")

        camera.release()
