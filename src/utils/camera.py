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


class StereoBinocularCamera(Camera):
    """USB双目摄像头（输出 1280×480 左右拼合图像，通过SGBM计算深度）"""

    # 出厂标定参数（来自商家提供的参考代码）
    LEFT_CAMERA_MATRIX = np.array([[416.841180253704, 0.0, 338.485167779639],
                                   [0., 416.465934495134, 230.419201769346],
                                   [0., 0., 1.]])
    LEFT_DISTORTION = np.array([[-0.0170280933781798, 0.0643596519467521,
                                  -0.00161785356900972, -0.00330684695473645, 0]])
    RIGHT_CAMERA_MATRIX = np.array([[417.765094485395, 0.0, 315.061245379892],
                                    [0., 417.845058291483, 238.181766936442],
                                    [0., 0., 1.]])
    RIGHT_DISTORTION = np.array([[-0.0394089328586398, 0.131112076868352,
                                   -0.00133793245429668, -0.00188957913931929, 0]])
    R = np.array([[0.999962872853149, 0.00187779299260463, -0.00840992323112715],
                  [-0.0018408858041373, 0.999988651353238, 0.00439412154902114],
                  [0.00841807904053251, -0.00437847669953504, 0.999954981430194]])
    T = np.array([[-120.326603502087], [0.199732192805711], [-0.203594457929446]])

    def __init__(self, config: dict):
        super().__init__(config)
        self.cap = None
        self.camera_index = config.get('camera_index', 1)

        # 预计算立体校正映射表
        size = (self.width, self.height)
        R1, R2, P1, P2, self.Q, self.valid_roi1, self.valid_roi2 = cv2.stereoRectify(
            self.LEFT_CAMERA_MATRIX, self.LEFT_DISTORTION,
            self.RIGHT_CAMERA_MATRIX, self.RIGHT_DISTORTION,
            size, self.R, self.T
        )
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.LEFT_CAMERA_MATRIX, self.LEFT_DISTORTION, R1, P1, size, cv2.CV_16SC2)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.RIGHT_CAMERA_MATRIX, self.RIGHT_DISTORTION, R2, P2, size, cv2.CV_16SC2)

        # SGBM 立体匹配器（与厂家参考代码保持一致）
        blockSize = 8
        img_channels = 3
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=1,
            numDisparities=64,
            blockSize=blockSize,
            P1=8 * img_channels * blockSize * blockSize,
            P2=32 * img_channels * blockSize * blockSize,
            disp12MaxDiff=-1,
            preFilterCap=140,
            uniquenessRatio=1,
            speckleWindowSize=100,
            speckleRange=100,
            mode=cv2.STEREO_SGBM_MODE_HH
        )

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"无法打开双目摄像头 (index={self.camera_index})")
            return False

        # 双目摄像头宽度是单目两倍
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width * 2)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 每只眼睛的实际宽度（总宽/2）
        self._eye_w = actual_w // 2
        self._eye_h = actual_h
        print(f"双目摄像头打开成功: 总分辨率 {actual_w}x{actual_h}，每眼 {self._eye_w}x{self._eye_h}")
        if self._eye_w != self.width or self._eye_h != self.height:
            print(f"  注意：实际分辨率与标定尺寸 {self.width}x{self.height} 不符，将自动缩放")
        self.is_opened = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.is_opened or self.cap is None:
            return False, None, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None, None

        # 分割左右图像（保留原始分辨率）
        left_bgr = frame[:, :self._eye_w, :]
        right_bgr = frame[:, self._eye_w:self._eye_w * 2, :]

        # 缩放到标定尺寸（640×480）进行立体匹配
        cal_w, cal_h = self.width, self.height
        if self._eye_w != cal_w or self._eye_h != cal_h:
            left_cal = cv2.resize(left_bgr, (cal_w, cal_h))
            right_cal = cv2.resize(right_bgr, (cal_w, cal_h))
        else:
            left_cal, right_cal = left_bgr, right_bgr

        # 转灰度 → 立体校正（在标定尺寸下）
        imgL = cv2.cvtColor(left_cal, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(right_cal, cv2.COLOR_BGR2GRAY)
        imgL_rect = cv2.remap(imgL, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        imgR_rect = cv2.remap(imgR, self.right_map1, self.right_map2, cv2.INTER_LINEAR)

        # 计算视差图（标定尺寸下）
        disparity = self.stereo.compute(imgL_rect, imgR_rect)
        three_d = cv2.reprojectImageTo3D(disparity, self.Q, handleMissingValues=True)
        three_d = three_d * 16
        depth_cal = three_d[:, :, 2].astype(np.float32) / 1000.0  # cal_w×cal_h 深度图

        # 彩色图直接用原始帧（不做立体校正 remap，避免图像弯曲变形）
        # 轻度锐化：补偿镜头软焦
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        color_image = cv2.filter2D(left_bgr, -1, kernel)
        if self._eye_w != cal_w or self._eye_h != cal_h:
            depth_map = cv2.resize(depth_cal, (self._eye_w, self._eye_h),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            depth_map = depth_cal

        return True, color_image, depth_map

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("双目摄像头已释放")

    def get_depth_colormap(self) -> Optional[np.ndarray]:
        """返回深度伪彩色图（仅供调试）"""
        ret, _, depth = self.read()
        if not ret or depth is None:
            return None
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)


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
    elif camera_type == 'stereo':
        return StereoBinocularCamera(config)
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
