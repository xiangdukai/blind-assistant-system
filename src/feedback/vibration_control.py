"""
震动控制模块
通过蓝牙或串口控制震动手环
"""

import time
import threading
from typing import Optional, Literal
from queue import Queue


class VibrationController:
    """震动控制器"""

    def __init__(self, config: dict):
        """
        初始化震动控制器

        Args:
            config: 配置字典
        """
        self.connection_type = config.get('connection_type', 'bluetooth')
        self.port = config.get('port', '')
        self.intensity_levels = config.get('intensity_levels', {
            'low': 100,
            'medium': 200,
            'high': 255
        })

        self.connection = None
        self.is_connected = False
        self.vibration_queue = Queue()

        # 初始化连接
        if self.connection_type == 'bluetooth':
            self._init_bluetooth()
        elif self.connection_type == 'serial':
            self._init_serial()

        # 启动震动控制线程
        self.control_thread = threading.Thread(target=self._control_worker, daemon=True)
        self.control_thread.start()

    def _init_bluetooth(self):
        """初始化蓝牙连接"""
        # TODO: 实现蓝牙连接
        print("蓝牙震动控制初始化")
        self.is_connected = False

    def _init_serial(self):
        """初始化串口连接"""
        try:
            import serial
            if self.port:
                self.connection = serial.Serial(self.port, 9600, timeout=1)
                self.is_connected = True
                print(f"串口连接成功: {self.port}")
            else:
                print("警告: 未指定串口端口")
                self.is_connected = False
        except Exception as e:
            print(f"串口连接失败: {e}")
            self.is_connected = False

    def vibrate(self, hand: Literal['left', 'right', 'both'],
               intensity: str = 'medium',
               duration: float = 0.3,
               pattern: Optional[list] = None):
        """
        触发震动

        Args:
            hand: 手环位置 ('left', 'right', 'both')
            intensity: 震动强度 ('low', 'medium', 'high')
            duration: 震动时长(秒)
            pattern: 震动模式，如 [0.2, 0.1, 0.2] 表示震0.2秒,停0.1秒,震0.2秒
        """
        vibration_cmd = {
            'hand': hand,
            'intensity': self.intensity_levels.get(intensity, 200),
            'duration': duration,
            'pattern': pattern
        }

        self.vibration_queue.put(vibration_cmd)

    def vibrate_danger(self, danger_info: dict):
        """
        根据危险信息触发震动

        Args:
            danger_info: 危险信息字典
        """
        level = danger_info['danger_level']
        direction = danger_info['direction']

        # 确定震动手环
        hand = 'both' if direction == 'front' else direction

        # 根据危险等级设置震动模式
        if level == 'high':
            # 连续急促震动
            pattern = [0.1, 0.05, 0.1, 0.05, 0.1, 0.05]
            self.vibrate(hand, 'high', pattern=pattern)
        elif level == 'medium':
            # 中等强度震动
            pattern = [0.2, 0.1, 0.2]
            self.vibrate(hand, 'medium', pattern=pattern)
        else:
            # 单次轻微震动
            self.vibrate(hand, 'low', 0.2)

    def vibrate_direction(self, direction: Literal['left', 'right', 'straight']):
        """
        方向引导震动

        Args:
            direction: 方向 ('left', 'right', 'straight')
        """
        if direction == 'left':
            self.vibrate('left', 'medium', 0.3)
        elif direction == 'right':
            self.vibrate('right', 'medium', 0.3)
        elif direction == 'straight':
            # 双手短震表示方向正确
            self.vibrate('both', 'low', 0.1)

    def _control_worker(self):
        """震动控制工作线程"""
        while True:
            try:
                cmd = self.vibration_queue.get()

                if cmd is None:
                    break

                self._execute_vibration(cmd)

            except Exception as e:
                print(f"震动控制错误: {e}")

    def _execute_vibration(self, cmd: dict):
        """
        执行震动命令

        Args:
            cmd: 震动命令字典
        """
        if not self.is_connected:
            # 降级方案：打印到控制台
            print(f"[震动] {cmd['hand']} 手, 强度: {cmd['intensity']}, "
                 f"时长: {cmd['duration']}秒")
            return

        hand = cmd['hand']
        intensity = cmd['intensity']
        pattern = cmd['pattern']

        if pattern is None:
            # 单次震动
            self._send_vibration_command(hand, intensity, cmd['duration'])
        else:
            # 按模式震动
            for i, duration in enumerate(pattern):
                if i % 2 == 0:
                    # 震动
                    self._send_vibration_command(hand, intensity, duration)
                else:
                    # 停止
                    time.sleep(duration)

    def _send_vibration_command(self, hand: str, intensity: int, duration: float):
        """
        发送震动命令到硬件

        Args:
            hand: 手环位置
            intensity: 强度值 (0-255)
            duration: 时长(秒)
        """
        if self.connection_type == 'serial' and self.connection is not None:
            try:
                # 构造命令协议（示例）
                # 格式: [hand_code, intensity, duration_ms]
                hand_code = {'left': 1, 'right': 2, 'both': 3}.get(hand, 0)
                duration_ms = int(duration * 1000)

                command = bytes([hand_code, intensity, duration_ms // 256, duration_ms % 256])
                self.connection.write(command)

                # 等待震动完成
                time.sleep(duration)

            except Exception as e:
                print(f"发送震动命令失败: {e}")

        elif self.connection_type == 'bluetooth':
            # TODO: 实现蓝牙震动命令
            time.sleep(duration)

    def stop(self):
        """停止所有震动"""
        # 清空队列
        while not self.vibration_queue.empty():
            try:
                self.vibration_queue.get_nowait()
            except:
                break

        # 发送停止命令
        if self.is_connected:
            self._send_vibration_command('both', 0, 0)

    def shutdown(self):
        """关闭震动控制器"""
        self.vibration_queue.put(None)
        self.control_thread.join(timeout=1)

        if self.connection is not None:
            try:
                self.connection.close()
            except:
                pass


if __name__ == "__main__":
    # 测试代码
    config = {
        'connection_type': 'serial',
        'port': '',  # 需要设置实际端口
        'intensity_levels': {
            'low': 100,
            'medium': 200,
            'high': 255
        }
    }

    controller = VibrationController(config)
    print("震动控制模块初始化成功")

    # 测试震动
    controller.vibrate('left', 'medium', 0.5)

    # 等待震动完成
    import time
    time.sleep(1)

    controller.shutdown()
