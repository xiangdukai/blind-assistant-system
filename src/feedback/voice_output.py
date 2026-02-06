"""
语音输出模块
使用pyttsx3或OpenAI TTS进行语音合成
"""

import threading
from typing import Optional
from queue import Queue


class VoiceOutput:
    """语音输出器"""

    def __init__(self, config: dict):
        """
        初始化语音输出器

        Args:
            config: 配置字典
        """
        self.engine_type = config.get('engine', 'pyttsx3')
        self.language = config.get('language', 'zh-CN')
        self.rate = config.get('rate', 150)  # 语速
        self.volume = config.get('volume', 1.0)  # 音量

        self.engine = None
        self.speech_queue = Queue()  # 语音队列
        self.is_speaking = False

        # 初始化TTS引擎
        if self.engine_type == 'pyttsx3':
            self._init_pyttsx3()
        elif self.engine_type == 'openai':
            self._init_openai()

        # 启动语音播报线程
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

    def _init_pyttsx3(self):
        """初始化pyttsx3引擎"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)

            # 设置中文语音
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'mandarin' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break

            print("pyttsx3引擎初始化成功")
        except Exception as e:
            print(f"pyttsx3初始化失败: {e}")
            self.engine = None

    def _init_openai(self):
        """初始化OpenAI TTS"""
        # TODO: 实现OpenAI TTS初始化
        print("OpenAI TTS初始化")
        pass

    def speak(self, text: str, priority: str = 'normal'):
        """
        添加语音到播报队列

        Args:
            text: 要播报的文字
            priority: 优先级 ('high', 'normal', 'low')
        """
        if not text:
            return

        # 高优先级直接插入队列前面
        if priority == 'high':
            # 清空队列并立即播报
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except:
                    break
            self.speech_queue.put(text)
        else:
            self.speech_queue.put(text)

    def speak_immediately(self, text: str):
        """
        立即播报（高优先级）

        Args:
            text: 要播报的文字
        """
        self.speak(text, priority='high')

    def _speech_worker(self):
        """语音播报工作线程"""
        while True:
            try:
                text = self.speech_queue.get()

                if text is None:
                    break

                self.is_speaking = True
                self._speak_text(text)
                self.is_speaking = False

            except Exception as e:
                print(f"语音播报错误: {e}")
                self.is_speaking = False

    def _speak_text(self, text: str):
        """
        实际执行语音播报

        Args:
            text: 文字
        """
        if self.engine_type == 'pyttsx3' and self.engine is not None:
            self.engine.say(text)
            self.engine.runAndWait()
        elif self.engine_type == 'openai':
            self._speak_openai(text)
        else:
            # 降级方案：打印到控制台
            print(f"[语音播报] {text}")

    def _speak_openai(self, text: str):
        """
        使用OpenAI TTS播报

        Args:
            text: 文字
        """
        # TODO: 实现OpenAI TTS播报
        pass

    def stop(self):
        """停止当前播报"""
        if self.engine_type == 'pyttsx3' and self.engine is not None:
            self.engine.stop()

        # 清空队列
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except:
                break

    def shutdown(self):
        """关闭语音输出器"""
        self.speech_queue.put(None)
        self.speech_thread.join(timeout=1)


class DangerVoiceGenerator:
    """危险语音生成器"""

    @staticmethod
    def generate_danger_warning(danger_info: dict) -> str:
        """
        生成危险预警语音

        Args:
            danger_info: 危险信息

        Returns:
            语音文字
        """
        level = danger_info['danger_level']
        time = danger_info['time_to_collision']
        direction = danger_info['direction']

        direction_text = {
            'front': '正前方',
            'left': '左前方',
            'right': '右前方'
        }.get(direction, '前方')

        if level == 'high':
            return f"危险！{direction_text}有物体快速接近，立即停止！"
        elif level == 'medium':
            return f"注意，{direction_text}有物体接近，{time:.1f}秒后可能碰撞"
        else:
            return f"{direction_text}有移动物体"

    @staticmethod
    def generate_stair_guidance(stair_info: dict) -> str:
        """
        生成楼梯引导语音

        Args:
            stair_info: 楼梯信息

        Returns:
            语音文字
        """
        direction = stair_info['direction']
        num_steps = stair_info['num_steps']

        direction_text = '上楼' if direction == 'up' else '下楼'
        return f"前方{direction_text}，约{num_steps}级台阶"

    @staticmethod
    def generate_crosswalk_guidance(crosswalk_info: dict) -> str:
        """
        生成斑马线引导语音

        Args:
            crosswalk_info: 斑马线信息

        Returns:
            语音文字
        """
        traffic_light = crosswalk_info['traffic_light']
        is_safe = crosswalk_info['is_safe']

        light_text = {
            'green': '绿灯',
            'red': '红灯',
            'yellow': '黄灯'
        }.get(traffic_light, '信号灯状态未知')

        if is_safe:
            return f"前方斑马线，{light_text}，可以通行"
        else:
            return f"前方斑马线，{light_text}，请等待"


if __name__ == "__main__":
    # 测试代码
    config = {
        'engine': 'pyttsx3',
        'language': 'zh-CN',
        'rate': 150,
        'volume': 1.0
    }

    voice = VoiceOutput(config)
    print("语音输出模块初始化成功")

    # 测试播报
    voice.speak("语音输出测试")

    # 等待播报完成
    import time
    time.sleep(2)

    voice.shutdown()
