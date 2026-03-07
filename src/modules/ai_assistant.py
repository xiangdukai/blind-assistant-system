"""
AI辅助感知模块
使用多模态大语言模型(MLLM)对复杂场景进行分析描述
适用于非紧急场景的智能环境认知辅助
"""

import base64
import threading
import cv2
import numpy as np
from typing import Optional
from queue import Queue, Empty

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    import json, urllib.request
    _HAS_REQUESTS = False


class AIAssistant:
    """AI辅助感知器，通过云端MLLM分析复杂场景"""

    def __init__(self, config: dict):
        """
        初始化AI辅助感知器

        Args:
            config: 配置字典，包含 api_url、api_key、model 等
        """
        self.enabled = config.get('enabled', False)
        self.api_url = config.get('api_url', '')
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'gpt-4o-mini')
        self.prompt = config.get(
            'prompt',
            '你是用于帮助盲人出行的眼睛，关注图片中潜在的危险，'
            '用一句话简短准确描述这张图片需要注意的重点，'
            '不要笼统宽泛，必须只针对这张图片特点分析，回答'
        )
        self.max_tokens = config.get('max_tokens', 100)
        self.timeout = config.get('timeout', 10)
        self.jpeg_quality = config.get('jpeg_quality', 70)

        # 结果队列（最多缓存1条）
        self._result_queue: Queue = Queue(maxsize=1)
        self._last_result: Optional[str] = None
        self._is_analyzing = False

        if self.enabled:
            print("AI辅助感知模块已启用")
        else:
            print("AI辅助感知模块未启用（如需启用请在配置中设置 ai_assistant.enabled: true）")

    def analyze_async(self, image: np.ndarray):
        """
        异步分析图像（不阻塞主循环）

        Args:
            image: BGR彩色图像
        """
        if not self.enabled or not self.api_url or not self.api_key:
            return

        # 避免并发请求堆积
        if self._is_analyzing:
            return

        thread = threading.Thread(
            target=self._analyze_worker,
            args=(image.copy(),),
            daemon=True
        )
        thread.start()

    def get_result(self) -> Optional[str]:
        """
        获取最新分析结果（非阻塞，取出后清空）

        Returns:
            分析文字描述，或 None（无新结果）
        """
        try:
            result = self._result_queue.get_nowait()
            self._last_result = result
            return result
        except Empty:
            return None

    def _analyze_worker(self, image: np.ndarray):
        """工作线程：压缩图像并调用MLLM API"""
        self._is_analyzing = True
        try:
            # 压缩图像为JPEG，减少传输数据量
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            _, buf = cv2.imencode('.jpg', image, encode_params)
            img_b64 = base64.b64encode(buf).decode()

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            body = {
                'model': self.model,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': self.prompt},
                        {'type': 'image_url', 'image_url': {
                            'url': f'data:image/jpeg;base64,{img_b64}',
                            'detail': 'low'   # 降低 token 用量，low 分辨率足够场景描述
                        }}
                    ]
                }],
                'max_tokens': self.max_tokens,
                'temperature': 0.1,
                'top_p': 0.9
            }

            if _HAS_REQUESTS:
                resp = _requests.post(
                    self.api_url, headers=headers, json=body,
                    timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
            else:
                import json, urllib.request
                req = urllib.request.Request(
                    self.api_url,
                    data=json.dumps(body).encode('utf-8'),
                    headers=headers, method='POST'
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode('utf-8'))

            result = data['choices'][0]['message']['content'].strip()

            # 放入队列（丢弃旧未读结果）
            try:
                self._result_queue.get_nowait()
            except Empty:
                pass
            self._result_queue.put(result)

        except Exception as e:
            print(f"AI辅助感知请求失败: {e}")
        finally:
            self._is_analyzing = False
