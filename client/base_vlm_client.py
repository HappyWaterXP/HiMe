import base64
import mimetypes
from typing import List, Dict, Any, Optional

import openai


class BaseVLMClient:
    """
    封装对 openai.ChatCompletion 的基础调用:
    - 提供通用的 messages 列表接口
    - 提供 base64 image_url 辅助函数
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        client: Optional[openai.OpenAI] = None,
    ) -> None:
        self.model = model
        self.client = client or openai.OpenAI()

    @staticmethod
    def encode_image_to_data_url(image_path: str) -> Optional[Dict[str, Any]]:
        """
        将本地图片编码为 data URL 形式的 image_url dict:
        {
            "type": "image_url",
            "image_url": {
                "url": "data:<mime_type>;base64,<data>"
            }
        }
        """
        import os

        if not image_path:
            return None

        if not os.path.exists(image_path):
            print(f"[BaseVLMClient] 警告：图片不存在: {image_path}")
            return None

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image"):
            print(
                f"[BaseVLMClient] 警告：无法确定 '{image_path}' 的图片类型，默认 'image/jpeg'"
            )
            mime_type = "image/jpeg"

        with open(image_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{b64_str}"
            },
        }

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
    ) -> str:
        """
        通用 chat 调用：
        - messages: 标准 OpenAI chat messages（可以包含 image_url）
        - 返回：模型回复的字符串 content
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""