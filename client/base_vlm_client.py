import base64
import mimetypes
import json
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

    @staticmethod
    def _extract_text_from_response(resp: Any) -> str:
        """
        Tolerant parser for OpenAI-compatible and non-standard providers.
        """
        # Some gateways may return raw text directly.
        if isinstance(resp, str):
            return resp

        # Standard OpenAI SDK response object.
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            pass

        # Dict-like response from some compatible backends.
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    text_parts: List[str] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text")
                            if isinstance(text, str):
                                text_parts.append(text)
                    if text_parts:
                        return "\n".join(text_parts)
            return json.dumps(resp, ensure_ascii=False)

        # Last-resort stringify to avoid crashing caller pipeline.
        return str(resp)

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
        return self._extract_text_from_response(resp)
