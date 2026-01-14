from typing import List, Dict, Any

from src.client.base_vlm_client import BaseVLMClient


class ObserverVLM:
    """
    专门给 Execution Observer 用的 VLM 封装：
    - 每次调用是“单轮”：system + user(plan text + images)
    - 不维持内部对话历史
    """

    def __init__(self, base_client: BaseVLMClient) -> None:
        self.base_client = base_client

    def call_once(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: List[str],
        max_tokens: int = 256,
    ) -> str:
        """
        单轮调用 observer：
        - system_prompt: 你的 observer.txt 内容
        - user_text:     这次 observer 需要的文本（plan list）
        - image_paths:   这次判断用到的所有图片路径
        """
        # 1. 构造 user content：先放文本，再放图片
        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": user_text,
            }
        ]

        for p in image_paths:
            image_part = self.base_client.encode_image_to_data_url(p)
            if image_part is not None:
                user_content.append(image_part)
            else:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"(注意：原本有一张图片 '{p}'，但加载失败)",
                    }
                )

        # 2. messages
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        # 3. 调用底层 client
        return self.base_client.chat(
            messages=messages,
            max_tokens=max_tokens,
        )