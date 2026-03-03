# src/client/planner_vlm.py
from typing import List, Dict, Any

from client.base_vlm_client import BaseVLMClient


class PlannerVLM:
    """
    给 Planner 使用的 VLM 封装：
    - 不维护对话历史
    - 不直接操作 Memory
    - 只负责把 messages 丢给 BaseVLMClient.chat
    """

    def __init__(self, base_client: BaseVLMClient) -> None:
        self.base_client = base_client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
    ) -> str:
        """
        - messages: 标准 OpenAI 风格的 messages 列表
        """
        return self.base_client.chat(
            messages=messages,
            max_tokens=max_tokens,
        )