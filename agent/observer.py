# src/agents/observer_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.prompt_loader import load_prompt
from src.extractor import parse_observer_output
from src.client.observer_vlm import ObserverVLM


def build_observer_user_message(plan_list: str) -> str:
    """
    与 observer.txt 对齐：

    system(在 observer.txt 中) 负责说明：
    - 你是 EXECUTION OBSERVER
    - 输入模式：Plan list + Combined images
    - 任务：只判断 CURRENT subtask 是否已完成
    - 输出：严格的 <status> XML

    因此 user 只要把 plan list 提供给模型即可。
    """
    plan_list_clean = plan_list.strip() or "(empty plan list)"

    return (
        "Here is the current plan list for the task.\n"
        "PLAN LIST:\n"
        f"{plan_list_clean}\n\n"
    ).strip()


@dataclass
class ObserverResult:
    status: str   # "done" | "not_done"
    raw_xml: str  # 原始模型输出（便于 debug）


class ObserverAgent:
    """
    Execution Observer Agent（单轮）：

    - system_prompt: src/prompt/observer.txt
    - user: plan list 文本
    - images: 这一次要看的所有 frame 图像
    """

    def __init__(self, vlm: ObserverVLM) -> None:
        self.vlm = vlm
        self.system_prompt: str = load_prompt("observer")

    def run(
        self,
        image_paths: List[str],
        plan_list: str,
        max_tokens: int = 256,
    ) -> ObserverResult:
        user_text = build_observer_user_message(plan_list)

        raw_xml = self.vlm.call_once(
            system_prompt=self.system_prompt,
            user_text=user_text,
            image_paths=image_paths,
            max_tokens=max_tokens,
        )

        parsed = parse_observer_output(raw_xml)
        status = parsed.get("status", "not_done")

        return ObserverResult(status=status, raw_xml=raw_xml)