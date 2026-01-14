from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from src.prompt_loader import load_prompt
from src.client.planner_vlm import PlannerVLM
from src.extractor import parse_planner_output, MemoryOperation
from src.memory.recorder import Memory


# ========== 构造一轮输入文本 ==========

def build_planner_user_message(
    user_instruction: Optional[str],
    current_plan_list: str,
    image_summary: Optional[str] = None,
) -> str:
    parts: List[str] = []

    if user_instruction:
        parts.append("USER INSTRUCTION:")
        parts.append(user_instruction.strip())
        parts.append("")

    plan_clean = current_plan_list.strip()
    if plan_clean:
        parts.append("CURRENT PLAN LIST:")
        parts.append(plan_clean)
        parts.append("")
    else:
        parts.append("CURRENT PLAN LIST:")
        parts.append("(empty)")
        parts.append("")

    if image_summary:
        parts.append("COMBINED IMAGES SUMMARY:")
        parts.append(image_summary.strip())
        parts.append("")

    parts.append(
        "You must respond strictly in the XML format defined in the system prompt, "
        "including <summary>, <memory_operations>, and <plan_list> sections."
    )

    return "\n".join(parts).strip()


# ========== 输出结构 ==========

@dataclass
class PlannerResult:
    summary: str
    memory_operations: List[MemoryOperation]
    plan_text: str      # 本次 refine 结束后的“最新计划”（即 self.plan_list）
    raw_xml: str


# ========== 核心 Agent ==========

class PlannerAgent:
    """
    PlannerAgent 的语义：

    - self.plan_list 代表整个任务当前“全局最新计划版本”。
    - 一次 run_refine() = 一次多轮对话 + memory 交互，用来对 plan 进行一次 refine。
    - 一个 task 完成 = 你在上层逻辑中多次调用 run_refine()。
    """

    def __init__(self, vlm: PlannerVLM, memory: Optional[Memory] = None) -> None:
        self.vlm = vlm
        self.memory = memory

        self.system_prompt: str = load_prompt("planner")
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        # 当前任务的“全局最新计划”
        self.plan_list: str = ""

        # 仅用于本次对话中，传递上一次 Memory QUERY 的结果
        self.last_query_results_text: str = ""

    # ---------- 状态操作 ----------

    def reset(self) -> None:
        """
        完全重置会话和内部状态（包括全局计划）。
        仅在你真的要“从头开始一个全新任务”时调用。
        """
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.plan_list = ""
        self.last_query_results_text = ""

    def set_plan(self, plan_text: str) -> None:
        """
        人工或外部逻辑直接指定当前“全局最新计划”。
        后续 refine 将在此基础上继续。
        """
        self.plan_list = (plan_text or "").strip()

    # ---------- 内部：Memory 操作 ----------

    @staticmethod
    def _safe_parse_rec_id(id_str: Optional[str]) -> Optional[int]:
        if id_str is None:
            return None
        try:
            return int(id_str.strip())
        except (ValueError, TypeError):
            return None

    def _apply_memory_operations(
        self,
        operations: List[MemoryOperation],
    ) -> str:
        """
        执行一轮内所有 memory ops，并返回“本轮所有 QUERY 的结果文本”，
        供下一轮 step 使用。
        """
        if self.memory is None:
            return ""

        query_results_lines: List[str] = []

        for op in operations:
            t = (op.type or "").upper().strip()

            # --- QUERY ---
            if t == "QUERY":
                if not op.reason:
                    continue
                records, scores = self.memory.query(
                    content=op.reason,
                    top_k=5,
                )
                query_results_lines.append(f"QUERY: {op.text}")
                if not records:
                    query_results_lines.append("  (no matching records)")
                else:
                    for rec in records:
                        score = scores.get(rec.id, 0.0)
                        data_type = rec.data.get("type")
                        value = rec.data.get("value")
                        if data_type == "text":
                            txt = str(value)
                        else:
                            txt = f"[{data_type}] {rec.obj_name}: {value}"
                        query_results_lines.append(
                            f"  id={rec.id}, obj_name={rec.obj_name}, "
                            f"score={score:.3f}, text={txt}"
                        )

            # --- CREATE ---
            elif t == "CREATE":
                if not op.obj_name or not op.text:
                    continue
                self.memory.create(
                    obj_name=op.obj_name,
                    data_type="text",
                    data_value=op.text,
                    text=op.text,
                )

            # --- UPDATE ---
            elif t == "UPDATE":
                rec_id = self._safe_parse_rec_id(op.id)
                if rec_id is None:
                    continue
                self.memory.update(
                    rec_id=rec_id,
                    obj_name=op.obj_name,
                    text=op.text,
                )

            # --- DELETE ---
            elif t == "DELETE":
                rec_id = self._safe_parse_rec_id(op.id)
                if rec_id is None:
                    continue
                self.memory.delete(rec_id=rec_id)

        return "\n".join(query_results_lines).strip()

    # ---------- 一轮 step（一次模型调用） ----------

    def step(
        self,
        user_instruction: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        current_plan_list: Optional[str] = None,
        max_tokens: int = 512,
        image_summary: Optional[str] = None,
        round_idx: int = 1
    ) -> PlannerResult:
        """
        一次模型调用（在一次 refine 内可能会有多次 step）：

        - 输入的 CURRENT PLAN =
            - 若 current_plan_list 不为 None，则用它（可以是 ""，表示从零开始）；
            - 否则用 self.plan_list（全局最新版本）。
        - 输出的 <plan_list> 若非空：
            - 直接覆盖 self.plan_list（代表这一次调用产生了一个新的“全局最新计划版本”）。
        """

        # 1) 选择本轮输入给模型的 plan：显式 current_plan_list 优先，其次用历史 self.plan_list
        if current_plan_list is None:
            effective_plan = self.plan_list
        else:
            effective_plan = current_plan_list  # 保留 "" 的语义

        user_text = f'Turn {round_idx}:\n'
        # 2) 构建文本提示
        if image_paths:
            user_text += build_planner_user_message(
                user_instruction=user_instruction,
                current_plan_list=effective_plan,
                image_summary=image_summary,
            )
        else:
            user_text = ""

        # 把上一轮 query 结果也附带进去
        if self.last_query_results_text:
            user_text += (
                "\n\nRESULTS OF PREVIOUS MEMORY QUERIES (for your reference):\n"
                + self.last_query_results_text
            )

        # 3) 组装 content（文本 + 图像）
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]

        if image_paths:
            for p in image_paths:
                if not p:
                    continue
                img_block = self.vlm.base_client.encode_image_to_data_url(p)
                if img_block is not None:
                    content.append(img_block)

        # 4) 写入对话历史 & 调模型
        self.messages.append({"role": "user", "content": content})
        raw_xml = self.vlm.chat(messages=self.messages, max_tokens=max_tokens)
        self.messages.append({"role": "assistant", "content": raw_xml})

        # import pdb; pdb.set_trace()
        # 5) 解析 XML
        parsed = parse_planner_output(raw_xml)
        summary: str = parsed.get("summary", "") or ""
        mem_ops: List[MemoryOperation] = parsed.get("memory_operations", []) or []
        new_plan: str = parsed.get("plan_list", "") or ""

        # 6) 执行 memory ops，拿到 query 结果
        self.last_query_results_text = self._apply_memory_operations(mem_ops)

        # print(f'assistant: {raw_xml}')
        # print(f'user: {self.last_query_results_text}')
        # 7) 如果模型这轮给出了非空的 <plan_list>，则更新“全局最新计划”
        new_plan_stripped = new_plan.strip()
        if new_plan_stripped:
            self.plan_list = new_plan_stripped

        return PlannerResult(
            summary=summary,
            memory_operations=mem_ops,
            plan_text=self.plan_list,
            raw_xml=raw_xml,
        )

    # ---------- 一次 refine（一次多轮对话） ----------

    def run_refine(
        self,
        image_paths: List[str],
        initial_plan_list: Optional[str] = None,
        user_instruction: Optional[str] = None,
        max_tokens: int = 512,
        max_inner_rounds: int = 10,
    ) -> PlannerResult:
        """
        一次 refine：在当前全局 plan 基础上（或显式 initial_plan_list），
        通过至多 max_inner_rounds 次 step + memory 交互，得到一个新的 plan 版本。

        - 第 1 轮：
            CURRENT PLAN =
                - initial_plan_list（若不为 None；可以是 "" 表示从零开始），
                - 否则用 self.plan_list（全局历史版本）。
            传入用户这次 refine 的指令 + 图片。
        - 后续轮：
            CURRENT PLAN = self.plan_list（上一轮 step 更新后的版本）
            不再传图片，一般也不再传 user_instruction（可选）。
        - 结束条件（单次 refine 视角）：
            - 某一轮 step 之后 self.plan_list 非空（说明有计划版本产生/更新），
            - 或达到 max_inner_rounds 上限。
        """

        last_result: Optional[PlannerResult] = None
        self.last_query_results_text = ""  # 这一次 refine 内重置 query 结果

        for round_idx in range(max_inner_rounds):
            if round_idx == 0:
                # 第 1 轮：确定起点计划
                if initial_plan_list is not None:
                    cur_plan_for_step = initial_plan_list  # "" => 从零生成
                else:
                    cur_plan_for_step = self.plan_list      # 历史版本
                cur_instr = user_instruction
                cur_images = image_paths
            else:
                # 后续轮：在最新 plan 上继续，通常不再传图片和额外指令
                cur_plan_for_step = None
                cur_instr = None
                cur_images = None

            last_result = self.step(
                user_instruction=cur_instr,
                image_paths=cur_images,
                current_plan_list=cur_plan_for_step,
                max_tokens=max_tokens,
                image_summary=None,
                round_idx=round_idx+1
            )

            # 在一次 refine 里：一旦有非空 plan，就认为这一轮 refine 有结果，可以结束
            if last_result.plan_text.strip():
                break

        # 即使最后没有产生非空计划（极端情况），也返回最后一轮结果
        return last_result