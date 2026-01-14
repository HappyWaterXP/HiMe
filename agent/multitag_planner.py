# src/agent/multitag_planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import re
import json
from datetime import datetime

from src.prompt_loader import load_prompt
from src.client.planner_vlm import PlannerVLM
from src.extractor import parse_planner_output, MemoryOperation
from src.memory.multitag_recorder import MultiTagMemory


# ========== 辅助函数：解析图片序号 ==========

def parse_image_indices(image_path_str: Optional[str]) -> List[int]:
    """
    从模型输出的 image_path 字段解析图片序号
    
    支持格式：
    - "1"
    - "1, 3"
    - "[1, 3, 5]"
    - "1,3,5"
    
    Returns:
        图片索引列表（0-based）
    """
    if not image_path_str:
        return []
    
    cleaned = image_path_str.strip().strip('[]')
    numbers = re.findall(r'\d+', cleaned)
    
    indices = []
    for num_str in numbers:
        try:
            idx = int(num_str) - 1  # 转换为 0-based
            if idx >= 0:
                indices.append(idx)
        except ValueError:
            continue
    
    return indices


# ========== 输出结构 ==========

@dataclass
class PlannerResult:
    summary: str
    memory_operations: List[MemoryOperation]
    plan_text: str
    refined_plan_text: str
    raw_xml: str


# ========== 核心 Agent ==========

class PlannerAgent:
    """
    PlannerAgent with MultiTagMemory and image support
    
    ✅ 核心功能：
    1. CREATE/UPDATE：模型输出图片序号 -> 代码映射到真实路径
    2. QUERY 结果：返回完整信息（tags + text），图片路径单独收集
    3. 图片序号从 1 开始（用户视角），内部转换为 0-based 索引
    
    ✅ 多轮对话逻辑：
    - Turn 1: 发送完整任务（instruction + plan + 原始图片）
    - Turn 2+: 发送查询结果（文本 + 检索到的图片编码）
    """

    def __init__(self, vlm: PlannerVLM, memory: Optional[MultiTagMemory] = None) -> None:
        self.vlm = vlm
        self.memory = memory

        self.system_prompt: str = load_prompt("multitag_planner")
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        self.plan_list: str = ""
        self.last_query_results_text: str = ""
        self.last_query_image_paths: List[str] = []
        
        # ✅ 当前 refine 会话的原始输入图片路径列表
        self.current_input_image_paths: List[str] = []

    # ---------- 状态操作 ----------

    def reset(self) -> None:
        """完全重置会话和内部状态"""
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.plan_list = ""
        self.last_query_results_text = ""
        self.last_query_image_paths = []
        self.current_input_image_paths = []

    def set_plan(self, plan_text: str) -> None:
        """人工指定当前全局最新计划"""
        self.plan_list = (plan_text or "").strip()

    # ---------- 图片处理 ----------

    def _resolve_image_paths(self, image_path_str: Optional[str]) -> List[str]:
        """
        将模型输出的图片序号转换为真实路径
        
        Args:
            image_path_str: 模型输出的序号字符串，如 "1" 或 "1,3,5"
            
        Returns:
            真实图片路径列表
        """
        if not image_path_str:
            return []
        
        indices = parse_image_indices(image_path_str)
        
        resolved_paths = []
        for idx in indices:
            if 0 <= idx < len(self.current_input_image_paths):
                resolved_paths.append(self.current_input_image_paths[idx])
            else:
                print(f"⚠️ Warning: Image index {idx+1} out of range (total: {len(self.current_input_image_paths)})")
        
        return resolved_paths

    # ---------- Memory 操作 ----------

    @staticmethod
    def _safe_parse_rec_id(id_str: Optional[str]) -> Optional[int]:
        """安全地解析记录 ID"""
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
        Execute memory operations and return all results as text.

        ✅ Key features:
        1. CREATE/UPDATE/DELETE: Return success/failure messages
        2. QUERY: Return complete record information (tags + text), collect image paths
        3. Image paths are saved to self.last_query_image_paths for encoding in step()
        """
        if self.memory is None:
            return ""

        result_lines: List[str] = []
        query_image_paths: List[str] = []

        for op in operations:
            t = (op.type or "").upper().strip()

            # --- QUERY ---
            if t == "QUERY":
                query_content = op.query
                if not query_content:
                    result_lines.append("\n❌ QUERY: Missing query content")
                    continue

                try:
                    records, scores = self.memory.query(
                        content=query_content,
                        top_k=5,
                    )

                    result_lines.append(f"\n--- QUERY: {query_content} ---")

                    if not records:
                        result_lines.append("  (No matching records found)")
                    else:
                        for rec in records:
                            score = scores.get(rec.id, 0.0)
                            data_type = rec.data.get("type")
                            value = rec.data.get("value")
                            tags_str = ", ".join(rec.tags)

                            # Display record basic info
                            result_lines.append(
                                f"  Record ID={rec.id} | Tags=[{tags_str}] | Score={score:.3f}"
                            )

                            # Display text/description content
                            if value:
                                text_preview = str(value)[:200]
                                if len(str(value)) > 200:
                                    text_preview += "..."
                                result_lines.append(f"    Content: {text_preview}")

                            # If has image, collect path (will be encoded later)
                            if data_type == "image" and rec.image_path:
                                query_image_paths.append(rec.image_path)
                                result_lines.append(
                                    f"    [Image: {Path(rec.image_path).name}]"
                                )
                except Exception as e:
                    result_lines.append(f"\n❌ QUERY failed: {str(e)}")
                    print(f"[Memory] QUERY error: {e}")

            # --- CREATE ---
            elif t == "CREATE":
                tags_to_use = op.tags if op.tags else ([op.obj_name] if op.obj_name else None)

                if not tags_to_use:
                    result_lines.append(f"\n❌ CREATE failed: Missing tags")
                    continue

                if not op.text:
                    result_lines.append(f"\n❌ CREATE failed: Missing text content")
                    continue

                try:
                    # Resolve image indices to real paths
                    resolved_paths = self._resolve_image_paths(op.image_path)

                    # Check if user specified image_path but resolution failed
                    if op.image_path and not resolved_paths:
                        result_lines.append(
                            f"\n⚠️  CREATE: Specified image_path='{op.image_path}' but no valid images found, creating text record instead"
                        )

                    if resolved_paths:
                        # Create one record per image
                        created_ids = []
                        for img_path in resolved_paths:
                            rec = self.memory.create(
                                tags=tags_to_use,
                                data_type="image",
                                data_value=op.text,
                                text=op.text,
                                image_path=img_path,
                            )
                            created_ids.append(rec.id)

                        tags_str = ", ".join(tags_to_use)
                        result_lines.append(
                            f"\n✅ CREATE: Created {len(created_ids)} image record(s) with IDs {created_ids} | Tags=[{tags_str}]"
                        )
                    else:
                        # Pure text record
                        rec = self.memory.create(
                            tags=tags_to_use,
                            data_type="text",
                            data_value=op.text,
                            text=op.text,
                            image_path=None,
                        )
                        tags_str = ", ".join(tags_to_use)
                        result_lines.append(
                            f"\n✅ CREATE: Created text record ID={rec.id} | Tags=[{tags_str}]"
                        )
                except Exception as e:
                    result_lines.append(f"\n❌ CREATE failed: {str(e)}")
                    print(f"[Memory] CREATE error: {e}")

            # --- UPDATE ---
            elif t == "UPDATE":
                rec_id = self._safe_parse_rec_id(op.id)
                if rec_id is None:
                    result_lines.append(f"\n❌ UPDATE failed: Invalid record ID '{op.id}'")
                    continue

                try:
                    tags_to_use = op.tags if op.tags else ([op.obj_name] if op.obj_name else None)

                    # Resolve image indices to real paths
                    resolved_paths = self._resolve_image_paths(op.image_path)

                    # UPDATE only uses first image (if multiple)
                    final_image_path = resolved_paths[0] if resolved_paths else None

                    updated_rec = self.memory.update(
                        rec_id=rec_id,
                        tags=tags_to_use,
                        text=op.text,
                        image_path=final_image_path,
                    )

                    if updated_rec:
                        tags_str = ", ".join(updated_rec.tags) if updated_rec.tags else "N/A"
                        result_lines.append(
                            f"\n✅ UPDATE: Updated record ID={rec_id} | Tags=[{tags_str}]"
                        )
                    else:
                        result_lines.append(f"\n❌ UPDATE failed: Record ID={rec_id} not found")
                except Exception as e:
                    result_lines.append(f"\n❌ UPDATE failed for ID={rec_id}: {str(e)}")
                    print(f"[Memory] UPDATE error: {e}")

            # --- DELETE ---
            elif t == "DELETE":
                rec_id = self._safe_parse_rec_id(op.id)
                if rec_id is None:
                    result_lines.append(f"\n❌ DELETE failed: Invalid record ID '{op.id}'")
                    continue

                try:
                    success = self.memory.delete(rec_id=rec_id)
                    if success:
                        result_lines.append(f"\n✅ DELETE: Deleted record ID={rec_id}")
                    else:
                        result_lines.append(f"\n❌ DELETE failed: Record ID={rec_id} not found")
                except Exception as e:
                    result_lines.append(f"\n❌ DELETE failed for ID={rec_id}: {str(e)}")
                    print(f"[Memory] DELETE error: {e}")

        # Save queried image paths for encoding in step()
        self.last_query_image_paths = query_image_paths

        return "\n".join(result_lines).strip()

    # ---------- 构造 User Prompt ----------

    def _build_user_prompt_turn1(
        self,
        instruction: str,
        plan_list: str,
        image_count: int
    ) -> str:
        """
        构建 Turn 1 的 User Prompt
        
        包含：完整任务描述 + 当前计划 + 原始图片说明
        """
        prompt_parts = ["=== TURN 1 ===\n"]
        
        prompt_parts.append("USER INSTRUCTION:")
        prompt_parts.append(instruction)
        
        prompt_parts.append("\nCURRENT PLAN:")
        if plan_list and plan_list.strip():
            prompt_parts.append(plan_list)
        else:
            prompt_parts.append("(No existing plan)")
        
        if image_count > 0:
            prompt_parts.append(f"\nINPUT IMAGES: {image_count} image(s) provided below, numbered 1 to {image_count}")
            prompt_parts.append('💡 When creating/updating memory with images, use: image_path="1" or image_path="2,3"')
        else:
            prompt_parts.append("\nINPUT IMAGES: None")
        
        prompt_parts.append("\n📋 REQUIRED OUTPUT FORMAT:")
        prompt_parts.append("You must respond in XML format with <summary>, <memory_operations>, and <plan_list> sections.")
        
        return "\n".join(prompt_parts)

    def _build_user_prompt_turn_n(
        self,
        turn_num: int,
        query_results_text: str
    ) -> str:
        """
        构建 Turn 2+ 的 User Prompt
        
        包含：上一轮的查询结果（文本 + 图片会在后面附加）
        """
        prompt_parts = [f"=== TURN {turn_num} ===\n"]
        
        if query_results_text:
            prompt_parts.append("=== MEMORY QUERY RESULTS ===")
            prompt_parts.append(query_results_text)
            prompt_parts.append("\n📌 Note: Retrieved images (if any) are shown below this text.")
        else:
            prompt_parts.append("=== NO MEMORY QUERIES PERFORMED ===")
            prompt_parts.append("Previous operations completed without queries.")
        
        prompt_parts.append("\n📋 NEXT STEPS:")
        prompt_parts.append("Based on the query results above, continue refining your plan and performing necessary memory operations.")
        prompt_parts.append("\n📋 REQUIRED OUTPUT FORMAT:")
        prompt_parts.append("You must respond in XML format with <summary>, <memory_operations>, and <plan_list> sections.")
        
        return "\n".join(prompt_parts)

    # ---------- 一轮 step ----------

    def step(
        self,
        user_instruction: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        current_plan_list: Optional[str] = None,
        max_tokens: int = 512,
        turn_num: int = 1
    ) -> PlannerResult:
        """
        一次模型调用
        
        Args:
            user_instruction: 用户指令（仅 Turn 1）
            image_paths: 原始输入图片路径列表（仅 Turn 1）
            current_plan_list: 当前计划文本（仅 Turn 1）
            max_tokens: 最大 token 数
            turn_num: 当前轮次（1, 2, 3...）
        
        Returns:
            PlannerResult 对象
        
        ✅ 核心逻辑：
        - Turn 1: 发送完整任务（instruction + plan + 原始图片）
        - Turn 2+: 发送查询结果（文本 + 检索到的图片编码）
        """
        
        # ✅ 获取上一轮 QUERY 返回的图片路径
        query_images_to_attach = self.last_query_image_paths.copy()
        self.last_query_image_paths = []  # 清空，避免重复使用

        # ========== 1. 构建文本 Prompt ==========
        
        if turn_num == 1:
            # Turn 1: 完整任务描述
            effective_plan = current_plan_list if current_plan_list is not None else self.plan_list
            user_text = self._build_user_prompt_turn1(
                instruction=user_instruction or "",
                plan_list=effective_plan,
                image_count=len(image_paths) if image_paths else 0
            )
        else:
            # Turn 2+: 查询结果
            user_text = self._build_user_prompt_turn_n(
                turn_num=turn_num,
                query_results_text=self.last_query_results_text
            )

        # ========== 2. 组装 Content（文本 + 图像）==========
        
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]

        # Turn 1: 添加用户提供的原始图片
        if turn_num == 1 and image_paths:
            for i, p in enumerate(image_paths, 1):
                if not p:
                    continue
                img_block = self.vlm.base_client.encode_image_to_data_url(p)
                if img_block is not None:
                    # 注入原始路径，便于导出/还原
                    if isinstance(img_block, dict):
                        img_block.setdefault("meta", {})
                        img_block["meta"]["source_path"] = p
                    content.append(img_block)

        # Turn 2+: Add images from Memory QUERY results
        # ✅ Directly attach encoded images without additional text markers
        if turn_num > 1 and query_images_to_attach:
            for img_path in query_images_to_attach:
                img_block = self.vlm.base_client.encode_image_to_data_url(img_path)
                if img_block is not None:
                    # Inject source path for export/restore
                    if isinstance(img_block, dict):
                        img_block.setdefault("meta", {})
                        img_block["meta"]["source_path"] = img_path
                    content.append(img_block)
                else:
                    print(f"⚠️  Warning: Failed to encode query image, skipping: {img_path}")

        # ========== 3. 调用模型 ==========
        
        self.messages.append({"role": "user", "content": content})
        raw_xml = self.vlm.chat(messages=self.messages, max_tokens=max_tokens)
        self.messages.append({"role": "assistant", "content": raw_xml})

        # ========== 4. 解析 XML ==========
        
        parsed = parse_planner_output(raw_xml)
        summary: str = parsed.get("summary", "") or ""
        mem_ops: List[MemoryOperation] = parsed.get("memory_operations", []) or []
        new_plan: str = parsed.get("plan_list", "") or ""

        # ========== 5. 执行 Memory Operations ==========
        
        self.last_query_results_text = self._apply_memory_operations(mem_ops)

        # ========== 6. 更新全局计划 ==========
        
        new_plan_stripped = new_plan.strip()
        if new_plan_stripped:
            self.plan_list = new_plan_stripped

        return PlannerResult(
            summary=summary,
            memory_operations=mem_ops,
            plan_text=self.plan_list,
            refined_plan_text=new_plan_stripped,
            raw_xml=raw_xml,
        )

    # ---------- 导出/还原：对话 JSON ----------

    def export_conversation_json(self, path: str, drop_images: bool = False) -> None:
        """
        将完整 messages 导出为 JSON。
        - drop_images=True 时，不写入 data URL，而是写入一个占位对象：
          {
            "type": "image_url",
            "image_url": {"url": "<REMOVED_DATA_URL>"},
            "meta": {"source_path": "<original_path_or_url>"}
          }
          这样后续可根据 meta.source_path 复原。
        """
        def sanitize_message(msg: Dict[str, Any]) -> Dict[str, Any]:
            role = msg.get("role")
            content = msg.get("content")
            if not drop_images:
                return msg

            out = {"role": role}
            if isinstance(content, list):
                filtered = []
                for item in content:
                    if not isinstance(item, dict):
                        filtered.append(item)
                        continue
                    t = item.get("type")
                    if t == "image_url":
                        # 构造可还原占位
                        source_path = None
                        if isinstance(item.get("meta"), dict):
                            source_path = item["meta"].get("source_path")
                        url_str = ""
                        if isinstance(item.get("image_url"), dict):
                            url_str = item["image_url"].get("url") or ""
                            if (not source_path) and isinstance(url_str, str) and url_str.startswith("file://"):
                                source_path = url_str[len("file://"):]
                        placeholder = {
                            "type": "image_url",
                            "image_url": {"url": "<REMOVED_DATA_URL>"},
                            "meta": {"source_path": source_path or "<UNKNOWN_SOURCE>"},
                        }
                        filtered.append(placeholder)
                    else:
                        filtered.append(item)
                out["content"] = filtered
            else:
                out["content"] = content
            return out

        data = [sanitize_message(m) for m in self.messages]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def restore_image_blocks_in_messages(self, messages: List[dict], strategy: str = "file_url") -> List[dict]:
        """
        将导入的 messages 中的 image_url 占位对象还原为可用的 URL。
        strategy:
          - "file_url": 使用 meta.source_path 生成 file://<path>
          - "data_url": 使用 base_client.encode_image_to_data_url 重新编码（需可访问该路径）
        """
        restored = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(content, list):
                restored.append(msg)
                continue
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url_dict = item.get("image_url", {})
                    if url_dict.get("url") == "<REMOVED_DATA_URL>":
                        meta = item.get("meta") or {}
                        src = meta.get("source_path")
                        if src:
                            if strategy == "file_url":
                                restored_item = {
                                    "type": "image_url",
                                    "image_url": {"url": f"file://{src}"},
                                    "meta": {"source_path": src},
                                }
                            elif strategy == "data_url":
                                encoded = self.vlm.base_client.encode_image_to_data_url(src)
                                if isinstance(encoded, dict):
                                    encoded.setdefault("meta", {})
                                    encoded["meta"]["source_path"] = src
                                    restored_item = encoded
                                else:
                                    restored_item = {
                                        "type": "image_url",
                                        "image_url": {"url": f"file://{src}"},
                                        "meta": {"source_path": src},
                                    }
                            else:
                                restored_item = item
                            new_content.append(restored_item)
                            continue
                new_content.append(item)
            restored.append({"role": role, "content": new_content})
        return restored

    # ---------- 一次 refine ----------

    def run_refine(
        self,
        image_paths: List[str],
        initial_plan_list: Optional[str] = None,
        user_instruction: Optional[str] = None,
        max_tokens: int = 512,
        max_inner_rounds: int = 10,
        # 新增：是否在开头重置
        do_reset: bool = True,
        # 新增：每轮结束是否打印完整对话
        print_full_interactions_each_round: bool = True,
        # 新增：每轮将完整对话保存到 JSON 的目录（若为 None 则不保存）
        log_interactions_json_dir: Optional[str] = None,
        # 新增：是否在每轮询问是否查看 memory（CLI）
        use_cli_prompt_for_memory_view: bool = False,
        # 新增：由调用方决定是否查看 memory 的回调（优先于 CLI）
        decide_view_memory: Optional[Callable[[int], bool]] = None,
        # 新增：保存 memory 轻量快照到 JSON 的目录（若为 None 则不保存）
        log_memory_json_dir: Optional[str] = None,
        # 新增：导出对话 JSON 时是否移除图片块（使用占位保留源路径）
        drop_images_in_json: bool = True,
    ) -> PlannerResult:
        """
        一次完整的 refine 流程：多轮对话 + memory 交互

        新增能力：
        - 每轮结束打印完整交互
        - 可选将每轮交互写入 JSON（图片以占位 + 源路径保存）
        - 每轮询问是否查看 memory（显示不含 embedding 的 all_light），并可选保存为 JSON
        - 可选自动 reset
        """
        
        last_result: Optional[PlannerResult] = None

        # 建议：每次新的 refine 会话前重置，避免历史状态干扰
        if do_reset:
            self.reset()

        # 设置原始输入图片路径列表，为本次 refine 会话固定
        self.current_input_image_paths = image_paths

        # 若给了 initial_plan_list，则设定
        if initial_plan_list is not None:
            self.plan_list = initial_plan_list.strip()

        # 轮次迭代
        for turn_num in range(1, max_inner_rounds + 1):
            
            if turn_num == 1:
                # Turn 1: 完整任务描述 + 原始图片
                last_result = self.step(
                    user_instruction=user_instruction,
                    image_paths=image_paths,
                    current_plan_list=self.plan_list,
                    max_tokens=max_tokens,
                    turn_num=1
                )
            else:
                # Turn 2+: 查询结果 + 继续迭代
                last_result = self.step(
                    user_instruction=None,
                    image_paths=None,
                    current_plan_list=None,
                    max_tokens=max_tokens,
                    turn_num=turn_num
                )

            # 每轮结束：打印完整交互记录（控制台）
            if print_full_interactions_each_round:
                self.print_conversation()

            # 每轮结束：可选保存完整交互为 JSON（图片占位，保留源路径）
            if log_interactions_json_dir:
                Path(log_interactions_json_dir).mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_path = str(Path(log_interactions_json_dir) / f"round_{turn_num}_{ts}.json")
                self.export_conversation_json(json_path, drop_images=drop_images_in_json)
                print(f"[LOG] Conversation saved to: {json_path}")

            # 每轮结束：询问是否查看 memory（不包含 embedding 的 all）
            view_memory = False
            if decide_view_memory is not None:
                try:
                    view_memory = bool(decide_view_memory(turn_num))
                except Exception:
                    view_memory = False
            elif use_cli_prompt_for_memory_view:
                try:
                    resp = input("View memory snapshot (without embeddings)? [y/N]: ").strip().lower()
                    view_memory = resp in ("y", "yes")
                except EOFError:
                    view_memory = False

            if view_memory and self.memory is not None:
                # 需要 MultiTagMemory 实现 all_light() / snapshot_light()
                try:
                    light = self.memory.all_light()
                except AttributeError:
                    # 兼容旧版本：退化为 snapshot() 但剔除 embedding 字段（尽力而为）
                    raw = getattr(self.memory, "snapshot", lambda: [])()
                    light = []
                    for rec in raw:
                        rec_copy = {
                            "id": rec.get("id"),
                            "tags": rec.get("tags"),
                            "data": rec.get("data"),
                            "image_path": rec.get("image_path"),
                        }
                        light.append(rec_copy)

                print("\n===== MEMORY SNAPSHOT (light, no embeddings) =====")
                for rec in light:
                    rid = rec.get("id")
                    tags = ", ".join(rec.get("tags") or [])
                    dtype = (rec.get("data") or {}).get("type")
                    val = (rec.get("data") or {}).get("value")
                    preview = (str(val)[:180] + "...") if (val and len(str(val)) > 180) else (str(val) if val is not None else "")
                    print(f"- ID={rid} | Tags=[{tags}] | type={dtype} | value={preview} | image={rec.get('image_path')}")
                print("==================================================\n")

                # 可选保存 memory 快照
                if log_memory_json_dir:
                    Path(log_memory_json_dir).mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    mem_path = str(Path(log_memory_json_dir) / f"memory_round_{turn_num}_{ts}.json")
                    with open(mem_path, "w", encoding="utf-8") as f:
                        json.dump(light, f, ensure_ascii=False, indent=2)
                    print(f"[LOG] Memory snapshot saved to: {mem_path}")

            # 终止条件：无新查询结果 且 有有效计划
            if not self.last_query_results_text and last_result and last_result.refined_plan_text.strip():
                # import pdb; pdb.set_trace()
                print(f"✅ Refine completed at Turn {turn_num}: No further queries needed.")
                break
        else:
            print(f"⚠️ Reached maximum rounds ({max_inner_rounds})")

        return last_result

    # ---------- 打印对话历史 ----------

    def print_conversation(self):
        """
        打印完整的 user-assistant 对话历史
        
        ✅ 格式化输出，方便调试和查看
        """
        print("\n" + "=" * 100)
        print("  COMPLETE CONVERSATION HISTORY")
        print("=" * 100 + "\n")

        turn_counter = 0

        for i, msg in enumerate(self.messages):
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                print(f"[SYSTEM PROMPT]")
                print("-" * 100)
                if isinstance(content, str):
                    print(content[:500] + "..." if len(content) > 500 else content)
                else:
                    print(str(content))
                print("\n")
                continue

            if role == "user":
                turn_counter += 1
                print(f"\n{'=' * 100}")
                print(f"USER (Turn {turn_counter})")
                print("=" * 100)
                
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            print(item.get("text", ""))
                        elif isinstance(item, dict) and item.get("type") == "image_url":
                            # 打印图像来源（如果可用）
                            source_path = None
                            if isinstance(item.get("meta"), dict):
                                source_path = item["meta"].get("source_path")
                            if source_path:
                                print(f"\n[IMAGE: {Path(source_path).name}]")
                            else:
                                url_show = ""
                                if isinstance(item.get("image_url"), dict):
                                    url_show = item["image_url"].get("url", "")[:30]
                                print(f"\n[IMAGE: {url_show}...]")
                        else:
                            # 其他未知结构
                            print(str(item))
                else:
                    print(content)

            elif role == "assistant":
                print(f"\n{'-' * 100}")
                print(f"ASSISTANT (Turn {turn_counter})")
                print("-" * 100)
                if isinstance(content, str):
                    print(content)
                else:
                    print(str(content))

        print("\n" + "=" * 100)
        print("  END OF CONVERSATION")
        print("=" * 100 + "\n")