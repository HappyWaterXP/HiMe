# src/agent/multitag_planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import re
import json
import time
from datetime import datetime

from prompt_loader import load_prompt
from client.planner_vlm import PlannerVLM
from extractor import parse_planner_output, MemoryOperation
from memory.multitag_recorder import MultiTagMemory


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


def parse_query_terms(query_text: Optional[str]) -> List[str]:
    """
    将 QUERY 内容按逗号拆成多个独立 tag。

    约定：
    - `apple, box, on_table` -> ["apple", "box", "on_table"]
    - 不按空格拆分，空格保留在单个 term 内
    - 为空时返回 []
    """
    if not query_text:
        return []

    parts = [part.strip() for part in str(query_text).split(",")]
    normalized_terms = []
    for part in parts:
        if not part:
            continue
        cleaned = part.strip().strip("[]\"'")
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
        if cleaned:
            normalized_terms.append(cleaned)
    return normalized_terms


def format_timestamp(ts: Optional[float]) -> str:
    if ts is None:
        return "unknown"
    try:
        ts_val = float(ts)
    except (TypeError, ValueError):
        return "unknown"
    if ts_val <= 0:
        return "unknown"
    dt = datetime.fromtimestamp(ts_val)
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(dt.microsecond / 1000):03d}"


def infer_frame_timestamp(path: Optional[str]) -> Optional[float]:
    """
    为输入帧推断本地时间戳。

    优先使用文件名里的 13 位毫秒时间戳；如果没有，则回退到文件 mtime。
    """
    if not path:
        return None

    match = re.search(r"(\d{13})", Path(path).name)
    if match:
        try:
            return int(match.group(1)) / 1000.0
        except ValueError:
            pass

    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None

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

    def __init__(
        self,
        vlm: PlannerVLM,
        memory: Optional[MultiTagMemory] = None,
        prompt_name: str = "multitag_planner",
        memory_op_policy: str = "allow_all",
        memory_mode: str = "mixed",
    ) -> None:
        self.vlm = vlm
        self.memory = memory
        self.memory_op_policy = (memory_op_policy or "allow_all").strip().lower()
        self.memory_mode = (memory_mode or "mixed").strip().lower()

        self.system_prompt: str = load_prompt(prompt_name)
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        self.plan_list: str = ""
        self.last_query_results_text: str = ""
        self.last_query_image_paths: List[str] = []

        # ✅ 当前 refine 会话的原始输入图片路径列表（Turn 1 执行帧）
        self.current_input_image_paths: List[str] = []
        # ✅ 当前 step 实际喂给模型的图片路径列表（仅用于调试/导出）
        self.current_turn_image_paths: List[str] = []

    def _memory_disabled(self) -> bool:
        return "MEMORY DISABLED" in self.system_prompt

    def _text_only_memory_mode(self) -> bool:
        return self.memory_mode == "text_only"

    def _image_only_memory_mode(self) -> bool:
        return self.memory_mode == "image_only"

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
        self.current_turn_image_paths = []

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
        # image_path 始终绑定到 Turn 1 的执行帧。
        # Turn 2 的检索附件只用于阅读 memory，不参与索引映射。
        source_paths = self.current_input_image_paths
        source_name = "turn1_input_images"
        if not source_paths:
            source_paths = self.current_turn_image_paths
            source_name = "current_turn_images"
        
        resolved_paths = []
        for idx in indices:
            if 0 <= idx < len(source_paths):
                resolved_paths.append(source_paths[idx])
            else:
                print(
                    f"⚠️ Warning: Image index {idx+1} out of range "
                    f"(source={source_name}, total={len(source_paths)})."
                )
        
        return resolved_paths

    def _resolve_representative_image_path(self, image_path_str: Optional[str]) -> Optional[str]:
        """
        解析 image_path，并只保留一个代表性帧。

        如果模型给出多个候选帧，默认取最后一个有效帧，
        作为当前事实在 recent frames 中最具代表性的证据图。
        """
        resolved_paths = self._resolve_image_paths(image_path_str)
        if not resolved_paths:
            return None
        return resolved_paths[-1]

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
        Execute memory operations and return all results.

        ✅ Key features:
        1. CREATE/UPDATE/DELETE: Return success/failure messages
        2. QUERY: Return complete record information (tags + text), collect image paths
        3. QUERY 去重：在一次 _apply_memory_operations 调用中，跨多次 QUERY
        对 record id 去重；对 image_path 去重（同一张图只附带一次）
        4. ✅ 使用“文本锚定”绑定图片：为每张去重后的图片分配 IMG#n，
        并在 step() 中在图片前插入一个文本块 `[ATTACHMENT IMG#n]`，保证模型可对齐。
        """
        if self.memory is None:
            self.last_query_image_attachments = []
            return "No Existing Memory.\n"

        allowed_by_policy = {
            "allow_all": None,
            "query_create_only": {"QUERY", "CREATE"},
            "disable_all": set(),
        }
        allowed_ops = allowed_by_policy.get(self.memory_op_policy)
        if allowed_ops is None and self.memory_op_policy not in allowed_by_policy:
            print(f"⚠️ [Planner] Unknown memory_op_policy='{self.memory_op_policy}', fallback to allow_all.")

        result_lines: List[str] = []

        # 跨整个 operations 列表去重（跨多个 QUERY）
        seen_record_ids: set[int] = set()

        # 图片去重 + ref 分配（跨多个 QUERY）
        image_path_to_ref: dict[str, str] = {}
        ref_to_path: dict[str, str] = {}
        ref_to_latest_ts: dict[str, float] = {}
        img_counter = 0

        for op in operations:
            t = (op.type or "").upper().strip()
            if allowed_ops is not None and t not in allowed_ops:
                print(f"⚠️ [Planner] SKIP {t}: blocked by memory_op_policy='{self.memory_op_policy}'")
                continue

            # --- QUERY ---
            if t == "QUERY":
                query_content = op.query
                if not query_content:
                    result_lines.append("\n❌ QUERY: Missing query content")
                    continue

                try:
                    query_terms = parse_query_terms(query_content)
                    if not query_terms:
                        result_lines.append("\n❌ QUERY: Missing query content")
                        continue

                    for query_term in query_terms:
                        records, scores = self.memory.query(
                            content=query_term,
                            top_k=5,
                        )

                        result_lines.append(f"\n--- QUERY: {query_term} ---")

                        if not records:
                            result_lines.append("  (No matching records found)")
                            continue

                        displayed_rank = 0
                        for rec in records:
                            # 跨 QUERY 对 record ID 去重：同一 rec（text/image）只展示一次
                            if rec.id in seen_record_ids:
                                continue
                            seen_record_ids.add(rec.id)
                            displayed_rank += 1

                            score = scores.get(rec.id, 0.0)
                            data_type = rec.data_type
                            value = rec.text
                            if self._image_only_memory_mode():
                                result_lines.append(
                                    f"  #{displayed_rank} Record ID={rec.id} | Type={data_type} | Score={score:.3f}"
                                )
                            else:
                                tags_str = ", ".join(rec.tags or [])
                                result_lines.append(
                                    f"  #{displayed_rank} Record ID={rec.id} | Type={data_type} | Tags=[{tags_str}] | Score={score:.3f}"
                                )
                            result_lines.append(
                                f"    UpdatedAt: {format_timestamp(getattr(rec, 'updated_at', None))}"
                            )

                            if value and not self._image_only_memory_mode():
                                value_str = str(value)
                                text_preview = value_str[:200] + ("..." if len(value_str) > 200 else "")
                                result_lines.append(f"    Content: {text_preview}")

                            # Image record: 允许不同 rec 引用同一张图，但图片只附带一次
                            if (not self._text_only_memory_mode()) and data_type == "image" and rec.image_path:
                                img_path = str(rec.image_path)

                                if img_path not in image_path_to_ref:
                                    img_counter += 1
                                    ref = f"IMG#{img_counter}"
                                    image_path_to_ref[img_path] = ref
                                    ref_to_path[ref] = img_path
                                else:
                                    ref = image_path_to_ref[img_path]

                                rec_updated_at = getattr(rec, "updated_at", 0.0) or 0.0
                                ref_to_latest_ts[ref] = max(ref_to_latest_ts.get(ref, 0.0), rec_updated_at)

                                # 在文本里输出 ref（模型将通过 step() 的文本锚定与图片对齐）
                                result_lines.append(
                                    f"    ImageRef: {ref} | file={Path(img_path).name}"
                                )

                        if displayed_rank == 0:
                            result_lines.append("  (All matching records were already shown above)")

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
                    representative_path = self._resolve_representative_image_path(op.image_path)

                    if self._text_only_memory_mode():
                        if op.image_path:
                            result_lines.append(
                                "\nℹ️  CREATE: TEXT_ONLY_MEMORY active, ignoring image_path and creating a text record"
                            )
                        representative_path = None
                    elif self._image_only_memory_mode() and not representative_path:
                        result_lines.append(
                            f"\n❌ CREATE failed: IMAGE_ONLY_MEMORY requires a valid image_path, got '{op.image_path}'"
                        )
                        continue

                    if op.image_path and not representative_path and not self._text_only_memory_mode():
                        result_lines.append(
                            f"\n⚠️  CREATE: Specified image_path='{op.image_path}' but no valid images found, creating text record instead"
                        )

                    if representative_path:
                        rec = self.memory.create(
                            tags=tags_to_use,
                            data_type="image",
                            data_value=op.text,
                            text=op.text,
                            image_path=representative_path,
                        )
                        tags_str = ", ".join(tags_to_use)
                        result_lines.append(
                            f"\n✅ CREATE: Created image record ID={rec.id} using representative frame | Tags=[{tags_str}]"
                        )
                    else:
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
                rec_id = self._safe_parse_rec_id(str(op.id) if op.id is not None else None)
                if rec_id is None:
                    result_lines.append(f"\n❌ UPDATE failed: Invalid record ID '{op.id}'")
                    continue

                try:
                    update_kwargs = {"rec_id": rec_id}

                    if op.text is not None:
                        update_kwargs["text"] = op.text

                    if self._text_only_memory_mode():
                        if op.image_path is not None:
                            result_lines.append(
                                "\nℹ️  UPDATE: TEXT_ONLY_MEMORY active, ignoring image_path and keeping memory text-only"
                            )
                        update_kwargs["data_type"] = "text"
                        update_kwargs["image_path"] = ""
                    elif op.image_path is not None:
                        representative_path = self._resolve_representative_image_path(op.image_path)
                        if representative_path is not None:
                            update_kwargs["image_path"] = representative_path
                        elif self._image_only_memory_mode():
                            result_lines.append(
                                f"\n❌ UPDATE failed: IMAGE_ONLY_MEMORY requires a valid image_path, got '{op.image_path}'"
                            )
                            continue

                    # tags：None/[] 都维持原 tags；只有非空才更新
                    if op.tags is not None:
                        if isinstance(op.tags, (list, tuple, set)):
                            normalized = [str(t).strip() for t in op.tags if str(t).strip()]
                        else:
                            s = str(op.tags).strip()
                            normalized = [s] if s else []

                        if normalized:
                            update_kwargs["tags"] = normalized

                    updated_rec = self.memory.update(**update_kwargs)

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

        # ✅ 跨所有 QUERY 去重后的“图片附件”（按 IMG#n 顺序）
        attachments: List[Dict[str, Any]] = []
        for i in range(1, img_counter + 1):
            ref = f"IMG#{i}"
            img_path = ref_to_path.get(ref)
            if img_path:
                attachments.append({
                    "ref": ref,
                    "image_path": img_path,
                    "updated_at": ref_to_latest_ts.get(ref, 0.0),
                    "updated_at_readable": format_timestamp(ref_to_latest_ts.get(ref, 0.0)),
                })

        self.last_query_image_attachments = attachments

        return "\n".join(result_lines).strip()


    # ---------- 构造 User Prompt ----------

    def _build_user_prompt_turn1(
        self,
        instruction: str,
        plan_list: str,
        image_count: int,
        current_timestamp_text: str,
    ) -> str:
        """
        构建 Turn 1 的 User Prompt

        包含：完整任务描述 + 当前计划 + 原始图片说明

        根据 system prompt 的内容决定是否包含 memory 相关的提示
        """
        prompt_parts = ["=== TURN 1 ===\n"]

        prompt_parts.append("USER INSTRUCTION:")
        prompt_parts.append(instruction)

        prompt_parts.append("\nCURRENT PLAN:")
        if plan_list and plan_list.strip():
            prompt_parts.append(plan_list)
        else:
            prompt_parts.append("(No existing plan)")
        prompt_parts.append(
            f"\nCURRENT TIME: {current_timestamp_text}"
        )
        prompt_parts.append(
            "\nTIME NOTE: The TURN 1 images below are the current world state. "
            "They are the authoritative source for what is happening now."
        )

        if image_count > 0:
            prompt_parts.append(f"\nINPUT IMAGES: {image_count} image(s) provided below, numbered 1 to {image_count}")
            prompt_parts.append(
                "Each image will be preceded by a text anchor showing only its frame index. "
                "The image order preserves temporal order within TURN 1. "
                "These TURN 1 frames are a longer-horizon, more sparsely sampled planning sequence, "
                "so neighboring frames may have larger time gaps. "
                "Each frame anchor will also show that frame's local timestamp. "
                "Use the full sequence to understand progress, but use the latest frames to judge the final current state. "
                "Middle TURN 1 frames often show intermediate motion states, while the latest frame(s) are most useful for judging the final result state. "
                "Use these TURN 1 images, not memory, to judge the current state."
            )

            # Check if memory is enabled by examining system prompt
            if not self._memory_disabled():
                if self._text_only_memory_mode():
                    prompt_parts.append("💡 TEXT_ONLY_MEMORY is active. Do NOT output image_path in memory operations.")
                elif self._image_only_memory_mode():
                    prompt_parts.append("💡 IMAGE_ONLY_MEMORY is active. Every CREATE memory must use exactly one image_path frame index.")
                else:
                    prompt_parts.append(
                        '💡 If you add image evidence, choose exactly one frame index in image_path.'
                    )
        else:
            prompt_parts.append("\nINPUT IMAGES: None")
            prompt_parts.append("Do NOT output image_path when no images are provided.")

        prompt_parts.append("\n📋 REQUIRED OUTPUT FORMAT:")

        # Determine output format based on system prompt
        if self._memory_disabled():
            # No memory mode
            if "<subtask>" in self.system_prompt and "<is_complete>" in self.system_prompt:
                # No plan mode (single subtask output)
                prompt_parts.append("You must respond in XML format with <summary>, <subtask>, and <is_complete> sections.")
            else:
                # No memory but still plan list mode
                prompt_parts.append("You must respond in XML format with <summary> and <plan_list> sections.")
        else:
            # Full memory mode
            prompt_parts.append("You must respond in XML format with <summary>, <memory_operations>, and <plan_list> sections.")

        return "\n".join(prompt_parts)

    def _build_user_prompt_turn_n(
        self,
        turn_num: int,
        query_results_text: str,
        attachment_count: int = 0,
    ) -> str:
        """
        构建 Turn 2+ 的 User Prompt

        包含：上一轮的查询结果（文本 + 图片会在后面附加）

        如果 memory 被禁用，则不应该有 Turn 2+（但为了兼容性保留此方法）
        """
        prompt_parts = [f"=== TURN {turn_num} ===\n"]

        # Check if memory is enabled
        if self._memory_disabled():
            # This should not happen in no-memory mode, but handle gracefully
            prompt_parts.append("=== SINGLE TURN MODE ===")
            prompt_parts.append("Memory is disabled. This is an unexpected second turn.")
        else:
            if query_results_text:
                prompt_parts.append("=== MEMORY QUERY RESULTS ===")
                prompt_parts.append(query_results_text)
                prompt_parts.append(
                    "\nIMPORTANT: Memory query results are historical records from the past. "
                    "They can help with context, but they are NOT the current world state."
                )
                if self.current_input_image_paths:
                    prompt_parts.append(
                        "For deciding what is true now, trust the TURN 1 execution frames over memory results."
                    )
                prompt_parts.append(
                    "Use UpdatedAt to judge recency inside memory: newer memory is usually more relevant than older memory, "
                    "but memory is still only reference and must not override TURN 1."
                )
                if self._text_only_memory_mode():
                    prompt_parts.append("TEXT_ONLY_MEMORY is active in this experiment. Use textual memory only, and do NOT output image_path in this turn.")
                elif self._image_only_memory_mode():
                    prompt_parts.append("IMAGE_ONLY_MEMORY is active in this experiment. New memory should be image-backed and use exactly one TURN 1 image_path when creating evidence-backed facts.")
                prompt_parts.append("\n📌 Note: Retrieved images (if any) are shown below this text.")
                if attachment_count > 0:
                    prompt_parts.append(
                        f"Retrieved image attachments in this turn: {attachment_count}. "
                        "These attachments are historical memory evidence for reading only. "
                        "If you output image_path in this turn, it MUST still index the original execution frames from TURN 1."
                    )
                else:
                    if self.current_input_image_paths:
                        prompt_parts.append(
                            "No retrieved images in this turn. "
                            "If you output image_path, it MUST still index the original execution frames from TURN 1."
                        )
                    else:
                        prompt_parts.append("No retrieved images in this turn. Do NOT output image_path.")
            else:
                prompt_parts.append("=== NO MEMORY QUERIES PERFORMED ===")
                prompt_parts.append("Previous operations completed without queries.")
                if self._text_only_memory_mode():
                    prompt_parts.append("TEXT_ONLY_MEMORY is active in this experiment. Do NOT output image_path in this turn.")
                elif self.current_input_image_paths:
                    prompt_parts.append(
                        "If you output image_path in this turn, it MUST still index the original execution frames from TURN 1."
                    )
                else:
                    prompt_parts.append("Do NOT output image_path in this turn.")

            prompt_parts.append("\n📋 NEXT STEPS:")
            prompt_parts.append("Based on the query results above, continue refining your plan and performing necessary memory operations.")

        prompt_parts.append("\n📋 REQUIRED OUTPUT FORMAT:")

        # Determine output format based on system prompt
        if self._memory_disabled():
            if "<subtask>" in self.system_prompt and "<is_complete>" in self.system_prompt:
                prompt_parts.append("You must respond in XML format with <summary>, <subtask>, and <is_complete> sections.")
            else:
                prompt_parts.append("You must respond in XML format with <summary> and <plan_list> sections.")
        else:
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
    ) -> "PlannerResult":
        """
        一次模型调用

        ✅ Turn 2+：
        - 发送查询结果文本
        - 然后按 IMG#n 顺序附图
        - ✅ 在每张图前插入文本锚：显示该图 ref + 被哪些 rec 引用（第三次出现引用关系）
        形式：`[ATTACHMENT IMG#n | ReferencedByRecIDs=[...]]`
        """

        query_attachments_to_attach = list(getattr(self, "last_query_image_attachments", []))
        self.last_query_image_attachments = []
        turn_visible_image_paths: List[str] = []

        # ========== 1. 构建文本 Prompt ==========
        if turn_num == 1:
            effective_plan = current_plan_list if current_plan_list is not None else self.plan_list
            valid_paths = [p for p in (image_paths or []) if p]
            current_timestamp_text = format_timestamp(
                infer_frame_timestamp(valid_paths[-1]) if valid_paths else time.time()
            )
            user_text = self._build_user_prompt_turn1(
                instruction=user_instruction or "",
                plan_list=effective_plan,
                image_count=len(image_paths) if image_paths else 0,
                current_timestamp_text=current_timestamp_text,
            )
        else:
            user_text = self._build_user_prompt_turn_n(
                turn_num=turn_num,
                query_results_text=self.last_query_results_text,
                attachment_count=len(query_attachments_to_attach),
            )

        # ========== 2. 组装 Content（文本 + 图像）==========
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]

        # Turn 1: 原始图片
        if turn_num == 1 and image_paths:
            for idx, p in enumerate(image_paths, start=1):
                if not p:
                    continue
                frame_timestamp = format_timestamp(infer_frame_timestamp(p))
                content.append({
                    "type": "text",
                    "text": f"[CURRENT FRAME {idx} | Timestamp={frame_timestamp}]"
                })
                img_block = self.vlm.base_client.encode_image_to_data_url(p)
                if img_block is not None:
                    if isinstance(img_block, dict):
                        img_block.setdefault("meta", {})
                        img_block["meta"]["source_path"] = p
                        img_block["meta"]["frame_index"] = idx
                        img_block["meta"]["frame_timestamp"] = frame_timestamp
                    content.append(img_block)
                    turn_visible_image_paths.append(p)

        # Turn 2+: 查询图片（文本锚定）
        if turn_num > 1 and query_attachments_to_attach:
            for att in query_attachments_to_attach:
                ref = att.get("ref") or "IMG#?"
                img_path = att.get("image_path")
                rec_ids = att.get("referenced_by", [])
                updated_at_readable = att.get("updated_at_readable", "unknown")
                if not img_path:
                    continue

                # ✅ 第二处（在附图阶段）再次显示引用关系：该图被哪些 rec 引用
                content.append({
                    "type": "text",
                    "text": (
                        f"[ATTACHMENT {ref} | ReferencedByRecIDs={rec_ids} "
                        f"| MemoryUpdatedAt={updated_at_readable}]"
                    )
                })

                img_block = self.vlm.base_client.encode_image_to_data_url(img_path)
                if img_block is not None:
                    if isinstance(img_block, dict):
                        img_block.setdefault("meta", {})
                        # meta 仅用于你调试/导出，不依赖模型读取
                        img_block["meta"]["source_path"] = img_path
                        img_block["meta"]["memory_image_ref"] = ref
                        img_block["meta"]["memory_referenced_by"] = rec_ids
                        img_block["meta"]["memory_updated_at"] = updated_at_readable
                    content.append(img_block)
                    turn_visible_image_paths.append(img_path)
                else:
                    print(f"⚠️  Warning: Failed to encode query image, skipping: {img_path}")


        # ========== 3. 调用模型 ==========
        
        # Keep the exact image list visible to the model in this step.
        self.current_turn_image_paths = turn_visible_image_paths
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
                        meta = dict(item.get("meta") or {}) if isinstance(item.get("meta"), dict) else {}
                        source_path = meta.get("source_path")
                        url_str = ""
                        if isinstance(item.get("image_url"), dict):
                            url_str = item["image_url"].get("url") or ""
                            if (not source_path) and isinstance(url_str, str) and url_str.startswith("file://"):
                                source_path = url_str[len("file://"):]
                                meta["source_path"] = source_path
                        placeholder = {
                            "type": "image_url",
                            "image_url": {"url": "<REMOVED_DATA_URL>"},
                            "meta": meta or {"source_path": source_path or "<UNKNOWN_SOURCE>"},
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
        # import pdb; pdb.set_trace()
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
            if turn_num == 2:
                view_memory = True
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
                            "data_type": rec.get("data_type"),
                            "text": rec.get("text"),
                            "image_path": rec.get("image_path"),
                        }
                        light.append(rec_copy)

                print("\n===== MEMORY SNAPSHOT (light, no embeddings) =====")
                for rec in light:
                    rid = rec.get("id")
                    tags = ", ".join(rec.get("tags") or [])
                    dtype = rec.get("data_type")
                    val = rec.get("text")
                    preview = (str(val)[:180] + "...") if (val and len(str(val)) > 180) else (str(val) if val is not None else "")
                    updated_at_readable = rec.get("updated_at_readable") or "unknown"
                    print(
                        f"- ID={rid} | Tags=[{tags}] | type={dtype} | updated_at={updated_at_readable} "
                        f"| value={preview} | image={rec.get('image_path')}"
                    )
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
