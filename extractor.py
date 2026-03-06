# src/extractor.py
import re
from typing import Dict, Any, Optional, List, Tuple
from xml.etree import ElementTree as ET


# ===================== 公共工具 =====================

def _strip_xml_comments(text: str) -> str:
    """Remove XML-style comments <!-- ... -->."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _extract_single_tag(text: str, tag: str) -> Optional[str]:
    """
    Extract the first <tag>...</tag> content (non-greedy).
    Returns inner text or None if not found.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1)


def _clean_text(s: str) -> str:
    return s.strip().replace("\r\n", "\n").replace("\r", "\n")


# ===================== Observer 解析 =====================

def parse_observer_output(raw_xml: str) -> Dict[str, Any]:
    """
    Parse observer XML.

    预期格式:

      <status>
      <!-- EXACTLY one of:
              done
              not_done
      -->
      </status>

    返回:
      {"status": "done" | "not_done"}

    任意解析失败 / 非法值统一回退为 "not_done"。
    """
    text = _strip_xml_comments(raw_xml)
    content = _extract_single_tag(text, "status")
    if content is None:
        return {"status": "not_done"}

    status = _clean_text(content).lower()
    if status not in {"done", "not_done"}:
        status = "not_done"

    return {"status": status}


# ===================== Planner 解析 =====================

class MemoryOperation:
    """
    一条从 Planner XML 中解析出的 memory 操作。
    
    ✅ 修改点：
    - 添加 tags: Optional[List[str]] 字段（替代单个 obj_name）
    - 添加 image_path: Optional[str] 字段
    - 保留 obj_name 用于向后兼容
    """
    def __init__(
        self,
        type: str,
        id: Optional[str],
        obj_name: Optional[str],
        text: Optional[str],
        reason: str,
        raw_xml: str,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,        # ✅ 新增
        image_path: Optional[str] = None,        # ✅ 新增
    ) -> None:
        self.type = type
        self.id = id
        self.obj_name = obj_name  # 保留以便向后兼容
        self.text = text
        self.reason = reason
        self.query = query
        self.raw_xml = raw_xml
        self.tags = tags              # ✅ 新增：多标签
        self.image_path = image_path  # ✅ 新增：图片路径

    def __repr__(self) -> str:
        return (
            f"MemoryOperation(type={self.type!r}, id={self.id!r}, "
            f"tags={self.tags!r}, obj_name={self.obj_name!r}, text={self.text!r}, "
            f"reason={self.reason!r}, query={self.query!r}, image_path={self.image_path!r})"
        )


def parse_planner_output(xml_text: str) -> Dict[str, Any]:
    """
    解析 planner 的 XML 输出，返回:
    {
        "summary": str,
        "memory_operations": List[MemoryOperation],
        "plan_list": str,
    }

    ✅ 修改点：
    - 解析 <tags> 字段（逗号分隔的标签列表）
    - 解析 <image_path> 字段
    - 兼容原有 <obj_name> 字段
    """
    # 尝试截取从 <summary> 到 </plan_list> 的主体
    match = re.search(
        r"<summary[\s\S]*?</plan_list>",
        xml_text,
        flags=re.IGNORECASE,
    )
    if not match:
        core = xml_text.strip()
    else:
        core = match.group(0)

    # 为了能用标准 XML 解析器，包上一个 root
    wrapped = f"<root>{core}</root>"

    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError as e:
        # Log parsing failure for debugging
        print(f"❌ [Extractor] XML parsing failed: {e}")
        print(f"   Raw XML (first 500 chars): {xml_text[:500]}")
        return {
            "summary": "",
            "memory_operations": [],
            "plan_list": "",
        }

    # ---------- 1. summary ----------
    summary_el = root.find("summary")
    summary_text = ""
    if summary_el is not None and summary_el.text:
        summary_text = summary_el.text.strip()

    # ---------- 2. memory_operations ----------
    mem_ops: List[MemoryOperation] = []
    mem_ops_el = root.find("memory_operations")
    if mem_ops_el is not None:
        for op_el in mem_ops_el.findall("operation"):
            raw_xml = ET.tostring(op_el, encoding="unicode")

            type_el = op_el.find("type")
            op_type = type_el.text.strip().upper() if type_el is not None and type_el.text else ""

            id_el = op_el.find("id")
            op_id = id_el.text.strip() if id_el is not None and id_el.text else None

            # ✅ 解析 tags（逗号分隔）
            tags_el = op_el.find("tags")
            tags_list: Optional[List[str]] = None
            if tags_el is not None and tags_el.text:
                tags_text = tags_el.text.strip()
                tags_list = [t.strip() for t in tags_text.split(",") if t.strip()]

            # ✅ 解析 image_path
            image_path_el = op_el.find("image_path")
            image_path = None
            if image_path_el is not None and image_path_el.text:
                image_path = image_path_el.text.strip()

            # ✅✅✅ 优先解析扁平化结构
            text_el = op_el.find("text")
            text_val = text_el.text.strip() if text_el is not None and text_el.text else None

            query_el = op_el.find("query")  # ✅ 新增：直接查找 <query>
            query_val = query_el.text.strip() if query_el is not None and query_el.text else None

            obj_name = None

            # ✅ 如果扁平化结构没找到，尝试嵌套结构 <content>
            if text_val is None and query_val is None:
                content_el = op_el.find("content")
                if content_el is not None:
                    if op_type == "QUERY":
                        if content_el.text:
                            query_val = content_el.text.strip()
                    else:
                        obj_el = content_el.find("obj_name")
                        txt_el = content_el.find("text")
                        if obj_el is not None and obj_el.text:
                            obj_name = obj_el.text.strip()
                        if txt_el is not None and txt_el.text:
                            text_val = txt_el.text.strip()

            reason_el = op_el.find("reason")
            reason = reason_el.text.strip() if reason_el is not None and reason_el.text else ""

            mem_ops.append(
                MemoryOperation(
                    type=op_type,
                    id=op_id,
                    obj_name=obj_name,
                    text=text_val,
                    reason=reason,
                    raw_xml=raw_xml,
                    query=query_val,  # ✅ 传递 query
                    tags=tags_list,
                    image_path=image_path,
                )
            )


    # ---------- 3. plan_list ----------
    plan_el = root.find("plan_list")
    plan_text = ""
    if plan_el is not None and plan_el.text:
        plan_text = plan_el.text.strip()

    # ---------- 4. single-subtask fallback ----------
    # Support prompts that output <subtask>/<is_complete> instead of <plan_list>.
    if not plan_text:
        subtask_el = root.find("subtask")
        is_complete_el = root.find("is_complete")

        subtask_text = ""
        if subtask_el is not None and subtask_el.text:
            subtask_text = subtask_el.text.strip()

        is_complete_text = ""
        if is_complete_el is not None and is_complete_el.text:
            is_complete_text = is_complete_el.text.strip().lower()

        if subtask_text or is_complete_text:
            if is_complete_text in {"yes", "true", "done"}:
                if subtask_text:
                    plan_text = f"[done] {subtask_text}"
                else:
                    plan_text = "[done] task complete"
            else:
                if subtask_text:
                    plan_text = f"[current] {subtask_text}"
                else:
                    plan_text = ""

    return {
        "summary": summary_text,
        "memory_operations": mem_ops,
        "plan_list": plan_text,
    }


# ===================== 新增：PlanList 辅助提取 =====================

_CURRENT_MARKER_RE = re.compile(r"\[current\]", flags=re.IGNORECASE)

def _strip_parentheses_note(s: str) -> str:
    """
    移除圆括号 () 内的附注（可能有多个），例如：
      'Step-2 (low-confidence)' -> 'Step-2'
      '推理(细化)' -> '推理'
    注意：仅处理 ()，不处理 [] 或 {}。
    """
    # 去除所有 () 内文本
    s_no_paren = re.sub(r"\s*\([^()]*\)", "", s)
    return s_no_paren.strip()

def _strip_step_prefix(s: str) -> str:
    """
    去除类似 "Step 1:", "step 2:", "Step-3:" 等前缀。
    支持的格式：
      - "Step 1: do something" -> "do something"
      - "step-2: action" -> "action"
      - "步骤1：" -> ""
    """
    # 匹配 "step" (大小写不敏感) + 数字/符号 + 冒号
    s = re.sub(r"^\s*step[\s\-]*\d+\s*[:：]\s*", "", s, flags=re.IGNORECASE)
    # 匹配中文 "步骤" + 数字 + 冒号
    s = re.sub(r"^\s*步骤\s*\d+\s*[:：]\s*", "", s)
    return s.strip()

def normalize_current_line(line: str) -> str:
    """
    清洗包含 [current] 的任务行，返回纯任务内容。

    处理步骤：
    1. 去除 [current] 标记（可能在行的任何位置）
    2. 去除 [done], [pending] 等其他状态标记
    3. 去除 "Step 1:", "step 2:" 等前缀
    4. 去除圆括号 () 及其内容
    5. 去除两端空白

    示例：
      "Step 1: Pick up object [current]" -> "Pick up object"
      "[current] Step 2: Place in box (carefully)" -> "Place in box"
      "step 3: inspect left box [current] (check contents)" -> "inspect left box"
    """
    # 1. 去除所有状态标记 [current], [done], [pending]
    core = re.sub(r"\[(current|done|pending)\]", "", line, flags=re.IGNORECASE)

    # 2. 去除 Step 前缀
    core = _strip_step_prefix(core)

    # 3. 去除圆括号内容
    core = _strip_parentheses_note(core)

    # 4. 去除两端空白
    return core.strip()

def extract_current_subtask(plan_text: str) -> Optional[str]:
    """
    从 plan_list 文本中提取包含 [current] 标记的子任务。

    特点：
    - [current] 可以在行的任何位置（开头、中间、结尾）
    - 自动去除 "Step 1:", "step 2:" 等前缀
    - 自动去除圆括号 () 及其内容
    - 返回清洗后的纯任务内容

    若找不到则返回 None。

    示例：
      输入: "Step 1: Pick up object [done]\nStep 2: Place in box [current] (carefully)\nStep 3: Verify [pending]"
      输出: "Place in box"
    """
    if not plan_text:
        return None
    lines = [ln.strip() for ln in plan_text.splitlines() if ln.strip()]
    for ln in lines:
        if _CURRENT_MARKER_RE.search(ln):
            return normalize_current_line(ln)
    return None

def extract_all_tasks_status(plan_text: str) -> List[Tuple[str, bool, bool]]:
    """
    解析每行任务，返回三元组列表：
      (task_text_clean, is_done, is_current)
    - 使用启发式匹配：
        - is_current: 行包含 [current]（大小写不敏感，可在任意位置）
        - is_done: 包含 '[done]'（大小写不敏感）
    - task_text_clean: 去除所有状态标记、Step 前缀、() 附注后的纯任务名
    """
    results: List[Tuple[str, bool, bool]] = []
    if not plan_text:
        return results

    lines = [ln.strip() for ln in plan_text.splitlines() if ln.strip()]
    for ln in lines:
        is_current = bool(_CURRENT_MARKER_RE.search(ln))
        is_done = bool(re.search(r"\[done\]", ln, flags=re.IGNORECASE))

        # 清洗任务内容
        core = normalize_current_line(ln)
        if core:
            results.append((core, is_done, is_current))
    return results

def is_plan_done(plan_text: str) -> bool:
    """
    粗粒度判断 plan_list 是否“整体完成”。
    规则（启发式，按需调整）：
      - 若含有全局 Done 提示，如 'all done', 'plan done', '<status>done</status>' 等，则视为完成；
      - 否则若解析到的所有子任务全为 done，则视为完成；
      - 否则视为未完成。
    """
    if not plan_text:
        return False
    text = plan_text.strip()

    # 全局标记
    if re.search(r"\ball\s*done\b|\bplan\s*done\b|\<status\>\s*done\s*\</status\>", text, flags=re.IGNORECASE):
        return True

    task_stats = extract_all_tasks_status(text)
    if task_stats and all(is_done for _, is_done, _ in task_stats):
        return True

    return False
