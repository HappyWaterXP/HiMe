#!/usr/bin/env python3
from __future__ import annotations

import os
import io
import json
import time
import uuid
import base64
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

import openai

# =========================================================
# Config
# =========================================================
os.environ.setdefault("OPENAI_API_KEY", "xx")
os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")

# MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-2024-08-06")
MODEL_NAME = os.environ.get("VLM_MODEL", "claude-sonnet-4-5-20250929")
REQUEST_TIMEOUT_S = int(os.environ.get("MEMER_OPENAI_TIMEOUT_S", "180"))

MERGE_DISTANCE_D = int(os.environ.get("MEMER_MERGE_DISTANCE_D", "5"))
MAX_KEYFRAMES = int(os.environ.get("MEMER_MAX_KEYFRAMES", "8"))
RECENT_MAX = int(os.environ.get("MEMER_RECENT_MAX", "8"))

COMPOSITE_LAYOUT = os.environ.get("MEMER_COMPOSITE_LAYOUT", "horizontal").lower()
COMPOSITE_MAX_SIDE = int(os.environ.get("MEMER_COMPOSITE_MAX_SIDE", "1024"))

LOG_ROOT = os.environ.get("MEMER_LOG_ROOT", "./memer_logs")
SAVE_KEYFRAMES = os.environ.get("MEMER_SAVE_KEYFRAMES", "1") not in {"0", "false", "False"}
SAVE_RECENT_FRAMES = os.environ.get("MEMER_SAVE_RECENT_FRAMES", "0") in {"1", "true", "True"}
LOG_JSONL = os.environ.get("MEMER_LOG_JSONL", "1") not in {"0", "false", "False"}

client = openai.OpenAI()
app = FastAPI(title="MemER Action Server (client-aligned)", version="4.0-client-aligned")

# =========================================================
# Global conversation history for memory retention
# =========================================================
# ✅ Store conversation history across tasks for memory continuity
GLOBAL_CONVERSATION_HISTORY: List[Dict[str, Any]] = []
ENABLE_CROSS_TASK_MEMORY = os.environ.get("MEMER_CROSS_TASK_MEMORY", "1") in {"1", "true", "True"}
MAX_CONVERSATION_HISTORY = int(os.environ.get("MEMER_MAX_CONVERSATION_HISTORY", "20"))

print(f"[MemER] Cross-task memory: {'Enabled' if ENABLE_CROSS_TASK_MEMORY else 'Disabled'}")
print(f"[MemER] Max conversation history: {MAX_CONVERSATION_HISTORY} turns")

# =========================================================
# JSON extraction (tolerant)
# =========================================================
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def pil_to_data_url(img: Image.Image) -> str:
    return png_bytes_to_data_url(pil_to_png_bytes(img))
def extract_first_json_object(text: str) -> str:
    if text is None:
        raise ValueError("empty model output")
    t = text.strip()
    m = _JSON_FENCE_RE.search(t)
    if m:
        candidate = m.group(1).strip()
        if candidate:
            t = candidate

    start = t.find("{")
    if start == -1:
        raise ValueError("no JSON object found")

    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]
    raise ValueError("unterminated JSON object")

def safe_parse_keyframe_positions(x: Any, k_max: int) -> List[int]:
    if not isinstance(x, list):
        return []
    out: List[int] = []
    for item in x:
        try:
            n = int(item)
        except Exception:
            continue
        if 1 <= n <= k_max:
            out.append(n)
    return sorted(set(out))

# =========================================================
# Data structures
# =========================================================
@dataclass
class StoredCompositeFrame:
    frame_id: int
    ts_ms: int
    data_url: str
    request_id: str
    local_pos_1idx: int

@dataclass
class TaskState:
    task_id: str
    global_instruction: str
    observer_window_size: int

    next_frame_id: int

    nominated_indices: List[int]
    clusters: List[List[int]]
    selected_keyframe_ids: List[int]

    current_subtask_description: str
    is_done: bool

TASKS: Dict[str, TaskState] = {}
FRAMES: Dict[str, Dict[int, StoredCompositeFrame]] = {}
SAVED_FRAME_IDS: Dict[str, set[int]] = {}

# =========================================================
# Global persistent keyframes for cross-task memory
# =========================================================
# ✅ Store persistent keyframes that survive across all tasks
# These keyframes will be prepended to every task's memory keyframes
PERSISTENT_KEYFRAMES: Dict[int, StoredCompositeFrame] = {}
PERSISTENT_KEYFRAME_IDS: List[int] = []
NEXT_PERSISTENT_ID: int = -1  # Use negative IDs to avoid conflicts with task frame IDs


def load_persistent_keyframes_on_startup():
    """
    Load persistent keyframes (already composite) from directory on server startup.

    Environment variables:
    - PERSISTENT_KEYFRAMES_DIR: Directory containing composite keyframe images
      Expected structure:
        keyframes/
        ├── keyframe_000001.png  (composite image)
        ├── keyframe_000002.png
        └── keyframe_000003.png
    """
    global NEXT_PERSISTENT_ID

    keyframes_dir = os.environ.get("PERSISTENT_KEYFRAMES_DIR", "/Users/makabaka/code/mem_vla/memer_logs/21034118-78e1-4357-8d34-3d29a6d13e25/keyframes").strip()
    if not keyframes_dir:
        print(f"[MemER] ℹ️  No PERSISTENT_KEYFRAMES_DIR set, starting with empty persistent memory")
        return

    if not os.path.exists(keyframes_dir):
        print(f"[MemER] ⚠️  PERSISTENT_KEYFRAMES_DIR not found: {keyframes_dir}")
        return

    print(f"[MemER] 📂 Loading persistent keyframes from: {keyframes_dir}")

    try:
        from pathlib import Path

        # Find composite keyframes
        keyframes_path = Path(keyframes_dir)
        composite_frames = sorted(keyframes_path.glob("keyframe_*.png"))

        if not composite_frames:
            print(f"[MemER] ⚠️  No keyframe_*.png files found in {keyframes_dir}")
            return

        # Load composite images
        loaded_count = 0
        for composite_path in composite_frames:
            try:
                # Load composite image directly
                composite_img = Image.open(composite_path).convert("RGB")
                data_url = pil_to_data_url(composite_img)

                # Store as persistent keyframe with negative ID
                fr = StoredCompositeFrame(
                    frame_id=NEXT_PERSISTENT_ID,
                    ts_ms=now_ms(),
                    data_url=data_url,
                    request_id="persistent_memory",
                    local_pos_1idx=loaded_count + 1,
                )

                PERSISTENT_KEYFRAMES[NEXT_PERSISTENT_ID] = fr
                PERSISTENT_KEYFRAME_IDS.append(NEXT_PERSISTENT_ID)
                NEXT_PERSISTENT_ID -= 1
                loaded_count += 1

            except Exception as e:
                print(f"[MemER] ❌ Failed to load {composite_path.name}: {e}")
                continue

        print(f"[MemER] ✅ Loaded {loaded_count} persistent keyframes (cross-task memory enabled)")

    except Exception as e:
        print(f"[MemER] ❌ Failed to load persistent keyframes: {e}")


# =========================================================
# Logging helpers
# =========================================================
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def now_ms() -> int:
    return int(time.time() * 1000)

def task_dir(task_id: str) -> str:
    return os.path.join(LOG_ROOT, task_id)

def task_keyframes_dir(task_id: str) -> str:
    return os.path.join(task_dir(task_id), "keyframes")

def task_jsonl_path(task_id: str) -> str:
    return os.path.join(task_dir(task_id), "events.jsonl")

def append_jsonl(task_id: str, record: Dict[str, Any]) -> None:
    if not LOG_JSONL:
        return
    _ensure_dir(task_dir(task_id))
    record = dict(record)
    record.setdefault("task_id", task_id)
    record.setdefault("ts_ms", now_ms())
    with open(task_jsonl_path(task_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def data_url_to_png_bytes(data_url: str) -> bytes:
    if "," not in data_url:
        raise ValueError("Invalid data_url")
    b64 = data_url.split(",", 1)[1]
    return base64.b64decode(b64)

def save_frame_png(task_id: str, frame: StoredCompositeFrame, subdir: str, filename: str) -> str:
    out_dir = os.path.join(task_dir(task_id), subdir)
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, filename)
    png = data_url_to_png_bytes(frame.data_url)
    with open(path, "wb") as f:
        f.write(png)
    return path

def maybe_save_recent_frames(task_id: str, recent_frames: List[StoredCompositeFrame]) -> None:
    if not SAVE_RECENT_FRAMES:
        return
    for fr in recent_frames:
        fname = f"frame_{fr.frame_id:06d}_req_{fr.request_id}_pos_{fr.local_pos_1idx}.png"
        save_frame_png(task_id, fr, "frames", fname)

def maybe_save_keyframes(task_id: str, keyframe_ids: List[int]) -> List[str]:
    if not SAVE_KEYFRAMES:
        return []
    if task_id not in SAVED_FRAME_IDS:
        SAVED_FRAME_IDS[task_id] = set()
    saved_paths: List[str] = []

    for fid in keyframe_ids:
        if fid in SAVED_FRAME_IDS[task_id]:
            continue
        fr = FRAMES.get(task_id, {}).get(fid)
        if fr is None:
            continue

        fname = f"keyframe_{fid:06d}.png"
        path = save_frame_png(task_id, fr, "keyframes", fname)

        meta_path = os.path.join(task_keyframes_dir(task_id), f"keyframe_{fid:06d}.json")
        _ensure_dir(task_keyframes_dir(task_id))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_id": task_id,
                    "frame_id": fr.frame_id,
                    "ts_ms": fr.ts_ms,
                    "request_id": fr.request_id,
                    "local_pos_1idx": fr.local_pos_1idx,
                    "composite_layout": COMPOSITE_LAYOUT,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        SAVED_FRAME_IDS[task_id].add(fid)
        saved_paths.append(path)

    return saved_paths

# =========================================================
# Image helpers
# =========================================================

def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return img.resize((nw, nh), Image.BILINEAR)

def make_composite(head: Image.Image, wrist: Image.Image) -> Image.Image:
    head = _resize_max_side(head, COMPOSITE_MAX_SIDE)
    wrist = _resize_max_side(wrist, COMPOSITE_MAX_SIDE)

    if COMPOSITE_LAYOUT == "vertical":
        target_w = max(head.size[0], wrist.size[0])

        def pad_to_w(img: Image.Image, tw: int) -> Image.Image:
            w, h = img.size
            if w == tw:
                return img
            out = Image.new("RGB", (tw, h), (0, 0, 0))
            out.paste(img, ((tw - w) // 2, 0))
            return out

        head2 = pad_to_w(head, target_w)
        wrist2 = pad_to_w(wrist, target_w)

        out = Image.new("RGB", (target_w, head2.size[1] + wrist2.size[1]), (0, 0, 0))
        out.paste(head2, (0, 0))
        out.paste(wrist2, (0, head2.size[1]))
        return out

    target_h = max(head.size[1], wrist.size[1])

    def pad_to_h(img: Image.Image, th: int) -> Image.Image:
        w, h = img.size
        if h == th:
            return img
        out = Image.new("RGB", (w, th), (0, 0, 0))
        out.paste(img, (0, (th - h) // 2))
        return out

    head2 = pad_to_h(head, target_h)
    wrist2 = pad_to_h(wrist, target_h)

    out = Image.new("RGB", (head2.size[0] + wrist2.size[0], target_h), (0, 0, 0))
    out.paste(head2, (0, 0))
    out.paste(wrist2, (head2.size[0], 0))
    return out

async def upload_to_pil_rgb(f: UploadFile) -> Image.Image:
    data = await f.read()
    return Image.open(io.BytesIO(data)).convert("RGB")

# =========================================================
# Load persistent keyframes on module import (server startup)
# =========================================================
load_persistent_keyframes_on_startup()
print(f"[MemER] ✅ Persistent memory initialized: {len(PERSISTENT_KEYFRAME_IDS)} keyframes")

# =========================================================
# MemER helpers
# =========================================================
def median_of_sorted_list(xs: List[int]) -> int:
    return xs[len(xs) // 2]

def rebuild_clusters_and_selected(nominated_indices: List[int], d: int) -> Tuple[List[List[int]], List[int]]:
    if not nominated_indices:
        return [], []
    G = sorted(nominated_indices)
    clusters: List[List[int]] = []
    cur = [G[0]]
    for i in range(1, len(G)):
        if G[i] - G[i - 1] <= d:
            cur.append(G[i])
        else:
            clusters.append(cur)
            cur = [G[i]]
    clusters.append(cur)
    selected = [median_of_sorted_list(c) for c in clusters]
    return clusters, selected

def trim_recent_keyframes(ids: List[int], max_k: int) -> List[int]:
    return ids if len(ids) <= max_k else ids[-max_k:]

# =========================================================
# Prompting (pure text action)
# =========================================================
def build_messages_for_action(global_instruction: str) -> Tuple[str, str]:
    system_text = (
        "You are a robot action selector.\n"
        "Each image is a COMPOSITE containing both main and wrist/waist views. Use both.\n"
        "Return JSON only. No markdown. No code fences.\n"
    )
    user_text = (
        "You will be given:\n"
        "- Memory keyframes: earlier composite images.\n"
        "- Recent frames: up to K composite images, indexed from 1..K.\n\n"
        f"Task:\n{global_instruction}\n\n"
        "Action format rules (VERY IMPORTANT):\n"
        "You MUST output exactly ONE action string in ONE of the following formats:\n"
        "Format A (inspection): inspect <target>\n"
        "Format B (pick and place): pick up <object> <preposition> the <source_location> and place it <preposition> the <target_location>\n"
        "Format C (reset): reset\n\n"
        "RESET action rules:\n"
        "- WHEN to reset: ONLY after a inspect action or a pick-and-place action is COMPLETELY FINISHED, the robot arm is above or extened into the box or above a recipe, and BEFORE starting a NEW pick-and-place action.\n"
        "- WHEN NOT to reset: If a pick-and-place is IN PROGRESS (object partially picked or being transported), continue the action directly - do NOT reset.\n"
        "- Reset completion criteria: In BOTH images, the robot arm is aligned parallel to what appears to be a rail or track, with the end effector open and facing downwards towards the table.\n"
        "- Sequence pattern: [pick-and-place complete] → [reset] → [new pick-and-place] → [reset] → [next pick-and-place]\n\n"
        "LOCATION naming:\n"
        "- You should distinguish the three plates using 'left', 'middle', or 'right'\n\n"
        "Do NOT add any extra prefixes or commentary.\n\n"
        "Output JSON with exactly:\n"
        '{ "action": string, "keyframe_positions": number[] }\n\n'
        "Rules for keyframe_positions:\n"
        "- unique, sorted\n"
        "- each integer must be in [1, K]\n"
        "No extra keys. No explanations.\n"
    )
    return system_text, user_text

# =========================================================
# VLM call (chat.completions) — tolerant
# =========================================================
def call_vlm_for_action(
    system_text: str,
    user_text: str,
    keyframe_data_urls: List[str],
    recent_data_urls: List[str],
) -> Tuple[str, List[int], Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]

    if keyframe_data_urls:
        user_content.append({"type": "text", "text": "\nMemory keyframes (older):"})
        for i, url in enumerate(keyframe_data_urls, start=1):
            user_content.append({"type": "text", "text": f"[Memory {i}]"})
            user_content.append({"type": "image_url", "image_url": {"url": url}})

    user_content.append({"type": "text", "text": "\nRecent frames (choose keyframe_positions from these):"})
    for i, url in enumerate(recent_data_urls, start=1):
        user_content.append({"type": "text", "text": f"[Recent {i}]"})
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        timeout=REQUEST_TIMEOUT_S,
    )

    raw_text = resp.choices[0].message.content or ""
    parsed: Optional[Dict[str, Any]] = None

    action = raw_text.strip()
    keyframe_positions: List[int] = []

    # Best-effort parse JSON; if fail, return raw text as action
    try:
        js = json.loads(extract_first_json_object(raw_text))
        if isinstance(js, dict):
            parsed = js
            if isinstance(js.get("action"), str) and js.get("action").strip():
                action = js["action"].strip()
            keyframe_positions = safe_parse_keyframe_positions(js.get("keyframe_positions"), k_max=len(recent_data_urls))
    except Exception:
        pass

    debug = {"raw_text": raw_text, "parsed_json": parsed}
    return action, keyframe_positions, debug

# =========================================================
# Public response model (client expects current_subtask_description)
# =========================================================
def public_state(ts: TaskState) -> Dict[str, Any]:
    return {
        "task_id": ts.task_id,
        "is_done": ts.is_done,
        "current_subtask_description": ts.current_subtask_description,
        # keep extra fields (harmless for client)
        "selected_keyframe_ids": ts.selected_keyframe_ids,
        "nominated_indices": ts.nominated_indices,
        "clusters": ts.clusters,
    }

class UserInstructionBody(BaseModel):
    user_new_instruction: str

# =========================================================
# Routes (client-aligned)
# =========================================================
@app.post("/tasks")
async def create_task(
    global_instruction: str = Form(..., description="High-level task description"),
    initial_waist_image: UploadFile = File(..., description="Initial waist/wrist image"),
    initial_image: UploadFile = File(..., description="Initial main/head image"),
    observer_window_size: int = Form(8),
    human_intervene_for_planner: bool = Form(False),  # accepted for compatibility; currently unused
):
    """
    创建新任务

    参数：
    - initial_waist_image, initial_image: 第一帧（必需）

    关键帧初始化：
    - 如果服务器启动时设置了 PERSISTENT_KEYFRAMES_DIR，预加载的关键帧会自动作为初始关键帧序列
    - 之后的 MemER 逻辑会继续更新这个序列
    """
    task_id = str(uuid.uuid4())

    # Decode initial images
    try:
        head0 = await upload_to_pil_rgb(initial_image)
        wrist0 = await upload_to_pil_rgb(initial_waist_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid initial image(s): {e}")

    # Init task state
    # ✅ 初始关键帧序列 = 预加载的 persistent keyframes
    ts = TaskState(
        task_id=task_id,
        global_instruction=global_instruction,
        observer_window_size=observer_window_size,
        next_frame_id=1,
        nominated_indices=[],
        clusters=[],
        selected_keyframe_ids=PERSISTENT_KEYFRAME_IDS.copy(),  # 直接使用预加载的关键帧
        current_subtask_description="",
        is_done=False,
    )
    TASKS[task_id] = ts
    FRAMES[task_id] = {}

    if PERSISTENT_KEYFRAME_IDS:
        print(f"[Task {task_id}] Initialized with {len(PERSISTENT_KEYFRAME_IDS)} persistent keyframes")

    # Store first composite frame
    request_id = str(uuid.uuid4())
    comp = make_composite(head0, wrist0)
    data_url = pil_to_data_url(comp)

    fr = StoredCompositeFrame(
        frame_id=ts.next_frame_id,
        ts_ms=now_ms(),
        data_url=data_url,
        request_id=request_id,
        local_pos_1idx=1,
    )
    FRAMES[task_id][fr.frame_id] = fr
    ts.next_frame_id += 1

    # Call VLM immediately with K=1 (recent frame) + keyframes
    system_text, user_text = build_messages_for_action(ts.global_instruction)

    # ✅ 使用当前的关键帧序列（包含预加载的）
    keyframe_data_urls = []
    for fid in ts.selected_keyframe_ids:
        if fid in PERSISTENT_KEYFRAMES:
            keyframe_data_urls.append(PERSISTENT_KEYFRAMES[fid].data_url)
        elif fid in FRAMES[task_id]:
            keyframe_data_urls.append(FRAMES[task_id][fid].data_url)

    action, keyframe_positions, debug = call_vlm_for_action(
        system_text=system_text,
        user_text=user_text,
        keyframe_data_urls=keyframe_data_urls,
        recent_data_urls=[fr.data_url],
    )
    ts.current_subtask_description = action

    append_jsonl(task_id, {
        "event": "create_task",
        "request_id": request_id,
        "global_instruction": global_instruction,
        "observer_window_size": observer_window_size,
        "human_intervene_for_planner": human_intervene_for_planner,
        "persistent_keyframe_count": len(PERSISTENT_KEYFRAME_IDS),
        "model_debug": debug,
        "returned_action": action,
        "returned_keyframe_positions": keyframe_positions,
    })

    return JSONResponse(public_state(ts))

@app.post("/tasks/{task_id}/step")
async def step(
    task_id: str,
    image: List[UploadFile] = File(..., description="Main/head camera images sequence"),
    waist_image: List[UploadFile] = File(..., description="Waist/wrist camera images sequence"),
):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    if not image:
        raise HTTPException(status_code=400, detail="No main images provided")
    if not waist_image:
        raise HTTPException(status_code=400, detail="No waist images provided")

    ts = TASKS[task_id]
    request_id = str(uuid.uuid4())

    # Decode lists (tolerant: skip bad frames)
    head_imgs: List[Image.Image] = []
    wrist_imgs: List[Image.Image] = []

    for f in image:
        try:
            head_imgs.append(await upload_to_pil_rgb(f))
        except Exception:
            continue

    for f in waist_image:
        try:
            wrist_imgs.append(await upload_to_pil_rgb(f))
        except Exception:
            continue

    if not head_imgs or not wrist_imgs:
        raise HTTPException(status_code=400, detail="Failed to decode head or waist image sequence")

    # Pair by min length (client usually sends same length)
    n = min(len(head_imgs), len(wrist_imgs))
    head_imgs = head_imgs[:n]
    wrist_imgs = wrist_imgs[:n]

    # Store composites as frames
    recent_frames: List[StoredCompositeFrame] = []
    for i in range(n):
        comp = make_composite(head_imgs[i], wrist_imgs[i])
        data_url = pil_to_data_url(comp)

        fid = ts.next_frame_id
        ts.next_frame_id += 1

        fr = StoredCompositeFrame(
            frame_id=fid,
            ts_ms=now_ms(),
            data_url=data_url,
            request_id=request_id,
            local_pos_1idx=i + 1,
        )
        FRAMES[task_id][fid] = fr
        recent_frames.append(fr)

    maybe_save_recent_frames(task_id, recent_frames)

    # Prepare recent window (last RECENT_MAX frames)
    all_frames_sorted = sorted(FRAMES[task_id].values(), key=lambda x: x.frame_id)
    recent_global = all_frames_sorted[-RECENT_MAX:]
    recent_data_urls = [fr.data_url for fr in recent_global]

    # ✅ 使用当前的关键帧序列（包含预加载的）
    keyframe_data_urls = []
    for fid in ts.selected_keyframe_ids:
        if fid in PERSISTENT_KEYFRAMES:
            keyframe_data_urls.append(PERSISTENT_KEYFRAMES[fid].data_url)
        elif fid in FRAMES[task_id]:
            keyframe_data_urls.append(FRAMES[task_id][fid].data_url)

    # Call VLM
    system_text, user_text = build_messages_for_action(ts.global_instruction)
    action, keyframe_positions, debug = call_vlm_for_action(
        system_text=system_text,
        user_text=user_text,
        keyframe_data_urls=keyframe_data_urls,
        recent_data_urls=recent_data_urls,
    )
    ts.current_subtask_description = action

    # MemER nominations: positions refer to recent_global window
    nominated_global_ids: List[int] = []
    for pos in keyframe_positions:
        if 1 <= pos <= len(recent_global):
            nominated_global_ids.append(recent_global[pos - 1].frame_id)

    ts.nominated_indices.extend(nominated_global_ids)
    ts.nominated_indices = sorted(set(ts.nominated_indices))

    # ✅ Clustering 只处理任务中产生的帧（正数 ID）
    ts.clusters, clustered_ids = rebuild_clusters_and_selected(ts.nominated_indices, MERGE_DISTANCE_D)
    clustered_ids = trim_recent_keyframes(clustered_ids, MAX_KEYFRAMES)

    # ✅ 保留预加载的关键帧 + 聚类后的关键帧
    ts.selected_keyframe_ids = PERSISTENT_KEYFRAME_IDS.copy() + clustered_ids

    saved_paths = maybe_save_keyframes(task_id, clustered_ids)  # 只保存任务产生的帧

    append_jsonl(task_id, {
        "event": "step",
        "request_id": request_id,
        "n_uploaded_pairs": n,
        "recent_window_frame_ids": [fr.frame_id for fr in recent_global],
        "model_debug": debug,
        "returned_action": action,
        "returned_keyframe_positions": keyframe_positions,
        "nominated_global_ids": nominated_global_ids,
        "clustered_keyframe_ids": clustered_ids,  # 任务产生的关键帧
        "selected_keyframe_ids": ts.selected_keyframe_ids,  # persistent + clustered
        "saved_keyframe_paths": saved_paths,
    })

    return JSONResponse(public_state(ts))

@app.post("/tasks/{task_id}/user_instruction")
async def user_instruction(task_id: str, body: UserInstructionBody):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    ts = TASKS[task_id]

    # 直接用 user instruction 覆盖全局指令（不再附加）
    ts.global_instruction = body.user_new_instruction.strip()

    append_jsonl(task_id, {
        "event": "user_instruction",
        "user_new_instruction": body.user_new_instruction,
        "global_instruction_after": ts.global_instruction,  # 可选：方便回放/排查
    })
    return JSONResponse(public_state(ts))