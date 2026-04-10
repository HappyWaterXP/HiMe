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
FLAT_MEMORY_API_KEY = os.environ.get("FLAT_MEMORY_OPENAI_API_KEY", "").strip()
FLAT_MEMORY_BASE_URL = os.environ.get("FLAT_MEMORY_OPENAI_BASE_URL", "").strip()
MODEL_NAME = os.environ.get("FLAT_MEMORY_MODEL_NAME", "").strip()
REQUEST_TIMEOUT_S = int(os.environ.get("FLAT_MEMORY_OPENAI_TIMEOUT_S", "180"))

COMPOSITE_LAYOUT = os.environ.get("FLAT_MEMORY_COMPOSITE_LAYOUT", "horizontal").lower()
COMPOSITE_MAX_SIDE = int(os.environ.get("FLAT_MEMORY_COMPOSITE_MAX_SIDE", "1024"))

missing = []
if not FLAT_MEMORY_API_KEY:
    missing.append("FLAT_MEMORY_OPENAI_API_KEY")
if not FLAT_MEMORY_BASE_URL:
    missing.append("FLAT_MEMORY_OPENAI_BASE_URL")
if not MODEL_NAME:
    missing.append("FLAT_MEMORY_MODEL_NAME")
if missing:
    raise ValueError(
        "Missing required Flat Memory config values: "
        + ", ".join(missing)
        + ". Set them via environment variables."
    )

client = openai.OpenAI(api_key=FLAT_MEMORY_API_KEY, base_url=FLAT_MEMORY_BASE_URL)
app = FastAPI(title="Flat Memory Action Server (no-keyframes)", version="1.0-no-keyframes")
print(f"[FlatMemory-no-memory] Model={MODEL_NAME}, base_url={FLAT_MEMORY_BASE_URL}")

# =========================================================
# JSON extraction (tolerant)
# =========================================================
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

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

# =========================================================
# Minimal state
# =========================================================
@dataclass
class TaskState:
    task_id: str
    global_instruction: str
    observer_window_size: int
    current_subtask_description: str
    is_done: bool

TASKS: Dict[str, TaskState] = {}

def now_ms() -> int:
    return int(time.time() * 1000)

# =========================================================
# Image helpers
# =========================================================
def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def pil_to_data_url(img: Image.Image) -> str:
    return png_bytes_to_data_url(pil_to_png_bytes(img))

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

    # horizontal
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
        "<target> should be a container (the left box, the middle box, the right box)\n"
        "Inspect is done when you observe that the robot arm is above a box and see its blck background.\n"
        "Format B (pick and place): pick up <object> <preposition> the <source_location> and place it <preposition> the <target_location>\n"
        "LOCATION naming:\n"
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
    recent_data_url: str,
) -> Tuple[str, Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": user_text},
        {"type": "text", "text": "\nMost recent composite frame:"},
        {"type": "image_url", "image_url": {"url": recent_data_url}},
    ]

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

    # Best-effort parse JSON; if fail, return raw text as action
    try:
        js = json.loads(extract_first_json_object(raw_text))
        if isinstance(js, dict):
            parsed = js
            if isinstance(js.get("action"), str) and js.get("action").strip():
                action = js["action"].strip()
    except Exception:
        pass

    debug = {"raw_text": raw_text, "parsed_json": parsed}
    return action, debug

# =========================================================
# Public response model
# =========================================================
def public_state(ts: TaskState) -> Dict[str, Any]:
    return {
        "task_id": ts.task_id,
        "is_done": ts.is_done,
        "current_subtask_description": ts.current_subtask_description,
        # keep field for compatibility
        "selected_keyframe_ids": [],
        "nominated_indices": [],
        "clusters": [],
    }

class UserInstructionBody(BaseModel):
    user_new_instruction: str

# =========================================================
# Routes
# =========================================================
@app.post("/tasks")
async def create_task(
    global_instruction: str = Form(..., description="High-level task description"),
    initial_waist_image: UploadFile = File(..., description="Initial waist/wrist image"),
    initial_image: UploadFile = File(..., description="Initial main/head image"),
    observer_window_size: int = Form(8),
    human_intervene_for_planner: bool = Form(False),  # accepted for compatibility; unused
):
    task_id = str(uuid.uuid4())

    # Decode initial images
    try:
        head0 = await upload_to_pil_rgb(initial_image)
        wrist0 = await upload_to_pil_rgb(initial_waist_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid initial image(s): {e}")

    ts = TaskState(
        task_id=task_id,
        global_instruction=global_instruction,
        observer_window_size=observer_window_size,
        current_subtask_description="",
        is_done=False,
    )
    TASKS[task_id] = ts

    # Make composite and call model once
    comp = make_composite(head0, wrist0)
    data_url = pil_to_data_url(comp)

    system_text, user_text = build_messages_for_action(ts.global_instruction)
    action, debug = call_vlm_for_action(system_text, user_text, data_url)
    ts.current_subtask_description = action

    return JSONResponse(public_state(ts))

@app.post("/tasks/{task_id}/step")
async def step(
    task_id: str,
    image: List[UploadFile] = File(..., description="Main/head camera images sequence"),
    waist_image: List[UploadFile] = File(..., description="Waist/wrist camera images sequence"),
):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    if not image or not waist_image:
        raise HTTPException(status_code=400, detail="No images provided")

    ts = TASKS[task_id]

    # Decode lists (tolerant)
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

    # Pair by min length, take ONLY the last pair
    n = min(len(head_imgs), len(wrist_imgs))
    head_last = head_imgs[n - 1]
    wrist_last = wrist_imgs[n - 1]

    comp = make_composite(head_last, wrist_last)
    data_url = pil_to_data_url(comp)

    system_text, user_text = build_messages_for_action(ts.global_instruction)
    action, debug = call_vlm_for_action(system_text, user_text, data_url)
    ts.current_subtask_description = action

    return JSONResponse(public_state(ts))

@app.post("/tasks/{task_id}/user_instruction")
async def user_instruction(task_id: str, body: UserInstructionBody):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    ts = TASKS[task_id]
    ts.global_instruction = body.user_new_instruction.strip()
    return JSONResponse(public_state(ts))
