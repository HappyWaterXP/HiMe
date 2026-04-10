"""Helpers to create and persist TaskRuntimeState.

- save_pil_to_dir: save a PIL image into a task's images directory.
- create_initial_task_state: allocate a new TaskRuntimeState with fresh dirs.
- save_task_state_json: persist task runtime state at a planner-stable boundary.
- load_task_state_json: restore task runtime state from a saved snapshot.
- fork_task_state_for_resume: branch a restored task into a fresh task directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from dataclasses import asdict
import json
import os
import time
from PIL import Image

from .schema import TaskRuntimeState, TaskConfig, make_task_dirs, TaskStateEnum


def save_pil_to_dir(images_dir: str, pil_img: Image.Image, prefix: str) -> str:
    """
    Save a PIL image into `images_dir` with a timestamped filename.

    Returns:
      Absolute path of the saved image.
    """
    ts = int(time.time() * 1000)
    fname = f"{prefix}_{ts}.png"
    path = os.path.join(images_dir, fname)
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    pil_img.save(path)
    return path


def create_initial_task_state(
    global_instruction: str,
    config: TaskConfig,
) -> TaskRuntimeState:
    """
    Create a fresh TaskRuntimeState with newly allocated dirs and default fields.
    """
    dirs = make_task_dirs()
    return TaskRuntimeState(
        task_id=dirs["task_id"],
        global_instruction=global_instruction.strip(),
        created_ts=time.time(),
        base_dir=dirs["base_dir"],
        images_dir=dirs["images_dir"],
        logs_dir=dirs["logs_dir"],
        config=config,
    )


def save_task_state_json(state: TaskRuntimeState, filepath: str) -> str:
    """
    Save a task snapshot at a planner-stable boundary.

    The snapshot is intended for resume-from-last-planner-output.
    """
    payload: Dict[str, Any] = {
        "version": "1.0",
        "task_id": state.task_id,
        "global_instruction": state.global_instruction,
        "created_ts": state.created_ts,
        "base_dir": state.base_dir,
        "images_dir": state.images_dir,
        "logs_dir": state.logs_dir,
        "plan_list": state.plan_list,
        "summary": state.summary,
        "is_done": state.is_done,
        "runtime_state": str(state.runtime_state),
        "current_subtask_description": state.current_subtask_description,
        "current_subtask_start_idx": state.current_subtask_start_idx,
        "image_paths": list(state.image_paths),
        "config": asdict(state.config),
        "extra": dict(state.extra),
        "planner_round_number": state.extra.get("last_completed_planner_round"),
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return filepath


def load_task_state_json(filepath: str) -> TaskRuntimeState:
    """Load a previously saved task snapshot."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = TaskConfig(**data.get("config", {}))
    runtime_state_raw = data.get("runtime_state", TaskStateEnum.OBSERVING.value)
    try:
        runtime_state = TaskStateEnum(runtime_state_raw)
    except Exception:
        runtime_state = TaskStateEnum.OBSERVING

    return TaskRuntimeState(
        task_id=data["task_id"],
        global_instruction=data.get("global_instruction", "").strip(),
        created_ts=float(data.get("created_ts", time.time())),
        base_dir=data["base_dir"],
        images_dir=data["images_dir"],
        logs_dir=data["logs_dir"],
        plan_list=data.get("plan_list", ""),
        summary=data.get("summary", ""),
        is_done=bool(data.get("is_done", False)),
        runtime_state=runtime_state,
        current_subtask_description=data.get("current_subtask_description"),
        current_subtask_start_idx=int(data.get("current_subtask_start_idx", 0)),
        image_paths=list(data.get("image_paths", [])),
        config=config,
        extra=dict(data.get("extra", {})),
    )


def fork_task_state_for_resume(state: TaskRuntimeState, root: str = "./_server_data") -> TaskRuntimeState:
    """
    Branch a restored task state into a new task directory for resume.

    This keeps the logical task state and historical image references, but all newly
    generated images/logs will be written under a fresh task_id/base_dir so that the
    original task folder remains immutable and can be resumed again later.
    """
    original_task_id = state.task_id
    original_base_dir = state.base_dir
    original_images_dir = state.images_dir
    original_logs_dir = state.logs_dir

    dirs = make_task_dirs(root=root)
    state.task_id = dirs["task_id"]
    state.base_dir = dirs["base_dir"]
    state.images_dir = dirs["images_dir"]
    state.logs_dir = dirs["logs_dir"]
    state.created_ts = time.time()
    state.round_logger = None

    state.extra = dict(state.extra)
    state.extra["resumed_from_task_id"] = original_task_id
    state.extra["resumed_from_base_dir"] = original_base_dir
    state.extra["resumed_from_images_dir"] = original_images_dir
    state.extra["resumed_from_logs_dir"] = original_logs_dir
    return state
