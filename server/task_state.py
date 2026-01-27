"""Helpers to create and persist TaskRuntimeState.

- save_pil_to_dir: save a PIL image into a task's images directory.
- create_initial_task_state: allocate a new TaskRuntimeState with fresh dirs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import os
import time
from PIL import Image

from .schema import TaskRuntimeState, TaskConfig, make_task_dirs


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