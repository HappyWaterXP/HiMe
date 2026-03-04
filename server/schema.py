"""Core task schemas for the server-side task runtime.

Defines:
- TaskConfig: configuration for observer / planner behavior.
- TaskRuntimeState: full server-side runtime state (includes internal fields).
- make_task_dirs: create per-task directory structure for logs & images.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, TYPE_CHECKING, Literal
import time
import os
import uuid

if TYPE_CHECKING:
    from server.round_logger import RoundLogger


class TaskStateEnum(str, Enum):
    OBSERVING = "observing"
    PLANNER_RUNNING = "planner_running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class TaskConfig:
    """
    Configuration that affects how Planner / Observer behave, but not
    the core task logic.

    All fields are optional knobs for tuning behavior.
    """
    # Sliding window size for Observer (in number of images)
    observer_window_size: int = 8

    # Whether to allow human intervention for planner calls
    # （这里只控制"是否可能有人类介入"，但不再包含 debug / 审批流程）
    human_intervene_for_planner: bool = False

    # Runtime switches for future extensions.
    # Keep defaults on the baseline main path for maintainability.
    use_observer: bool = True
    use_memory: bool = True

    # Planner execution behavior:
    # - sync: block request until planner returns
    # - async: run planner in background for user-instruction refinement
    planner_execution_mode: Literal["sync", "async"] = "sync"


@dataclass
class TaskRuntimeState:
    """
    Full internal runtime state of a task.

    *Execution-critical fields*:
      - task_id
      - global_instruction
      - is_done
      - plan_list
      - summary
      - current_subtask_description
      - current_subtask_start_idx
      - image_paths

    *Logging / bookkeeping fields*:
      - created_ts
      - base_dir, images_dir, logs_dir
      - config
      - pending_user_instruction
      - extra
    """
    # Identity
    task_id: str
    global_instruction: str          # initial high-level user instruction
    created_ts: float

    # Storage dirs (for logs / images)
    base_dir: str
    images_dir: str
    logs_dir: str

    # Planner outputs
    plan_list: str = ""              # full multi-step plan
    summary: str = ""                # high-level summary
    is_done: bool = False            # whether entire task is done
    runtime_state: TaskStateEnum = TaskStateEnum.OBSERVING

    # Current subtask description
    current_subtask_description: Optional[str] = None

    # In image_paths, where the current subtask's image segment starts
    current_subtask_start_idx: int = 0

    # Global combined image sequence (in chronological order)
    image_paths: List[str] = field(default_factory=list)

    # Execution configuration
    config: TaskConfig = field(default_factory=TaskConfig)

    # For any extra metadata / debug info
    extra: Dict[str, Any] = field(default_factory=dict)

    # Round logger for tracking interactions
    round_logger: Optional["RoundLogger"] = None


def make_task_dirs(root: str = "./_server_data") -> Dict[str, str]:
    """
    Create per-task directory structure under `root`.

    Returns:
      - task_id
      - base_dir
      - images_dir
      - logs_dir
    """
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    task_id = f"task_{ts_str}_{uuid.uuid4().hex[:8]}"
    base_dir = os.path.join(root, task_id)
    images_dir = os.path.join(base_dir, "images")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    return {
        "task_id": task_id,
        "base_dir": base_dir,
        "images_dir": images_dir,
        "logs_dir": logs_dir,
    }
