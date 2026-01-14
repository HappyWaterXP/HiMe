"""Core task schemas for the server-side task runtime.

Defines:
- TaskStateEnum: high-level task state.
- TaskConfig: configuration for observer / planner behavior.
- TaskRuntimeState: full server-side runtime state (includes internal fields).
- make_task_dirs: create per-task directory structure for logs & images.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import time
import os
import uuid
import threading


class TaskStateEnum(str, Enum):
    """High-level task state."""
    PLANNING = "PLANNING"   # Planner is running or about to run
    OBSERVING = "OBSERVING" # Waiting for / processing new images
    IDLE = "IDLE"           # Task finished or paused
    # Debug mode: waiting for approval
    PENDING_OBSERVER_APPROVAL = "PENDING_OBSERVER_APPROVAL"
    PENDING_PLANNER_APPROVAL = "PENDING_PLANNER_APPROVAL"


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
    human_intervene_for_planner: bool = False

    # (Optional) server-side consecutive done control, reserved for extension
    use_server_consecutive: bool = False
    required_consecutive_done: int = 2

    # Debug mode configuration (default: disabled)
    debug_mode: bool = False
    pause_on_observer: bool = False
    pause_on_planner: bool = False


@dataclass
class PendingApproval:
    """Pending agent output waiting for user approval (debug mode only)"""
    agent_type: str  # "observer" | "planner"
    timestamp: float
    raw_output: str
    parsed_output: Dict[str, Any]
    input_context: Dict[str, Any]


@dataclass
class TaskRuntimeState:
    """
    Full internal runtime state of a task.

    *Execution-critical fields*:
      - task_id
      - global_instruction
      - state
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
      - consecutive_done_count
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

    # Current subtask (no index is used in logic anymore)
    current_subtask_description: Optional[str] = None

    # In image_paths, where the current subtask's image segment starts
    current_subtask_start_idx: int = 0

    # Global combined image sequence (in chronological order)
    image_paths: List[str] = field(default_factory=list)

    # Optional server-side consecutive done count (currently unused)
    consecutive_done_count: int = 0

    # Current high-level state of the task
    state: TaskStateEnum = TaskStateEnum.PLANNING

    # Pending user instruction waiting to be applied by planner
    pending_user_instruction: Optional[str] = None

    # Execution configuration
    config: TaskConfig = field(default_factory=TaskConfig)

    # For any extra metadata / debug info
    extra: Dict[str, Any] = field(default_factory=dict)

    # Debug mode: pending approval data
    pending_approval: Optional[PendingApproval] = None
    approval_event: Optional[threading.Event] = None
    approved_result: Dict[str, Any] = field(default_factory=dict)


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