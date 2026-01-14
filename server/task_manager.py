"""In-memory task manager coordinating Planner and Observer.

Responsibilities:
- Keep a mapping {task_id -> TaskRuntimeState}.
- For each task:
  - Manage creation with initial combined image.
  - Append new combined images on each step from robot.
  - Call Observer to detect subtask completion.
  - Call Planner to refine plan_list (with or without new user input).
- Hide index-based subtask logic: current subtask is always derived from `plan_list`
  using `extract_current_subtask(plan_list)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from PIL import Image
import threading
import time

from .schema import (
    TaskRuntimeState,
    TaskConfig,
    TaskStateEnum,
    PendingApproval,
)
from .task_state import create_initial_task_state, save_pil_to_dir
from .image_utils import combine_two_pil_horizontally

from src.agent.multitag_planner import PlannerAgent
from src.agent.observer import ObserverAgent
from src.extractor import extract_current_subtask, is_plan_done

from src.infer.loop import (
    build_planner_user_instruction,
    sample_up_to_n_evenly,
    observer_loop_with_scheduler_window,
)

from .image_utils import RobotImageInput

class ServerTaskManager:
    """
    Central in-memory manager for all running tasks.

    - Stores TaskRuntimeState objects keyed by task_id.
    - Exposes methods for:
        - create_task
        - add_step_and_maybe_refine_robot
        - refine_with_user_instruction
    - Hides all low-level details: image paths, observer/planner invocation, etc.
    """

    def __init__(self):
        self.tasks: Dict[str, TaskRuntimeState] = {}

        # Agents are injected from outside (e.g., FastAPI startup hook).
        self.planner_agent: Optional[PlannerAgent] = None
        self.observer_agent: Optional[ObserverAgent] = None

    # ---------- Agent injection ----------

    def set_agents(self, planner: PlannerAgent, observer: ObserverAgent) -> None:
        """Inject pre-initialized PlannerAgent and ObserverAgent."""
        self.planner_agent = planner
        self.observer_agent = observer

    # ---------- Public: create task ----------

    def create_task(
        self,
        *,
        global_instruction: str,
        initial_robot_input: RobotImageInput,
        config: Optional[TaskConfig] = None,
    ) -> TaskRuntimeState:
        """
        Create a new task:

        Steps:
        - Allocate a new TaskRuntimeState.
        - Combine initial waist + main image into a single combined image.
        - Save combined image to disk and append to state.image_paths.
        - Call Planner (first round) to obtain initial plan_list + summary.
        - Extract current subtask via extract_current_subtask(plan_list).
        """
        assert self.planner_agent is not None, "PlannerAgent not set"
        assert self.observer_agent is not None, "ObserverAgent not set"

        cfg = config or TaskConfig()
        state = create_initial_task_state(global_instruction, cfg)

        # Save initial combined image
        combined_path = self._save_robot_input_as_combined_image(
            state=state,
            robot_input=initial_robot_input,
            prefix="init",
        )
        planner_images = [combined_path] if combined_path else []

        # First planner call
        user_instruction = build_planner_user_instruction(
            base_instruction=state.global_instruction,
            current_plan_list="",
            user_new_input=None,
            is_first_round=True,
        )

        res = self.planner_agent.run_refine(
            image_paths=planner_images,
            initial_plan_list="",
            user_instruction=user_instruction,
            max_tokens=4096,
            max_inner_rounds=10,
            do_reset=True,
            print_full_interactions_each_round=True,
            log_interactions_json_dir=None,
            use_cli_prompt_for_memory_view=False,
            decide_view_memory=None,
            log_memory_json_dir=None,
            drop_images_in_json=True,
        )

        state.plan_list = (res.plan_text or "").strip()
        state.summary = (res.summary or "").strip()

        # Decide if whole plan is already done
        if is_plan_done(state.plan_list):
            state.is_done = True
            state.state = TaskStateEnum.IDLE
            state.current_subtask_description = None
        else:
            subtask_desc = extract_current_subtask(state.plan_list)
            if subtask_desc:
                state.current_subtask_description = subtask_desc
                # current subtask image segment starts at the last appended index
                state.current_subtask_start_idx = len(state.image_paths) - 1 if state.image_paths else 0
                state.state = TaskStateEnum.OBSERVING
            else:
                # Plan has no recognizable subtask; treat as done
                state.is_done = True
                state.state = TaskStateEnum.IDLE
                state.current_subtask_description = None

        self.tasks[state.task_id] = state
        return state

    # ---------- Public: robot step + maybe refine ----------

    def add_step_and_maybe_refine_robot(
        self,
        task_id: str,
        robot_input: RobotImageInput,
    ) -> TaskRuntimeState:
        """
        Handle one step of robot observation:

        - Combine waist + main into one "combined" PIL image.
        - Save combined to images_dir, append path to image_paths.
        - If task is OBSERVING:
            - Run Observer on global image_paths with sliding window.
            - If Observer returns done (and no pending user instruction):
                - Run planner refine (without user_new_instruction).
        - If task is not OBSERVING:
            - Only append image; do not run observer/planner.
        """
        assert self.planner_agent is not None
        assert self.observer_agent is not None

        state = self._get_task(task_id)

        # If the task is already completed, just store the image (for logging)
        if state.is_done:
            self._save_robot_input_as_combined_image(
                state=state,
                robot_input=robot_input,
                prefix="step_done",
            )
            return state

        # If not in OBSERVING state, don't run Observer; just store image.
        if state.state != TaskStateEnum.OBSERVING:
            self._save_robot_input_as_combined_image(
                state=state,
                robot_input=robot_input,
                prefix="step_passive",
            )
            return state

        # 1. Save combined image as a new step
        combined_path = self._save_robot_input_as_combined_image(
            state=state,
            robot_input=robot_input,
            prefix="step",
        )
        if not combined_path:
            # Should not happen; we require at least main image
            return state

        # 2. Decide whether to skip observer result (if we have a pending user instruction)
        skip_observer_result = state.pending_user_instruction is not None

        # 3. Run observer on all combined images with window
        # all_imgs = state.image_paths[:]  # in chronological order
        # if not all_imgs:
        #     return state

        start = state.current_subtask_start_idx
        end = len(state.image_paths)
        segment = state.image_paths[start:end] if end > start else []

        status, seen_imgs, _last_xml = observer_loop_with_scheduler_window(
            observer=self.observer_agent,
            plan_list=state.plan_list,
            sampled_imgs=segment,
            window_size_w=state.config.observer_window_size,
            max_tokens=512,
        )

        # Debug mode: pause for approval after observer
        if state.config.debug_mode and state.config.pause_on_observer:
            state.pending_approval = PendingApproval(
                agent_type="observer",
                timestamp=time.time(),
                raw_output=_last_xml,
                parsed_output={"status": status},
                input_context={
                    "image_paths": segment,
                    "plan_list": state.plan_list
                }
            )
            state.state = TaskStateEnum.PENDING_OBSERVER_APPROVAL

            event = threading.Event()
            state.approval_event = event

            print(f"[Debug] Observer output pending approval (task_id={task_id})")
            event.wait()  # Block until user approves

            # Get approved result (possibly modified)
            status = state.approved_result.get("status", status)
            state.pending_approval = None
            state.approval_event = None
            state.approved_result = {}

        if not skip_observer_result:
            if status == "done":
                # Current subtask is completed; run planner refine without new user input
                self._run_planner_refine_without_user_instruction(state)
            else:
                # Not done; keep observing
                pass
        else:
            # If user instruction is pending, ignore this observer status
            pass

        return state

    # ---------- Public: user instruction refine ----------

    def refine_with_user_instruction(
        self,
        task_id: str,
        user_new_instruction: str,
    ) -> TaskRuntimeState:
        """
        Apply an additional user instruction to refine the current plan_list.

        - Does not accept a new image.
        - Planner will see:
           - global_instruction
           - existing plan_list
           - this new user_new_instruction
           - images belonging to the current subtask segment.
        """
        assert self.planner_agent is not None

        state = self._get_task(task_id)
        if state.is_done:
            return state

        state.pending_user_instruction = user_new_instruction.strip()
        self._run_planner_refine_with_user_instruction(state)
        state.pending_user_instruction = None
        return state

    # ---------- Internal: image handling ----------

    def _save_robot_input_as_combined_image(
        self,
        state: TaskRuntimeState,
        robot_input: RobotImageInput,
        prefix: str,
    ) -> Optional[str]:
        """
        Merge RobotImageInput into a single combined image and save.

        - If both waist_image and image are present:
            - horizontally concatenate (waist on the left, main on the right)
        - If only image is present:
            - use image directly
        - If only waist_image is present (should be rare):
            - use waist_image directly

        Returns:
          Path to the saved combined image (and appends to state.image_paths).
        """
        waist = robot_input.waist_image
        main = robot_input.image

        if waist is None and main is None:
            # No usable image
            return None

        if waist is not None and main is not None:
            combined_pil = combine_two_pil_horizontally(waist, main)
        else:
            combined_pil = main if main is not None else waist

        combined_path = save_pil_to_dir(
            state.images_dir,
            combined_pil,
            prefix=prefix,
        )
        state.image_paths.append(combined_path)
        return combined_path

    # ---------- Internal: planner refine without new instruction ----------

    def _run_planner_refine_without_user_instruction(
        self,
        state: TaskRuntimeState,
    ) -> None:
        """
        Planner refine when observer says current subtask is done.

        Logic:
        - Use images from current_subtask_start_idx to the latest as the segment.
        - sample up to 8 images from this segment.
        - Build planner user_instruction (no user_new_input).
        - Call planner.run_refine.
        - Update plan_list, summary.
        - Derive new current_subtask_description via extract_current_subtask.
        - If plan is done or no subtask found -> mark task as done + IDLE.
        - Otherwise:
            - set new current_subtask_description
            - set current_subtask_start_idx = len(image_paths)
            - set state to OBSERVING.
        """
        if state.is_done:
            return

        state.state = TaskStateEnum.PLANNING

        start = state.current_subtask_start_idx
        end = len(state.image_paths)
        segment = state.image_paths[start:end] if end > start else []
        planner_images = sample_up_to_n_evenly(segment, 8) if segment else []

        user_instruction = build_planner_user_instruction(
            base_instruction=state.global_instruction,
            current_plan_list=state.plan_list,
            user_new_input=None,
            is_first_round=False,
        )

        res = self.planner_agent.run_refine(
            image_paths=planner_images,
            initial_plan_list=state.plan_list,
            user_instruction=user_instruction,
            max_tokens=4096,
            max_inner_rounds=10,
            do_reset=True,
            print_full_interactions_each_round=True,
            log_interactions_json_dir=None,
            use_cli_prompt_for_memory_view=False,
            decide_view_memory=None,
            log_memory_json_dir=None,
            drop_images_in_json=True,
        )

        # Debug mode: pause for approval after planner
        if state.config.debug_mode and state.config.pause_on_planner:
            state.pending_approval = PendingApproval(
                agent_type="planner",
                timestamp=time.time(),
                raw_output=res.raw_xml,
                parsed_output={
                    "summary": res.summary,
                    "plan_text": res.plan_text,
                    "memory_operations": [op.__dict__ for op in res.memory_operations]
                },
                input_context={
                    "image_paths": planner_images,
                    "user_instruction": user_instruction,
                    "current_plan_list": state.plan_list
                }
            )
            state.state = TaskStateEnum.PENDING_PLANNER_APPROVAL

            event = threading.Event()
            state.approval_event = event

            print(f"[Debug] Planner output pending approval (task_id={state.task_id})")
            event.wait()  # Block until user approves

            # Get approved result (possibly modified)
            approved = state.approved_result
            plan_text = approved.get("plan_text", res.plan_text)
            summary = approved.get("summary", res.summary)
            state.pending_approval = None
            state.approval_event = None
            state.approved_result = {}
        else:
            plan_text = res.plan_text
            summary = res.summary

        state.plan_list = (plan_text or "").strip()
        state.summary = (summary or "").strip()

        if is_plan_done(state.plan_list):
            state.is_done = True
            state.state = TaskStateEnum.IDLE
            state.current_subtask_description = None
            return

        new_sub_desc = extract_current_subtask(state.plan_list)
        if not new_sub_desc:
            state.is_done = True
            state.state = TaskStateEnum.IDLE
            state.current_subtask_description = None
            return

        state.current_subtask_description = new_sub_desc
        state.current_subtask_start_idx = len(state.image_paths)
        state.state = TaskStateEnum.OBSERVING

    # ---------- Internal: planner refine with user instruction ----------

    def _run_planner_refine_with_user_instruction(
        self,
        state: TaskRuntimeState,
    ) -> None:
        """
        Planner refine when a new user instruction is provided.

        - Uses images from current_subtask_start_idx onward as context.
        - Builds planner user_instruction with user_new_input filled in.
        - Planner is allowed to significantly change the plan_list structure.
        - After refine:
            - If plan is done or no subtask: mark done + IDLE.
            - Else: extract current subtask from new plan_list and continue OBSERVING.
        """
        state.state = TaskStateEnum.PLANNING

        start = state.current_subtask_start_idx
        end = len(state.image_paths)
        segment = state.image_paths[start:end] if end > start else []
        planner_images = sample_up_to_n_evenly(segment, 8) if segment else []

        ui = (state.pending_user_instruction or "").strip()

        user_instruction = build_planner_user_instruction(
            base_instruction=state.global_instruction,
            current_plan_list=state.plan_list,
            user_new_input=ui,
            is_first_round=False,
        )

        res = self.planner_agent.run_refine(
            image_paths=planner_images,
            initial_plan_list=state.plan_list,
            user_instruction=user_instruction,
            max_tokens=4096,
            max_inner_rounds=10,
            do_reset=True,
            print_full_interactions_each_round=True,
            log_interactions_json_dir=None,
            use_cli_prompt_for_memory_view=False,
            decide_view_memory=None,
            log_memory_json_dir=None,
            drop_images_in_json=True,
        )

        # Debug mode: pause for approval after planner
        if state.config.debug_mode and state.config.pause_on_planner:
            state.pending_approval = PendingApproval(
                agent_type="planner",
                timestamp=time.time(),
                raw_output=res.raw_xml,
                parsed_output={
                    "summary": res.summary,
                    "plan_text": res.plan_text,
                    "memory_operations": [op.__dict__ for op in res.memory_operations]
                },
                input_context={
                    "image_paths": planner_images,
                    "user_instruction": user_instruction,
                    "current_plan_list": state.plan_list
                }
            )
            state.state = TaskStateEnum.PENDING_PLANNER_APPROVAL

            event = threading.Event()
            state.approval_event = event

            print(f"[Debug] Planner output pending approval (task_id={state.task_id})")
            event.wait()  # Block until user approves

            # Get approved result (possibly modified)
            approved = state.approved_result
            plan_text = approved.get("plan_text", res.plan_text)
            summary = approved.get("summary", res.summary)
            state.pending_approval = None
            state.approval_event = None
            state.approved_result = {}
        else:
            plan_text = res.plan_text
            summary = res.summary

        state.plan_list = (plan_text or "").strip()
        state.summary = (summary or "").strip()

        if is_plan_done(state.plan_list):
            state.is_done = True
            state.state = TaskStateEnum.IDLE
            state.current_subtask_description = None
            return

        new_sub_desc = extract_current_subtask(state.plan_list)
        if not new_sub_desc:
            state.is_done = True
            state.state = TaskStateEnum.IDLE
            state.current_subtask_description = None
            return

        # When plan structure may change significantly, we always just take
        # "the current subtask" from the fresh plan_list.
        state.current_subtask_description = new_sub_desc
        state.current_subtask_start_idx = len(state.image_paths)
        state.state = TaskStateEnum.OBSERVING

    # ---------- Internal: task lookup ----------

    def _get_task(self, task_id: str) -> TaskRuntimeState:
        """Retrieve a TaskRuntimeState by id or raise KeyError."""
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")
        return self.tasks[task_id]