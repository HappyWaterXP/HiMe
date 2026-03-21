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
- Log all interactions in round-based format.
"""

from __future__ import annotations

from typing import Dict, Optional, List, Any
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
import threading

from .schema import (
    TaskRuntimeState,
    TaskConfig,
    TaskStateEnum,
)
from .task_state import create_initial_task_state, save_pil_to_dir
from .image_utils import combine_two_pil_horizontally
from .round_logger import RoundLogger

from agent.multitag_planner import PlannerAgent
from agent.observer import ObserverAgent
from extractor import extract_current_subtask, is_plan_done, ensure_plan_has_current

from .image_utils import RobotImageInput

PLANNER_PREFIX_EN = (
    "Your current plan list represents the latest plan you have made."
    "Based on the new input, update this plan list by adding, modifying, or marking items as completed as needed."
    "You must preserve previously completed tasks to reflect the full workflow, and your new plan must be an update of the previous plan, even when a new task arrives"
)
PLANNER_DONE_EXTENSION_EN = (
    "The current plan list is fully completed, but the user has now provided a new instruction."
    "Keep the completed plan as history only, and extend it with any new subtasks required by the new instruction."
    "In your final plan_list, preserve the old completed lines under a past-history divider, then write the new active plan under a current-plan divider."
    "Do not assume that an old [done] line is automatically [done] for the new instruction."
    "If the new instruction requires reversing, changing, or building on the old result, write new subtasks for that work."
    "Any subtask newly introduced by the new instruction must start as [current] or [pending] unless the current TURN 1 images directly show it is already complete."
    "If there is no current TURN 1 image evidence, do not mark newly introduced subtasks as [done]."
    "Every plan line must follow the fixed pick-and-place sentence pattern from the system prompt."
    "Do not output abstract plan lines about satisfying preferences or goals."
    "Do not return an all-done plan unless the new instruction is already satisfied by the current world state."
)

DONE_TASK_REPLAN_TAIL_FRAMES = 1
PAST_PLAN_DIVIDER = "----- Past Plan History -----"
CURRENT_PLAN_DIVIDER = "----- Current Active Plan -----"


@dataclass
class AsyncPlannerJob:
    """Background planner job metadata."""
    task_id: str
    user_instruction: str
    planner_images: List[str]
    initial_plan_list: str
    is_user_instruction_update: bool
    target_global_instruction: Optional[str] = None
    mode: str = "update_plan"  # "update_plan" | "direct_instruction"

def build_planner_user_instruction(
    base_instruction: str,
    current_plan_list: str,
    is_first_round: bool,
) -> str:
    """
    构建 Planner 的用户指令

    - 每次调用都包含 base_instruction（即当前的 global_instruction）
    - 首轮：只有 base_instruction
    - 非首轮：base_instruction + 前缀 + 当前计划

    注意：不再需要 user_new_input 参数，因为新的用户指令会直接替换 global_instruction
    """
    if is_first_round:
        # 首轮：只传 global instruction
        return base_instruction

    # 非首轮：global instruction + 前缀 + 当前计划
    parts: List[str] = []
    parts.append(base_instruction)
    parts.append("\n")
    parts.append(PLANNER_PREFIX_EN)
    normalized_plan = (current_plan_list or "").strip()
    if normalized_plan:
        if PAST_PLAN_DIVIDER in normalized_plan or CURRENT_PLAN_DIVIDER in normalized_plan:
            parts.append("\n")
            parts.append(normalized_plan)
        else:
            parts.append(f"\n{CURRENT_PLAN_DIVIDER}")
            parts.append(normalized_plan)

    return "\n".join(parts)


def merge_past_plan_history(past_plan: str, new_plan: str) -> str:
    past = (past_plan or "").strip()
    current = (new_plan or "").strip()
    if not past:
        return current
    if not current:
        return "\n".join([PAST_PLAN_DIVIDER, past]).strip()
    if PAST_PLAN_DIVIDER in current:
        return current
    return "\n".join([PAST_PLAN_DIVIDER, past, CURRENT_PLAN_DIVIDER, current]).strip()

def sample_up_to_n_evenly(paths: List[str], n: int) -> List[str]:
    """
    If len(paths) > n, return exactly n samples evenly spaced across the list,
    always keeping the last element. If len(paths) <= n, return paths as-is.
    """
    L = len(paths)
    if n <= 0 or L == 0:
        return []
    if L <= n:
        return paths
    if n == 1:
        return [paths[-1]]

    # Evenly spaced indices from 0..L-1, inclusive; last index always L-1.
    # i in [0, n-1]
    indices = [(i * (L - 1)) // (n - 1) for i in range(n)]
    # indices 单调不减，且 indices[-1] == L-1

    return [paths[i] for i in indices]

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

        # Async planner infrastructure (used when task config mode == "async").
        self._planner_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="planner-bg")
        self._planner_jobs: Dict[str, Future] = {}
        self._planner_job_meta: Dict[str, AsyncPlannerJob] = {}
        self._planner_call_lock = threading.RLock()

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
        # import pdb; pdb.set_trace()
        print(f"[TaskManager] Create task: config={config}")
        assert self.planner_agent is not None, "PlannerAgent not set"
        assert self.observer_agent is not None, "ObserverAgent not set"

        cfg = config or TaskConfig()
        state = create_initial_task_state(global_instruction, cfg)
        state.extra.setdefault("planner_status", "idle")
        state.extra.setdefault("planner_last_error", None)
        state.extra.setdefault("needs_initial_instruction", cfg.planner_execution_mode == "async")
        state.runtime_state = TaskStateEnum.PLANNER_RUNNING

        # Initialize round logger
        state.round_logger = RoundLogger(state.logs_dir)
        state.round_logger.start_round()  # Start first round

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
            is_first_round=True,
        )

        res = self._call_planner_run_refine(
            state=state,
            planner_images=planner_images,
            user_instruction=user_instruction,
            initial_plan_list=None,
        )

        self._log_planner_interaction(
            state=state,
            planner_images=planner_images,
            user_instruction=user_instruction,
            initial_plan_list="",
            planner_result=res,
            ensure_round_started=False,
            end_round=True,
        )

        print(f"[TaskManager] First planner done")
        self._apply_planner_result_to_task_state(
            state,
            plan_text=res.plan_text or "",
            summary=res.summary or "",
        )
        if state.is_done:
            print(f"[TaskManager] Task already complete")
        else:
            print(
                f"[TaskManager] Task initialized: subtask='{state.current_subtask_description}', "
                f"start_idx={state.current_subtask_start_idx}"
            )

        self.tasks[state.task_id] = state
        print(f"[TaskManager] Task {state.task_id} created and stored")
        return state

    # ---------- Public: robot step + maybe refine ----------

    def add_step_and_maybe_refine_robot(
        self,
        task_id: str,
        robot_input: RobotImageInput,
    ) -> TaskRuntimeState:
        """
        Handle robot observation (supports both single frame and image buffer):

        1. Normalize input (single image -> list).
        2. Iterate through all images in the buffer:
           - Combine waist + main.
           - Save to disk and update state.image_paths.
        3. Run Observer/Planner logic ONCE based on the latest state.
        """
        # Reuse the single-frame save path by wrapping buffer items.
        from server.image_utils import RobotImageInput as SingleFrameInput 

        assert self.planner_agent is not None
        assert self.observer_agent is not None

        state = self._get_task(task_id)
        self._flush_async_planner_if_ready(state)
        
        # --- 1. Normalize Inputs to Lists ---
        # 无论由 create_task 传入单图，还是 step 传入 buffer，统一转为 list
        main_imgs = robot_input.image if isinstance(robot_input.image, list) else [robot_input.image]
        
        waist_imgs = robot_input.waist_image
        if waist_imgs is None:
            waist_imgs = [None] * len(main_imgs)
        elif not isinstance(waist_imgs, list):
            waist_imgs = [waist_imgs] # 单图转列表
            
        # 简单的长度对齐检查
        if len(main_imgs) != len(waist_imgs):
            print(f"[TaskManager] Warning: Mismatch in image buffer lengths. Main: {len(main_imgs)}, Waist: {len(waist_imgs)}")
            # 取最短长度，防止 crash
            min_len = min(len(main_imgs), len(waist_imgs))
            main_imgs = main_imgs[:min_len]
            waist_imgs = waist_imgs[:min_len]

        print(f"\n[TaskManager] add_step: task_id={task_id}, is_done={state.is_done}, "
              f"buffer_size={len(main_imgs)}, total_history={len(state.image_paths)}")

        # --- 2. Process Buffer (Save all images) ---
        # 如果任务已完成，我们只保存图片用于日志，不运行逻辑
        prefix = "step_done" if state.is_done else "step"
        
        for i, (m_img, w_img) in enumerate(zip(main_imgs, waist_imgs)):
            # 构造一个临时的单帧 Input，复用现有的保存逻辑
            # 注意：这里假设 _save_robot_input_as_combined_image 内部处理了 state.image_paths 的 append
            single_input = SingleFrameInput(waist_image=w_img, image=m_img)
            
            self._save_robot_input_as_combined_image(
                state=state,
                robot_input=single_input,
                prefix=prefix,
            )

        if state.is_done:
            print(f"[TaskManager] Task already done, images stored.")
            state.runtime_state = TaskStateEnum.DONE
            return state

        # Async bootstrap: first step must block until planner produces a runnable instruction.
        if self._is_async_mode(state) and bool(state.extra.get("needs_initial_instruction", False)):
            state.extra["needs_initial_instruction"] = False
            print("[TaskManager] Async bootstrap step: waiting planner output to ensure runnable instruction.")
            self._wait_for_running_async_planner(state)
            if not state.is_done:
                self._run_planner_refine_without_observer(state)
            return state

        # --- 3. Optional observer bypass ---
        if not state.config.use_observer:
            print(f"[TaskManager] Observer disabled, calling planner directly")
            if self._is_async_mode(state):
                self._schedule_async_plan_refresh(state, mode="direct_instruction")
            else:
                self._run_planner_refine_without_observer(state)
            return state

        # --- 4. STANDARD MODE: Run observer ---
        # 此时 state.image_paths 已经包含了刚才 buffer 里的所有新图片
        print(f"[TaskManager] Buffer saved, now running observer")

        # Get all images from current subtask start to end
        start = state.current_subtask_start_idx
        end = len(state.image_paths)
        full_segment = state.image_paths[start:end] if end > start else []

        # For observer, only use latest window_size images
        w = state.config.observer_window_size
        if len(full_segment) > w:
            observer_images = full_segment[-w:]
            print(f"[TaskManager] Full segment: {len(full_segment)}, giving observer latest {len(observer_images)}")
        else:
            observer_images = full_segment

        if not observer_images:
            print("[TaskManager] Warning: No images for observer.")
            return state

        print(f"[TaskManager] Running observer on {len(observer_images)} images")

        # Run observer agent
        r = self.observer_agent.run(
            image_paths=observer_images,
            plan_list=state.plan_list,
            max_tokens=512,
        )
        status = r.status.strip().lower() if r.status else "not_done"

        # Log observer interaction
        if state.round_logger:
            if not state.round_logger.current_round:
                state.round_logger.start_round()
            state.round_logger.add_observer_interaction(
                image_paths=observer_images,
                subtask=state.current_subtask_description or state.plan_list,
                status=status,
                raw_output=r.raw_xml or "",
                timestamp=None,
            )

        print(f"[TaskManager] Observer returned status='{status}'")

        # --- 5. Decisions ---
        
        # If observer says done
        if status == "done":
            print(f"[TaskManager] Observer says done, calling planner refine")
            if self._is_async_mode(state):
                self._schedule_async_plan_refresh(state, mode="update_plan")
            else:
                self._run_planner_refine_without_user_instruction(state)
        
        # Stuck detection (check total steps in current subtask)
        # 注意：这里 full_segment 包含了刚刚加入的一整个 buffer，所以如果 buffer 很大，
        # 可能会立即触发 stuck 逻辑，这是符合预期的（动作太多了还没做完）
        elif len(full_segment) > 50:
            print(f"[TaskManager] Task maybe stuck (segment len {len(full_segment)} > 50), calling planner refine")
            if self._is_async_mode(state):
                self._schedule_async_plan_refresh(state, mode="update_plan")
            else:
                self._run_planner_refine_without_user_instruction(state)
        else:
            # Keep observing with current instruction.
            state.runtime_state = TaskStateEnum.OBSERVING

        return state

    # ---------- Public: user instruction refine ----------

    def refine_with_user_instruction(
        self,
        task_id: str,
        user_new_instruction: str,
    ) -> TaskRuntimeState:
        """
        Apply an additional user instruction to refine the current plan_list.

        - Directly replaces global_instruction with the new instruction.
        - Planner will see:
           - new global_instruction (replaced)
           - existing plan_list
           - images belonging to the current subtask segment.

        Note: This method now allows refinement even if the task is marked as done,
        enabling users to extend or modify completed tasks.
        """
        assert self.planner_agent is not None

        state = self._get_task(task_id)
        self._flush_async_planner_if_ready(state)

        # Allow refinement even if task is done - user may want to add more work
        # If task was done, we need to reactivate it
        if state.is_done:
            print(f"[TaskManager] Task was done, reactivating for user instruction")
            state.is_done = False
            state.runtime_state = TaskStateEnum.OBSERVING
            state.summary = ""
            state.current_subtask_description = None
            state.current_subtask_start_idx = max(0, len(state.image_paths) - DONE_TASK_REPLAN_TAIL_FRAMES)
            state.extra["extend_from_done"] = state.config.use_memory

        user_new_instruction = user_new_instruction.strip()
        # User instruction must be blocking. If async planner is running, wait for it first.
        self._wait_for_running_async_planner(state)

        # Apply instruction immediately and block on planner.
        state.global_instruction = user_new_instruction
        self._run_planner_refine_with_user_instruction(state)
        return state

    # ---------- Internal: image handling ----------

    def _is_async_mode(self, state: TaskRuntimeState) -> bool:
        return state.config.planner_execution_mode == "async"

    def _planner_job_running(self, task_id: str) -> bool:
        fut = self._planner_jobs.get(task_id)
        return fut is not None and not fut.done()

    def _collect_current_segment_images(self, state: TaskRuntimeState, max_n: int = 8) -> List[str]:
        start = state.current_subtask_start_idx
        end = len(state.image_paths)
        segment = state.image_paths[start:end] if end > start else []
        return sample_up_to_n_evenly(segment, max_n) if segment else []

    def _collect_recent_segment_images(self, state: TaskRuntimeState, max_n: int = 8) -> List[str]:
        start = state.current_subtask_start_idx
        end = len(state.image_paths)
        segment = state.image_paths[start:end] if end > start else []
        return segment[-max_n:] if segment else []

    def _collect_planner_images(self, state: TaskRuntimeState, max_n: int = 8) -> List[str]:
        if state.config.planner_image_mode == "latest_frame":
            return [state.image_paths[-1]] if state.image_paths else []
        if state.config.planner_image_mode == "recent_window":
            return self._collect_recent_segment_images(state, max_n=max_n)
        return self._collect_current_segment_images(state, max_n=max_n)

    def _planner_sees_history(self, state: TaskRuntimeState) -> bool:
        return state.config.use_memory

    def _call_planner_run_refine(
        self,
        *,
        state: TaskRuntimeState,
        planner_images: List[str],
        user_instruction: str,
        initial_plan_list: Optional[str],
    ):
        assert self.planner_agent is not None
        max_rounds = 2 if state.config.use_memory else 1
        with self._planner_call_lock:
            return self.planner_agent.run_refine(
                image_paths=planner_images,
                initial_plan_list=initial_plan_list,
                user_instruction=user_instruction,
                max_tokens=4096,
                max_inner_rounds=max_rounds,
                do_reset=True,
                print_full_interactions_each_round=False,
                log_interactions_json_dir=state.logs_dir + "/interactions",
                use_cli_prompt_for_memory_view=False,
                decide_view_memory=None,
                log_memory_json_dir=state.logs_dir + "/memory",
                drop_images_in_json=True,
            )

    @staticmethod
    def _serialize_memory_operations(memory_operations: Any) -> List[Dict[str, Any]]:
        if not memory_operations:
            return []
        return [op.__dict__ for op in memory_operations]

    def _log_planner_interaction(
        self,
        *,
        state: TaskRuntimeState,
        planner_images: List[str],
        user_instruction: str,
        initial_plan_list: Optional[str],
        planner_result: Any,
        ensure_round_started: bool,
        end_round: bool,
    ) -> None:
        if not state.round_logger:
            return
        if ensure_round_started and not state.round_logger.current_round:
            state.round_logger.start_round()

        state.round_logger.add_planner_interaction(
            image_paths=planner_images,
            user_instruction=user_instruction,
            initial_plan_list=(initial_plan_list or ""),
            result_plan_list=planner_result.plan_text or "",
            result_summary=planner_result.summary or "",
            raw_output=planner_result.raw_xml or "",
            memory_operations=self._serialize_memory_operations(planner_result.memory_operations),
        )
        if end_round:
            state.round_logger.end_round()

    def _apply_planner_result_to_task_state(
        self,
        state: TaskRuntimeState,
        *,
        plan_text: str,
        summary: str,
    ) -> None:
        normalized_plan = ensure_plan_has_current((plan_text or "").strip())
        state.plan_list = normalized_plan
        state.summary = (summary or "").strip()

        if is_plan_done(state.plan_list):
            state.is_done = True
            state.current_subtask_description = None
            state.runtime_state = TaskStateEnum.DONE
            return

        new_sub_desc = extract_current_subtask(state.plan_list)
        if not new_sub_desc:
            state.is_done = True
            state.current_subtask_description = None
            state.runtime_state = TaskStateEnum.DONE
            return

        state.is_done = False
        state.runtime_state = TaskStateEnum.OBSERVING
        state.current_subtask_description = new_sub_desc
        state.current_subtask_start_idx = len(state.image_paths)

    def _schedule_async_plan_refresh(
        self,
        state: TaskRuntimeState,
        *,
        mode: str,
    ) -> None:
        """
        Queue planner refine triggered by observer/stuck/runtime flow.
        mode:
          - update_plan: update based on current plan + planner prefix
          - direct_instruction: use global instruction directly
        """
        task_id = state.task_id
        if self._planner_job_running(task_id):
            # Observer/stuck/no-observer triggered refresh should NOT queue in async mode.
            # Keep running with old instruction until current planner job finishes.
            return

        planner_images = self._collect_planner_images(state, max_n=8)
        if not self._planner_sees_history(state):
            user_instruction = state.global_instruction
            initial_plan_list = None
        elif mode == "direct_instruction":
            user_instruction = state.global_instruction
            initial_plan_list = state.plan_list
        else:
            user_instruction = build_planner_user_instruction(
                base_instruction=state.global_instruction,
                current_plan_list=state.plan_list,
                is_first_round=False,
            )
            initial_plan_list = state.plan_list

        job = AsyncPlannerJob(
            task_id=task_id,
            user_instruction=user_instruction,
            planner_images=planner_images,
            initial_plan_list=initial_plan_list,
            is_user_instruction_update=False,
            mode=mode,
        )
        self._submit_async_job(state, job)

    def _submit_async_job(self, state: TaskRuntimeState, job: AsyncPlannerJob) -> None:
        task_id = state.task_id

        def _worker():
            return self._call_planner_run_refine(
                state=state,
                planner_images=job.planner_images,
                user_instruction=job.user_instruction,
                initial_plan_list=job.initial_plan_list,
            )

        fut = self._planner_executor.submit(_worker)
        self._planner_jobs[task_id] = fut
        self._planner_job_meta[task_id] = job
        state.extra["planner_status"] = "running"
        state.extra["planner_last_error"] = None
        state.runtime_state = TaskStateEnum.PLANNER_RUNNING

    def _flush_async_planner_if_ready(self, state: TaskRuntimeState) -> None:
        task_id = state.task_id
        fut = self._planner_jobs.get(task_id)
        if fut is None or not fut.done():
            return

        job = self._planner_job_meta.pop(task_id, None)
        self._planner_jobs.pop(task_id, None)
        if job is None:
            return

        try:
            res = fut.result()
        except Exception as e:
            state.extra["planner_status"] = "failed"
            state.extra["planner_last_error"] = str(e)
            state.summary = f"Async planner failed: {e}"
            state.runtime_state = TaskStateEnum.FAILED
            return

        if job.is_user_instruction_update and job.target_global_instruction:
            state.global_instruction = job.target_global_instruction

        self._log_planner_interaction(
            state=state,
            planner_images=job.planner_images,
            user_instruction=job.user_instruction,
            initial_plan_list=job.initial_plan_list,
            planner_result=res,
            ensure_round_started=True,
            end_round=True,
        )

        self._apply_planner_result_to_task_state(
            state,
            plan_text=res.plan_text or "",
            summary=res.summary or "",
        )
        state.extra["planner_status"] = "idle"
        state.extra["planner_last_error"] = None

    def _wait_for_running_async_planner(self, state: TaskRuntimeState) -> None:
        """Block until current async planner job finishes, then flush result into state."""
        task_id = state.task_id
        fut = self._planner_jobs.get(task_id)
        if fut is None:
            return
        if not fut.done():
            try:
                fut.result()
            except Exception as e:
                state.extra["planner_status"] = "failed"
                state.extra["planner_last_error"] = str(e)
                state.summary = f"Async planner failed: {e}"
                state.runtime_state = TaskStateEnum.FAILED
                raise
        self._flush_async_planner_if_ready(state)

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
        """

        if state.is_done:
            return
        state.runtime_state = TaskStateEnum.PLANNER_RUNNING

        planner_images = self._collect_planner_images(state, max_n=8)
        if self._planner_sees_history(state):
            user_instruction = build_planner_user_instruction(
                base_instruction=state.global_instruction,
                current_plan_list=state.plan_list,
                is_first_round=False,
            )
            initial_plan_list = state.plan_list
        else:
            user_instruction = state.global_instruction
            initial_plan_list = None
        res = self._call_planner_run_refine(
            state=state,
            planner_images=planner_images,
            user_instruction=user_instruction,
            initial_plan_list=initial_plan_list,
        )

        self._log_planner_interaction(
            state=state,
            planner_images=planner_images,
            user_instruction=user_instruction,
            initial_plan_list=initial_plan_list,
            planner_result=res,
            ensure_round_started=False,
            end_round=True,
        )

        self._apply_planner_result_to_task_state(
            state,
            plan_text=res.plan_text or "",
            summary=res.summary or "",
        )

    # ---------- Internal: planner refine without observer ----------

    def _run_planner_refine_without_observer(
        self,
        state: TaskRuntimeState,
    ) -> None:
        """
        Planner refine when observer is disabled.

        This is called every time a new image arrives (no observer filtering).

        Logic:
        - Use images from current_subtask_start_idx to the latest.
        - Call planner to refresh plan list directly.
        - Update state with new subtask and completion status.
        - Use runtime config to control inner rounds.
        """
        if state.is_done:
            return
        state.runtime_state = TaskStateEnum.PLANNER_RUNNING

        planner_images = self._collect_planner_images(state, max_n=8)

        user_instruction = state.global_instruction
        initial_plan_list = state.plan_list
        if not state.config.use_memory:
            initial_plan_list = None

        res = self._call_planner_run_refine(
            state=state,
            planner_images=planner_images,
            user_instruction=user_instruction,
            initial_plan_list=initial_plan_list,
        )

        self._log_planner_interaction(
            state=state,
            planner_images=planner_images,
            user_instruction=user_instruction,
            initial_plan_list=initial_plan_list,
            planner_result=res,
            ensure_round_started=True,
            end_round=False,
        )

        self._apply_planner_result_to_task_state(
            state,
            plan_text=res.plan_text or "",
            summary=res.summary or "",
        )
        new_sub_desc = state.current_subtask_description
        if state.is_done:
            print(f"[TaskManager] Task marked as complete")
        else:
            print(f"[TaskManager] Next subtask: {new_sub_desc}")

        # End round after planner completes
        if state.round_logger:
            state.round_logger.end_round()

    # ---------- Internal: planner refine with user instruction ----------

    def _run_planner_refine_with_user_instruction(
        self,
        state: TaskRuntimeState,
    ) -> None:
        """
        Planner refine when a new user instruction is provided.

        - Uses images from current_subtask_start_idx onward as context.
        - Builds planner user_instruction with updated global_instruction.
        - Planner is allowed to significantly change the plan_list structure.
        - After refine:
            - If plan is done or no subtask: mark done + IDLE.
            - Else: extract current subtask from new plan_list and continue OBSERVING.
        """
        state.runtime_state = TaskStateEnum.PLANNER_RUNNING
        planner_images = self._collect_planner_images(state, max_n=8)
        extend_from_done = bool(state.extra.pop("extend_from_done", False))

        # Note: global_instruction has already been updated to the new instruction
        if not self._planner_sees_history(state):
            user_instruction = state.global_instruction
            initial_plan_list = None
            extend_from_done = False
        elif extend_from_done:
            past_plan_history = (state.plan_list or "").strip()
            user_instruction = "\n".join(
                [
                    state.global_instruction,
                    "",
                    PLANNER_DONE_EXTENSION_EN,
                    PAST_PLAN_DIVIDER,
                    past_plan_history,
                ]
            )
            initial_plan_list = state.plan_list
        else:
            user_instruction = build_planner_user_instruction(
                base_instruction=state.global_instruction,
                current_plan_list=state.plan_list,
                is_first_round=False,
            )
            initial_plan_list = state.plan_list

        res = self._call_planner_run_refine(
            state=state,
            planner_images=planner_images,
            user_instruction=user_instruction,
            initial_plan_list=initial_plan_list,
        )

        final_plan_text = res.plan_text or ""
        if extend_from_done:
            final_plan_text = merge_past_plan_history(past_plan_history, final_plan_text)
            res.plan_text = final_plan_text

        self._log_planner_interaction(
            state=state,
            planner_images=planner_images,
            user_instruction=user_instruction,
            initial_plan_list=initial_plan_list,
            planner_result=res,
            ensure_round_started=False,
            end_round=False,
        )

        self._apply_planner_result_to_task_state(
            state,
            plan_text=final_plan_text,
            summary=res.summary or "",
        )

    # ---------- Internal: task lookup ----------

    def _get_task(self, task_id: str) -> TaskRuntimeState:
        """Retrieve a TaskRuntimeState by id or raise KeyError."""
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")
        return self.tasks[task_id]
