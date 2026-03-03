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

from typing import Dict, Optional, List

from .schema import (
    TaskRuntimeState,
    TaskConfig,
)
from .task_state import create_initial_task_state, save_pil_to_dir
from .image_utils import combine_two_pil_horizontally
from .round_logger import RoundLogger

from agent.multitag_planner import PlannerAgent
        # noqa
from agent.observer import ObserverAgent
from extractor import extract_current_subtask, is_plan_done

from .image_utils import RobotImageInput

PLANNER_PREFIX_EN = (
    "Your current plan list represents the latest plan you have made."
    "Based on the new input, update this plan list by adding, modifying, or marking items as completed as needed."
    "You must preserve previously completed tasks to reflect the full workflow, and your new plan must be an update of the previous plan, even when a new task arrives"
)

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
    parts.append("\n----- Current Plan List -----")
    parts.append((current_plan_list or "").strip())

    return "\n".join(parts)

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

        # Determine max_inner_rounds based on runtime config.
        max_rounds = 2 if state.config.use_memory else 1

        res = self.planner_agent.run_refine(
            image_paths=planner_images,
            initial_plan_list=None,
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

        state.plan_list = (res.plan_text or "").strip()
        state.summary = (res.summary or "").strip()

        # Log the initial planner interaction
        if state.round_logger:
            state.round_logger.add_planner_interaction(
                image_paths=planner_images,
                user_instruction=user_instruction,
                initial_plan_list="",
                result_plan_list=state.plan_list,
                result_summary=state.summary,
                raw_output=res.raw_xml or "",
                memory_operations=[op.__dict__ for op in res.memory_operations] if res.memory_operations else [],
            )
            # End first round and start observing
            state.round_logger.end_round()

        print(f"[TaskManager] First planner done")

        if is_plan_done(state.plan_list):
            state.is_done = True
            state.current_subtask_description = None
            print(f"[TaskManager] Task already complete")
        else:
            subtask_desc = extract_current_subtask(state.plan_list)
            if subtask_desc:
                state.current_subtask_description = subtask_desc
                # current subtask image segment starts at the last appended index
                state.current_subtask_start_idx = len(state.image_paths) - 1 if state.image_paths else 0
                print(f"[TaskManager] Task initialized: subtask='{subtask_desc}', start_idx={state.current_subtask_start_idx}")
            else:
                state.is_done = True
                state.current_subtask_description = None
                print(f"[TaskManager] No subtask found, marking as done")

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
            return state

        # --- 3. Optional observer bypass ---
        if not state.config.use_observer:
            print(f"[TaskManager] Observer disabled, calling planner directly")
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
            self._run_planner_refine_without_user_instruction(state)
        
        # Stuck detection (check total steps in current subtask)
        # 注意：这里 full_segment 包含了刚刚加入的一整个 buffer，所以如果 buffer 很大，
        # 可能会立即触发 stuck 逻辑，这是符合预期的（动作太多了还没做完）
        elif len(full_segment) > 50:
            print(f"[TaskManager] Task maybe stuck (segment len {len(full_segment)} > 10), calling planner refine")
            self._run_planner_refine_without_user_instruction(state)

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

        # Allow refinement even if task is done - user may want to add more work
        # If task was done, we need to reactivate it
        if state.is_done:
            print(f"[TaskManager] Task was done, reactivating for user instruction")
            state.is_done = False

        # ✅ Directly replace global_instruction
        state.global_instruction = user_new_instruction.strip()

        self._run_planner_refine_with_user_instruction(state)
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
        """

        if state.is_done:
            return

        # ---- collect segment images ----
        start = state.current_subtask_start_idx
        end = len(state.image_paths)
        segment = state.image_paths[start:end] if end > start else []

        planner_images = sample_up_to_n_evenly(segment, 8) if segment else []
        user_instruction = build_planner_user_instruction(
            base_instruction=state.global_instruction,
            current_plan_list=state.plan_list,
            is_first_round=False,
        )
        initial_plan_list = state.plan_list
        max_rounds = 2 if state.config.use_memory else 1

        # ---- call planner ----
        res = self.planner_agent.run_refine(
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

        # ---- logging ----
        if state.round_logger:
            state.round_logger.add_planner_interaction(
                image_paths=planner_images,
                user_instruction=user_instruction,
                initial_plan_list=initial_plan_list or "",
                result_plan_list=res.plan_text or "",
                result_summary=res.summary or "",
                raw_output=res.raw_xml or "",
                memory_operations=[op.__dict__ for op in res.memory_operations] if res.memory_operations else [],
            )
            state.round_logger.end_round()

        state.plan_list = (res.plan_text or "").strip()
        state.summary = (res.summary or "").strip()

        if is_plan_done(state.plan_list):
            state.is_done = True
            state.current_subtask_description = None
            return

        new_sub_desc = extract_current_subtask(state.plan_list)
        if not new_sub_desc:
            state.is_done = True
            state.current_subtask_description = None
            return

        state.current_subtask_description = new_sub_desc
        state.current_subtask_start_idx = len(state.image_paths)

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

        start = state.current_subtask_start_idx
        end = len(state.image_paths)
        segment = state.image_paths[start:end] if end > start else []
        planner_images = sample_up_to_n_evenly(segment, 8) if segment else []

        user_instruction = state.global_instruction

        max_rounds = 2 if state.config.use_memory else 1

        res = self.planner_agent.run_refine(
            image_paths=planner_images,
            initial_plan_list=state.plan_list,
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

        # Log planner interaction
        if state.round_logger:
            if not state.round_logger.current_round:
                state.round_logger.start_round()
            state.round_logger.add_planner_interaction(
                image_paths=planner_images,
                user_instruction=user_instruction,
                initial_plan_list=state.plan_list,
                result_plan_list=res.plan_text or "",
                result_summary=res.summary or "",
                raw_output=res.raw_xml or "",
                    memory_operations=[op.__dict__ for op in res.memory_operations] if res.memory_operations else [],
            )

        state.plan_list = (res.plan_text or "").strip()
        state.summary = (res.summary or "").strip()

        if is_plan_done(state.plan_list):
            state.is_done = True
            new_sub_desc = None
            state.current_subtask_description = None
            print(f"[TaskManager] Task marked as complete")
        else:
            new_sub_desc = extract_current_subtask(state.plan_list)
            state.current_subtask_description = new_sub_desc
            print(f"[TaskManager] Next subtask: {new_sub_desc}")

        # End round after planner completes
        if state.round_logger:
            state.round_logger.end_round()

        state.current_subtask_description = new_sub_desc
        state.current_subtask_start_idx = len(state.image_paths)

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
        start = state.current_subtask_start_idx
        end = len(state.image_paths)
        segment = state.image_paths[start:end] if end > start else []
        planner_images = sample_up_to_n_evenly(segment, 8) if segment else []

        # Note: global_instruction has already been updated to the new instruction
        user_instruction = build_planner_user_instruction(
            base_instruction=state.global_instruction,
            current_plan_list=state.plan_list,
            is_first_round=False,
        )

        # Determine max_inner_rounds based on runtime config
        max_rounds = 2 if state.config.use_memory else 1

        res = self.planner_agent.run_refine(
            image_paths=planner_images,
            initial_plan_list=state.plan_list,
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

        # Log planner interaction
        if state.round_logger:
            state.round_logger.add_planner_interaction(
                image_paths=planner_images,
                user_instruction=user_instruction,
                initial_plan_list=state.plan_list,
                result_plan_list=res.plan_text or "",
                result_summary=res.summary or "",
                raw_output=res.raw_xml or "",
                memory_operations=[op.__dict__ for op in res.memory_operations] if res.memory_operations else [],
            )

        plan_text = res.plan_text
        summary = res.summary

        state.plan_list = (plan_text or "").strip()
        state.summary = (summary or "").strip()

        if is_plan_done(state.plan_list):
            state.is_done = True
            state.current_subtask_description = None
            return

        new_sub_desc = extract_current_subtask(state.plan_list)
        if not new_sub_desc:
            state.is_done = True
            state.current_subtask_description = None
            return

        # When plan structure may change significantly, we always just take
        # "the current subtask" from the fresh plan_list.
        state.current_subtask_description = new_sub_desc
        state.current_subtask_start_idx = len(state.image_paths)

    # ---------- Internal: task lookup ----------

    def _get_task(self, task_id: str) -> TaskRuntimeState:
        """Retrieve a TaskRuntimeState by id or raise KeyError."""
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")
        return self.tasks[task_id]
