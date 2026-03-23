"""FastAPI HTTP server for multi-step robot visual task execution.

- Exposes task lifecycle APIs:
  - POST /tasks: create a new task with initial combined image.
  - POST /tasks/{task_id}/step: upload next robot observation (waist + main image).
  - POST /tasks/{task_id}/user_instruction: refine plan with new user instruction.

- Converts HTTP file uploads to PIL images.
- Delegates all task logic to ServerTaskManager.
- Returns a *clean* execution state to client (no internal-only fields).

Startup Commands:
    1) Start app first:
       uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

    2) Then start widow infer client:
       uv run python widow/infer.py \
           --task_server_base_url http://127.0.0.1:8000 \
           --policy_host 192.168.1.103 \
           --policy_port 8000 \
           --policy_trace_npz_folder ./_policy_traces
"""

from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import io
import os
import openai
from pathlib import Path

from .task_manager import ServerTaskManager
from .schema import TaskConfig, TaskRuntimeState, TaskStateEnum
from .image_utils import RobotImageInput
from .config import load_server_model_config
from .ablation import load_ablation_setting
from .task_state import load_task_state_json
# VLM client & agents
from client.base_vlm_client import BaseVLMClient
from client.planner_vlm import PlannerVLM
from client.observer_vlm import ObserverVLM
from agent.multitag_planner import PlannerAgent
from agent.observer import ObserverAgent
from memory.multitag_recorder import MultiTagMemory
from memory.encoder import OpenAIEmbeddingEncoder

app = FastAPI()
task_manager = ServerTaskManager()


def load_memory_from_resume_dir(resume_dir: str, encoder: OpenAIEmbeddingEncoder) -> MultiTagMemory:
    """
    Resume memory from a task logs folder.
    Source:
    - latest memory snapshot under */memory/memory_round_*.json
    """
    base = Path(resume_dir)
    if not base.exists():
        raise FileNotFoundError(f"MEMORY_RESUME_DIR not found: {resume_dir}")

    # Only load snapshots under */memory/
    snapshot_files = sorted(
        p for p in base.rglob("memory_round_*.json") if p.parent.name == "memory"
    )
    if not snapshot_files:
        raise FileNotFoundError(
            f"No memory snapshot found under {resume_dir} (expected */memory/memory_round_*.json)"
        )
    latest = snapshot_files[-1]
    print(f"[App] 📂 Resuming memory from snapshot: {latest}")
    return MultiTagMemory.resume_from_json(str(latest), encoder)


def load_task_state_from_resume_dir(resume_dir: str) -> TaskRuntimeState:
    """
    Resume task runtime state from a task logs folder.
    Source:
    - latest task snapshot under */task_state/latest_task_state.json
    """
    base = Path(resume_dir)
    if not base.exists():
        raise FileNotFoundError(f"TASK_RESUME_DIR not found: {resume_dir}")

    snapshot_files = sorted(
        p for p in base.rglob("latest_task_state.json") if p.parent.name == "task_state"
    )
    if not snapshot_files:
        raise FileNotFoundError(
            f"No task state snapshot found under {resume_dir} (expected */task_state/latest_task_state.json)"
        )
    latest = snapshot_files[-1]
    print(f"[App] 📂 Resuming task state from snapshot: {latest}")
    return load_task_state_json(str(latest))


def init_agents_once() -> None:
    """
    Initialize PlannerAgent and ObserverAgent once on startup,
    and inject them into the global task_manager.

    Memory persistence:
    - Memory is created once and shared across all tasks
    - Resume source: MEMORY_RESUME_DIR (task logs folder, memory snapshot files)
    """
    if task_manager.planner_agent is not None:
        return

    cfg = load_server_model_config()
    planner_openai_client = openai.OpenAI(
        api_key=cfg.planner_api_key,
        base_url=cfg.planner_base_url,
    )
    observer_openai_client = openai.OpenAI(
        api_key=cfg.observer_api_key,
        base_url=cfg.observer_base_url,
    )

    ablation = load_ablation_setting()
    prompt_name = os.environ.get("PLANNER_PROMPT_NAME", "").strip() or ablation.prompt_name
    observer_prompt_name = os.environ.get("OBSERVER_PROMPT_NAME", "").strip() or "observer"
    memory_op_policy = os.environ.get("PLANNER_MEMORY_OP_POLICY", "").strip() or ablation.memory_op_policy
    memory_mode = os.environ.get("PLANNER_MEMORY_MODE", "").strip() or ablation.memory_mode
    memory_max_records_raw = os.environ.get("PLANNER_MEMORY_MAX_RECORDS", "").strip()
    if memory_max_records_raw:
        memory_max_records = int(memory_max_records_raw)
    else:
        memory_max_records = ablation.memory_max_records
    print(f"[App] Ablation profile={ablation.profile}")
    print(f"[App] Using planner prompt: {prompt_name}")
    print(f"[App] Using observer prompt: {observer_prompt_name}")
    print(f"[App] Planner memory_op_policy={memory_op_policy}")
    print(f"[App] Planner memory_mode={memory_mode}")
    print(f"[App] Planner memory_max_records={memory_max_records}")
    print(f"[App] Planner image_mode={ablation.planner_image_mode}")
    print(
        f"[App] Planner model={cfg.planner_model}, planner_base_url={cfg.planner_base_url}"
    )
    print(
        f"[App] Observer model={cfg.observer_model}, observer_base_url={cfg.observer_base_url}"
    )
    print(
        f"[App] Embedding model={cfg.embedding_model}, embedding_base_url={cfg.embedding_base_url}, embedding_dim={cfg.embedding_dim}"
    )

    planner_client = BaseVLMClient(
        model=cfg.planner_model,
        client=planner_openai_client,
    )
    observer_base_client = BaseVLMClient(
        model=cfg.observer_model,
        client=observer_openai_client,
    )

    # Shared memory instance (persisted across tasks during runtime)
    task_resume_dir = os.environ.get("TASK_RESUME_DIR", "").strip()
    memory_resume_dir = os.environ.get("MEMORY_RESUME_DIR", "").strip()
    effective_memory_resume_dir = memory_resume_dir or task_resume_dir
    embedding_encoder = OpenAIEmbeddingEncoder(
        embedding_dim=cfg.embedding_dim,
        model=cfg.embedding_model,
        api_key=cfg.embedding_api_key,
        base_url=cfg.embedding_base_url,
    )

    if effective_memory_resume_dir:
        try:
            multitag_memory = load_memory_from_resume_dir(effective_memory_resume_dir, embedding_encoder)
            multitag_memory.set_max_records(memory_max_records)
            print(f"[App] ✅ Memory resumed: {len(multitag_memory.all())} records loaded")
        except Exception as e:
            print(f"[App] ❌ Failed to resume memory: {e}")
            print(f"[App] ℹ️  Creating fresh memory instead")
            multitag_memory = MultiTagMemory(embedding_encoder, max_records=memory_max_records)
    else:
        multitag_memory = MultiTagMemory(embedding_encoder, max_records=memory_max_records)
        print(f"[App] ✅ Initialized fresh shared memory")

    planner_vlm = PlannerVLM(base_client=planner_client)
    observer_vlm = ObserverVLM(base_client=observer_base_client)

    planner = PlannerAgent(
        vlm=planner_vlm,
        memory=multitag_memory,
        prompt_name=prompt_name,
        memory_op_policy=memory_op_policy,
        memory_mode=memory_mode,
    )
    observer = ObserverAgent(vlm=observer_vlm, prompt_name=observer_prompt_name)

    task_manager.set_agents(planner, observer)

    if task_resume_dir:
        try:
            resumed_state = load_task_state_from_resume_dir(task_resume_dir)
            task_manager.add_resumed_task(resumed_state)
            print(f"[App] ✅ Task resumed: task_id={resumed_state.task_id}")
        except Exception as e:
            print(f"[App] ❌ Failed to resume task state: {e}")


def file_to_pil(upload: UploadFile) -> Image.Image:
    """Convert an UploadFile to RGB PIL.Image."""
    data = upload.file.read()
    img = Image.open(io.BytesIO(data))
    return img.convert("RGB")


# ========= Pydantic Schemas exposed via HTTP =========

class TaskPublicState(BaseModel):
    """Minimal public view of a task runtime state for HTTP responses."""
    task_id: str
    is_done: bool
    runtime_state: TaskStateEnum
    planner_status: str
    plan_list: str
    summary: str
    current_subtask_description: Optional[str]


class CreateTaskResponse(TaskPublicState):
    pass


class StepResponse(TaskPublicState):
    pass


class UserInstructionBody(BaseModel):
    user_new_instruction: str


class UserInstructionResponse(TaskPublicState):
    pass


class ResumeStateResponse(BaseModel):
    has_resumed_task: bool
    task: Optional[TaskPublicState] = None


def state_to_public(state: TaskRuntimeState) -> TaskPublicState:
    """
    Map internal TaskRuntimeState to a public response model.
    Internal-only fields (paths, logs, indices, etc.) are hidden.
    """
    return TaskPublicState(
        task_id=state.task_id,
        is_done=state.is_done,
        runtime_state=state.runtime_state,
        planner_status=str(state.extra.get("planner_status", "idle")),
        plan_list=state.plan_list,
        summary=state.summary,
        current_subtask_description=state.current_subtask_description,
    )


# ========= Routes =========

@app.on_event("startup")
async def on_startup():
    init_agents_once()


@app.get("/resume_state", response_model=ResumeStateResponse)
async def get_resume_state():
    if not task_manager.tasks:
        return ResumeStateResponse(has_resumed_task=False, task=None)

    # Resume flow is intended for one active restored task.
    state = next(iter(task_manager.tasks.values()))
    return ResumeStateResponse(
        has_resumed_task=True,
        task=state_to_public(state),
    )


@app.post("/tasks", response_model=CreateTaskResponse)
async def create_task(
    global_instruction: str = Form(..., description="High-level task description for the robot"),
    initial_waist_image: Optional[UploadFile] = File(
        ..., description="Initial waist camera image"
    ),
    initial_image: UploadFile = File(
        ..., description="Initial main camera image"
    ),
    observer_window_size: int = Form(8),
    human_intervene_for_planner: bool = Form(False),
    use_observer: Optional[bool] = Form(None),
    use_memory: Optional[bool] = Form(None),
    planner_execution_mode: str = Form("sync"),
    # ✅ 控制是否重置 Planner 对话历史
    reset_planner_conversation: bool = Form(True, description="Whether to reset planner conversation history for new task (default: True). Set to False to preserve conversation context across tasks."),
):
    """
    Create a new task with an initial robot observation.

    Parameters:
    - initial_waist_image: waist view
    - initial_image: main view
    - reset_planner_conversation: (default: True) whether to reset planner's conversation history
      - True: Each task starts with fresh conversation history
      - False: Preserve conversation history across tasks

    Memory behavior:
    - Memory is shared globally across all tasks
    - Memory can be pre-loaded on server startup via MEMORY_RESUME_DIR
    - If reset_planner_conversation=False, conversation context is also preserved

    Server will combine images into a single 'combined' image,
    then run the planner for the initial plan_list + summary.
    """
    try:
        main_img = file_to_pil(initial_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid initial main image: {e}")

    try:
        waist_img = file_to_pil(initial_waist_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid initial waist image: {e}")

    # ✅ Reset planner conversation if requested
    if reset_planner_conversation and task_manager.planner_agent is not None:
        print(f"[App] 🔄 Resetting planner conversation history for new task")
        task_manager.planner_agent.reset()

    planner_execution_mode = (planner_execution_mode or "sync").strip().lower()
    if planner_execution_mode not in {"sync", "async"}:
        raise HTTPException(status_code=400, detail="planner_execution_mode must be 'sync' or 'async'")

    ablation = load_ablation_setting()
    effective_use_observer = ablation.use_observer if use_observer is None else use_observer
    effective_use_memory = ablation.use_memory if use_memory is None else use_memory

    cfg = TaskConfig(
        observer_window_size=observer_window_size,
        human_intervene_for_planner=human_intervene_for_planner,
        use_observer=effective_use_observer,
        use_memory=effective_use_memory,
        planner_execution_mode=planner_execution_mode,
        planner_image_mode=ablation.planner_image_mode,
    )

    # RobotImageInput for create_task usually takes single images (start state)
    robot_input = RobotImageInput(
        waist_image=waist_img,
        image=main_img,
    )

    state = task_manager.create_task(
        global_instruction=global_instruction,
        initial_robot_input=robot_input,
        config=cfg,
    )
    return state_to_public(state)


@app.post("/tasks/{task_id}/step", response_model=StepResponse)
async def upload_step_image(
    task_id: str,
    # 关键修改：接收 List[UploadFile]，默认值为空列表以处理可选情况
    waist_image: List[UploadFile] = File(
        ..., description="Waist camera images sequence for this step"
    ),
    image: List[UploadFile] = File(
        ..., description="Main camera images sequence for this step"
    ),
):
    """
    Upload a new robot observation step (Sequence of images).

    The robot sends a list of images (buffer) captured during policy execution.
    Server will:
    - Receive all images.
    - Pass them to task manager (which should handle image sequences for Observer).
    """
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    # 1. Process Main Images List
    if not image:
        raise HTTPException(status_code=400, detail="No main images provided")
    
    try:
        main_imgs_pil = [file_to_pil(img) for img in image]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid main image in sequence: {e}")

    # 2. Process Waist Images List
    try:
        waist_imgs_pil = [file_to_pil(img) for img in waist_image]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid waist image in sequence: {e}")

    # 3. Construct Robot Input
    
    # 如果 RobotImageInput 只接受单图，但你想传列表，你需要修改 RobotImageInput 的定义。
    # 这里为了兼容性，我构造了一个包含列表的输入，请确保后端逻辑能处理它。
    
    # 临时处理：如果后端只支持单图，我们取最后一张（最新状态），但理想情况是传列表
    # 为了让 Observer 看到过程，建议修改 RobotImageInput 支持 list。
    # 这里我按传递 List 编写：
    
    robot_input = RobotImageInput(
        waist_image=waist_imgs_pil if waist_imgs_pil else None,
        image=main_imgs_pil, # Passing the list of images
    )

    state = task_manager.add_step_and_maybe_refine_robot(
        task_id=task_id,
        robot_input=robot_input,
    )

    return state_to_public(state)


@app.post("/tasks/{task_id}/user_instruction", response_model=UserInstructionResponse)
async def user_instruction(task_id: str, body: UserInstructionBody):
    """
    Refine the current plan_list with an additional user instruction.
    """
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    state = task_manager.refine_with_user_instruction(
        task_id=task_id,
        user_new_instruction=body.user_new_instruction,
    )

    return state_to_public(state)


# ========= Memory Management =========

# class SaveMemoryBody(BaseModel):
#     output_path: str


# @app.post("/tasks/{task_id}/memory/save")
# async def save_memory(task_id: str, body: SaveMemoryBody):
#     """
#     Save current memory state to JSON file.

#     Args:
#         task_id: Task ID (for compatibility, currently all tasks share the same memory)
#         body.output_path: Path where to save the memory JSON file

#     Returns:
#         {
#             "success": true,
#             "output_path": "path/to/file.json",
#             "record_count": 10
#         }
#     """
#     if task_id not in task_manager.tasks:
#         raise HTTPException(status_code=404, detail="Task not found")

#     if task_manager.planner_agent is None or task_manager.planner_agent.memory is None:
#         raise HTTPException(status_code=500, detail="Memory not initialized")

#     try:
#         memory = task_manager.planner_agent.memory
#         memory.save_to_json(body.output_path)

#         return {
#             "success": True,
#             "output_path": body.output_path,
#             "record_count": len(memory.all()),
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to save memory: {str(e)}")


# @app.get("/tasks/{task_id}/memory")
# async def get_memory(task_id: str):
#     """
#     Get current memory state (lightweight view without embeddings).

#     Args:
#         task_id: Task ID (for compatibility, currently all tasks share the same memory)

#     Returns:
#         {
#             "record_count": 10,
#             "records": [
#                 {
#                     "id": 1,
#                     "tags": ["apple", "fruit"],
#                     "data": {"type": "text", "value": "..."},
#                     "image_path": null
#                 },
#                 ...
#             ],
#             "tag_stats": {"apple": 1, "fruit": 2, ...}
#         }
#     """
#     if task_id not in task_manager.tasks:
#         raise HTTPException(status_code=404, detail="Task not found")

#     if task_manager.planner_agent is None or task_manager.planner_agent.memory is None:
#         raise HTTPException(status_code=500, detail="Memory not initialized")

#     try:
#         memory = task_manager.planner_agent.memory
#         records = memory.all_light()
#         tag_stats = memory.get_tag_stats()

#         return {
#             "record_count": len(records),
#             "records": records,
#             "tag_stats": tag_stats,
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get memory: {str(e)}")
