"""FastAPI HTTP server for multi-step robot visual task execution.

- Exposes task lifecycle APIs:
  - POST /tasks: create a new task with initial combined image.
  - POST /tasks/{task_id}/step: upload next robot observation (waist + main image).
  - POST /tasks/{task_id}/user_instruction: refine plan with new user instruction.

- Converts HTTP file uploads to PIL images.
- Delegates all task logic to ServerTaskManager.
- Returns a *clean* execution state to client (no internal-only fields).

Usage:
    uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import io
import os
import openai

from .task_manager import ServerTaskManager
from .schema import TaskConfig, TaskRuntimeState
from .image_utils import RobotImageInput
from .config import load_server_model_config
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


def init_agents_once() -> None:
    """
    Initialize PlannerAgent and ObserverAgent once on startup,
    and inject them into the global task_manager.

    ✅ Memory persistence:
    - Memory is created once and shared across all tasks
    - If MEMORY_RESUME_PATH is set, memory will be auto-resumed on startup
    - This enables cross-task knowledge retention
    """
    if task_manager.planner_agent is not None:
        return

    cfg = load_server_model_config()
    client = openai.OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )

    prompt_name = os.environ.get("PLANNER_PROMPT_NAME", "multitag_planner").strip() or "multitag_planner"
    print(f"[App] Using planner prompt: {prompt_name}")
    print(
        f"[App] Using planner model={cfg.planner_model}, "
        f"observer model={cfg.observer_model}, base_url={cfg.base_url}"
    )

    planner_client = BaseVLMClient(
        model=cfg.planner_model,
        client=client,
    )
    observer_base_client = BaseVLMClient(
        model=cfg.observer_model,
        client=client,
    )

    # ✅ Create shared memory instance (persists across all tasks)
    # Check if we should resume memory from a file
    # memory_resume_path = os.environ.get("MEMORY_RESUME_PATH", "/Users/makabaka/code/mem_vla/_server_data/task_20260127_205643_df7e6dfa/logs/memory/memory_round_2_20260127_210037.json").strip()
    memory_resume_path = os.environ.get("MEMORY_RESUME_PATH", "").strip()
    if memory_resume_path and os.path.exists(memory_resume_path):
        print(f"[App] 📂 Resuming memory from: {memory_resume_path}")
        try:
            multitag_memory = MultiTagMemory.resume_from_json(
                memory_resume_path,
                OpenAIEmbeddingEncoder()
            )
            print(f"[App] ✅ Memory resumed: {len(multitag_memory.all())} records loaded")
        except Exception as e:
            print(f"[App] ❌ Failed to resume memory: {e}")
            print(f"[App] ℹ️  Creating fresh memory instead")
            multitag_memory = MultiTagMemory(OpenAIEmbeddingEncoder())
    else:
        multitag_memory = MultiTagMemory(OpenAIEmbeddingEncoder())
        if memory_resume_path:
            print(f"[App] ⚠️  MEMORY_RESUME_PATH set but file not found: {memory_resume_path}")
        print(f"[App] ✅ Initialized fresh shared memory")

    planner_vlm = PlannerVLM(base_client=planner_client)
    observer_vlm = ObserverVLM(base_client=observer_base_client)

    planner = PlannerAgent(vlm=planner_vlm, memory=multitag_memory, prompt_name=prompt_name)
    observer = ObserverAgent(vlm=observer_vlm)

    task_manager.set_agents(planner, observer)


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


def state_to_public(state: TaskRuntimeState) -> TaskPublicState:
    """
    Map internal TaskRuntimeState to a public response model.
    Internal-only fields (paths, logs, indices, etc.) are hidden.
    """
    return TaskPublicState(
        task_id=state.task_id,
        is_done=state.is_done,
        plan_list=state.plan_list,
        summary=state.summary,
        current_subtask_description=state.current_subtask_description,
    )


# ========= Routes =========

@app.on_event("startup")
async def on_startup():
    init_agents_once()


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
    use_observer: bool = Form(True),
    use_memory: bool = Form(True),
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
    - Memory can be pre-loaded on server startup via MEMORY_RESUME_PATH environment variable
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

    cfg = TaskConfig(
        observer_window_size=observer_window_size,
        human_intervene_for_planner=human_intervene_for_planner,
        use_observer=use_observer,
        use_memory=use_memory,
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
