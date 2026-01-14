"""FastAPI HTTP server for multi-step robot visual task execution.

- Exposes task lifecycle APIs:
  - POST /tasks: create a new task with initial combined image.
  - POST /tasks/{task_id}/step: upload next robot observation (waist + main image).
  - POST /tasks/{task_id}/user_instruction: refine plan with new user instruction.

- Converts HTTP file uploads to PIL images.
- Delegates all task logic to ServerTaskManager.
- Returns a *clean* execution state to client (no internal-only fields).
uv run python -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io
import os
import openai

from .task_manager import ServerTaskManager
from .schema import TaskConfig, TaskRuntimeState
from .image_utils import RobotImageInput
# VLM client & agents
from src.client.base_vlm_client import BaseVLMClient
from src.client.planner_vlm import PlannerVLM
from src.client.observer_vlm import ObserverVLM
from src.agent.multitag_planner import PlannerAgent
from src.agent.observer import ObserverAgent


app = FastAPI()
task_manager = ServerTaskManager()


def init_agents_once() -> None:
    """
    Initialize PlannerAgent and ObserverAgent once on startup,
    and inject them into the global task_manager.
    """
    if task_manager.planner_agent is not None:
        return

    # Load OpenAI config from env (set defaults for local dev)
    os.environ.setdefault("OPENAI_API_KEY", "xx")
    os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")
    client = openai.OpenAI()

    base_client = BaseVLMClient(
        model="claude-sonnet-4-5-20250929",  # replace with your actual model
        client=client,
    )
    planner_vlm = PlannerVLM(base_client=base_client)
    observer_vlm = ObserverVLM(base_client=base_client)

    planner = PlannerAgent(vlm=planner_vlm)
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
    state: str           # TaskStateEnum value
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
        state=state.state.value,
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
        None, description="Initial waist camera image (optional)"
    ),
    initial_image: UploadFile = File(
        ..., description="Initial main camera image (required)"
    ),
    observer_window_size: int = Form(8),
    human_intervene_for_planner: bool = Form(False),
    debug_mode: bool = Form(False),
    pause_on_observer: bool = Form(False),
    pause_on_planner: bool = Form(False),
):
    """
    Create a new task with an initial robot observation.
    The robot may provide:
    - initial_waist_image: waist view
    - initial_image: main view (required)

    Server will first combine these into a single 'combined' image,
    then run the planner for the initial plan_list + summary.
    """
    try:
        main_img = file_to_pil(initial_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid initial main image: {e}")

    waist_img: Optional[Image.Image] = None
    if initial_waist_image is not None:
        try:
            waist_img = file_to_pil(initial_waist_image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid initial waist image: {e}")

    cfg = TaskConfig(
        observer_window_size=observer_window_size,
        human_intervene_for_planner=human_intervene_for_planner,
        debug_mode=debug_mode,
        pause_on_observer=pause_on_observer,
        pause_on_planner=pause_on_planner,
    )

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
    waist_image: Optional[UploadFile] = File(
        None, description="Waist camera image for this step (optional)"
    ),
    image: UploadFile = File(
        ..., description="Main camera image for this step (required)"
    ),
):
    """
    Upload a new robot observation step.

    The robot can send:
    - waist_image (optional)
    - image (required main camera)

    Server will:
    - combine them into a single 'combined' image
    - append it to task state
    - run Observer to decide whether the current subtask is done
    - if done, run Planner refine
    """
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        main_img = file_to_pil(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid main image: {e}")

    waist_img: Optional[Image.Image] = None
    if waist_image is not None:
        try:
            waist_img = file_to_pil(waist_image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid waist image: {e}")

    robot_input = RobotImageInput(
        waist_image=waist_img,
        image=main_img,
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

    - Does *not* upload any image.
    - Planner will re-evaluate plan_list based on:
      - global_instruction
      - existing plan_list
      - new user_new_instruction
      - current subtask's image segment.
    """
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    state = task_manager.refine_with_user_instruction(
        task_id=task_id,
        user_new_instruction=body.user_new_instruction,
    )
    return state_to_public(state)


# ========= Debug Mode APIs =========

@app.get("/tasks/{task_id}/pending_approval")
async def get_pending_approval(task_id: str):
    """Get pending agent output waiting for approval (debug mode)"""
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    state = task_manager.tasks[task_id]

    if state.pending_approval is None:
        return {"has_pending": False}

    return {
        "has_pending": True,
        "agent_type": state.pending_approval.agent_type,
        "timestamp": state.pending_approval.timestamp,
        "raw_output": state.pending_approval.raw_output,
        "parsed_output": state.pending_approval.parsed_output,
        "input_context": state.pending_approval.input_context,
    }


class ApproveBody(BaseModel):
    modifications: Optional[Dict[str, Any]] = None


@app.post("/tasks/{task_id}/approve")
async def approve_agent_output(task_id: str, body: ApproveBody):
    """Approve agent output and resume execution (debug mode)"""
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    state = task_manager.tasks[task_id]

    if state.pending_approval is None:
        raise HTTPException(status_code=400, detail="No pending approval")

    # Save approved result (possibly modified)
    if body.modifications:
        state.approved_result = body.modifications
    else:
        state.approved_result = state.pending_approval.parsed_output

    # Wake up the blocked thread
    if state.approval_event:
        state.approval_event.set()

    return {"status": "approved"}


@app.get("/tasks/{task_id}/memory")
async def get_memory(task_id: str):
    """Get current memory state (debug mode)"""
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    if task_manager.planner_agent is None:
        raise HTTPException(status_code=500, detail="Planner agent not initialized")

    snapshot = task_manager.planner_agent.get_memory_snapshot()
    return snapshot


@app.get("/tasks/{task_id}/planner_conversation")
async def get_planner_conversation(task_id: str):
    """Get planner conversation history (debug mode)"""
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    if task_manager.planner_agent is None:
        raise HTTPException(status_code=500, detail="Planner agent not initialized")

    return {
        "messages": task_manager.planner_agent.messages,
        "memory_snapshot": task_manager.planner_agent.get_memory_snapshot()
    }
