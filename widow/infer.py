#!/usr/bin/env python3
"""Widow robot inference client.

This script is an example client implementation that connects a robot policy
to the task server. It:
1) Creates a task with initial observations.
2) Sends step image sequences during execution.
3) Optionally sends user instructions while running.

You must adapt this file to match your real robot deployment (camera drivers,
policy interface, and control loop).

Startup Command:
    uv run python widow/infer.py \
        --task_server_base_url <TASK_SERVER_BASE_URL> \
        --policy_host <POLICY_HOST> \
        --policy_port <POLICY_PORT> \
        --policy_trace_npz_folder ./_policy_traces
"""
from __future__ import annotations

import argparse
import io
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Full, Queue
from typing import Optional, Dict, Any, List

import numpy as np
import requests
from PIL import Image

from openpi_client import image_tools, websocket_client_policy
from wx250s_ros2_client import WX250sRos2Client
from camera_client import RealSenseRos2Client
from utils import euler_angle_to_axis_angle


# ==========================
# Tools: Euler angles to axis-angle conversion
# ==========================

def eef_euler2axis(eef_pose: np.ndarray) -> np.ndarray:
    """Convert 6D EEF Euler pose [x,y,z, roll,pitch,yaw] to [x,y,z, rx,ry,rz]."""
    euler_rotation = eef_pose[3:6]
    rx, ry, rz = euler_angle_to_axis_angle(
        euler_rotation[0], euler_rotation[1], euler_rotation[2]
    )
    return np.concatenate([eef_pose[:3], np.array([rx, ry, rz])])


# ==========================
# RobotClient: Task server (images)
# ==========================

@dataclass
class RobotClientConfig:
    base_url: str = ""
    timeout: int = 180
    observer_window_size: int = 8
    human_intervene_for_planner: bool = False
    planner_execution_mode: str = "sync"


class RobotClient:
    """HTTP client for the task server.

    API surface used here:
    - POST /tasks
      - form fields:
        global_instruction, observer_window_size, human_intervene_for_planner,
        planner_execution_mode
      - files:
        initial_image, initial_waist_image
      - response: TaskPublicState
    - POST /tasks/{task_id}/step
      - files:
        image[] (main camera sequence), waist_image[] (waist camera sequence)
      - response: TaskPublicState
    - GET /resume_state
      - response: { has_resumed_task, task }
    """

    def __init__(self, config: Optional[RobotClientConfig] = None):
        self.config = config or RobotClientConfig()
        if not self.config.base_url:
            raise ValueError("task server base_url is required")
        self.base_url = self.config.base_url.rstrip("/")
        self.timeout = self.config.timeout

    @staticmethod
    def _to_pil(img) -> Image.Image:
        if img is None:
            raise ValueError("Image is None")
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                return Image.fromarray(img, mode="L").convert("RGB")
            if img.ndim == 3:
                return Image.fromarray(img).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(img)}")

    @staticmethod
    def _pil_to_png_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def create_task(
        self,
        global_instruction: str,
        initial_image,
        initial_waist_image,
    ) -> Dict[str, Any]:
        """Create a task with initial images and the global instruction.

        Returns a TaskPublicState dict with:
        task_id, is_done, runtime_state, planner_status,
        plan_list, summary, current_subtask_description.
        """
        url = f"{self.base_url}/tasks"

        main_pil = self._to_pil(initial_image)
        main_bytes = self._pil_to_png_bytes(main_pil)
        waist_pil = self._to_pil(initial_waist_image)
        waist_bytes = self._pil_to_png_bytes(waist_pil)

        files = {
            "initial_image": (
                "initial_image.png",
                io.BytesIO(main_bytes),
                "image/png",
            ),
            "initial_waist_image": (
                "initial_waist_image.png",
                io.BytesIO(waist_bytes),
                "image/png",
            ),
        }

        data = {
            "global_instruction": global_instruction,
            "observer_window_size": self.config.observer_window_size,
            "human_intervene_for_planner": self.config.human_intervene_for_planner,
            "planner_execution_mode": self.config.planner_execution_mode,
        }

        resp = requests.post(url, files=files, data=data, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def send_step(
        self,
        task_id: str,
        images: List[Any],
        waist_images: List[Any],
    ) -> Dict[str, Any]:
        """Send a sequence of images for a single robot step.

        images and waist_images are lists in chronological order.
        Returns a TaskPublicState dict.
        """
        url = f"{self.base_url}/tasks/{task_id}/step"

        # We use a list of tuples for 'files' to send multiple files with the same key (or distinct keys)
        # Assuming the server accepts 'image' as a list or multiple fields.
        # Standard requests format for multiple files: [('field_name', (filename, bytes, content_type)), ...]
        files = []

        # 1. Process Head Images
        if not images:
            raise ValueError("send_step called with empty image list")
        if not waist_images:
            raise ValueError("send_step called with empty waist image list")
        if len(waist_images) != len(images):
            raise ValueError(
                f"waist image count ({len(waist_images)}) must equal head image count ({len(images)})"
            )

        for idx, img in enumerate(images):
            pil_img = self._to_pil(img)
            img_bytes = self._pil_to_png_bytes(pil_img)
            # We use the key "image" repeatedly. Most server frameworks (FastAPI/Flask) parse this as a list.
            files.append((
                "image",
                (f"step_image_{idx}.png", io.BytesIO(img_bytes), "image/png")
            ))

        # 2. Process Waist Images
        for idx, img in enumerate(waist_images):
            pil_img = self._to_pil(img)
            img_bytes = self._pil_to_png_bytes(pil_img)
            files.append((
                "waist_image",
                (f"waist_step_image_{idx}.png", io.BytesIO(img_bytes), "image/png")
            ))

        # Send request
        resp = requests.post(url, files=files, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_resume_state(self) -> Dict[str, Any]:
        """Check whether the server has a resumed task state."""
        url = f"{self.base_url}/resume_state"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


# ==========================
# UserClient: Task server (user instructions)
# ==========================

@dataclass
class UserClientConfig:
    base_url: str = ""
    timeout: int = 180


class UserClient:
    """HTTP client for user instruction updates.

    API surface:
    - POST /tasks/{task_id}/user_instruction
      - json: { "user_new_instruction": "<text>" }
      - response: TaskPublicState
    """

    def __init__(self, config: Optional[UserClientConfig] = None):
        self.config = config or UserClientConfig()
        if not self.config.base_url:
            raise ValueError("task server base_url is required")
        self.base_url = self.config.base_url.rstrip("/")
        self.timeout = self.config.timeout

    def send_user_instruction(self, task_id: str, instruction: str) -> Dict[str, Any]:
        """POST /tasks/{task_id}/user_instruction."""
        url = f"{self.base_url}/tasks/{task_id}/user_instruction"
        body = {"user_new_instruction": instruction}
        resp = requests.post(url, json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


# ==========================
# User input thread
# ==========================

def user_input_loop(
    user_client: UserClient,
    get_task_id,
    stop_flag: threading.Event,
    reset_robot,
    get_current_prompt,
    save_current_views,
):
    """Read user instructions from stdin and push to server.

    This is optional and can be replaced with your own UI or tooling.
    """
    print("\n" + "="*60)
    print("[UserInput] User input thread ready.")
    print("[UserInput] Enter instructions at any time, or 'quit' to exit.")
    print("="*60 + "\n")

    while not stop_flag.is_set():
        try:
            text = input("[UserInput] > ").strip()
        except EOFError:
            print("\n[UserInput] EOF detected, exiting.")
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            print("[UserInput] Exit command received.")
            stop_flag.set()
            break
        if text.lower() == "reset":
            cur_prompt = get_current_prompt() or ""
            print("[UserInput] Reset command received. Calling robot.reset_to_home()...")
            if cur_prompt:
                print(f"[UserInput] Current instruction (unchanged): {cur_prompt}")
            try:
                reset_robot()
                print("[UserInput] ✓ Reset complete.")
            except Exception as e:
                print(f"[UserInput] ✗ Reset failed: {e}")
            continue
        if text.lower() == "save":
            print("[UserInput] Save command received. Capturing current third view and wrist view...")
            try:
                saved_paths = save_current_views()
                print("[UserInput] ✓ Saved current views.")
                print(f"[UserInput] Third view: {saved_paths['third_view']}")
                print(f"[UserInput] Wrist view: {saved_paths['wrist_view']}")
            except Exception as e:
                print(f"[UserInput] ✗ Save failed: {e}")
            continue

        task_id = get_task_id()
        if not task_id:
            print("[UserInput] No task_id yet, ignoring.")
            continue

        try:
            resp = user_client.send_user_instruction(task_id, text)
            print(f"\n[UserInput] ✓ Instruction sent.")
            print(f"[UserInput] New subtask: {resp.get('current_subtask_description')}\n")
        except Exception as e:
            print(f"\n[UserInput] ✗ Send failed: {e}\n")


def is_pick_place_subtask(subtask: Optional[str]) -> bool:
    if not subtask:
        return False
    text = subtask.lower().strip()
    return text.startswith("pick up ") and " and place it " in text


def save_policy_trace_npz(
    output_root: str,
    task_id: str,
    head_image: np.ndarray,
    wrist_image: np.ndarray,
    state: np.ndarray,
    task_description: str,
    actions: Any,
) -> Optional[str]:
    """Save one policy request/response sample as compressed npz."""
    if not output_root:
        return None

    task_dir = os.path.join(output_root, task_id)
    os.makedirs(task_dir, exist_ok=True)

    timestamp_ms = int(time.time() * 1000)
    timestamp_ns_tail = time.time_ns() % 1_000_000
    filename = f"{timestamp_ms}_{timestamp_ns_tail:06d}.npz"
    save_path = os.path.join(task_dir, filename)

    np.savez_compressed(
        save_path,
        task_description=np.asarray(task_description),
        image=head_image,
        wrist_image=wrist_image,
        state=np.asarray(state),
        action=np.asarray(actions),
    )
    return save_path


class AsyncPolicyTraceWriter:
    """Background writer for policy trace buffers.

    Saves trace arrays to disk without blocking the control loop.
    """
    """Best-effort async writer to avoid blocking policy/action loop on disk I/O."""

    def __init__(self, output_root: str, task_id: str, max_queue: int = 256):
        self.output_root = output_root
        self.task_id = task_id
        self._queue: "Queue[Dict[str, Any]]" = Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._dropped = 0
        self._thread.start()

    def enqueue(
        self,
        head_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        task_description: str,
        actions: Any,
    ) -> None:
        item = {
            "head_image": head_image,
            "wrist_image": wrist_image,
            "state": state,
            "task_description": task_description,
            "actions": actions,
        }
        try:
            self._queue.put_nowait(item)
        except Full:
            self._dropped += 1
            if self._dropped % 20 == 1:
                print(
                    f"[Main] Policy trace queue full, dropped {self._dropped} samples"
                )

    def close(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout_s)

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                save_policy_trace_npz(
                    output_root=self.output_root,
                    task_id=self.task_id,
                    head_image=item["head_image"],
                    wrist_image=item["wrist_image"],
                    state=item["state"],
                    task_description=item["task_description"],
                    actions=item["actions"],
                )
            except Exception as e:
                print(f"[Main] Failed to save policy trace npz (async): {e}")
            finally:
                self._queue.task_done()


# ==========================
# Main logic
# ==========================

def main():
    """Main robot loop.

    Connects to robot policy, captures camera frames, posts to server,
    and executes the returned action plan.
    """
    parser = argparse.ArgumentParser()
    # Low-level policy server
    parser.add_argument("--policy_host", default="")
    parser.add_argument("--policy_port", type=int, default=8000)
    parser.add_argument("--task_prompt", default="pick up a red flower, and place it in the vase on the left.")
    parser.add_argument("--control_hz", type=float, default=25.0)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--policy_calls_per_observation", type=int, default=10,
                        help="Number of policy calls (n) between observer communications")
    parser.add_argument("--actions_per_policy_call", type=int, default=10,
                        help="Number of actions (k) to execute per policy call")

    # Camera topics
    parser.add_argument("--head_camera", default="/head_camera/camera/color/image_raw")
    parser.add_argument("--head_camera_info", default="/head_camera/camera/color/camera_info")
    parser.add_argument("--wrist_camera", default="/wrist_camera/camera/color/image_raw")
    parser.add_argument("--wrist_camera_info", default="/wrist_camera/camera/color/camera_info")

    # Policy modes
    parser.add_argument("--policy_proprio_mode", default="libero_eef",
                        choices=["libero_eef", "joints_positions"])
    parser.add_argument("--policy_action_mode", default="absolute_eef",
                        choices=["relative_eef", "absolute_eef", "joint_positions"])

    # Task server
    parser.add_argument("--task_server_base_url", default="")
    parser.add_argument("--task_server_timeout", type=int, default=180)
    parser.add_argument("--observer_window_size", type=int, default=8)
    parser.add_argument("--human_intervene_for_planner", action="store_true")
    parser.add_argument(
        "--planner_execution_mode",
        default="sync",
        choices=["sync", "async"],
        help="Planner execution mode on task server.",
    )
    parser.add_argument(
        "--reset_on_new_pick_place_subtask",
        action="store_true",
        help="If enabled, when the current subtask changes to a new pick-and-place action, execute one reset cycle before sending that subtask to the policy server.",
    )
    parser.add_argument(
        "--policy_trace_npz_folder",
        default="",
        help="Default off. If provided, save policy trace npz under <folder>/<task_id>/<timestamp>.npz",
    )
    parser.add_argument(
        "--policy_trace_npz_dir",
        default="",
        help="Deprecated alias of --policy_trace_npz_folder",
    )
    parser.add_argument(
        "--manual_save_dir",
        default="./_manual_saves",
        help="Directory for images captured via the `save` user command.",
    )

    args = parser.parse_args()
    dt = 1.0 / args.control_hz

    # 1. Connect to robot arm
    robot = WX250sRos2Client()
    assert robot.connect(), "Failed to connect to WidowX 250s"
    robot.reset_to_home()

    # 2. Connect to cameras
    head_cam = RealSenseRos2Client(
        camera_id="head",
        rgb_topic=args.head_camera,
        camera_info_topic=args.head_camera_info,
    )
    assert head_cam.connect(), "Failed to connect to head RealSense"
    assert head_cam.wait_for_image(5.0), "No image from head camera"

    wrist_cam = RealSenseRos2Client(
        camera_id="wrist",
        rgb_topic=args.wrist_camera,
        camera_info_topic=args.wrist_camera_info,
    )
    assert wrist_cam.connect(), "Failed to connect to wrist RealSense"
    assert wrist_cam.wait_for_image(5.0), "No image from wrist camera"
    camera_lock = threading.Lock()

    def capture_current_views() -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with camera_lock:
            head_img = head_cam.capture_image()
            wrist_img = wrist_cam.capture_image()
        return head_img, wrist_img

    def save_image_array(img: np.ndarray, path: str) -> None:
        Image.fromarray(img).save(path)

    def save_current_views() -> Dict[str, str]:
        head_img, wrist_img = capture_current_views()
        if head_img is None or wrist_img is None:
            raise RuntimeError("Failed to capture one or both camera views")

        save_dir = os.path.abspath(args.manual_save_dir)
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        third_path = os.path.join(save_dir, f"{ts}_third_view.png")
        wrist_path = os.path.join(save_dir, f"{ts}_wrist_view.png")
        save_image_array(head_img, third_path)
        save_image_array(wrist_img, wrist_path)
        return {
            "third_view": third_path,
            "wrist_view": wrist_path,
        }

    # 3. Connect to policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
    )
    print(f"[Main] Connected to policy server at {args.policy_host}:{args.policy_port}")

    # 4. Create task server clients
    robot_client = RobotClient(
        RobotClientConfig(
            base_url=args.task_server_base_url,
            timeout=args.task_server_timeout,
            observer_window_size=args.observer_window_size,
            human_intervene_for_planner=args.human_intervene_for_planner,
            planner_execution_mode=args.planner_execution_mode,
        )
    )

    user_client = UserClient(
        UserClientConfig(
            base_url=args.task_server_base_url,
            timeout=args.task_server_timeout,
        )
    )

    # 5. Resume existing task if server already restored one; otherwise create task.
    init_resp: Dict[str, Any]
    try:
        resume_state = robot_client.get_resume_state()
    except Exception as e:
        print(f"[Main] Resume-state probe failed, falling back to create_task: {e}")
        resume_state = {"has_resumed_task": False}

    if resume_state.get("has_resumed_task") and isinstance(resume_state.get("task"), dict):
        init_resp = resume_state["task"]
        task_id = init_resp.get("task_id")
        print("[Main] Resume mode: attached to existing task.")
        print("  task_id:", task_id)
        print("  current_subtask:", init_resp.get("current_subtask_description"))
        print("  is_done:", init_resp.get("is_done"))
    else:
        head0, wrist0 = capture_current_views()
        assert head0 is not None and wrist0 is not None, "Failed to grab initial images"

        init_resp = robot_client.create_task(
            global_instruction=args.task_prompt,
            initial_image=head0,
            initial_waist_image=wrist0,
        )
        task_id = init_resp.get("task_id")
        if not task_id:
            print("[Main] task server did not return task_id, abort.")
            return

        print("[Main] Fresh mode: created new task.")
        print("  task_id:", task_id)
        print("  current_subtask:", init_resp.get("current_subtask_description"))

    policy_trace_npz_dir = (args.policy_trace_npz_folder or args.policy_trace_npz_dir).strip()
    if policy_trace_npz_dir:
        task_npz_dir = os.path.join(policy_trace_npz_dir, task_id)
        os.makedirs(task_npz_dir, exist_ok=True)
        print(f"[Main] Policy trace npz enabled: {task_npz_dir}")
    trace_writer = (
        AsyncPolicyTraceWriter(policy_trace_npz_dir, task_id)
        if policy_trace_npz_dir
        else None
    )

    # Current policy prompt
    current_policy_prompt = init_resp.get("current_subtask_description") or args.task_prompt
    last_server_subtask = current_policy_prompt.strip() if isinstance(current_policy_prompt, str) else ""
    pending_reset_subtask: Optional[str] = None

    # Track task completion status
    task_is_done = init_resp.get("is_done", False)
    last_reported_done_state = task_is_done

    # Shared task_id & stop_flag for user input thread
    task_id_lock = threading.Lock()
    stop_flag = threading.Event()

    def get_task_id() -> Optional[str]:
        with task_id_lock:
            return task_id

    def reset_robot():
        robot.reset_to_home()

    def get_current_prompt() -> str:
        # Read-only access for user thread; instruction is unchanged by reset.
        return current_policy_prompt

    user_thread = threading.Thread(
        target=user_input_loop,
        args=(user_client, get_task_id, stop_flag, reset_robot, get_current_prompt, save_current_views),
        daemon=True,
    )
    user_thread.start()

    # Buffers to store images during policy execution
    head_img_buffer = []
    wrist_img_buffer = []

    # 6. Main loop
    try:
        while not stop_flag.is_set():
            observation_cycle_start = time.perf_counter()
            total_actions_executed = 0

            # Clear buffers at start of cycle
            head_img_buffer = []
            wrist_img_buffer = []

            # 6.1 Execute n policy calls (only if task not done)
            if not task_is_done:
                if pending_reset_subtask:
                    print("\n[Main] ↻ Executing pre-subtask reset.")
                    print("[Main] → Calling robot.reset_to_home()...")
                    try:
                        robot.reset_to_home()
                        time.sleep(1.0)
                    except Exception as e:
                        print(f"[Main] ✗ Reset execution failed: {e}")
                        stop_flag.set()
                        break
                    current_policy_prompt = pending_reset_subtask
                    pending_reset_subtask = None

                # === NORMAL POLICY EXECUTION ===
                for policy_call_idx in range(args.policy_calls_per_observation):
                    if stop_flag.is_set():
                        break

                    # Get current robot state
                    try:
                        all_state = robot.get_state()
                    except Exception as e:
                        time.sleep(0.1)
                        continue

                    eef_state = all_state["end_effector_pose"]

                    if args.policy_proprio_mode == "joints_positions":
                        joint_state = all_state["joint_states"]
                        gripper_state = all_state["gripper_state"]
                        input_state = np.concatenate([joint_state, gripper_state[1:]])
                    else:
                        gripper_state = all_state["gripper_state"]
                        input_state = np.concatenate(
                            [eef_euler2axis(eef_state), gripper_state[1:]]
                        )

                    # Capture fresh images for this policy call as policy input.
                    # These are pre-action observations and should not be sent to the task server.
                    head_img, wrist_img = capture_current_views()
                    if head_img is None or wrist_img is None:
                        time.sleep(0.1)
                        continue

                    # Resize images for policy
                    head_img_p = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(head_img, 224, 224)
                    )
                    wrist_img_p = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, 224, 224)
                    )

                    observation = {
                        "observation/state": input_state,
                        "observation/image": head_img_p,
                        "observation/wrist_image": wrist_img_p,
                        "prompt": current_policy_prompt,
                    }

                    # Call policy
                    try:
                        result = policy_client.infer(observation)
                        actions = result["actions"]
                    except Exception as e:
                        continue

                    if trace_writer is not None:
                        trace_writer.enqueue(
                            head_image=head_img_p,
                            wrist_image=wrist_img_p,
                            state=input_state,
                            task_description=current_policy_prompt,
                            actions=actions,
                        )

                    # Execute actions
                    executed = 0
                    for a in actions:
                        if stop_flag.is_set(): break
                        if executed >= args.actions_per_policy_call: break

                        step_start = time.perf_counter()
                        try:
                            if args.policy_action_mode == "relative_eef":
                                delta_eef = a[:6]
                                base_eef_axis = eef_euler2axis(eef_state)
                                target_pose = base_eef_axis + delta_eef
                                robot.move_to_pose(target_pose, blocking=False)
                                robot.set_gripper(a[6], blocking=False)
                            elif args.policy_action_mode == "absolute_eef":
                                target_pose = a[:6]
                                robot.move_to_pose(target_pose, blocking=False)
                                robot.set_gripper(a[6], blocking=False)
                            elif args.policy_action_mode == "joint_positions":
                                target_joints = a[:6]
                                robot.move_to_joint_positions(target_joints, blocking=False)
                                robot.set_gripper(a[6], blocking=False)
                        except Exception as e:
                            print(f"[Main] Robot control error: {e}")
                            stop_flag.set()
                            break

                        executed += 1
                        total_actions_executed += 1
                        elapsed = time.perf_counter() - step_start
                        sleep_t = dt - elapsed
                        if sleep_t > 0: time.sleep(sleep_t)

                    # Capture post-action images for server/planner.
                    # These should reflect the state after this policy call's action chunk.
                    post_head_img, post_wrist_img = capture_current_views()
                    if post_head_img is not None and post_wrist_img is not None:
                        head_img_buffer.append(post_head_img)
                        wrist_img_buffer.append(post_wrist_img)
            else:
                # Task is done, robot idle
                if not last_reported_done_state:
                    print("\n[Main] ✓ Task completed. Robot idle, waiting for new instructions...")
                    last_reported_done_state = True

                # If idle, we still need to capture one image to show the observer the current state
                head_img, wrist_img = capture_current_views()
                if head_img is not None and wrist_img is not None:
                    head_img_buffer.append(head_img)
                    wrist_img_buffer.append(wrist_img)

            # 6.2 Communicate with task server (Send Buffer)
            # If buffer is empty (e.g. camera errors), skip
            if not head_img_buffer:
                if task_is_done: time.sleep(1.0)
                continue

            try:
                # Send the entire sequence of images captured during policy execution
                step_resp = robot_client.send_step(
                    task_id=task_id,
                    images=head_img_buffer,
                    waist_images=wrist_img_buffer,
                )

                cur_subtask = step_resp.get("current_subtask_description")
                prev_subtask = current_policy_prompt
                next_server_subtask = cur_subtask.strip() if isinstance(cur_subtask, str) and cur_subtask.strip() else ""

                prev_done_state = task_is_done
                task_is_done = step_resp.get("is_done", False)

                if next_server_subtask:
                    should_reset_for_switch = (
                        args.reset_on_new_pick_place_subtask
                        and not task_is_done
                        and is_pick_place_subtask(next_server_subtask)
                        and next_server_subtask != last_server_subtask
                    )
                    last_server_subtask = next_server_subtask
                    if should_reset_for_switch:
                        pending_reset_subtask = next_server_subtask
                        current_policy_prompt = next_server_subtask
                        print("\n[Main] ↻ New pick-and-place subtask detected.")
                        print(f"[Main] → Will reset once before executing: '{next_server_subtask}'")
                    else:
                        current_policy_prompt = next_server_subtask
                else:
                    current_policy_prompt = args.task_prompt

                if prev_done_state != task_is_done or prev_subtask != current_policy_prompt:
                    print(f"\n[TaskServer] Done: {task_is_done}")
                    print(f"[TaskServer] Current subtask: {cur_subtask}")
                    if not task_is_done and prev_done_state:
                        print("[Main] → Resuming task execution...")
                        last_reported_done_state = False

                if task_is_done:
                    time.sleep(2.0)

            except Exception as e:
                print(f"\n[Main] ⚠ Communication error: {e}\n")

    finally:
        print("[Main] Stopping robot & closing resources...")
        stop_flag.set()
        if trace_writer is not None:
            trace_writer.close(timeout_s=2.0)
        try:
            robot.go_to_sleep()
        except Exception: pass
        robot.disconnect()
        head_cam.disconnect()
        wrist_cam.disconnect()
        try:
            user_thread.join(timeout=2.0)
        except Exception: pass


if __name__ == "__main__":
    main()
