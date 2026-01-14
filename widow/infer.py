#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

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
    base_url: str = "http://localhost:8000"
    timeout: int = 60
    observer_window_size: int = 8
    human_intervene_for_planner: bool = False


class RobotClient:
    """Robot-side HTTP client: sends images and config to task server."""

    def __init__(self, config: Optional[RobotClientConfig] = None):
        self.config = config or RobotClientConfig()
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
        initial_waist_image=None,
    ) -> Dict[str, Any]:
        """POST /tasks: create task on server with initial images + instruction."""
        url = f"{self.base_url}/tasks"

        main_pil = self._to_pil(initial_image)
        main_bytes = self._pil_to_png_bytes(main_pil)

        files = {
            "initial_image": (
                "initial_image.png",
                io.BytesIO(main_bytes),
                "image/png",
            )
        }

        if initial_waist_image is not None:
            waist_pil = self._to_pil(initial_waist_image)
            waist_bytes = self._pil_to_png_bytes(waist_pil)
            files["initial_waist_image"] = (
                "initial_waist_image.png",
                io.BytesIO(waist_bytes),
                "image/png",
            )

        data = {
            "global_instruction": global_instruction,
            "observer_window_size": str(self.config.observer_window_size),
            "human_intervene_for_planner": (
                "true" if self.config.human_intervene_for_planner else "false"
            ),
        }

        resp = requests.post(url, files=files, data=data, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def send_step(
        self,
        task_id: str,
        image,
        waist_image=None,
    ) -> Dict[str, Any]:
        """POST /tasks/{task_id}/step: upload current observation images for each step."""
        url = f"{self.base_url}/tasks/{task_id}/step"

        main_pil = self._to_pil(image)
        main_bytes = self._pil_to_png_bytes(main_pil)

        files = {
            "image": (
                "step_image.png",
                io.BytesIO(main_bytes),
                "image/png",
            )
        }

        if waist_image is not None:
            waist_pil = self._to_pil(waist_image)
            waist_bytes = self._pil_to_png_bytes(waist_pil)
            files["waist_image"] = (
                "waist_step_image.png",
                io.BytesIO(waist_bytes),
                "image/png",
            )

        resp = requests.post(url, files=files, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


# ==========================
# UserClient: Task server (user instructions)
# ==========================

@dataclass
class UserClientConfig:
    base_url: str = "http://localhost:8000"
    timeout: int = 60


class UserClient:
    """Robot-side HTTP client: sends new user instructions to task server."""

    def __init__(self, config: Optional[UserClientConfig] = None):
        self.config = config or UserClientConfig()
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
):
    print("[UserInput] User input thread started. Enter natural language instructions; 'quit'/'exit' to stop.")
    while not stop_flag.is_set():
        try:
            text = input("\n[UserInput] > ").strip()
        except EOFError:
            print("[UserInput] EOF detected, exiting.")
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            print("[UserInput] Exit command received.")
            stop_flag.set()
            break

        task_id = get_task_id()
        if not task_id:
            print("[UserInput] No task_id yet, ignoring.")
            continue

        try:
            resp = user_client.send_user_instruction(task_id, text)
            print("[UserInput] Instruction sent. Current subtask:")
            print("  ", resp.get("current_subtask_description"))
        except Exception as e:
            print(f"[UserInput] Send failed: {e}")


# ==========================
# Main logic
# ==========================

def main():
    parser = argparse.ArgumentParser()
    # Low-level policy server
    parser.add_argument("--policy_host", default="192.168.1.103")
    parser.add_argument("--policy_port", type=int, default=8000)
    parser.add_argument("--task_prompt", default="pick up a red flower, and place it in the vase on the left.")
    parser.add_argument("--control_hz", type=float, default=25.0)
    parser.add_argument("--max_steps", type=int, default=5)

    # Camera topics
    parser.add_argument("--head_camera", default="/head_camera/camera/color/image_raw")
    parser.add_argument("--head_camera_info", default="/head_camera/camera/color/camera_info")
    parser.add_argument("--wrist_camera", default="/wrist_camera/camera/color/image_raw")
    parser.add_argument("--wrist_camera_info", default="/wrist_camera/camera/color/camera_info")

    # Policy modes
    parser.add_argument("--policy_proprio_mode", default="libero_eef",
                        choices=["libero_eef", "joints_positions"])
    parser.add_argument("--policy_action_mode", default="relative_eef",
                        choices=["relative_eef", "absolute_eef", "joint_positions"])

    # Task server
    parser.add_argument("--task_server_base_url", default="http://localhost:8000")
    parser.add_argument("--task_server_timeout", type=int, default=60)
    parser.add_argument("--observer_window_size", type=int, default=8)
    parser.add_argument("--human_intervene_for_planner", action="store_true")

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
        )
    )
    user_client = UserClient(
        UserClientConfig(
            base_url=args.task_server_base_url,
            timeout=args.task_server_timeout,
        )
    )

    # 5. Create task (using current frame)
    head0 = head_cam.capture_image()
    wrist0 = wrist_cam.capture_image()
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

    print("[Main] Task created:")
    print("  task_id:", task_id)
    print("  current_subtask:", init_resp.get("current_subtask_description"))

    # Current policy prompt: will be dynamically updated by task server's current_subtask
    current_policy_prompt = init_resp.get("current_subtask_description") or args.task_prompt

    # Shared task_id & stop_flag for user input thread
    task_id_lock = threading.Lock()
    stop_flag = threading.Event()

    def get_task_id() -> Optional[str]:
        with task_id_lock:
            return task_id

    user_thread = threading.Thread(
        target=user_input_loop,
        args=(user_client, get_task_id, stop_flag),
        daemon=True,
    )
    user_thread.start()

    # 6. Main loop
    try:
        while not stop_flag.is_set():
            chunk_start = time.perf_counter()

            # 6.1 Get robot state
            try:
                all_state = robot.get_state()
            except Exception as e:
                print(f"[Main] Failed to get robot state: {e}")
                time.sleep(0.1)
                continue

            if args.policy_proprio_mode == "joints_positions":
                joint_state = all_state["joint_states"]      # (6,)
                gripper_state = all_state["gripper_state"]   # (2,)
                input_state = np.concatenate([joint_state, gripper_state[1:]])  # (8,)
                eef_state = all_state["end_effector_pose"]   # Still keep for relative_eef
            else:  # libero_eef
                eef_state = all_state["end_effector_pose"]   # (6,)
                gripper_state = all_state["gripper_state"]   # (2,)
                input_state = np.concatenate(
                    [eef_euler2axis(eef_state), gripper_state[1:]]
                )  # (8,)

            print("Input proprioception:", input_state)

            # 6.2 Capture images
            head_img = head_cam.capture_image()
            wrist_img = wrist_cam.capture_image()
            if head_img is None or wrist_img is None:
                print("[Main] Failed to capture images, retrying...")
                time.sleep(0.1)
                continue

            # Resize for policy (224x224)
            head_img_p = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(head_img, 224, 224)
            )
            wrist_img_p = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, 224, 224)
            )

            # 6.3 Send observation to task server, get current subtask
            try:
                step_resp = robot_client.send_step(
                    task_id=task_id,
                    image=head_img,
                    waist_image=wrist_img,
                )
                cur_subtask = step_resp.get("current_subtask_description")
                if isinstance(cur_subtask, str) and cur_subtask.strip():
                    current_policy_prompt = cur_subtask.strip()
                else:
                    current_policy_prompt = args.task_prompt

                print("\n[Main] Task server step:")
                print("  state:", step_resp.get("state"))
                print("  is_done:", step_resp.get("is_done"))
                print("  current_subtask:", cur_subtask)

                if step_resp.get("is_done"):
                    print("[Main] Task server reports done. Stopping.")
                    stop_flag.set()
                    break
            except Exception as e:
                print(f"[Main] send_step failed: {e}")
                print("[Main] Using previous prompt and continuing...")

            # 6.4 Construct observation with current subtask as prompt
            observation = {
                "observation/state": input_state,
                "observation/image": head_img_p,
                "observation/wrist_image": wrist_img_p,
                "prompt": current_policy_prompt,
            }

            # 6.5 Call policy to get action sequence
            try:
                result = policy_client.infer(observation)
                actions = result["actions"]  # Shape (H, 7)
            except Exception as e:
                print(f"[Main] Policy inference failed: {e}")
                print("[Main] Skipping this control cycle...")
                time.sleep(0.1)
                continue

            executed = 0
            for a in actions:
                if stop_flag.is_set():
                    break
                if executed >= args.max_steps:
                    break

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

                    else:
                        raise ValueError(f"Unknown policy_action_mode: {args.policy_action_mode}")

                except Exception as e:
                    print(f"[Main] Robot control error: {e}")
                    print("[Main] Stopping for safety...")
                    stop_flag.set()
                    break

                executed += 1

                elapsed = time.perf_counter() - step_start
                sleep_t = dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            chunk_time = time.perf_counter() - chunk_start
            hz = executed / chunk_time if chunk_time > 0 and executed > 0 else float("nan")
            print(f"[Chunk] steps={executed}, time={chunk_time:.3f}s, freq={hz:.2f}Hz")

    finally:
        print("[Main] Stopping robot & closing resources...")
        stop_flag.set()
        try:
            robot.go_to_sleep()
        except Exception as e:
            print(f"[Main] Failed to put robot to sleep: {e}")
        robot.disconnect()
        head_cam.disconnect()
        wrist_cam.disconnect()
        try:
            user_thread.join(timeout=2.0)
        except Exception as e:
            print(f"[Main] Failed to join user thread: {e}")


if __name__ == "__main__":
    main()