#!/usr/bin/env python3

"""
WidowX 250s 的同步推理执行脚本（单线程、按 chunk 运行）。

流程：
- 连接机械臂与 head/wrist 两个 RealSense 相机；
- 每次循环读取当前 joint/gripper 状态与两路图像，构造 observation 调用 pi0.5 policy；
- policy 返回一段动作序列 actions(H,7)，每步为 [dx,dy,dz,drx,dry,drz,gripper]；
- 将每步动作视为“相对当前时刻 t0 的 EEF 增量”，转换为绝对目标位姿 eef_state + delta，
  并以固定 control_hz 依次下发（move_to_pose + set_gripper）。
"""


import argparse
import time
import numpy as np

from openpi_client import image_tools, websocket_client_policy
from wx250s_ros2_client import WX250sRos2Client
from camera_client import RealSenseRos2Client

from utils import euler_angle_to_axis_angle


def eef_euler2axis(action):
    """
    将 action 中的 EEF 旋转部分从欧拉角转换为轴角表示。
    action: (7,) array-like, [x, y, z, roll, pitch, yaw, ...]
    """
    euler_rotation = action[3:6]
    rx, ry, rz = euler_angle_to_axis_angle(euler_rotation[0], euler_rotation[1], euler_rotation[2])
    axis_eef = np.concatenate([action[:3], np.array([rx, ry, rz]), action[6:]])
    return axis_eef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_host", default="192.168.1.103", help="pi0.5 server IP")
    parser.add_argument("--policy_port", type=int, default=8000)
    parser.add_argument(
        "--task_prompt",
        default="pick up a red flower, and place it in the vase on the left.",
    )
    parser.add_argument("--control_hz", type=float, default=25.0)
    parser.add_argument("--max_steps", type=int, default=5, help="max steps per chunk")
    parser.add_argument("--head_camera", default="/head_camera/camera/color/image_raw")
    parser.add_argument("--head_camera_info", default="/head_camera/camera/color/camera_info")
    parser.add_argument("--wrist_camera", default="/wrist_camera/camera/color/image_raw")
    parser.add_argument("--wrist_camera_info", default="/wrist_camera/camera/color/camera_info")
    parser.add_argument("--policy_proprio_mode", default="libero_eef", choices=["libero_eef", "joints_positions"])
    parser.add_argument("--policy_action_mode", default="relative_eef", choices=["relative_eef", "absolute_eef", "joint_positions"])
    # parser.add_argument("--top_camera", default="/top_camera/camera/color/image_raw")
    # parser.add_argument("--top_camera_info", default="/top_camera/camera/color/camera_info")

    args = parser.parse_args()

    dt = 1.0 / args.control_hz

    # 1. 连接机械臂
    robot = WX250sRos2Client()
    assert robot.connect(), "Failed to connect to WidowX 250s"
    robot.reset_to_home()

    # 2. 连接相机
    head_camera = RealSenseRos2Client(
        camera_id="head",
        rgb_topic=args.head_camera,
        camera_info_topic=args.head_camera_info,
    )
    assert head_camera.connect(), "Failed to connect to head RealSense camera"
    if not head_camera.wait_for_image(timeout=5.0):
        print("No image received from head_camera within timeout.")
        return

    wrist_camera = RealSenseRos2Client(
        camera_id="wrist",
        rgb_topic=args.wrist_camera,
        camera_info_topic=args.wrist_camera_info,
    )
    assert wrist_camera.connect(), "Failed to connect to wrist RealSense camera"
    if not wrist_camera.wait_for_image(timeout=5.0):
        print("No image received from wrist_camera within timeout.")
        return

    # 3. 连接 policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
    )
    print(f"Connected to policy server at {args.policy_host}:{args.policy_port}")
    print(f"Task prompt: {args.task_prompt}")

    try:
        while True:
            # ---- t0：构造 observation，并拿到 t0 的 EEF 绝对位姿 ----
            chunk_start = time.perf_counter()

            all_state = robot.get_state()

            if args.policy_proprio_mode == "joint_positions":
                joint_state = all_state["joint_states"]  # (6,)
                eef_state = all_state["end_effector_pose"]
                gripper_state = all_state["gripper_state"]
                input_state = np.concatenate([joint_state, gripper_state[1:]])  # (8,)
            elif args.policy_proprio_mode == "libero_eef":
                eef_state = all_state["end_effector_pose"]  # (6,)
                gripper_state = all_state["gripper_state"]  # (2,)
                input_state = np.concatenate([eef_euler2axis(eef_state), gripper_state[1:]])  # (8,)
            else:
                raise ValueError(f"Unknown policy_proprio_mode: {args.policy_proprio_mode}")

            print(f"Input proprioception: {input_state}")

            head_img = head_camera.capture_image()
            if head_img is None:
                print("No RGB image available from head_camera.")
                return
            head_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(head_img, 224, 224)
            )

            wrist_img = wrist_camera.capture_image()
            if wrist_img is None:
                print("No RGB image available from wrist_camera.")
                return
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, 224, 224)
            )

            observation = {
                "observation/state": input_state,
                "observation/image": head_img,
                "observation/wrist_image": wrist_img,
                "prompt": args.task_prompt,
            }

            #inf_start = time.perf_counter()
            # ---- 调一次 policy，得到一个 chunk ----
            result = policy_client.infer(observation)

            #print(f"Inference time: {time.perf_counter() - inf_start:.3f}s")
            actions = result["actions"]  # 预期形状 (H, 7)

            # if actions.ndim != 2 or actions.shape[1] != 7:
            #     raise ValueError(f"Unexpected action shape: {actions.shape}, expected (H, 7)")

            # ---- 在这个 chunk 里，动作都是“相对 t0 的 EEF 变化量” ----
            executed = 0

            for a in actions:
                if executed >= args.max_steps:
                    break

                step_start = time.perf_counter()
                # import pdb; pdb.set_trace()
                # >>> Moving start
                if args.policy_action_mode == "relative_eef":
                    # a[:6] 是相对 t0 EEF 的 delta -> 目标 EEF = base_eef + delta
                    delta_eef = a[:6]

                    # eef_state here is euler angles, while action is axis-angle
                    base_eef = eef_euler2axis(eef_state)  # (6,)

                    target_pose = base_eef + delta_eef  # (6,)

                    # 绝对 set 到这个位姿
                    robot.move_to_pose(target_pose, blocking=False)
                    robot.set_gripper(a[6], blocking=False)

                elif args.policy_action_mode == "absolute_eef":
                    # 直接把 a[:6] 作为目标 EEF 下发
                    target_pose = a[:6]
                    robot.move_to_pose(target_pose, blocking=False)
                    robot.set_gripper(a[6], blocking=False)

                elif args.policy_action_mode == "joint_positions":
                    # 直接把 a 作为目标 joint positions 下发
                    target_joints = a[:6]
                    robot.move_to_joint_positions(target_joints, blocking=False)
                    robot.set_gripper(a[6], blocking=False)

                else:
                    raise ValueError(f"Unknown policy_action_mode: {args.policy_action_mode}")

                # gripper_state = robot.get_gripper_state()
                # robot.set_gripper(0.0 if a[6] <= gripper_state[0] else 1.0, blocking=False)
                # print(gripper_state[0], a[6])
                # <<< Moving end

                executed += 1

                # 控制频率
                elapsed = time.perf_counter() - step_start
                sleep_t = dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            chunk_end = time.perf_counter()
            chunk_time = chunk_end - chunk_start
            chunk_hz = executed / chunk_time if chunk_time > 0 and executed > 0 else float("nan")

            print(
                f"[Chunk] steps_executed={executed}, "
                f"used_time={chunk_time:.3f}s, "
                f"control_freq={chunk_hz:.2f} Hz"
            )

    finally:
        print("Stopping robot & closing resources...")
        try:
            robot.go_to_sleep()
        except Exception:
            pass
        robot.disconnect()
        wrist_camera.disconnect()
        head_camera.disconnect()


if __name__ == "__main__":
    main()