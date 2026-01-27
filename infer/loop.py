# src/test_infer_loop.py
import os
import glob
import pathlib
import time
from typing import Dict, Any, Optional, List, Tuple
import openai

from src.client.base_vlm_client import BaseVLMClient
from src.client.planner_vlm import PlannerVLM
from src.client.observer_vlm import ObserverVLM

from src.agent.multitag_planner import PlannerAgent
from src.agent.observer import ObserverAgent

from src.extractor import (
    extract_current_subtask,
    is_plan_done,
)


# =============== 工具函数 ===============

def natural_sorted_images(folder: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))

    def natural_key(p: str):
        name = pathlib.Path(p).name
        parts, buf = [], ""
        for ch in name:
            if ch.isdigit():
                buf += ch
            else:
                if buf:
                    parts.append(int(buf))
                    buf = ""
                parts.append(ch)
        if buf:
            parts.append(int(buf))
        return parts

    paths.sort(key=natural_key)
    return paths


def stride_sample(paths: List[str], k: int) -> List[str]:
    if k <= 0:
        return paths
    return [p for i, p in enumerate(paths) if i % k == 0]


def sample_up_to_n_evenly(paths: List[str], n: int) -> List[str]:
    length = len(paths)
    if length <= n:
        return paths
    if n <= 1:
        # 只保留首元素
        return [paths[-1]]

    # 必须保留首尾
    result = [paths[0]]
    # 中间还需要取多少个
    middle_count = n - 2

    if middle_count <= 0:
        # 只保留首尾中的一个（此时 n==1 已在上面处理，这里理论上不会进）
        return [paths[0], paths[-1]][:n]

    # 在 (0, length-1) 之间均匀选择 middle_count 个中间索引
    start = 1
    end = length - 2  # 最后一个索引留给 paths[-1]
    if end < start or middle_count == 0:
        # 没有足够的中间元素，直接补上最后一个
        result.append(paths[-1])
        return result

    step = (end - start + 1) / (middle_count + 1)
    for i in range(1, middle_count + 1):
        idx = round(start + (i - 1) * step)
        # 双重保护，避免越界
        idx = max(start, min(end, idx))
        result.append(paths[idx])

    result.append(paths[-1])
    return result


# =============== Observer 调度（窗口在调度层实现） ===============
def observer_loop_with_scheduler_window(
    observer,
    plan_list: str,
    sampled_imgs: List[str],
    window_size_w: int,
    max_tokens: int = 512,
) -> Tuple[str, List[str], str, List[Dict[str, Any]]]:
    """
    Run the Observer with a growing window then a fixed-size sliding window.
    Only when Observer returns 'done' for 3 consecutive calls do we treat it as truly done.

    Returns:
        (final_status, seen_image_paths, last_raw_xml, all_observer_calls)
        - final_status: "done" or "not_done"
        - seen_image_paths: list of image paths that have been fed to Observer so far
        - last_raw_xml: the latest raw XML (if observer provides it), else ""
        - all_observer_calls: list of all Observer calls with their results
    """
    n = len(sampled_imgs)
    if n == 0:
        print("⚠️ 本轮没有可用图片。")
        return "not_done", [], "", []

    w = max(1, int(window_size_w))
    seen: List[str] = []
    final_status = "not_done"
    last_raw_xml = ""
    all_observer_calls: List[Dict[str, Any]] = []

    # Observer 连续 done 计数器
    consecutive_done_obs = 0
    REQUIRED_CONSECUTIVE_DONE = 1  # 如需可配置，可将其做为函数参数

    # 处理一次 Observer 返回并更新计数
    def handle_observer_result(r, batch: List[str]) -> bool:
        nonlocal last_raw_xml, consecutive_done_obs
        status_str = str(getattr(r, "status", "")).strip().lower()
        raw_xml = getattr(r, "raw_xml", "") or ""

        print(f"status: {status_str}")
        if raw_xml:
            print("--- raw_xml ---")
            print(raw_xml)
            print("---------------")
            last_raw_xml = raw_xml

        # Record this Observer call
        all_observer_calls.append({
            "image_paths": batch.copy(),
            "status": status_str,
            "raw_output": raw_xml,
            "timestamp": time.time(),
        })

        if status_str == "done":
            consecutive_done_obs += 1
            print(f"↪️ Observer done streak: {consecutive_done_obs}/{REQUIRED_CONSECUTIVE_DONE}")
            if consecutive_done_obs >= REQUIRED_CONSECUTIVE_DONE:
                return True  # 真正 done
        else:
            if consecutive_done_obs > 0:
                print("↪️ Done streak broken, reset to 0.")
            consecutive_done_obs = 0
        return False

    # 阶段 1：从 1 增长到 min(n, w)
    grow_limit = min(n, w)
    for size in range(1, grow_limit + 1):
        batch = sampled_imgs[:size]
        print(f"[Observer] Growing window size={size}, frames=({pathlib.Path(batch[0]).name} ... {pathlib.Path(batch[-1]).name})")
        r = observer.run(
            image_paths=batch,
            plan_list=plan_list,
            max_tokens=max_tokens,
        )

        for p in batch:
            if p not in seen:
                seen.append(p)

        if handle_observer_result(r, batch):
            final_status = "done"
            return final_status, seen, last_raw_xml, all_observer_calls

    # 阶段 2：固定长度 w 的滑动窗口
    if n > w:
        for start in range(1, n - w + 1):
            batch = sampled_imgs[start:start + w]
            first, last = pathlib.Path(batch[0]).name, pathlib.Path(batch[-1]).name
            print(f"[Observer] Sliding window [{start}:{start+w}) frames=({first} ... {last})")
            r = observer.run(
                image_paths=batch,
                plan_list=plan_list,
                max_tokens=max_tokens,
            )

            for p in batch:
                if p not in seen:
                    seen.append(p)

            if handle_observer_result(r, batch):
                final_status = "done"
                break

    return final_status, seen, last_raw_xml, all_observer_calls


def ask_nonempty(prompt: str) -> str:
    while True:
        s = input(prompt).strip()
        if s:
            return s
        print("该项为必填，请重新输入。")


# =============== 英文前缀（严谨简洁） ===============

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


def main():
    # ========= 1. 配置 API =========
    os.environ.setdefault("OPENAI_API_KEY", "xx")
    os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")
    client = openai.OpenAI()

    # ========= 2. 初始化 VLM & Agents =========
    base_client = BaseVLMClient(
        model="claude-sonnet-4-5-20250929",  # 按你的可用模型替换
        # model="gpt-4o",
        client=client,
    )
    planner_vlm = PlannerVLM(base_client=base_client)
    observer_vlm = ObserverVLM(base_client=base_client)

    planner = PlannerAgent(vlm=planner_vlm)
    observer = ObserverAgent(vlm=observer_vlm)

    # ========= 3. 初始化：必填 instruction + 初始单张图片 =========
    print("== 初始化 Planner ==")
    global_instruction = ask_nonempty("请输入初始 user instruction（必填）: ") 
    # Use the robot arm to move the toy duck and the snack from the table into the correct boxes. Please verify the boxes before placing items. Lucy likes the purple toy duck.
    initial_image = ask_nonempty("请输入初始图片路径（单张，必填）: ")
    # /inspire/hdd/global_user/gongjingjing-25039/lji/memory/lerobot-dataset/images/combined_image/episode_000000/frame_000000.png
    if not os.path.isfile(initial_image):
        print(f"⚠️ 初始图片不存在：{initial_image}（将以空图列表调用）")

    # 首次 Planner： initial image + instruction -> 初始 plan_list
    print("➡️ 调用首次 Planner ...")
    first_user_instruction = build_planner_user_instruction(
        base_instruction=global_instruction,
        current_plan_list="",
        is_first_round=True,
    )

    # 可选：从环境变量控制日志输出目录（默认关闭）
    interactions_dir = os.getenv("PLANNER_LOG_INTERACTIONS_DIR")  # e.g. "./_logs/interactions"
    memory_dir = os.getenv("PLANNER_LOG_MEMORY_DIR")              # e.g. "./_logs/memory"

    first_plan_res = planner.run_refine(
        image_paths=[initial_image] if os.path.isfile(initial_image) else [],
        initial_plan_list="",
        user_instruction=first_user_instruction,
        max_tokens=4096,
        max_inner_rounds=10,
        # 利用新参数（均为可选）
        do_reset=True,
        print_full_interactions_each_round=True,
        log_interactions_json_dir=interactions_dir if interactions_dir else None,
        use_cli_prompt_for_memory_view=False,
        decide_view_memory=None,
        log_memory_json_dir=memory_dir if memory_dir else None,
        drop_images_in_json=True,
    )
    current_plan_list = (first_plan_res.plan_text or "").strip()

    print("\n== 初始 PLAN ==")
    print(current_plan_list or "(No plan)")
    print("\n== 初始 SUMMARY ==")
    print(first_plan_res.summary or "(No summary)")

    # 完成性检查：若初始即 done，直接结束
    if is_plan_done(current_plan_list):
        print("\n✅ 计划已完成（初始即 done）。流程结束。")
        return

    # ========= 4. 主循环 =========
    cycle = 1
    while True:
        print(f"\n==== 循环 #{cycle} ====")
        folder = input("本轮图片文件夹（q 退出）: ").strip()
        if folder.lower() in ("q", "quit", "exit"):
            print("退出。")
            break
        if not os.path.isdir(folder):
            print(f"❌ 文件夹不存在: {folder}")
            continue

        # 采样间隔 k
        k_str = input("采样间隔 k（默认 16）: ").strip()
        k = 16
        if k_str:
            try:
                k = int(k_str)
            except ValueError:
                print("⚠️ 非法输入，使用默认 k=4")

        # 滑动窗口大小 w（调度层控制）
        w_str = input("Observer 滑动窗口大小 w（默认 8）: ").strip()
        if w_str:
            try:
                window_size_w = max(1, int(w_str))
            except ValueError:
                print("⚠️ 非法输入，使用默认 w=8")
                window_size_w = 8
        else:
            window_size_w = 8

        # 本轮用户可能提供的新输入（留空则沿用）
        maybe_new_instr = input("本轮是否更新 user instruction？留空则沿用默认: ").strip()
        if maybe_new_instr:
            global_instruction = maybe_new_instr  # 直接替换 global_instruction

        # 采样图片
        all_imgs = natural_sorted_images(folder)
        sampled_imgs = stride_sample(all_imgs, k)
        if not sampled_imgs:
            print("⚠️ 该目录按当前 k 未采样到图片，跳过该轮。")
            continue

        # Observer（窗口调度）
        status, seen_images, _last_xml = observer_loop_with_scheduler_window(
            observer=observer,
            plan_list=current_plan_list,
            sampled_imgs=sampled_imgs,
            window_size_w=window_size_w,
            max_tokens=512,
        )
        print(f"Observer 最终状态: {status}")

        # Planner user_instruction 构造：使用当前的 global_instruction + 前缀 + 当前 plan list
        user_instruction_for_planner = build_planner_user_instruction(
            base_instruction=global_instruction,
            current_plan_list=current_plan_list,
            is_first_round=False,
        )

        # Planner：从本轮 seen images 采样至 8 张 refine
        planner_images = sample_up_to_n_evenly(seen_images if seen_images else sampled_imgs, 8)
        print("➡️ 调用 Planner refine（使用本轮 seen images 采样至 8 张） ...")
        plan_res = planner.run_refine(
            image_paths=planner_images,
            initial_plan_list=current_plan_list,
            user_instruction=user_instruction_for_planner,
            max_tokens=4096,
            max_inner_rounds=10,
            # 同样带上可选日志参数
            do_reset=True,
            print_full_interactions_each_round=True,
            log_interactions_json_dir=interactions_dir if interactions_dir else None,
            use_cli_prompt_for_memory_view=False,
            decide_view_memory=None,
            log_memory_json_dir=memory_dir if memory_dir else None,
            drop_images_in_json=True,
        )
        current_plan_list = (plan_res.plan_text or "").strip()

        print("\n== 更新后的 PLAN ==")
        print(current_plan_list or "(No plan)")
        print("\n== SUMMARY ==")
        print(plan_res.summary or "(No summary)")

        # 提取 [current] 子任务（去除 () 附注）
        current_subtask = extract_current_subtask(current_plan_list)
        if current_subtask:
            print(f"\n>> 当前子任务（已清洗）: {current_subtask}")

        # 结束条件：若当前 plan_list 已完成，结束整个 Loop
        if is_plan_done(current_plan_list):
            print("\n✅ 计划已完成（判定为 done）。流程结束。")
            break

        cycle += 1


if __name__ == "__main__":
    main()