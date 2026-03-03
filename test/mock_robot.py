#!/usr/bin/env python3
"""Mock robot client for testing task server without real robot.

This script generates synthetic test images at regular intervals
to simulate robot observations.
""" 
from __future__ import annotations

import argparse
import time
import requests
from PIL import Image, ImageDraw
import io
import numpy as np
from typing import Optional, Dict, Any


def generate_test_image(
    step: int,
    width: int = 640,
    height: int = 480,
) -> Image.Image:
    """Generate a simple synthetic test image with step counter."""
    # Create gradient background
    gradient = np.linspace(50, 200, height, dtype=np.uint8)
    img_array = np.tile(gradient, (width, 1)).T
    img_array = np.stack([img_array] * 3, axis=-1)

    # Add visual content
    center_x, center_y = width // 2, height // 2
    box_size = 100 + (step % 5) * 10
    x1 = center_x - box_size // 2
    y1 = center_y - box_size // 2
    x2 = center_x + box_size // 2
    y2 = center_y + box_size // 2

    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], fill=(0, 255, 0), outline=(0, 200, 0), width=3)
    draw.text((20, 20), f"Step {step}", fill=(255, 255, 255))
    draw.text((20, 45), f"Time: {time.strftime('%H:%M:%S')}", fill=(255, 255, 255))

    # Moving red circle
    circle_x = 100 + (step * 20) % (width - 150)
    circle_y = height // 3
    draw.ellipse([circle_x, circle_y, circle_x+20, circle_y+20], fill=(255, 0, 0))

    return img


def _to_pil(img) -> Image.Image:
    """Convert image to RGB PIL image."""
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


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    """Convert PIL image to PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_task(
    base_url: str,
    global_instruction: str,
    initial_image,
    initial_waist_image=None,
    observer_window_size: int = 8,
) -> Dict[str, Any]:
    """POST /tasks: create task on server with initial images + instruction."""
    url = f"{base_url}/tasks"
    main_pil = _to_pil(initial_image)
    main_bytes = _pil_to_png_bytes(main_pil)
    files = {"initial_image": ("initial_image.png", io.BytesIO(main_bytes), "image/png")}

    # server currently requires waist image. If not provided, mirror main image.
    waist_source = initial_waist_image if initial_waist_image is not None else initial_image
    waist_pil = _to_pil(waist_source)
    waist_bytes = _pil_to_png_bytes(waist_pil)
    files["initial_waist_image"] = ("initial_waist_image.png", io.BytesIO(waist_bytes), "image/png")

    data = {
        "global_instruction": global_instruction,
        "observer_window_size": observer_window_size,
        "human_intervene_for_planner": False,
    }
    resp = requests.post(url, files=files, data=data, timeout=180)
    resp.raise_for_status()
    return resp.json()


def send_step(
    base_url: str,
    task_id: str,
    image,
    waist_image=None,
    timeout: int = 180,
) -> Dict[str, Any]:
    """POST /tasks/{task_id}/step: upload current observation images."""
    url = f"{base_url}/tasks/{task_id}/step"
    main_pil = _to_pil(image)
    main_bytes = _pil_to_png_bytes(main_pil)
    files = {"image": ("step_image.png", io.BytesIO(main_bytes), "image/png")}

    # server currently requires waist image for each step.
    waist_source = waist_image if waist_image is not None else image
    waist_pil = _to_pil(waist_source)
    waist_bytes = _pil_to_png_bytes(waist_pil)
    files["waist_image"] = ("waist_step_image.png", io.BytesIO(waist_bytes), "image/png")

    resp = requests.post(url, files=files, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Mock robot client with synthetic images")
    parser.add_argument("--task_server_base_url", default="http://localhost:8000",
                        help="Base URL for task server")
    parser.add_argument("--observer_window_size", type=int, default=8, help="Observer window size")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Interval between step uploads (seconds)")
    parser.add_argument("--steps", type=int, default=20, help="Total number of steps to send")
    parser.add_argument("--task_prompt", default="There is a green rectangle and red circle. Observe the changes.",
                        help="Initial task prompt")

    args = parser.parse_args()
    print(f"[MockRobot] Starting: {args.task_server_base_url}")
    print(f"[MockRobot] Task: {args.task_prompt}\n")

    initial_image = generate_test_image(0)
    try:
        init_resp = create_task(
            base_url=args.task_server_base_url,
            global_instruction=args.task_prompt,
            initial_image=initial_image,
            observer_window_size=args.observer_window_size,
        )
        task_id = init_resp.get("task_id")

        if not task_id:
            print(f"[MockRobot] Failed to get task_id!")
            return

        print(f"[MockRobot] Task created: task_id={task_id}")
        print(f"[MockRobot] Current subtask: {init_resp.get('current_subtask_description')}")
        print(f"[MockRobot] State: {init_resp.get('state')}, Done: {init_resp.get('is_done')}\n")

        for step in range(1, args.steps + 1):
            print(f"[MockRobot] Step {step}/{args.steps}...")
            head_image = generate_test_image(step)

            resp = send_step(
                base_url=args.task_server_base_url,
                task_id=task_id,
                image=head_image,
                timeout=60,
            )

            print(f"  -> State: {resp.get('state')}, Subtask: {resp.get('current_subtask_description')}")

            if resp.get("is_done", False):
                print(f"\n[MockRobot] Task completed at step {step}!")
                break

            sleep_time = max(0.0, args.interval)
            if sleep_time > 0:
                time.sleep(sleep_time)
            print()

        print(f"\n[MockRobot] Test completed")

    except Exception as e:
        print(f"[MockRobot] Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"[MockRobot] Exit")


if __name__ == "__main__":
    main()
