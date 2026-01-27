#!/usr/bin/env python3
"""
VLM Performance Profiler - Test different models to measure API call latency

This script tests:
1. Observer calls with varying number of images (1, 4, 8, 16)
2. Planner calls with varying turns (1, 2, 5, 10)
3. Different models (configurable)

Usage:
    uv run python src/test/test_vlm_profile.py --model gpt-4o --test observer
    uv run python src/test/test_vlm_profile.py --model qwen3-vl-235b-a22b-instruct --test planner
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import json
from typing import Dict, Any, List
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import openai
from src.client.base_vlm_client import BaseVLMClient
from src.client.planner_vlm import PlannerVLM
from src.client.observer_vlm import ObserverVLM
from src.agent.multitag_planner import PlannerAgent
from src.agent.observer import ObserverAgent


class VLMPerformanceProfiler:
    """Performance profiler for VLM API calls."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_client = None
        self.planner_vlm = None
        self.observer_vlm = None
        self.planner = None
        self.observer = None

    def initialize(self):
        """Initialize VLM clients and agents."""
        print(f"[Profiler] Initializing with model: {self.model_name}")

        os.environ.setdefault("OPENAI_API_KEY", "xx")
        os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")
        client = openai.OpenAI()

        self.base_client = BaseVLMClient(
            model=self.model_name,
            client=client,
        )

        self.planner_vlm = PlannerVLM(base_client=self.base_client)
        self.observer_vlm = ObserverVLM(base_client=self.base_client)

        self.planner = PlannerAgent(vlm=self.planner_vlm)
        self.observer = ObserverAgent(vlm=self.observer_vlm)

        print(f"✅ Initialization complete")
        return self

    def create_test_images(self, count: int) -> List[Image.Image]:
        """Create test images for profiling."""
        images = []
        for i in range(count):
            # Create images with varying colors
            color = (100 + i * 10, 150 + i * 5, 200 - i * 8)
            img = Image.new('RGB', (640, 480), color=color)
            images.append(img)
        return images

    def save_images_to_disk(self, images: List[Image.Image], temp_dir: Path) -> List[str]:
        """Save images to disk and return their paths."""
        temp_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, img in enumerate(images):
            path = (temp_dir / f"test_image_{i:03d}.png")
            img.save(path, format="PNG")
            paths.append(str(path))
        return paths

    def profile_observer_call(self, num_images: int, run: int) -> Dict[str, Any]:
        """
        Profile a single Observer call with varying numbers of images.

        Returns:
            A dictionary with profile results
        """
        plan_list = """Step 1: Find the red block [current]
Step 2: Place it in the blue box [pending]
Step 3: Verify placement [pending]"""

        # Prepare images
        temp_dir = Path("/tmp/vlm_profile_images")
        images = self.create_test_images(num_images)
        image_paths = self.save_images_to_disk(images, temp_dir)

        # Profile the call
        start_time = time.perf_counter()
        try:
            result = self.observer.run(
                image_paths=image_paths,
                plan_list=plan_list,
                max_tokens=512,
            )
            elapsed = time.perf_counter() - start_time
            status = "success"
            error = None
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            status = "error"
            error = str(e)

        # Cleanup temp images
        for path in image_paths:
            try:
                os.remove(path)
            except:
                pass

        return {
            "run_number": run,
            "test_type": "observer",
            "model": self.model_name,
            "num_images": num_images,
            "status": status,
            "elapsed_seconds": elapsed,
            "error": error,
        }

    def profile_planner_single_turn(self, num_images: int, run: int) -> Dict[str, Any]:
        """
        Profile a single Planner turn (initial turn only, no memory operations).

        Returns:
            A dictionary with profile results
        """
        plan_list = """Step 1: Find the red block [current]
Step 2: Place it in the blue box [pending]
Step 3: Verify placement [pending]"""

        user_instruction = "Your current plan list represents the active task. Based on the user's new input, adjust the plan list by adding or modifying items as needed. You must not completely change or replace its overall content.\n" \
                         "----- Current Plan List -----\n" \
                         "Step 1: Find the red block [current]\n" \
                         "Step 2: Place it in the blue box [pending]" \
                         "\n----- User New Input -----\n" \
                         ""

        # Reset planner internal state (important)
        self.planner.reset()

        # Prepare images
        temp_dir = Path("/tmp/vlm_profile_images")
        images = self.create_test_images(num_images)
        image_paths = self.save_images_to_disk(images, temp_dir)

        self.planner.current_input_image_paths = image_paths

        # Reset and setup for single turn
        self.planner.reset()
        self.planner.plan_list = plan_list

        # Profile the call
        start_time = time.perf_counter()
        try:
            # Single turn only (first turn)
            from src.client.planner_vlm import PlannerVLM
            raw_xml = self.planner_vlm.chat_first_turn(
                system_prompt=self.planner.system_prompt,
                user_instruction=user_instruction,
                image_paths=image_paths,
                max_tokens=4096,
            )
            elapsed = time.perf_counter() - start_time
            status = "success"
            error = None
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            status = "error"
            error = str(e)

        # Cleanup temp images
        for path in image_paths:
            try:
                os.remove(path)
            except:
                pass

        return {
            "run_number": run,
            "test_type": "planner_single_turn",
            "model": self.model_name,
            "num_images": num_images,
            "status": status,
            "elapsed_seconds": elapsed,
            "error": error,
        }

    def profile_planner_multi_turn(self, num_images: int, num_turns: int, run: int) -> Dict[str, Any]:
        """
        Profile a full Planner refine with multiple turns (no actual memory).

        Returns:
            A dictionary with profile results
        """
        plan_list = """Step 1: Find the red block [current]
Step 2: Place it in the blue box [pending]
Step 3: Verify placement [pending]"""

        user_instruction = "Your current plan list represents the active task. Based on the user's new input, adjust the plan list by adding or modifying items as needed. You must not completely change or replace its overall content.\n" \
                         "----- Current Plan List -----\n" \
                         "Step 1: Find the red block [current]\n" \
                         "Step 2: Place it in the blue box [pending]" \
                         "\n----- User New Input -----\n" \
                         ""

        # Reset planner internal state
        self.planner.reset()

        # Prepare images
        temp_dir = Path("/tmp/vlm_profile_images")
        images = self.create_test_images(num_images)
        image_paths = self.save_images_to_disk(images, temp_dir)

        self.planner.current_input_image_paths = image_paths

        # Profile the call
        start_time = time.perf_counter()
        try:
            res = self.planner.run_refine(
                image_paths=image_paths,
                initial_plan_list=plan_list,
                user_instruction=user_instruction,
                max_tokens=4096,
                max_inner_rounds=num_turns,
                do_reset=True,
                print_full_interactions_each_round=False,
            )
            elapsed = time.perf_counter() - start_time
            status = "success"
            error = None
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            status = "error"
            error = str(e)

        # Cleanup temp images
        for path in image_paths:
            try:
                os.remove(path)
            except:
                pass

        return {
            "run_number": run,
            "test_type": "planner_multi_turn",
            "model": self.model_name,
            "num_images": num_images,
            "num_turns": num_turns,
            "status": status,
            "elapsed_seconds": elapsed,
            "error": error,
        }

    def run_observer_tests(self, image_counts: List[int], repeats: int = 3) -> List[Dict[str, Any]]:
        """Run Observer performance tests with different image counts."""
        print(f"\n{'='*70}")
        print(f"OBSERVER PERFORMANCE TESTS")
        print(f"{'='*70}")

        results = []
        for num_images in image_counts:
            print(f"\n[Observer] Testing with {num_images} images...")
            for run in range(1, repeats + 1):
                print(f"  Run {run}/{repeats}... ", end="", flush=True)
                result = self.profile_observer_call(num_images, run)
                results.append(result)
                print(f"✅ {result['elapsed_seconds']:.2f}s")

        return results

    def run_planner_single_turn_tests(self, image_counts: List[int], repeats: int = 3) -> List[Dict[str, Any]]:
        """Run Planner single-turn performance tests."""
        print(f"\n{'='*70}")
        print(f"PLANNER SINGLE-TURN TESTS")
        print(f"{'='*70}")

        results = []
        for num_images in image_counts:
            print(f"\n[Planner] Testing with {num_images} images...")
            for run in range(1, repeats + 1):
                print(f"  Run {run}/{repeats}... ", end="", flush=True)
                result = self.profile_planner_single_turn(num_images, run)
                results.append(result)
                if result['status'] == 'success':
                    print(f"✅ {result['elapsed_seconds']:.2f}s")
                else:
                    print(f"❌ {result['error']}")

        return results

    def run_planner_multi_turn_tests(
        self, image_counts: List[int], turn_counts: List[int], repeats: int = 2
    ) -> List[Dict[str, Any]]:
        """Run Planner multi-turn performance tests."""
        print(f"\n{'='*70}")
        print(f"PLANNER MULTI-TURN TESTS")
        print(f"{'='*70}")

        results = []
        for num_images in image_counts:
            for num_turns in turn_counts:
                print(f"\n[Planner] Testing with {num_images} images, {num_turns} turns...")
                for run in range(1, repeats + 1):
                    print(f"  Run {run}/{repeats}... ", end="", flush=True)
                    result = self.profile_planner_multi_turn(num_images, num_turns, run)
                    results.append(result)
                    if result['status'] == 'success':
                        print(f"✅ {result['elapsed_seconds']:.2f}s")
                    else:
                        print(f"❌ {result['error']}")

        return results


def main():
    parser = argparse.ArgumentParser(description="VLM Performance Profiler")
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Model to test",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["observer", "planner_single", "planner_multi", "all"],
        default="all",
        help="Which tests to run",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of times to repeat each test",
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"VLM Performance Profiler")
    print(f"Model: {args.model}")
    print(f"{'='*70}")

    # Initialize profiler
    profiler = VLMPerformanceProfiler(args.model)
    profiler.initialize()

    # Run tests
    results = []

    if args.test in ["observer", "all"]:
        print(f"\n⚠️ NOTE: Observer tests may take a while (checking token limits)...")
        observer_results = profiler.run_observer_tests(
            image_counts=[1, 4, 8, 16],
            repeats=args.repeats,
        )
        results.extend(observer_results)

    if args.test in ["planner_single", "all"]:
        planner_single_results = profiler.run_planner_single_turn_tests(
            image_counts=[1, 4, 8, 16],
            repeats=min(args.repeats, 2),  # Limit to 2 to save time
        )
        results.extend(planner_single_results)

    if args.test in ["planner_multi", "all"]:
        planner_multi_results = profiler.run_planner_multi_turn_tests(
            image_counts=[1, 4, 8],
            turn_counts=[1, 2, 5, 10],
            repeats=min(args.repeats, 1),  # Limit to 1 to save time (these are slow!)
        )
        results.extend(planner_multi_results)

    # Save results
    output_file = f"vlm_profile_{args.model.replace('/', '_')}_{args.test}_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"\nSaved {len(results)} test results to: {output_file}")

    # Print summary
    success_results = [r for r in results if r['status'] == 'success']
    if success_results:
        avg_time = sum(r['elapsed_seconds'] for r in success_results) / len(success_results)
        print(f"Average successful call time: {avg_time:.2f}s")
        print(f"Total successful calls: {len(success_results)}/{len(results)}")

    # Print by test type
    print(f"\nBreakdown by test type:")
    for test_type in set(r['test_type'] for r in results):
        test_results = [r for r in results if r['test_type'] == test_type and r['status'] == 'success']
        if test_results:
            avg_time = sum(r['elapsed_seconds'] for r in test_results) / len(test_results)
            print(f"  {test_type}: {len(test_results)} calls, avg {avg_time:.2f}s")


if __name__ == "__main__":
    main()