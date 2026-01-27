#!/usr/bin/env python3
"""
Integration test for server functionality.

This test verifies:
1. Task creation with initial images
2. Observer execution with multiple steps
3. Planner refinement when Observer returns "done"
4. Round-based logging functionality

Usage:
    uv run python src/test/test_server_integration.py
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from typing import Optional, List
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.server.task_manager import ServerTaskManager
from src.server.schema import TaskConfig, TaskStateEnum
from src.server.image_utils import RobotImageInput
from src.agent.multitag_planner import PlannerAgent
from src.agent.observer import ObserverAgent
from src.client.base_vlm_client import BaseVLMClient
from src.client.planner_vlm import PlannerVLM
from src.client.observer_vlm import ObserverVLM
import openai


class MockVLMClient:
    """Mock VLM client for testing without actual API calls."""

    def __init__(self):
        self.call_count = 0

    def call(self, *args, **kwargs):
        self.call_count += 1
        # Return a mock response based on call count
        if self.call_count == 1:
            # First planner call - return initial plan
            return """<summary>
Initial task analysis. Creating plan to complete the task.
</summary>

<memory_operations>
<operation>
  <type>QUERY</type>
  <query>test_object</query>
  <reason>Check existing memory</reason>
</operation>
</memory_operations>

<plan_list>
Step 1: Pick up object from table [current]
Step 2: Place object in box [pending]
</plan_list>"""
        elif self.call_count <= 4:
            # Observer calls - return not_done for first few calls
            return "<status>not_done</status>"
        elif self.call_count == 5:
            # Observer call - return done
            return "<status>done</status>"
        else:
            # Subsequent planner calls - refine plan
            return """<summary>
Task progress update. Moving to next subtask.
</summary>

<memory_operations>
</memory_operations>

<plan_list>
Step 1: Pick up object from table [done]
Step 2: Place object in box [current]
Step 3: Verify placement [pending]
</plan_list>"""

    def encode_image_to_data_url(self, path: str) -> str:
        """Mock image encoding."""
        return "data:image/png;base64,mock_base64_data"


class MockBaseClient:
    """Mock base client for VLM."""

    def __init__(self, mock_client):
        self.mock_client = mock_client

    def encode_image_to_data_url(self, path: str) -> str:
        return self.mock_client.encode_image_to_data_url(path)

    def chat(self, *args, **kwargs):
        return self.mock_client.call(*args, **kwargs)


class MockPlannerVLM(PlannerVLM):
    """Mock Planner VLM for testing."""

    def __init__(self, mock_client):
        self.mock_client = mock_client
        self.base_client = MockBaseClient(mock_client)

    def call_multi_turn(self, *args, **kwargs):
        return self.mock_client.call(*args, **kwargs)


class MockObserverVLM(ObserverVLM):
    """Mock Observer VLM for testing."""

    def __init__(self, mock_client):
        self.mock_client = mock_client
        self.base_client = MockBaseClient(mock_client)

    def call_once(self, *args, **kwargs):
        return self.mock_client.call(*args, **kwargs)


def create_test_image() -> Image.Image:
    """Create a simple test image."""
    return Image.new('RGB', (640, 480), color=(100, 150, 200))


def load_existing_image(path: str) -> Optional[Image.Image]:
    """Load an existing image if it exists."""
    if os.path.exists(path):
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Failed to load image {path}: {e}")
    return None


def simulate_approval_thread(task_manager: ServerTaskManager, task_id: str, delay: float = 2.0):
    """
    Simulate user approval in a separate thread after a delay.

    This function waits for a pending approval and then approves it automatically.
    """
    time.sleep(delay)

    state = task_manager.tasks.get(task_id)
    if not state:
        print(f"[ApprovalThread] Task {task_id} not found")
        return

    if state.pending_approval:
        print(f"[ApprovalThread] Approving {state.pending_approval.agent_type} output")

        # Approve with original output (no modifications)
        state.approved_result = state.pending_approval.parsed_output

        # Wake up the blocked thread
        if state.approval_event:
            state.approval_event.set()

        print(f"[ApprovalThread] Approval completed")
    else:
        print(f"[ApprovalThread] No pending approval found")


def test_basic_workflow(use_mock: bool = True, test_images: Optional[List[str]] = None):
    """
    Test basic task execution workflow.

    Args:
        use_mock: If True, use mock VLM client. If False, use real API (requires API key)
        test_images: List of image paths to use for testing. If None, creates test images.
    """
    print("\n" + "=" * 80)
    print("  TEST: Basic Workflow")
    print("=" * 80)

    # Initialize task manager
    task_manager = ServerTaskManager()

    if use_mock:
        print("\n[Test] Using mock VLM client")
        mock_client = MockVLMClient()
        planner_vlm = MockPlannerVLM(mock_client)
        observer_vlm = MockObserverVLM(mock_client)
    else:
        print("\n[Test] Using real VLM client")
        os.environ.setdefault("OPENAI_API_KEY", "xx")
        os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")
        client = openai.OpenAI()

        base_client = BaseVLMClient(
            model="claude-sonnet-4-5-20250929",
            client=client,
        )
        planner_vlm = PlannerVLM(base_client=base_client)
        observer_vlm = ObserverVLM(base_client=base_client)

    planner = PlannerAgent(vlm=planner_vlm)
    observer = ObserverAgent(vlm=observer_vlm)
    task_manager.set_agents(planner, observer)

    # Create task
    print("\n[Test] Step 1: Creating task...")

    if test_images and len(test_images) > 0:
        initial_image = load_existing_image(test_images[0])
        if initial_image is None:
            print(f"⚠️ Failed to load initial image, creating test image")
            initial_image = create_test_image()
    else:
        initial_image = create_test_image()

    robot_input = RobotImageInput(
        waist_image=None,
        image=initial_image,
    )

    config = TaskConfig(
        observer_window_size=8,
    )

    state = task_manager.create_task(
        global_instruction="Pick up the object and place it in the box",
        initial_robot_input=robot_input,
        config=config,
    )

    print(f"✅ Task created: {state.task_id}")
    print(f"   State: {state.state}")
    print(f"   Plan: {state.plan_list[:100]}...")

    # Verify round logger was initialized
    assert state.round_logger is not None, "Round logger should be initialized"
    print(f"✅ Round logger initialized")

    # Check that first round was logged
    rounds_dir = os.path.join(state.logs_dir, "rounds")
    round_files = list(Path(rounds_dir).glob("round_*.json"))
    print(f"✅ Found {len(round_files)} round log file(s)")

    # Send multiple steps
    print("\n[Test] Step 2: Sending observation steps...")

    for i in range(3):
        print(f"\n--- Sending step {i+1} ---")

        if test_images and len(test_images) > i + 1:
            step_image = load_existing_image(test_images[i + 1])
            if step_image is None:
                step_image = create_test_image()
        else:
            step_image = create_test_image()

        robot_input = RobotImageInput(
            waist_image=None,
            image=step_image,
        )

        state = task_manager.add_step_and_maybe_refine_robot(
            task_id=state.task_id,
            robot_input=robot_input,
        )

        print(f"   State after step {i+1}: {state.state}")
        print(f"   Current subtask: {state.current_subtask_description}")

    # Verify round logs
    round_files = sorted(Path(rounds_dir).glob("round_*.json"))
    print(f"\n[Test] Verification: Found {len(round_files)} round log file(s)")

    for round_file in round_files:
        with open(round_file, 'r') as f:
            round_data = json.load(f)

        print(f"\n--- Round {round_data['round_number']} ---")
        print(f"   Observer interactions: {round_data['observer_count']}")
        print(f"   Has planner interaction: {round_data['planner_interaction'] is not None}")
        print(f"   Duration: {round_data.get('duration_seconds', 'N/A')} seconds")

    print("\n" + "=" * 80)
    print("✅ TEST PASSED: Basic Workflow")
    print("=" * 80)


def main():
    """Run all integration tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Integration test for server functionality")
    parser.add_argument(
        "--use-real-api",
        action="store_true",
        help="Use real VLM API instead of mock (requires API key)"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/Users/makabaka/code/mem_vla/_server_data/task_20260120_005357_2fc609ab/images",
        help="Directory containing test images"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["basic", "all"],
        default="all",
        help="Which test to run (default: all)"
    )

    args = parser.parse_args()

    use_mock = not args.use_real_api

    # Load test images if directory exists
    test_images = None
    if os.path.isdir(args.images_dir):
        image_files = sorted([
            os.path.join(args.images_dir, f)
            for f in os.listdir(args.images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if image_files:
            test_images = image_files
            print(f"\n[Setup] Found {len(test_images)} test images in {args.images_dir}")
        else:
            print(f"\n[Setup] No images found in {args.images_dir}, will create test images")
    else:
        print(f"\n[Setup] Images directory {args.images_dir} not found, will create test images")

    try:
        if args.test in ["basic", "all"]:
            test_basic_workflow(use_mock=use_mock, test_images=test_images)

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
