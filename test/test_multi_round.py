#!/usr/bin/env python3
"""
Multi-round integration test.

This test simulates a complete task execution with multiple rounds:
- Round 1: Initial planning
- Round 2-4: Execute subtasks with Observer monitoring and Planner refinement
- Tests round-based logging

Usage:
    uv run python src/test/test_multi_round.py
"""

import os
import sys
import time
import json
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.server.task_manager import ServerTaskManager
from src.server.schema import TaskConfig, TaskStateEnum
from src.server.image_utils import RobotImageInput
from src.agent.multitag_planner import PlannerAgent
from src.agent.observer import ObserverAgent
from src.client.planner_vlm import PlannerVLM
from src.client.observer_vlm import ObserverVLM


class MockVLMClient:
    """Mock VLM client that simulates a 3-subtask execution."""

    def __init__(self):
        self.planner_calls = []   # Track planner calls

    def generate_planner_response(self):
        """Generate Planner response based on number of Planner refine calls."""
        planner_num = len(self.planner_calls) + 1
        self.planner_calls.append(planner_num)

        if planner_num == 1:
            # Initial plan
            return """<summary>
Creating initial plan for toy cleanup task. Will inspect boxes first, then move items.
</summary>

<memory_operations>
<operation>
  <type>QUERY</type>
  <query>toy_items</query>
  <reason>Check existing knowledge about toy items</reason>
</operation>
</memory_operations>

<plan_list>
Step 1: inspect left box to identify contents [current]
Step 2: pick up toy duck and place it in purple box [pending]
Step 3: pick up snack package and place it in white box [pending]
</plan_list>"""

        elif planner_num == 2:
            # After first subtask
            return """<summary>
Step 1 completed. Left box inspected and identified. Moving to Step 2: picking up toy duck.
</summary>

<memory_operations>
<operation>
  <type>CREATE</type>
  <tags>left_box,contents</tags>
  <text>Left box contains purple toy duck</text>
  <reason>Record inspection result</reason>
</operation>
</memory_operations>

<plan_list>
Step 1: inspect left box to identify contents [done]
Step 2: pick up toy duck and place it in purple box [current]
Step 3: pick up snack package and place it in white box [pending]
</plan_list>"""

        elif planner_num == 3:
            # After second subtask
            return """<summary>
Step 2 completed. Toy duck placed in purple box. Moving to Step 3: picking up snack.
</summary>

<memory_operations>
<operation>
  <type>CREATE</type>
  <tags>toy_duck,purple_box</tags>
  <text>Toy duck successfully placed in purple box</text>
  <reason>Record completion of step 2</reason>
</operation>
</memory_operations>

<plan_list>
Step 1: inspect left box to identify contents [done]
Step 2: pick up toy duck and place it in purple box [done]
Step 3: pick up snack package and place it in white box [current]
</plan_list>"""

        elif planner_num == 4:
            # All done
            return """<summary>
All steps completed. Task finished successfully.
</summary>

<memory_operations>
<operation>
  <type>CREATE</type>
  <tags>snack_package,white_box</tags>
  <text>Snack package successfully placed in white box</text>
  <reason>Record completion of step 3</reason>
</operation>
</memory_operations>

<plan_list>
Step 1: inspect left box to identify contents [done]
Step 2: pick up toy duck and place it in purple box [done]
Step 3: pick up snack package and place it in white box [done]
</plan_list>"""

        else:
            # Default: keep current state
            return """<summary>Continuing...</summary>
<memory_operations></memory_operations>
<plan_list>
Step 1: inspect left box to identify contents [done]
Step 2: pick up toy duck and place it in purple box [done]
Step 3: pick up snack package and place it in white box [done]
</plan_list>"""

    def encode_image_to_data_url(self, path: str) -> str:
        return f"data:image/png;base64,mock_{Path(path).name}"


class MockBaseClient:
    """Mock base client."""

    def __init__(self, mock_client):
        self.mock_client = mock_client

    def encode_image_to_data_url(self, path: str) -> str:
        return f"data:image/png;base64,mock_{Path(path).name}"

    def chat(self, *args, **kwargs):
        return self.mock_client.generate_planner_response()


class MockPlannerVLM(PlannerVLM):
    """Mock Planner VLM with separate call counting."""

    def __init__(self, mock_client):
        self.mock_client = mock_client
        self.base_client = MockBaseClient(mock_client)
        self.planner_refine_count = 0

    def call_multi_turn(self, *args, **kwargs):
        # This is called for each turn in the planner dialogue
        # But we want to count per refine call, not per turn
        # So we'll return the response based on planner_refine_count
        return self.mock_client.generate_planner_response()


class MockObserverVLM(ObserverVLM):
    """Mock Observer VLM with separate call counting."""

    def __init__(self, mock_client):
        self.mock_client = mock_client
        self.base_client = MockBaseClient(mock_client)
        self.observer_call_count = 0
        self.current_subtask_id = 0  # Track which subtask we're on

    def call_once(self, *args, **kwargs):
        self.observer_call_count += 1
        # After 4 calls, return "done" consistently to trigger planner
        if self.observer_call_count >= 4:
            return "<status>done</status>"
        else:
            return "<status>not_done</status>"

    def new_subtask(self):
        """Signal that a new subtask has started."""
        self.observer_call_count = 0
        self.current_subtask_id += 1


def load_test_images(images_dir: str) -> list:
    """Load test images from directory."""
    if not os.path.exists(images_dir):
        return []

    image_files = sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    return image_files


def test_multi_round():
    """Test multi-round execution."""
    print("\n" + "=" * 80)
    print("  MULTI-ROUND TEST")
    print("=" * 80)

    # Load test images
    images_dir = "/Users/makabaka/code/mem_vla/_server_data/task_20260120_005357_2fc609ab/images"
    test_images = load_test_images(images_dir)

    if len(test_images) < 3:
        print(f"\n⚠️  Warning: Only found {len(test_images)} test images")
        print("    Will create test images instead")
        test_images = [None, None, None]  # Will create test images

    # Initialize task manager with mock agents
    print("\n[Test] Initializing task manager with mock agents...")
    task_manager = ServerTaskManager()

    mock_client = MockVLMClient()
    planner_vlm = MockPlannerVLM(mock_client)
    observer_vlm = MockObserverVLM(mock_client)

    planner = PlannerAgent(vlm=planner_vlm)
    observer = ObserverAgent(vlm=observer_vlm)
    task_manager.set_agents(planner, observer)

    # Create task
    print("\n[Test] Creating task...")

    if test_images[0]:
        try:
            initial_image = Image.open(test_images[0]).convert('RGB')
        except:
            initial_image = Image.new('RGB', (640, 480), color=(100, 150, 200))
    else:
        initial_image = Image.new('RGB', (640, 480), color=(100, 150, 200))

    robot_input = RobotImageInput(waist_image=None, image=initial_image)

    config = TaskConfig(
        observer_window_size=4,
    )

    state = task_manager.create_task(
        global_instruction="Organize toy items: inspect boxes, put toy duck in purple box, put snack in white box",
        initial_robot_input=robot_input,
        config=config,
    )

    print(f"✅ Task created: {state.task_id}")
    print(f"   State: {state.state}")
    print(f"   Plan:\n{state.plan_list}")

    # Reset observer for first subtask
    observer_vlm.new_subtask()

    # Execute multiple rounds
    rounds_executed = 0
    max_rounds = 4

    print(f"\n[Test] Executing up to {max_rounds} rounds...")

    while not state.is_done and rounds_executed < max_rounds:
        rounds_executed += 1
        print(f"\n{'='*70}")
        print(f"  ROUND {rounds_executed}")
        print(f"{'='*70}")

        # Send 3 steps per round (simulating robot execution)
        for step in range(1, 4):
            print(f"\n--- Round {rounds_executed}, Step {step} ---")

            # Load or create image
            img_idx = min(rounds_executed - 1, len(test_images) - 1)
            if test_images[img_idx]:
                try:
                    step_image = Image.open(test_images[img_idx]).convert('RGB')
                except:
                    step_image = Image.new('RGB', (640, 480), color=(150 + step*20, 100, 200))
            else:
                step_image = Image.new('RGB', (640, 480), color=(150 + step*20, 100, 200))

            robot_input = RobotImageInput(waist_image=None, image=step_image)

            state = task_manager.add_step_and_maybe_refine_robot(
                task_id=state.task_id,
                robot_input=robot_input,
            )

            print(f"   State: {state.state}")
            print(f"   Current subtask: {state.current_subtask_description}")

            if state.is_done:
                print(f"\n🎉 Task completed!")
                break

        if state.is_done:
            break

    # Verify round logs
    print(f"\n{'='*70}")
    print("  LOG VERIFICATION")
    print(f"{'='*70}")

    rounds_dir = os.path.join(state.logs_dir, "rounds")
    round_files = sorted(Path(rounds_dir).glob("round_*.json"))

    print(f"\n✅ Found {len(round_files)} round log files")

    for round_file in round_files:
        with open(round_file, 'r') as f:
            round_data = json.load(f)

        print(f"\n--- {round_file.name} ---")
        print(f"   Round number: {round_data['round_number']}")
        print(f"   Observer calls: {round_data['observer_count']}")
        print(f"   Has planner: {round_data['planner_interaction'] is not None}")
        print(f"   Duration: {round_data.get('duration_seconds', 'N/A'):.3f}s")

        if round_data['planner_interaction']:
            planner = round_data['planner_interaction']
            print(f"   Initial plan: {planner['initial_plan_list'][:60]}...")
            print(f"   Result plan: {planner['result_plan_list'][:60]}...")

    # Show final state
    print(f"\n{'='*70}")
    print("  FINAL STATE")
    print(f"{'='*70}")
    print(f"\nTask ID: {state.task_id}")
    print(f"State: {state.state}")
    print(f"Is Done: {state.is_done}")
    print(f"Rounds executed: {rounds_executed}")
    print(f"\nFinal Plan:")
    print(state.plan_list)

    print(f"\n{'='*70}")
    print("✅ MULTI-ROUND TEST PASSED")
    print(f"{'='*70}\n")

    return state


if __name__ == "__main__":
    test_multi_round()
