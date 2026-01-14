#!/usr/bin/env python3
"""
Interactive test client for the task server.

This script provides an interactive CLI to test all server endpoints:
- POST /tasks: Create a new task with initial images
- POST /tasks/{task_id}/step: Upload observation images for each step
- POST /tasks/{task_id}/user_instruction: Send user refinement instructions

Usage:
    1. Start the FastAPI server: uvicorn server.app:app --reload
    2. Run this script: python -m src.test.test_server
    3. Follow the interactive prompts to test the workflow
"""

import requests
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import io


class ServerTestClient:
    """Interactive test client for task server."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.task_id: Optional[str] = None
        self.step_count: int = 0

    def _print_separator(self, title: str = ""):
        """Print a visual separator."""
        print("\n" + "=" * 80)
        if title:
            print(f"  {title}")
            print("=" * 80)
        print()

    def _print_response(self, response: Dict[str, Any]):
        """Pretty print server response."""
        print("📥 Server Response:")
        print("-" * 80)
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print("-" * 80)

    def _load_image(self, image_path: str) -> bytes:
        """Load image and convert to PNG bytes."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def create_task(
        self,
        global_instruction: str,
        initial_image_path: str,
        initial_waist_image_path: Optional[str] = None,
        observer_window_size: int = 8,
        human_intervene: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a new task.

        Args:
            global_instruction: High-level task description
            initial_image_path: Path to initial main camera image
            initial_waist_image_path: Path to initial waist camera image (optional)
            observer_window_size: Observer window size
            human_intervene: Whether to enable human intervention

        Returns:
            Server response dict
        """
        self._print_separator("CREATING NEW TASK")

        print(f"📝 Global Instruction: {global_instruction}")
        print(f"🖼️  Main Image: {initial_image_path}")
        if initial_waist_image_path:
            print(f"🖼️  Waist Image: {initial_waist_image_path}")
        print(f"⚙️  Observer Window Size: {observer_window_size}")
        print(f"⚙️  Human Intervene: {human_intervene}")

        # Prepare files
        main_bytes = self._load_image(initial_image_path)
        files = {
            "initial_image": ("initial_image.png", io.BytesIO(main_bytes), "image/png")
        }

        if initial_waist_image_path:
            waist_bytes = self._load_image(initial_waist_image_path)
            files["initial_waist_image"] = (
                "initial_waist_image.png",
                io.BytesIO(waist_bytes),
                "image/png"
            )

        data = {
            "global_instruction": global_instruction,
            "observer_window_size": str(observer_window_size),
            "human_intervene_for_planner": "true" if human_intervene else "false",
        }

        print("\n🚀 Sending request to server...")
        url = f"{self.base_url}/tasks"

        try:
            resp = requests.post(url, files=files, data=data, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()

            self.task_id = result.get("task_id")
            self.step_count = 0

            self._print_response(result)

            print(f"\n✅ Task created successfully!")
            print(f"   Task ID: {self.task_id}")
            print(f"   State: {result.get('state')}")
            print(f"   Current Subtask: {result.get('current_subtask_description')}")

            return result

        except requests.exceptions.RequestException as e:
            print(f"\n❌ Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Status Code: {e.response.status_code}")
                print(f"   Response: {e.response.text}")
            raise

    def send_step(
        self,
        image_path: str,
        waist_image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send observation step.

        Args:
            image_path: Path to main camera image
            waist_image_path: Path to waist camera image (optional)

        Returns:
            Server response dict
        """
        if not self.task_id:
            raise ValueError("No task created yet. Call create_task() first.")

        self.step_count += 1
        self._print_separator(f"STEP {self.step_count}")

        print(f"🖼️  Main Image: {image_path}")
        if waist_image_path:
            print(f"🖼️  Waist Image: {waist_image_path}")

        # Prepare files
        main_bytes = self._load_image(image_path)
        files = {
            "image": ("step_image.png", io.BytesIO(main_bytes), "image/png")
        }

        if waist_image_path:
            waist_bytes = self._load_image(waist_image_path)
            files["waist_image"] = (
                "waist_step_image.png",
                io.BytesIO(waist_bytes),
                "image/png"
            )

        print("\n🚀 Sending step to server...")
        url = f"{self.base_url}/tasks/{self.task_id}/step"

        try:
            resp = requests.post(url, files=files, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()

            self._print_response(result)

            print(f"\n✅ Step {self.step_count} processed!")
            print(f"   State: {result.get('state')}")
            print(f"   Is Done: {result.get('is_done')}")
            print(f"   Current Subtask: {result.get('current_subtask_description')}")

            return result

        except requests.exceptions.RequestException as e:
            print(f"\n❌ Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Status Code: {e.response.status_code}")
                print(f"   Response: {e.response.text}")
            raise

    def send_user_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Send user instruction to refine plan.

        Args:
            instruction: User refinement instruction

        Returns:
            Server response dict
        """
        if not self.task_id:
            raise ValueError("No task created yet. Call create_task() first.")

        self._print_separator("USER INSTRUCTION")

        print(f"💬 Instruction: {instruction}")

        body = {"user_new_instruction": instruction}

        print("\n🚀 Sending instruction to server...")
        url = f"{self.base_url}/tasks/{self.task_id}/user_instruction"

        try:
            resp = requests.post(url, json=body, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()

            self._print_response(result)

            print(f"\n✅ Instruction processed!")
            print(f"   State: {result.get('state')}")
            print(f"   Current Subtask: {result.get('current_subtask_description')}")

            return result

        except requests.exceptions.RequestException as e:
            print(f"\n❌ Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Status Code: {e.response.status_code}")
                print(f"   Response: {e.response.text}")
            raise


def interactive_mode(client: ServerTestClient):
    """Interactive CLI for testing the server."""

    print("\n" + "=" * 80)
    print("  🤖 TASK SERVER INTERACTIVE TEST CLIENT")
    print("=" * 80)
    print("\nCommands:")
    print("  create  - Create a new task")
    print("  step    - Send an observation step")
    print("  user    - Send a user instruction")
    print("  status  - Show current task status")
    print("  help    - Show this help message")
    print("  quit    - Exit the test client")
    print("\n" + "=" * 80 + "\n")

    while True:
        try:
            cmd = input("\n💻 Command> ").strip().lower()

            if not cmd:
                continue

            if cmd in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            elif cmd == "help":
                print("\nAvailable commands:")
                print("  create  - Create a new task with initial images")
                print("  step    - Send observation step with images")
                print("  user    - Send user refinement instruction")
                print("  status  - Show current task ID and step count")
                print("  quit    - Exit the test client")

            elif cmd == "status":
                if client.task_id:
                    print(f"\n📊 Current Status:")
                    print(f"   Task ID: {client.task_id}")
                    print(f"   Steps Sent: {client.step_count}")
                else:
                    print("\n⚠️  No task created yet.")

            elif cmd == "create":
                print("\n--- Create New Task ---")

                instruction = input("Global Instruction: ").strip()
                if not instruction:
                    print("❌ Instruction cannot be empty.")
                    continue

                main_img = input("Main Camera Image Path: ").strip()
                if not main_img:
                    print("❌ Main image path cannot be empty.")
                    continue

                waist_img = input("Waist Camera Image Path (optional, press Enter to skip): ").strip()
                waist_img = waist_img if waist_img else None

                window_size = input("Observer Window Size (default: 8): ").strip()
                window_size = int(window_size) if window_size else 8

                human_intervene = input("Human Intervene (y/n, default: n): ").strip().lower()
                human_intervene = human_intervene == "y"

                try:
                    client.create_task(
                        global_instruction=instruction,
                        initial_image_path=main_img,
                        initial_waist_image_path=waist_img,
                        observer_window_size=window_size,
                        human_intervene=human_intervene,
                    )
                except Exception as e:
                    print(f"\n❌ Error: {e}")

            elif cmd == "step":
                if not client.task_id:
                    print("\n⚠️  No task created yet. Use 'create' command first.")
                    continue

                print(f"\n--- Send Step {client.step_count + 1} ---")

                main_img = input("Main Camera Image Path: ").strip()
                if not main_img:
                    print("❌ Main image path cannot be empty.")
                    continue

                waist_img = input("Waist Camera Image Path (optional, press Enter to skip): ").strip()
                waist_img = waist_img if waist_img else None

                try:
                    result = client.send_step(
                        image_path=main_img,
                        waist_image_path=waist_img,
                    )

                    if result.get("is_done"):
                        print("\n🎉 Task completed! You can create a new task or exit.")

                except Exception as e:
                    print(f"\n❌ Error: {e}")

            elif cmd == "user":
                if not client.task_id:
                    print("\n⚠️  No task created yet. Use 'create' command first.")
                    continue

                print("\n--- Send User Instruction ---")

                instruction = input("User Instruction: ").strip()
                if not instruction:
                    print("❌ Instruction cannot be empty.")
                    continue

                try:
                    client.send_user_instruction(instruction)
                except Exception as e:
                    print(f"\n❌ Error: {e}")

            else:
                print(f"\n❌ Unknown command: {cmd}")
                print("   Type 'help' to see available commands.")

        except EOFError:
            print("\n\n👋 EOF detected, exiting...")
            break

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user, exiting...")
            break


def batch_test_mode(client: ServerTestClient, args):
    """Run a predefined test sequence."""

    print("\n" + "=" * 80)
    print("  🤖 BATCH TEST MODE")
    print("=" * 80 + "\n")

    # Step 1: Create task
    print("Step 1: Creating task...")
    try:
        client.create_task(
            global_instruction=args.instruction,
            initial_image_path=args.initial_main,
            initial_waist_image_path=args.initial_waist,
            observer_window_size=args.window_size,
            human_intervene=args.human_intervene,
        )
    except Exception as e:
        print(f"❌ Failed to create task: {e}")
        return

    # Step 2: Send observation steps
    if args.step_images:
        step_pairs = []
        images = args.step_images.split(",")

        for i in range(0, len(images), 2):
            main = images[i].strip()
            waist = images[i + 1].strip() if i + 1 < len(images) else None
            step_pairs.append((main, waist))

        for idx, (main, waist) in enumerate(step_pairs, 1):
            print(f"\nStep 2.{idx}: Sending observation step...")
            try:
                result = client.send_step(
                    image_path=main,
                    waist_image_path=waist,
                )

                if result.get("is_done"):
                    print("\n🎉 Task completed!")
                    break

                input("\nPress Enter to continue to next step...")

            except Exception as e:
                print(f"❌ Failed to send step: {e}")
                break

    # Step 3: Send user instruction (optional)
    if args.user_instruction:
        print("\nStep 3: Sending user instruction...")
        try:
            client.send_user_instruction(args.user_instruction)
        except Exception as e:
            print(f"❌ Failed to send user instruction: {e}")

    print("\n" + "=" * 80)
    print("  BATCH TEST COMPLETED")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive test client for task server"
    )
    parser.add_argument(
        "--base_url",
        default="http://localhost:8000",
        help="Task server base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)"
    )

    # Batch mode arguments
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode with predefined test sequence"
    )
    parser.add_argument(
        "--instruction",
        help="Global instruction for batch mode"
    )
    parser.add_argument(
        "--initial_main",
        help="Initial main camera image path for batch mode"
    )
    parser.add_argument(
        "--initial_waist",
        help="Initial waist camera image path for batch mode"
    )
    parser.add_argument(
        "--step_images",
        help="Comma-separated list of step images (main1,waist1,main2,waist2,...)"
    )
    parser.add_argument(
        "--user_instruction",
        help="User refinement instruction for batch mode"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=8,
        help="Observer window size (default: 8)"
    )
    parser.add_argument(
        "--human_intervene",
        action="store_true",
        help="Enable human intervention for planner"
    )

    args = parser.parse_args()

    # Create client
    client = ServerTestClient(base_url=args.base_url, timeout=args.timeout)

    # Test server connection
    try:
        resp = requests.get(f"{args.base_url}/docs", timeout=5)
        print(f"✅ Server is running at {args.base_url}")
    except requests.exceptions.RequestException:
        print(f"⚠️  Warning: Could not connect to server at {args.base_url}")
        print(f"   Make sure the server is running with: uvicorn server.app:app --reload")
        choice = input("\nContinue anyway? (y/n): ").strip().lower()
        if choice != "y":
            return

    # Run in appropriate mode
    if args.batch:
        if not args.instruction or not args.initial_main:
            print("❌ Batch mode requires --instruction and --initial_main arguments")
            return
        batch_test_mode(client, args)
    else:
        interactive_mode(client)


if __name__ == "__main__":
    main()
