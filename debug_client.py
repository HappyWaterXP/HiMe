#!/usr/bin/env python3
"""
Debug Client for Task Approval

Usage:
    python debug_client.py --server http://localhost:8000 --task-id task_xxx

This client polls the server for pending approvals and provides an interactive
interface for users to review and approve agent outputs.
"""

import argparse
import time
import requests
import json
from typing import Optional, Dict, Any


class DebugClient:
    """Interactive debug client for approving agent outputs"""

    def __init__(self, server_url: str, task_id: str):
        self.server_url = server_url.rstrip("/")
        self.task_id = task_id
        self.session = requests.Session()

    def check_pending_approval(self) -> Optional[Dict[str, Any]]:
        """Check if there's a pending approval"""
        try:
            url = f"{self.server_url}/tasks/{self.task_id}/pending_approval"
            resp = self.session.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            if data.get("has_pending"):
                return data
            return None
        except Exception as e:
            print(f"[Error] Failed to check pending approval: {e}")
            return None

    def get_memory(self) -> Optional[Dict[str, Any]]:
        """Get current memory state"""
        try:
            url = f"{self.server_url}/tasks/{self.task_id}/memory"
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[Error] Failed to get memory: {e}")
            return None

    def get_conversation(self) -> Optional[Dict[str, Any]]:
        """Get planner conversation history"""
        try:
            url = f"{self.server_url}/tasks/{self.task_id}/planner_conversation"
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[Error] Failed to get conversation: {e}")
            return None

    def approve(self, modifications: Optional[Dict[str, Any]] = None) -> bool:
        """Submit approval"""
        try:
            url = f"{self.server_url}/tasks/{self.task_id}/approve"
            body = {"modifications": modifications}
            resp = self.session.post(url, json=body, timeout=10)
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"[Error] Failed to approve: {e}")
            return False

    def display_pending_approval(self, data: Dict[str, Any]):
        """Display pending approval information"""
        print("\n" + "=" * 70)
        print(f"🔔 {data['agent_type'].upper()} OUTPUT NEEDS APPROVAL")
        print("=" * 70)
        print(f"\nTimestamp: {time.ctime(data['timestamp'])}")
        print(f"\n--- Raw Output ---")
        print(data['raw_output'][:500])  # Limit output length
        if len(data['raw_output']) > 500:
            print("... (truncated)")
        print(f"\n--- Parsed Output ---")
        print(json.dumps(data['parsed_output'], indent=2))
        print("=" * 70)

    def interactive_approval(self, data: Dict[str, Any]):
        """Interactive approval interface"""
        while True:
            print("\nOptions:")
            print("  [a] Approve (use original output)")
            print("  [m] Modify and approve")
            print("  [v] View memory")
            print("  [c] View conversation history")
            print("  [r] Refresh (check again)")
            print("  [q] Quit")

            choice = input("\nYour choice: ").strip().lower()

            if choice == 'a':
                print("\n✅ Approving with original output...")
                if self.approve(None):
                    print("✅ Approved successfully!")
                    return
                else:
                    print("❌ Approval failed, please try again")

            elif choice == 'm':
                self.modify_and_approve(data)
                return

            elif choice == 'v':
                self.view_memory()

            elif choice == 'c':
                self.view_conversation()

            elif choice == 'r':
                print("\n🔄 Refreshing...")
                return

            elif choice == 'q':
                print("\n👋 Exiting without approval")
                exit(0)

            else:
                print("❌ Invalid choice, please try again")

    def modify_and_approve(self, data: Dict[str, Any]):
        """Modify output and approve"""
        agent_type = data['agent_type']
        parsed = data['parsed_output']

        print("\n--- Modify Output ---")

        if agent_type == "observer":
            print(f"Current status: {parsed.get('status')}")
            new_status = input("New status (done/not_done) [press Enter to keep]: ").strip()
            if new_status:
                modifications = {"status": new_status}
            else:
                modifications = None

        elif agent_type == "planner":
            print("Current plan:")
            print(parsed.get('plan_text', '')[:200])
            print("\nModification options:")
            print("  [1] Modify plan_text")
            print("  [2] Modify summary")
            print("  [3] Keep original")

            mod_choice = input("Your choice: ").strip()

            modifications = {}
            if mod_choice == '1':
                print("\nEnter new plan_text (multi-line, end with Ctrl+D or empty line):")
                lines = []
                try:
                    while True:
                        line = input()
                        if not line:
                            break
                        lines.append(line)
                except EOFError:
                    pass
                if lines:
                    modifications['plan_text'] = '\n'.join(lines)

            elif mod_choice == '2':
                new_summary = input("New summary: ").strip()
                if new_summary:
                    modifications['summary'] = new_summary

            if not modifications:
                modifications = None
        else:
            modifications = None

        print(f"\n✅ Approving with modifications: {modifications}")
        if self.approve(modifications):
            print("✅ Approved successfully!")
        else:
            print("❌ Approval failed")

    def view_memory(self):
        """View current memory state"""
        print("\n--- Memory State ---")
        memory = self.get_memory()
        if memory:
            print(f"Total records: {len(memory.get('all_records', []))}")
            print(f"Tags: {memory.get('all_tags', [])}")
            print("\nRecords:")
            for record in memory.get('all_records', [])[:5]:  # Show first 5
                print(f"  ID {record['id']}: {record['tags']}")
            if len(memory.get('all_records', [])) > 5:
                print(f"  ... and {len(memory['all_records']) - 5} more")
        print("---")

    def view_conversation(self):
        """View planner conversation history"""
        print("\n--- Conversation History ---")
        conv = self.get_conversation()
        if conv:
            messages = conv.get('messages', [])
            print(f"Total messages: {len(messages)}")
            for i, msg in enumerate(messages[-3:]):  # Show last 3
                role = msg.get('role', 'unknown')
                content = str(msg.get('content', ''))[:100]
                print(f"\n[{i}] {role}: {content}...")
        print("---")

    def run(self, poll_interval: float = 2.0):
        """Main loop: poll for pending approvals"""
        print(f"🚀 Debug Client started")
        print(f"   Server: {self.server_url}")
        print(f"   Task ID: {self.task_id}")
        print(f"   Poll interval: {poll_interval}s")
        print("\n⏳ Waiting for pending approvals...\n")

        try:
            while True:
                pending = self.check_pending_approval()
                if pending:
                    self.display_pending_approval(pending)
                    self.interactive_approval(pending)
                    print("\n⏳ Waiting for next approval...\n")
                else:
                    print(".", end="", flush=True)
                    time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\n\n👋 Debug client stopped by user")


def main():
    parser = argparse.ArgumentParser(description="Debug client for task approval")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--task-id", required=True, help="Task ID to monitor")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Poll interval in seconds")

    args = parser.parse_args()

    client = DebugClient(args.server, args.task_id)
    client.run(args.poll_interval)


if __name__ == "__main__":
    main()
