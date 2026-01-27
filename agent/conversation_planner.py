# src/agent/conversation_planner.py
"""
Conversation-based Planner Agent that maintains multi-turn dialogue history.

This is the main experimental branch replacing external memory with conversation history.
All prompt variants (including memory-disabled ones) can be tested with this agent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
from datetime import datetime
import json

from src.prompt_loader import load_prompt
from src.client.planner_vlm import PlannerVLM
from src.extractor import parse_planner_output, MemoryOperation


@dataclass
class PlannerResult:
    """Result structure matching PlannerAgent's output for compatibility"""
    summary: str
    memory_operations: List[MemoryOperation]  # Will always be empty for this agent
    plan_text: str
    refined_plan_text: str
    raw_xml: str


class ConversationPlannerAgent:
    """
    Conversation-based Planner Agent - MAIN EXPERIMENTAL BRANCH

    Replaces external memory (MultiTagMemory) with conversation history.
    All ablation studies use this agent by selecting different prompts.

    Key differences from PlannerAgent:
    - No external memory module (memory operations handled within conversation)
    - Maintains conversation history up to max_history_rounds
    - Automatically truncates old messages when limit exceeded
    - Can use ANY planner prompt (memory-enabled or memory-disabled)

    Prompt compatibility:
    - multitag_planner.txt (memory-enabled, uses memory operations in conversation)
    - multitag_planner_no_image.txt (memory without image storage)
    - multitag_planner_no_delete.txt (memory without deletion)
    - multitag_planner_no_memory.txt (memory-disabled, no memory operations)
    - multitag_planner_single_subtask.txt (single-step prediction with memory)

    Configuration:
    - prompt_name: Selects which prompt variant to use
    - max_history_rounds: Controls conversation history retention
    """

    def __init__(
        self,
        vlm: PlannerVLM,
        prompt_name: str = "multitag_planner_no_memory",  # Default to memory-disabled
        max_history_rounds: int = 10,
    ) -> None:
        """
        Initialize ConversationPlannerAgent.

        Args:
            vlm: PlannerVLM client for making model calls
            prompt_name: Name of the prompt file to load (without .txt extension)
            max_history_rounds: Maximum number of user-assistant rounds to keep in history
                              Old rounds are dropped when exceeded (keeps system + latest N rounds)
        """
        self.vlm = vlm
        self.max_history_rounds = max_history_rounds

        # Load the specified prompt
        self.system_prompt: str = load_prompt(prompt_name)
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        self.plan_list: str = ""

    # ---------- State Management ----------

    def reset(self) -> None:
        """Completely reset conversation and internal state"""
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.plan_list = ""

    def set_plan(self, plan_text: str) -> None:
        """Manually set the current global plan"""
        self.plan_list = (plan_text or "").strip()

    # ---------- Conversation History Truncation ----------

    def _truncate_history(self) -> None:
        """
        Truncate conversation history to keep only the last max_history_rounds rounds.

        Message format:
        - messages[0]: system prompt
        - messages[1:]: alternating user/assistant pairs (each pair = 1 round)

        Truncation logic:
        - Always keep system prompt (messages[0])
        - Calculate total rounds: (len(messages) - 1) / 2
        - If rounds > max_history_rounds:
            - Drop oldest user/assistant pairs
            - Keep newest max_history_rounds pairs
        """
        total_messages = len(self.messages)
        if total_messages <= 1:
            return  # Only system prompt, nothing to truncate

        total_rounds = (total_messages - 1) // 2
        if total_rounds <= self.max_history_rounds:
            return  # Within limit, no truncation needed

        # Calculate how many rounds to drop
        rounds_to_keep = self.max_history_rounds
        messages_to_drop = (total_rounds - rounds_to_keep) * 2

        # New messages: system + last N rounds
        new_messages = [self.messages[0]] + self.messages[messages_to_drop + 1:]
        self.messages = new_messages

        print(f"[ConversationPlanner] Truncated history: {total_rounds} -> {rounds_to_keep} rounds")

    # ---------- Turn Logic ----------

    def _build_user_prompt(
        self,
        turn_num: int,
        instruction: str,
        plan_list: str,
        image_count: int
    ) -> str:
        """
        Build user prompt for a single turn.

        Args:
            turn_num: Current turn number
            instruction: User instruction for this turn
            plan_list: Current plan text
            image_count: Number of images provided
        """
        prompt_parts = [f"=== TURN {turn_num} ===\n"]

        prompt_parts.append("USER INSTRUCTION:")
        prompt_parts.append(instruction)

        prompt_parts.append("\nCURRENT PLAN:")
        if plan_list and plan_list.strip():
            prompt_parts.append(plan_list)
        else:
            prompt_parts.append("(No existing plan)")

        if image_count > 0:
            prompt_parts.append(f"\nINPUT IMAGES: {image_count} image(s) provided below, numbered 1 to {image_count}")
        else:
            prompt_parts.append("\nINPUT IMAGES: None")

        prompt_parts.append("\n📋 REQUIRED OUTPUT FORMAT:")
        prompt_parts.append("Respond with <summary> and <plan_list> sections.")

        return "\n".join(prompt_parts)

    def _single_turn(
        self,
        instruction: str,
        image_paths: List[str],
        current_plan_list: str,
        max_tokens: int = 512,
    ) -> PlannerResult:
        """
        Perform a single turn of conversation.

        Args:
            instruction: User instruction for this turn
            image_paths: List of image paths to include
            current_plan_list: Current plan text
            max_tokens: Max tokens for model response

        Returns:
            PlannerResult with parsed output
        """
        # Build text prompt
        user_text = self._build_user_prompt(
            turn_num=1,
            instruction=instruction,
            plan_list=current_plan_list,
            image_count=len(image_paths)
        )

        # Build content with images
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]

        # Add images if provided
        if image_paths:
            for p in image_paths:
                if not p:
                    continue
                img_block = self.vlm.base_client.encode_image_to_data_url(p)
                if img_block is not None:
                    # Store source path for export/debugging
                    if isinstance(img_block, dict):
                        img_block.setdefault("meta", {})
                        img_block["meta"]["source_path"] = p
                    content.append(img_block)

        # Call model
        self.messages.append({"role": "user", "content": content})
        raw_xml = self.vlm.chat(messages=self.messages, max_tokens=max_tokens)
        self.messages.append({"role": "assistant", "content": raw_xml})

        # Truncate history if needed
        self._truncate_history()

        # Parse output
        parsed = parse_planner_output(raw_xml)
        summary: str = parsed.get("summary", "") or ""
        mem_ops: List[MemoryOperation] = []  # Memory operations handled in conversation
        new_plan: str = parsed.get("plan_list", "") or ""

        # Update internal plan
        if new_plan.strip():
            self.plan_list = new_plan.strip()

        return PlannerResult(
            summary=summary,
            memory_operations=mem_ops,
            plan_text=new_plan,
            refined_plan_text=new_plan,
            raw_xml=raw_xml
        )

    # ---------- Main API ----------

    def run_refine(
        self,
        image_paths: List[str],
        initial_plan_list: Optional[str] = None,
        user_instruction: Optional[str] = None,
        max_tokens: int = 512,
        max_inner_rounds: int = 1,
        do_reset: bool = True,
        print_full_interactions_each_round: bool = True,
        log_interactions_json_dir: Optional[str] = None,
        use_cli_prompt_for_memory_view: bool = False,
        decide_view_memory: Optional[Callable[[int], bool]] = None,
        log_memory_json_dir: Optional[str] = None,
        drop_images_in_json: bool = True,
    ) -> PlannerResult:
        """
        Run a refinement session.

        Args:
            image_paths: List of image paths for context
            initial_plan_list: Optional starting plan text
            user_instruction: User's instruction/refinement request
            max_tokens: Max tokens for response
            max_inner_rounds: Number of inner rounds (default 1 for memory-disabled prompts)
            do_reset: Whether to reset conversation before processing
            print_full_interactions_each_round: Print conversation after processing
            log_interactions_json_dir: Optional directory to save conversation JSON
            use_cli_prompt_for_memory_view: Ignored (no external memory)
            decide_view_memory: Ignored (no external memory)
            log_memory_json_dir: Ignored (no external memory)
            drop_images_in_json: Whether to drop image data from JSON export

        Returns:
            PlannerResult with the refined plan
        """
        # Reset if requested
        if do_reset:
            self.reset()

        # Set initial plan if provided
        if initial_plan_list is not None:
            self.plan_list = initial_plan_list.strip()

        # Single turn for now (memory-free mode)
        turn_num = 1
        result = self._single_turn(
            instruction=user_instruction or "",
            image_paths=image_paths,
            current_plan_list=self.plan_list,
            max_tokens=max_tokens,
        )

        # Print conversation if requested
        if print_full_interactions_each_round:
            self.print_conversation()

        # Save conversation JSON if requested
        if log_interactions_json_dir:
            Path(log_interactions_json_dir).mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = str(Path(log_interactions_json_dir) / f"round_{turn_num}_{ts}.json")
            self.export_conversation_json(json_path, drop_images=drop_images_in_json)
            print(f"[LOG] Conversation saved to: {json_path}")

        return result

    # ---------- Conversation Export ----------

    def print_conversation(self) -> None:
        """Print full conversation history to console"""
        print("\n" + "=" * 100)
        print("  COMPLETE CONVERSATION HISTORY")
        print("=" * 100 + "\n")

        turn_counter = 0

        for i, msg in enumerate(self.messages):
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                print(f"[SYSTEM PROMPT]")
                print("-" * 100)
                if isinstance(content, str):
                    print(content[:500] + "..." if len(content) > 500 else content)
                else:
                    print(str(content))
                print("\n")
                continue

            if role == "user":
                turn_counter += 1
                print(f"\n{'=' * 100}")
                print(f"USER (Turn {turn_counter})")
                print("=" * 100)

                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            print(item.get("text", ""))
                        elif isinstance(item, dict) and item.get("type") == "image_url":
                            source_path = None
                            if isinstance(item.get("meta"), dict):
                                source_path = item["meta"].get("source_path")
                            if source_path:
                                print(f"\n[IMAGE: {Path(source_path).name}]")
                            else:
                                url_show = ""
                                if isinstance(item.get("image_url"), dict):
                                    url_show = item["image_url"].get("url", "")[:30]
                                print(f"\n[IMAGE: {url_show}...]")
                        else:
                            print(str(item))
                else:
                    print(content)

            elif role == "assistant":
                print(f"\n{'-' * 100}")
                print(f"ASSISTANT (Turn {turn_counter})")
                print("-" * 100)
                print(content)
                print()

    def export_conversation_json(
        self,
        json_path: str,
        drop_images: bool = True
    ) -> None:
        """
        Export conversation history to JSON file.

        Args:
            json_path: Output file path
            drop_images: If True, replace image data URLs with placeholders
        """
        export_messages = []

        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]

            if drop_images and role == "user" and isinstance(content, list):
                # Process content list, replacing images with placeholders
                processed_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        # Replace with placeholder
                        source_path = None
                        if isinstance(item.get("meta"), dict):
                            source_path = item["meta"].get("source_path")
                        if source_path:
                            processed_content.append({
                                "type": "image_url",
                                "placeholder": f"[IMAGE: {Path(source_path).name}]",
                                "source_path": source_path
                            })
                        else:
                            processed_content.append({
                                "type": "image_url",
                                "placeholder": "[IMAGE: data URL]"
                            })
                    else:
                        processed_content.append(item)
                export_messages.append({"role": role, "content": processed_content})
            else:
                export_messages.append({"role": role, "content": content})

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_messages, f, ensure_ascii=False, indent=2)