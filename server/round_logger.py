"""Round-based logger for tracking Observer and Planner interactions.

This module provides functionality to log all interactions within a "round":
- A round starts when we begin observing a subtask
- Multiple Observer calls may occur as new images arrive
- A round completes when Planner refines the plan (after Observer returns "done")

Each round is logged as a single JSON file containing:
- Round metadata (round number, timestamps)
- All Observer interactions (inputs, outputs, timestamps)
- The final Planner interaction (inputs, outputs, timestamp)
"""

from __future__ import annotations
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ObserverInteraction:
    """Record of a single Observer call."""
    timestamp: float
    image_paths: List[str]
    # plan_list: str
    subtask: str
    status: str  # "done" | "not_done"
    raw_output: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
            "image_paths": self.image_paths,
            # "plan_list": self.plan_list,
            "subtask": self.subtask,
            "status": self.status,
            "raw_output": self.raw_output,
        }


@dataclass
class PlannerInteraction:
    """Record of a single Planner call."""
    timestamp: float
    image_paths: List[str]
    user_instruction: str
    initial_plan_list: str
    result_plan_list: str
    result_summary: str
    raw_output: str
    memory_operations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
            "image_paths": self.image_paths,
            "user_instruction": self.user_instruction,
            "initial_plan_list": self.initial_plan_list,
            "result_plan_list": self.result_plan_list,
            "result_summary": self.result_summary,
            "raw_output": self.raw_output,
            "memory_operations": self.memory_operations,
        }


@dataclass
class RoundRecord:
    """Complete record of one execution round."""
    round_number: int
    start_timestamp: float
    end_timestamp: Optional[float] = None
    observer_interactions: List[ObserverInteraction] = field(default_factory=list)
    planner_interaction: Optional[PlannerInteraction] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "start_timestamp": self.start_timestamp,
            "start_timestamp_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_timestamp)),
            "end_timestamp": self.end_timestamp,
            "end_timestamp_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_timestamp)) if self.end_timestamp else None,
            "duration_seconds": (self.end_timestamp - self.start_timestamp) if self.end_timestamp else None,
            "observer_count": len(self.observer_interactions),
            "observer_interactions": [obs.to_dict() for obs in self.observer_interactions],
            "planner_interaction": self.planner_interaction.to_dict() if self.planner_interaction else None,
        }


class RoundLogger:
    """
    Logger for tracking execution rounds.

    A round consists of:
    1. Multiple Observer calls (as new images arrive)
    2. One Planner call (when Observer returns "done")

    Usage:
        logger = RoundLogger(logs_dir="/path/to/logs")
        logger.start_round()
        logger.add_observer_interaction(...)
        logger.add_observer_interaction(...)
        logger.add_planner_interaction(...)
        logger.end_round()  # Saves to disk
    """

    def __init__(self, logs_dir: str):
        """
        Initialize round logger.

        Args:
            logs_dir: Base directory for logs (will create "rounds" subdirectory)
        """
        self.logs_dir = logs_dir
        self.rounds_dir = os.path.join(logs_dir, "rounds")
        Path(self.rounds_dir).mkdir(parents=True, exist_ok=True)

        self.current_round: Optional[RoundRecord] = None
        self.round_counter: int = 0

    def start_round(self) -> int:
        """
        Start a new round.

        Returns:
            The round number
        """
        if self.current_round is not None:
            # Auto-save previous round if not ended
            self.end_round()

        self.round_counter += 1
        self.current_round = RoundRecord(
            round_number=self.round_counter,
            start_timestamp=time.time()
        )
        return self.round_counter

    def add_observer_interaction(
        self,
        image_paths: List[str],
        # plan_list: str,
        subtask: str,
        status: str,
        raw_output: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Add an Observer interaction to the current round.

        Args:
            image_paths: Image paths passed to Observer
            # plan_list: Plan list passed to Observer
            subtask: subtask passed to Observer
            status: Observer result ("done" or "not_done")
            raw_output: Raw XML output from Observer
            timestamp: Optional timestamp (defaults to current time)
        """
        if self.current_round is None:
            # Auto-start round if not started
            self.start_round()

        interaction = ObserverInteraction(
            timestamp=timestamp if timestamp is not None else time.time(),
            image_paths=image_paths,
            # plan_list=plan_list,
            subtask=subtask,
            status=status,
            raw_output=raw_output,
        )
        self.current_round.observer_interactions.append(interaction)

    def add_planner_interaction(
        self,
        image_paths: List[str],
        user_instruction: str,
        initial_plan_list: str,
        result_plan_list: str,
        result_summary: str,
        raw_output: str,
        memory_operations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add a Planner interaction to the current round.

        Args:
            image_paths: Image paths passed to Planner
            user_instruction: User instruction passed to Planner
            initial_plan_list: Plan list before refinement
            result_plan_list: Plan list after refinement
            result_summary: Summary from Planner
            raw_output: Raw XML output from Planner
            memory_operations: Memory operations performed
        """
        if self.current_round is None:
            # Auto-start round if not started
            self.start_round()

        interaction = PlannerInteraction(
            timestamp=time.time(),
            image_paths=image_paths,
            user_instruction=user_instruction,
            initial_plan_list=initial_plan_list,
            result_plan_list=result_plan_list,
            result_summary=result_summary,
            raw_output=raw_output,
            memory_operations=memory_operations or [],
        )
        self.current_round.planner_interaction = interaction

    def end_round(self) -> Optional[str]:
        """
        End the current round and save to disk.

        Returns:
            Path to the saved JSON file, or None if no round was active
        """
        if self.current_round is None:
            return None

        self.current_round.end_timestamp = time.time()

        # Save to disk
        filename = f"round_{self.current_round.round_number:03d}_{int(self.current_round.start_timestamp)}.json"
        filepath = os.path.join(self.rounds_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.current_round.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"[RoundLogger] Saved round {self.current_round.round_number} to {filepath}")

        self.current_round = None
        return filepath

    def get_current_round_number(self) -> Optional[int]:
        """Get the current round number, or None if no round is active."""
        return self.current_round.round_number if self.current_round else None
