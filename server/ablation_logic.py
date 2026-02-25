"""
Ablation-aware task runtime schema and logic.

This module extends the original TaskManager to support different planner modes:
- "plan_list": classic multi-step plan list (Baseline, Ablation 1/2/3)
- "single_subtask": single-step subtask (Ablation 4)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from .schema import TaskRuntimeState as _TaskRuntimeState


# ============================================================
#  Ablation-Specific Schema Extensions
# ============================================================

@dataclass
class AblationConfig:
    """Configuration that determines which ablation is active."""
    planner_mode: str = "plan_list"  # "plan_list" | "single_subtask"
    use_observer: bool = True  # Whether to use observer for subtask completion detection
    use_memory: bool = True  # Whether to use external memory module
    # Other ablation flags could be added here (e.g., "no_image", "no_delete")


def get_ablation_config() -> AblationConfig:
    """
    Read ablation configuration from environment variables.

    Environment variables:
        ABLATION_STUDY:
            - "no_plan" -> single_subtask mode, NO observer, NO memory
                          (periodic trigger, direct VLM output)
            - "single_step" -> single_subtask mode with observer and memory
                              (backward compatibility)
            - "no_reasoning" -> plan_list mode, no summary/annotations
            - "no_memory" -> plan_list mode, no memory operations
            - "baseline" or others -> plan_list mode (full system)
    """
    ablation = os.environ.get("ABLATION_STUDY", "baseline")

    if ablation == "no_plan":
        # No plan: single subtask, no observer, no memory
        return AblationConfig(
            planner_mode="single_subtask",
            use_observer=True,
            use_memory=False
        )
    elif ablation == "single_step":
        # Single step: single subtask with observer and memory (backward compat)
        return AblationConfig(
            planner_mode="single_subtask",
            use_observer=True,
            use_memory=True
        )
    elif ablation == "no_memory":
        # No memory: plan list without memory
        return AblationConfig(
            planner_mode="plan_list",
            use_observer=True,
            use_memory=False
        )
    elif ablation == "no_obs":
        # No memory: plan list without memory
        return AblationConfig(
            planner_mode="plan_list",
            use_observer=False,
            use_memory=True
        )
    else:
        # Default: full system (baseline, no_reasoning, etc.)
        return AblationConfig(
            planner_mode="plan_list",
            use_observer=True,
            use_memory=True
        )



# ============================================================
#  Ablation-Aware Logic
# ============================================================

def should_task_be_done(state: _TaskRuntimeState, ablation: AblationConfig) -> bool:
    """
    Judge whether the task is done based on current ablation mode.

    For "plan_list" mode:
        - Use is_plan_done(plan_list) from src.extractor

    For "single_subtask" mode:
        - If state.is_task_complete is True (from <is_complete>yes tag) -> done
        - Otherwise -> not done
    """
    from extractor import is_plan_done

    if ablation.planner_mode == "single_subtask":
        # Ablation 4: use explicit completion flag
        return getattr(state, 'is_task_complete', False)
    else:
        # Baseline / Ablation 1/2/3: use plan list parsing
        return is_plan_done(state.plan_list)


def extract_current_subtask_description(state: _TaskRuntimeState, ablation: AblationConfig) -> Optional[str]:
    """
    Extract the description of the current subtask.

    For "plan_list" mode:
        - Use extract_current_subtask(plan_list) from src.extractor

    For "single_subtask" mode:
        - Use getattr(state, 'current_subtask', None) (instead of plan list)
    """
    from extractor import extract_current_subtask

    if ablation.planner_mode == "single_subtask":
        # Ablation 4: direct subtask string (no parsing needed)
        return getattr(state, 'current_subtask', None)
    else:
        # Baseline / Ablation 1/2/3: parse from plan_list
        return extract_current_subtask(state.plan_list)


def apply_planner_result_to_state(
    state: _TaskRuntimeState,
    planner_result,
    ablation: AblationConfig
) -> None:
    """
    Apply planner.run_refine() result to state, handling different ablation modes.

    For "plan_list" mode:
        - planner_result.plan_text -> state.plan_list
        - summary -> state.summary
        - is_done computed via is_plan_done(plan_list)

    For "single_subtask" mode:
        - Parse raw_xml to extract <subtask> and <is_complete>
        - subtask_text -> state.current_subtask
        - is_complete="yes" -> state.is_task_complete = True
        - summary -> state.summary
        - state.plan_list remains empty or placeholder
    """
    from extractor import _extract_single_tag, is_plan_done

    # Summary is universal
    state.summary = (planner_result.summary or "").strip()

    if ablation.planner_mode == "single_subtask":
        # Ablation 4: Parse subtask + completion flag from raw XML
        raw_xml = planner_result.raw_xml or ""

        subtask_text = _extract_single_tag(raw_xml, "subtask")
        is_complete_text = _extract_single_tag(raw_xml, "is_complete")

        if subtask_text:
            state.current_subtask = subtask_text.strip()

        if is_complete_text and is_complete_text.strip().lower() == "yes":
            state.is_task_complete = True
        else:
            state.is_task_complete = False

        # Plan list is not used in this mode
        # state.plan_list = ""  # optional: clear to avoid confusion

    else:
        # Baseline / Ablation 1/2/3: Use plan_list
        plan_text = (planner_result.plan_text or "").strip()
        if plan_text:
            state.plan_list = plan_text

        state.is_done = is_plan_done(state.plan_list)


# ============================================================
#  Schema Patches for Ablation 4
# ============================================================

# These fields should be added to TaskRuntimeState when Ablation 4 is used.
# Ideally TaskRuntimeState should be extended, but we provide them as optional
# attributes (monkey-patched at runtime or added via subclassing).

def ensure_ablation_fields(state: _TaskRuntimeState, ablation: AblationConfig) -> None:
    """
    Ensure that all required fields exist on the state object.

    For "single_subtask" mode:
        - state.current_subtask: str | None
        - state.is_task_complete: bool
    """
    if ablation.planner_mode == "single_subtask":
        if not hasattr(state, 'current_subtask'):
            state.current_subtask = None
        if not hasattr(state, 'is_task_complete'):
            state.is_task_complete = False