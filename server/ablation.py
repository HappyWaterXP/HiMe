from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AblationSetting:
    profile: str
    prompt_name: str
    use_observer: bool
    use_memory: bool
    memory_op_policy: str


ABLATION_PROFILE_SETTINGS = {
    "baseline": AblationSetting(
        profile="baseline",
        prompt_name="multitag_planner",
        use_observer=True,
        use_memory=True,
        memory_op_policy="allow_all",
    ),
    "no_observer": AblationSetting(
        profile="no_observer",
        prompt_name="multitag_planner",
        use_observer=False,
        use_memory=True,
        memory_op_policy="allow_all",
    ),
    "no_memory_interaction": AblationSetting(
        profile="no_memory_interaction",
        prompt_name="multitag_planner_no_memory",
        use_observer=True,
        use_memory=False,
        memory_op_policy="disable_all",
    ),
    "no_memory_modify_delete": AblationSetting(
        profile="no_memory_modify_delete",
        prompt_name="multitag_planner_query_create_only",
        use_observer=True,
        use_memory=True,
        memory_op_policy="query_create_only",
    ),
    "no_delete": AblationSetting(
        profile="no_delete",
        prompt_name="multitag_planner_no_delete",
        use_observer=True,
        use_memory=True,
        memory_op_policy="allow_all",
    ),
    "no_text_memory": AblationSetting(
        profile="no_text_memory",
        prompt_name="multitag_planner_no_text",
        use_observer=True,
        use_memory=True,
        memory_op_policy="allow_all",
    ),
    "no_image_memory": AblationSetting(
        profile="no_image_memory",
        prompt_name="multitag_planner_no_image",
        use_observer=True,
        use_memory=True,
        memory_op_policy="allow_all",
    ),
    "single_subtask_mode": AblationSetting(
        profile="single_subtask_mode",
        prompt_name="multitag_planner_single_subtask",
        use_observer=True,
        use_memory=True,
        memory_op_policy="allow_all",
    ),
    "no_plan_no_memory": AblationSetting(
        profile="no_plan_no_memory",
        prompt_name="multitag_planner_no_plan_no_memory",
        use_observer=False,
        use_memory=False,
        memory_op_policy="disable_all",
    ),
}

AVAILABLE_ABLATION_PROFILES = tuple(ABLATION_PROFILE_SETTINGS.keys())


def load_ablation_setting() -> AblationSetting:
    requested = os.environ.get("ABLATION_PROFILE", "baseline").strip() or "baseline"
    cfg = ABLATION_PROFILE_SETTINGS.get(requested)
    if cfg is None:
        print(
            f"[Ablation] Unknown ABLATION_PROFILE='{requested}', fallback to 'baseline'. "
            f"Available={list(AVAILABLE_ABLATION_PROFILES)}"
        )
        cfg = ABLATION_PROFILE_SETTINGS["baseline"]
    return cfg
