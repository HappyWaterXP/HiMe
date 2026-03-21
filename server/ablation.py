from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class AblationSetting:
    profile: str
    prompt_name: str
    use_observer: bool
    use_memory: bool
    memory_op_policy: str
    memory_mode: str
    planner_image_mode: Literal["segment", "recent_window", "latest_frame"] = "segment"
    memory_max_records: Optional[int] = None


ABLATION_PROFILE_SETTINGS = {
    "baseline": AblationSetting(
        profile="baseline",
        prompt_name="task3_v2",
        use_observer=True,
        use_memory=True,
        memory_op_policy="allow_all",
        memory_mode="mixed",
        planner_image_mode="segment",
    ),
    "baseline_wo_observer": AblationSetting(
        profile="baseline_wo_observer",
        prompt_name="task3_no_observer",
        use_observer=False,
        use_memory=True,
        memory_op_policy="allow_all",
        memory_mode="mixed",
        planner_image_mode="recent_window",
    ),
    "baseline_wo_memory": AblationSetting(
        profile="baseline_wo_memory",
        prompt_name="task3_no_memory",
        use_observer=True,
        use_memory=False,
        memory_op_policy="disable_all",
        memory_mode="mixed",
        planner_image_mode="latest_frame",
    ),
    "baseline_wo_memory_wo_observer": AblationSetting(
        profile="baseline_wo_memory_wo_observer",
        prompt_name="task3_no_memory_no_observer",
        use_observer=False,
        use_memory=False,
        memory_op_policy="disable_all",
        memory_mode="mixed",
        planner_image_mode="latest_frame",
    ),
    "no_text_memory": AblationSetting(
        profile="no_text_memory",
        prompt_name="task3_no_text",
        use_observer=True,
        use_memory=True,
        memory_op_policy="allow_all",
        memory_mode="image_only",
        planner_image_mode="segment",
    ),
    "no_image_memory": AblationSetting(
        profile="no_image_memory",
        prompt_name="task3_no_image",
        use_observer=True,
        use_memory=True,
        memory_op_policy="allow_all",
        memory_mode="text_only",
        planner_image_mode="segment",
    ),
    "no_delete_update": AblationSetting(
        profile="no_delete_update",
        prompt_name="task3_no_delete_update",
        use_observer=True,
        use_memory=True,
        memory_op_policy="query_create_only",
        memory_mode="mixed",
        planner_image_mode="segment",
    ),
    "fifo": AblationSetting(
        profile="fifo",
        prompt_name="task3_fifo",
        use_observer=True,
        use_memory=True,
        memory_op_policy="query_create_only",
        memory_mode="mixed",
        planner_image_mode="segment",
        memory_max_records=20,
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
