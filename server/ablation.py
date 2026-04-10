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
    "hime": AblationSetting(
        profile="hime",
        prompt_name="task3",
        use_observer=True,
        use_memory=True,
        memory_op_policy="allow_all",
        memory_mode="mixed",
        planner_image_mode="segment",
    ),
    "hime_wo_sentry": AblationSetting(
        profile="hime_wo_sentry",
        prompt_name="task3_hime_wo_sentry",
        use_observer=False,
        use_memory=True,
        memory_op_policy="allow_all",
        memory_mode="mixed",
        planner_image_mode="recent_window",
    ),
    "transient_memory": AblationSetting(
        profile="transient_memory",
        prompt_name="task3_transient_memory",
        use_observer=True,
        use_memory=False,
        memory_op_policy="disable_all",
        memory_mode="mixed",
        planner_image_mode="latest_frame",
    ),
    "transient_memory_wo_sentry": AblationSetting(
        profile="transient_memory_wo_sentry",
        prompt_name="task3_transient_memory_wo_sentry",
        use_observer=False,
        use_memory=False,
        memory_op_policy="disable_all",
        memory_mode="mixed",
        planner_image_mode="latest_frame",
    ),
    "only_image": AblationSetting(
        profile="only_image",
        prompt_name="task3_only_image",
        use_observer=True,
        use_memory=True,
        memory_op_policy="allow_all",
        memory_mode="image_only",
        planner_image_mode="segment",
    ),
    "only_text": AblationSetting(
        profile="only_text",
        prompt_name="task3_only_text",
        use_observer=True,
        use_memory=True,
        memory_op_policy="allow_all",
        memory_mode="text_only",
        planner_image_mode="segment",
    ),
    "no_management": AblationSetting(
        profile="no_management",
        prompt_name="task3_no_management",
        use_observer=True,
        use_memory=True,
        memory_op_policy="query_create_only",
        memory_mode="mixed",
        planner_image_mode="segment",
    ),
    "FIFO": AblationSetting(
        profile="FIFO",
        prompt_name="task3_FIFO",
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
    requested_raw = os.environ.get("ABLATION_PROFILE", "hime").strip() or "hime"
    alias_map = {
        "baseline": "hime",
        "baseline_wo_observer": "hime_wo_sentry",
        "baseline_wo_memory": "transient_memory",
        "baseline_wo_memory_wo_observer": "transient_memory_wo_sentry",
        "no_text_memory": "only_image",
        "no_image_memory": "only_text",
        "no_delete_update": "no_management",
        "fifo": "FIFO",
    }
    requested = alias_map.get(requested_raw, requested_raw)
    cfg = ABLATION_PROFILE_SETTINGS.get(requested)
    if cfg is None:
        print(
            f"[Ablation] Unknown ABLATION_PROFILE='{requested}', fallback to 'hime'. "
            f"Available={list(AVAILABLE_ABLATION_PROFILES)}"
        )
        cfg = ABLATION_PROFILE_SETTINGS["hime"]
    return cfg
