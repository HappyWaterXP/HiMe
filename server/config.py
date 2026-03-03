from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerModelConfig:
    api_key: str
    base_url: str
    planner_model: str
    observer_model: str


def load_server_model_config() -> ServerModelConfig:
    """
    Single source of truth for model endpoint/auth/model names used by server agents.

    Priority:
    - OPENAI_* controls auth + endpoint
    - VLM_MODEL controls both planner/observer by default
    - OBSERVER_VLM_MODEL optionally overrides observer model only
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip() or "xx"
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or "https://aigc.x-see.cn/v1"
    default_model = os.environ.get("VLM_MODEL", "").strip() or "qwen3-vl-30b-a3b-instruct"
    observer_model = os.environ.get("OBSERVER_VLM_MODEL", "").strip() or default_model

    return ServerModelConfig(
        api_key=api_key,
        base_url=base_url,
        planner_model=default_model,
        observer_model=observer_model,
    )
