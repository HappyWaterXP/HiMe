from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerModelConfig:
    planner_api_key: str
    planner_base_url: str
    planner_model: str
    observer_api_key: str
    observer_base_url: str
    observer_model: str


def load_server_model_config() -> ServerModelConfig:
    """
    Single source of truth for model endpoint/auth/model names used by server agents.

    Priority:
    - OPENAI_* controls shared auth + endpoint defaults
    - VLM_MODEL controls shared model default
    - PLANNER_* / OBSERVER_* can override API key, base URL, and model independently
    """
    shared_api_key = os.environ.get("OPENAI_API_KEY", "").strip() or "xx"
    shared_base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or "https://aigc.x-see.cn/v1"
    shared_model = os.environ.get("VLM_MODEL", "").strip() or "qwen3-vl-30b-a3b-instruct"

    planner_api_key = os.environ.get("PLANNER_OPENAI_API_KEY", "").strip() or shared_api_key
    planner_base_url = os.environ.get("PLANNER_OPENAI_BASE_URL", "").strip() or shared_base_url
    planner_model = os.environ.get("PLANNER_VLM_MODEL", "").strip() or shared_model

    observer_api_key = os.environ.get("OBSERVER_OPENAI_API_KEY", "").strip() or shared_api_key
    observer_base_url = os.environ.get("OBSERVER_OPENAI_BASE_URL", "").strip() or shared_base_url
    observer_model = os.environ.get("OBSERVER_VLM_MODEL", "").strip() or shared_model

    return ServerModelConfig(
        planner_api_key=planner_api_key,
        planner_base_url=planner_base_url,
        planner_model=planner_model,
        observer_api_key=observer_api_key,
        observer_base_url=observer_base_url,
        observer_model=observer_model,
    )
