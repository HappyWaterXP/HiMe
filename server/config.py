from __future__ import annotations

import os
from dataclasses import dataclass

from . import model_config as py_model_config


@dataclass(frozen=True)
class ServerModelConfig:
    planner_api_key: str
    planner_base_url: str
    planner_model: str
    observer_api_key: str
    observer_base_url: str
    observer_model: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    embedding_dim: int


def load_server_model_config() -> ServerModelConfig:
    """
    Single source of truth for model endpoint/auth/model names used by server agents.

    Priority:
    1) Environment variables (if set and non-empty)
    2) server/model_config.py
    3) hardcoded safe defaults
    """
    def env_or_py(env_name: str, py_val: str, default: str) -> str:
        env_val = os.environ.get(env_name, "").strip()
        if env_val:
            return env_val
        py_clean = (py_val or "").strip()
        if py_clean:
            return py_clean
        return default

    def int_env_or_py(env_name: str, py_val: object, default: int) -> int:
        env_val = os.environ.get(env_name, "").strip()
        if env_val:
            try:
                return int(env_val)
            except Exception:
                pass
        try:
            if py_val is not None and str(py_val).strip() != "":
                return int(py_val)
        except Exception:
            pass
        return int(default)

    shared_api_key = env_or_py("OPENAI_API_KEY", py_model_config.OPENAI_API_KEY, "xx")
    shared_base_url = env_or_py(
        "OPENAI_BASE_URL", py_model_config.OPENAI_BASE_URL, "https://aigc.x-see.cn/v1"
    )
    shared_model = env_or_py("VLM_MODEL", py_model_config.VLM_MODEL, "qwen3-vl-30b-a3b-instruct")

    planner_api_key = env_or_py(
        "PLANNER_OPENAI_API_KEY", py_model_config.PLANNER_OPENAI_API_KEY, shared_api_key
    )
    planner_base_url = env_or_py(
        "PLANNER_OPENAI_BASE_URL", py_model_config.PLANNER_OPENAI_BASE_URL, shared_base_url
    )
    planner_model = env_or_py("PLANNER_VLM_MODEL", py_model_config.PLANNER_VLM_MODEL, shared_model)

    observer_api_key = env_or_py(
        "OBSERVER_OPENAI_API_KEY", py_model_config.OBSERVER_OPENAI_API_KEY, shared_api_key
    )
    observer_base_url = env_or_py(
        "OBSERVER_OPENAI_BASE_URL", py_model_config.OBSERVER_OPENAI_BASE_URL, shared_base_url
    )
    observer_model = env_or_py(
        "OBSERVER_VLM_MODEL", py_model_config.OBSERVER_VLM_MODEL, shared_model
    )
    embedding_api_key = env_or_py(
        "EMBEDDING_OPENAI_API_KEY", py_model_config.EMBEDDING_OPENAI_API_KEY, shared_api_key
    )
    embedding_base_url = env_or_py(
        "EMBEDDING_OPENAI_BASE_URL", py_model_config.EMBEDDING_OPENAI_BASE_URL, shared_base_url
    )
    embedding_model = env_or_py(
        "EMBEDDING_MODEL", py_model_config.EMBEDDING_MODEL, "text-embedding-3-large"
    )
    embedding_dim = int_env_or_py("EMBEDDING_DIM", getattr(py_model_config, "EMBEDDING_DIM", 0), 0)

    return ServerModelConfig(
        planner_api_key=planner_api_key,
        planner_base_url=planner_base_url,
        planner_model=planner_model,
        observer_api_key=observer_api_key,
        observer_base_url=observer_base_url,
        observer_model=observer_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
