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
    3) otherwise raise with a clear message
    """
    def env_or_py(env_name: str, py_val: str) -> str:
        env_val = os.environ.get(env_name, "").strip()
        if env_val:
            return env_val
        py_clean = (py_val or "").strip()
        if py_clean:
            return py_clean
        return ""

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

    planner_api_key = env_or_py("PLANNER_OPENAI_API_KEY", py_model_config.PLANNER_OPENAI_API_KEY)
    planner_base_url = env_or_py("PLANNER_OPENAI_BASE_URL", py_model_config.PLANNER_OPENAI_BASE_URL)
    planner_model = env_or_py("PLANNER_VLM_MODEL", py_model_config.PLANNER_VLM_MODEL)

    observer_api_key = env_or_py("OBSERVER_OPENAI_API_KEY", py_model_config.OBSERVER_OPENAI_API_KEY)
    observer_base_url = env_or_py("OBSERVER_OPENAI_BASE_URL", py_model_config.OBSERVER_OPENAI_BASE_URL)
    observer_model = env_or_py("OBSERVER_VLM_MODEL", py_model_config.OBSERVER_VLM_MODEL)
    embedding_api_key = env_or_py("EMBEDDING_OPENAI_API_KEY", py_model_config.EMBEDDING_OPENAI_API_KEY)
    embedding_base_url = env_or_py("EMBEDDING_OPENAI_BASE_URL", py_model_config.EMBEDDING_OPENAI_BASE_URL)
    embedding_model = env_or_py("EMBEDDING_MODEL", py_model_config.EMBEDDING_MODEL)
    embedding_dim = int_env_or_py("EMBEDDING_DIM", getattr(py_model_config, "EMBEDDING_DIM", 0), 0)

    missing = []
    if not planner_api_key:
        missing.append("PLANNER_OPENAI_API_KEY")
    if not planner_base_url:
        missing.append("PLANNER_OPENAI_BASE_URL")
    if not planner_model:
        missing.append("PLANNER_VLM_MODEL")
    if not observer_api_key:
        missing.append("OBSERVER_OPENAI_API_KEY")
    if not observer_base_url:
        missing.append("OBSERVER_OPENAI_BASE_URL")
    if not observer_model:
        missing.append("OBSERVER_VLM_MODEL")
    if not embedding_api_key:
        missing.append("EMBEDDING_OPENAI_API_KEY")
    if not embedding_base_url:
        missing.append("EMBEDDING_OPENAI_BASE_URL")
    if not embedding_model:
        missing.append("EMBEDDING_MODEL")
    if missing:
        raise ValueError(
            "Missing required model config values: "
            + ", ".join(missing)
            + ". Set them via environment variables or server/model_config.py."
        )

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
