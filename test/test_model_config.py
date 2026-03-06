import os
import unittest
from contextlib import contextmanager

from server import model_config as py_model_config
from server.config import load_server_model_config


@contextmanager
def _temp_env(overrides):
    old = {}
    for k, v in overrides.items():
        old[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def _temp_py_config(overrides):
    old = {}
    for k, v in overrides.items():
        old[k] = getattr(py_model_config, k)
        setattr(py_model_config, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(py_model_config, k, v)


class ModelConfigPriorityTest(unittest.TestCase):
    def test_env_overrides_python_config(self):
        py_overrides = {
            "OPENAI_API_KEY": "py-shared-key",
            "OPENAI_BASE_URL": "https://py.shared/v1",
            "VLM_MODEL": "py-shared-model",
            "PLANNER_OPENAI_API_KEY": "py-planner-key",
            "PLANNER_OPENAI_BASE_URL": "https://py.planner/v1",
            "PLANNER_VLM_MODEL": "py-planner-model",
            "OBSERVER_OPENAI_API_KEY": "py-observer-key",
            "OBSERVER_OPENAI_BASE_URL": "https://py.observer/v1",
            "OBSERVER_VLM_MODEL": "py-observer-model",
            "EMBEDDING_OPENAI_API_KEY": "py-emb-key",
            "EMBEDDING_OPENAI_BASE_URL": "https://py.embedding/v1",
            "EMBEDDING_MODEL": "py-emb-model",
        }
        env_overrides = {
            "OPENAI_API_KEY": "env-shared-key",
            "OPENAI_BASE_URL": "https://env.shared/v1",
            "VLM_MODEL": "env-shared-model",
            "PLANNER_OPENAI_API_KEY": "env-planner-key",
            "PLANNER_OPENAI_BASE_URL": "https://env.planner/v1",
            "PLANNER_VLM_MODEL": "env-planner-model",
            "OBSERVER_OPENAI_API_KEY": "env-observer-key",
            "OBSERVER_OPENAI_BASE_URL": "https://env.observer/v1",
            "OBSERVER_VLM_MODEL": "env-observer-model",
            "EMBEDDING_OPENAI_API_KEY": "env-emb-key",
            "EMBEDDING_OPENAI_BASE_URL": "https://env.embedding/v1",
            "EMBEDDING_MODEL": "env-emb-model",
        }
        with _temp_py_config(py_overrides), _temp_env(env_overrides):
            cfg = load_server_model_config()
            self.assertEqual(cfg.planner_api_key, "env-planner-key")
            self.assertEqual(cfg.planner_base_url, "https://env.planner/v1")
            self.assertEqual(cfg.planner_model, "env-planner-model")
            self.assertEqual(cfg.observer_api_key, "env-observer-key")
            self.assertEqual(cfg.observer_base_url, "https://env.observer/v1")
            self.assertEqual(cfg.observer_model, "env-observer-model")
            self.assertEqual(cfg.embedding_api_key, "env-emb-key")
            self.assertEqual(cfg.embedding_base_url, "https://env.embedding/v1")
            self.assertEqual(cfg.embedding_model, "env-emb-model")

    def test_python_config_fallback_when_env_empty(self):
        py_overrides = {
            "OPENAI_API_KEY": "py-shared-key",
            "OPENAI_BASE_URL": "https://py.shared/v1",
            "VLM_MODEL": "py-shared-model",
            "PLANNER_OPENAI_API_KEY": "",
            "PLANNER_OPENAI_BASE_URL": "",
            "PLANNER_VLM_MODEL": "",
            "OBSERVER_OPENAI_API_KEY": "py-observer-key",
            "OBSERVER_OPENAI_BASE_URL": "",
            "OBSERVER_VLM_MODEL": "",
            "EMBEDDING_OPENAI_API_KEY": "",
            "EMBEDDING_OPENAI_BASE_URL": "https://py.embedding/v1",
            "EMBEDDING_MODEL": "text-embedding-3-large",
        }
        env_clear = {
            "OPENAI_API_KEY": None,
            "OPENAI_BASE_URL": None,
            "VLM_MODEL": None,
            "PLANNER_OPENAI_API_KEY": None,
            "PLANNER_OPENAI_BASE_URL": None,
            "PLANNER_VLM_MODEL": None,
            "OBSERVER_OPENAI_API_KEY": None,
            "OBSERVER_OPENAI_BASE_URL": None,
            "OBSERVER_VLM_MODEL": None,
            "EMBEDDING_OPENAI_API_KEY": None,
            "EMBEDDING_OPENAI_BASE_URL": None,
            "EMBEDDING_MODEL": None,
        }
        with _temp_py_config(py_overrides), _temp_env(env_clear):
            cfg = load_server_model_config()
            self.assertEqual(cfg.planner_api_key, "py-shared-key")
            self.assertEqual(cfg.planner_base_url, "https://py.shared/v1")
            self.assertEqual(cfg.planner_model, "py-shared-model")
            self.assertEqual(cfg.observer_api_key, "py-observer-key")
            self.assertEqual(cfg.observer_base_url, "https://py.shared/v1")
            self.assertEqual(cfg.observer_model, "py-shared-model")
            self.assertEqual(cfg.embedding_api_key, "py-shared-key")
            self.assertEqual(cfg.embedding_base_url, "https://py.embedding/v1")
            self.assertEqual(cfg.embedding_model, "text-embedding-3-large")


if __name__ == "__main__":
    unittest.main()
