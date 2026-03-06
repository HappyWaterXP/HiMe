import os
import unittest
from contextlib import contextmanager

from server.ablation import AVAILABLE_ABLATION_PROFILES, load_ablation_setting


@contextmanager
def _temp_env(name: str, value):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


class AblationProfileTest(unittest.TestCase):
    def test_default_is_baseline(self):
        with _temp_env("ABLATION_PROFILE", None):
            cfg = load_ablation_setting()
        self.assertEqual(cfg.profile, "baseline")

    def test_no_memory_modify_delete_profile(self):
        with _temp_env("ABLATION_PROFILE", "no_memory_modify_delete"):
            cfg = load_ablation_setting()
        self.assertEqual(cfg.prompt_name, "multitag_planner_query_create_only")
        self.assertEqual(cfg.memory_op_policy, "query_create_only")
        self.assertTrue(cfg.use_observer)
        self.assertTrue(cfg.use_memory)

    def test_no_reasoning_profile_removed(self):
        self.assertNotIn("no_reasoning", AVAILABLE_ABLATION_PROFILES)


if __name__ == "__main__":
    unittest.main()
