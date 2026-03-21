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
        self.assertEqual(cfg.prompt_name, "task3_v2")

    def test_no_delete_update_profile(self):
        with _temp_env("ABLATION_PROFILE", "no_delete_update"):
            cfg = load_ablation_setting()
        self.assertEqual(cfg.prompt_name, "task3_no_delete_update")
        self.assertEqual(cfg.memory_op_policy, "query_create_only")
        self.assertTrue(cfg.use_observer)
        self.assertTrue(cfg.use_memory)

    def test_removed_profiles_are_not_available(self):
        self.assertNotIn("no_reasoning", AVAILABLE_ABLATION_PROFILES)
        self.assertNotIn("no_memory_modify_delete", AVAILABLE_ABLATION_PROFILES)
        self.assertNotIn("no_observer", AVAILABLE_ABLATION_PROFILES)


if __name__ == "__main__":
    unittest.main()
