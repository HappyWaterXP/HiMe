import os
import unittest

from server.ablation import AVAILABLE_ABLATION_PROFILES, load_ablation_setting


EXPECTED_PROFILES = (
    "baseline",
    "baseline_wo_observer",
    "baseline_wo_memory",
    "baseline_wo_memory_wo_observer",
    "no_text_memory",
    "no_image_memory",
    "no_delete_update",
    "fifo",
)


class AblationProfileMemoryModeTest(unittest.TestCase):
    def test_available_profiles_are_trimmed_to_expected_set(self):
        self.assertEqual(AVAILABLE_ABLATION_PROFILES, EXPECTED_PROFILES)

    def test_baseline_profile_uses_task3_v2(self):
        old = os.environ.get("ABLATION_PROFILE")
        try:
            os.environ["ABLATION_PROFILE"] = "baseline"
            cfg = load_ablation_setting()
            self.assertEqual(cfg.prompt_name, "task3_v2")
            self.assertEqual(cfg.planner_image_mode, "segment")
            self.assertTrue(cfg.use_observer)
            self.assertTrue(cfg.use_memory)
        finally:
            if old is None:
                os.environ.pop("ABLATION_PROFILE", None)
            else:
                os.environ["ABLATION_PROFILE"] = old

    def test_baseline_wo_observer_profile_disables_observer_only(self):
        old = os.environ.get("ABLATION_PROFILE")
        try:
            os.environ["ABLATION_PROFILE"] = "baseline_wo_observer"
            cfg = load_ablation_setting()
            self.assertEqual(cfg.prompt_name, "task3_no_observer")
            self.assertFalse(cfg.use_observer)
            self.assertTrue(cfg.use_memory)
            self.assertEqual(cfg.planner_image_mode, "recent_window")
        finally:
            if old is None:
                os.environ.pop("ABLATION_PROFILE", None)
            else:
                os.environ["ABLATION_PROFILE"] = old

    def test_baseline_wo_memory_profile_uses_latest_frame_and_no_memory_prompt(self):
        old = os.environ.get("ABLATION_PROFILE")
        try:
            os.environ["ABLATION_PROFILE"] = "baseline_wo_memory"
            cfg = load_ablation_setting()
            self.assertEqual(cfg.prompt_name, "task3_no_memory")
            self.assertTrue(cfg.use_observer)
            self.assertFalse(cfg.use_memory)
            self.assertEqual(cfg.memory_op_policy, "disable_all")
            self.assertEqual(cfg.planner_image_mode, "latest_frame")
        finally:
            if old is None:
                os.environ.pop("ABLATION_PROFILE", None)
            else:
                os.environ["ABLATION_PROFILE"] = old

    def test_baseline_wo_memory_wo_observer_profile_uses_direct_subtask_prompt(self):
        old = os.environ.get("ABLATION_PROFILE")
        try:
            os.environ["ABLATION_PROFILE"] = "baseline_wo_memory_wo_observer"
            cfg = load_ablation_setting()
            self.assertEqual(cfg.prompt_name, "task3_no_memory_no_observer")
            self.assertFalse(cfg.use_observer)
            self.assertFalse(cfg.use_memory)
            self.assertEqual(cfg.memory_op_policy, "disable_all")
            self.assertEqual(cfg.planner_image_mode, "latest_frame")
        finally:
            if old is None:
                os.environ.pop("ABLATION_PROFILE", None)
            else:
                os.environ["ABLATION_PROFILE"] = old

    def test_no_image_memory_profile_uses_text_only_mode(self):
        old = os.environ.get("ABLATION_PROFILE")
        try:
            os.environ["ABLATION_PROFILE"] = "no_image_memory"
            cfg = load_ablation_setting()
            self.assertEqual(cfg.prompt_name, "task3_no_image")
            self.assertEqual(cfg.memory_mode, "text_only")
            self.assertEqual(cfg.planner_image_mode, "segment")
        finally:
            if old is None:
                os.environ.pop("ABLATION_PROFILE", None)
            else:
                os.environ["ABLATION_PROFILE"] = old

    def test_no_text_memory_profile_uses_image_only_mode(self):
        old = os.environ.get("ABLATION_PROFILE")
        try:
            os.environ["ABLATION_PROFILE"] = "no_text_memory"
            cfg = load_ablation_setting()
            self.assertEqual(cfg.prompt_name, "task3_no_text")
            self.assertEqual(cfg.memory_mode, "image_only")
            self.assertEqual(cfg.planner_image_mode, "segment")
        finally:
            if old is None:
                os.environ.pop("ABLATION_PROFILE", None)
            else:
                os.environ["ABLATION_PROFILE"] = old

    def test_no_delete_update_profile_uses_v2_based_prompt_and_query_create_only(self):
        old = os.environ.get("ABLATION_PROFILE")
        try:
            os.environ["ABLATION_PROFILE"] = "no_delete_update"
            cfg = load_ablation_setting()
            self.assertEqual(cfg.prompt_name, "task3_no_delete_update")
            self.assertEqual(cfg.memory_op_policy, "query_create_only")
            self.assertIsNone(cfg.memory_max_records)
        finally:
            if old is None:
                os.environ.pop("ABLATION_PROFILE", None)
            else:
                os.environ["ABLATION_PROFILE"] = old

    def test_fifo_profile_uses_query_create_only_and_capacity(self):
        old = os.environ.get("ABLATION_PROFILE")
        try:
            os.environ["ABLATION_PROFILE"] = "fifo"
            cfg = load_ablation_setting()
            self.assertEqual(cfg.prompt_name, "task3_fifo")
            self.assertEqual(cfg.memory_op_policy, "query_create_only")
            self.assertEqual(cfg.memory_max_records, 20)
        finally:
            if old is None:
                os.environ.pop("ABLATION_PROFILE", None)
            else:
                os.environ["ABLATION_PROFILE"] = old


if __name__ == "__main__":
    unittest.main()
