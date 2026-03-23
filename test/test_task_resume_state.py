import json
import os
import tempfile
import unittest

from server.round_logger import RoundLogger
from server.schema import TaskConfig, TaskStateEnum, TaskRuntimeState
from server.task_state import save_task_state_json, load_task_state_json


class TaskResumeStateTest(unittest.TestCase):
    def test_save_and_load_task_state_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = os.path.join(tmp, "logs")
            images_dir = os.path.join(tmp, "images")
            os.makedirs(logs_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)

            state = TaskRuntimeState(
                task_id="task_x",
                global_instruction="do something",
                created_ts=123.0,
                base_dir=tmp,
                images_dir=images_dir,
                logs_dir=logs_dir,
                plan_list="[done] a\n[current] b",
                summary="summary",
                is_done=False,
                runtime_state=TaskStateEnum.OBSERVING,
                current_subtask_description="b",
                current_subtask_start_idx=7,
                image_paths=[os.path.join(images_dir, "1.png"), os.path.join(images_dir, "2.png")],
                config=TaskConfig(
                    observer_window_size=6,
                    use_observer=False,
                    use_memory=True,
                    planner_execution_mode="async",
                    planner_image_mode="recent_window",
                ),
                extra={"planner_status": "idle", "x": 1},
            )

            snapshot_path = os.path.join(logs_dir, "task_state", "latest_task_state.json")
            save_task_state_json(state, snapshot_path)
            restored = load_task_state_json(snapshot_path)

            self.assertEqual(restored.task_id, state.task_id)
            self.assertEqual(restored.global_instruction, state.global_instruction)
            self.assertEqual(restored.plan_list, state.plan_list)
            self.assertEqual(restored.summary, state.summary)
            self.assertEqual(restored.current_subtask_description, state.current_subtask_description)
            self.assertEqual(restored.current_subtask_start_idx, state.current_subtask_start_idx)
            self.assertEqual(restored.image_paths, state.image_paths)
            self.assertEqual(restored.runtime_state, TaskStateEnum.OBSERVING)
            self.assertEqual(restored.config.planner_image_mode, "recent_window")
            self.assertEqual(restored.extra["x"], 1)

    def test_round_logger_respects_existing_round_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            rounds_dir = os.path.join(tmp, "rounds")
            os.makedirs(rounds_dir, exist_ok=True)
            with open(os.path.join(rounds_dir, "round_007_123.json"), "w", encoding="utf-8") as f:
                json.dump({}, f)

            logger = RoundLogger(tmp)
            round_number = logger.start_round()
            self.assertEqual(round_number, 8)


if __name__ == "__main__":
    unittest.main()
