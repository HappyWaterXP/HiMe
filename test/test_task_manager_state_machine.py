import time
import unittest
from types import SimpleNamespace

from PIL import Image

from server.image_utils import RobotImageInput
from server.schema import TaskConfig, TaskStateEnum
from server.task_manager import ServerTaskManager


def _make_img(color=(120, 200, 80)):
    return Image.new("RGB", (32, 32), color)


class FakePlannerAgent:
    def __init__(self, non_initial_sleep_s: float = 0.0):
        self.non_initial_sleep_s = non_initial_sleep_s
        self.calls = []

    def run_refine(
        self,
        *,
        image_paths,
        initial_plan_list,
        user_instruction,
        **kwargs,
    ):
        self.calls.append(
            {
                "initial_plan_list": initial_plan_list,
                "user_instruction": user_instruction,
                "image_count": len(image_paths or []),
            }
        )
        if initial_plan_list is not None and self.non_initial_sleep_s > 0:
            time.sleep(self.non_initial_sleep_s)
        return SimpleNamespace(
            plan_text="Step 1: inspect the workspace [current]\nStep 2: finish [pending]",
            summary="ok",
            raw_xml="<planner/>",
            memory_operations=[],
        )


class FakeObserverAgent:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.call_count = 0

    def run(self, *, image_paths, plan_list, max_tokens):
        idx = self.call_count
        self.call_count += 1
        status = self.statuses[min(idx, len(self.statuses) - 1)] if self.statuses else "not_done"
        return SimpleNamespace(status=status, raw_xml="<observer/>")


class FakePendingOnlyPlannerAgent(FakePlannerAgent):
    def run_refine(
        self,
        *,
        image_paths,
        initial_plan_list,
        user_instruction,
        **kwargs,
    ):
        self.calls.append(
            {
                "initial_plan_list": initial_plan_list,
                "user_instruction": user_instruction,
                "image_count": len(image_paths or []),
            }
        )
        return SimpleNamespace(
            plan_text=(
                "[pending] pick up the toy croissant from the left plate and place it in the box\n"
                "[pending] pick up the toy mushroom from the right plate and place it in the box\n"
                "[pending] pick up the toy bread from the table and place it in the box"
            ),
            summary="ok",
            raw_xml="<planner/>",
            memory_operations=[],
        )


class FakeSingleNewPlanPlannerAgent(FakePlannerAgent):
    def run_refine(
        self,
        *,
        image_paths,
        initial_plan_list,
        user_instruction,
        **kwargs,
    ):
        self.calls.append(
            {
                "initial_plan_list": initial_plan_list,
                "user_instruction": user_instruction,
                "image_count": len(image_paths or []),
            }
        )
        return SimpleNamespace(
            plan_text="[current] pick up the toy croissant from the box and place it on the left plate",
            summary="ok",
            raw_xml="<planner/>",
            memory_operations=[],
        )


class TaskManagerStateMachineTest(unittest.TestCase):
    def test_async_done_trigger_is_not_queued(self):
        manager = ServerTaskManager()
        planner = FakePlannerAgent(non_initial_sleep_s=0.8)
        observer = FakeObserverAgent(statuses=["done", "done", "not_done"])
        manager.set_agents(planner, observer)

        cfg = TaskConfig(use_observer=True, planner_execution_mode="async", observer_window_size=4, use_memory=False)
        state = manager.create_task(
            global_instruction="test async done",
            initial_robot_input=RobotImageInput(waist_image=_make_img(), image=_make_img()),
            config=cfg,
        )
        self.assertEqual(state.runtime_state, TaskStateEnum.OBSERVING)

        t1 = time.time()
        manager.add_step_and_maybe_refine_robot(
            state.task_id,
            RobotImageInput(waist_image=[_make_img()], image=[_make_img()]),
        )
        d1 = time.time() - t1

        t2 = time.time()
        manager.add_step_and_maybe_refine_robot(
            state.task_id,
            RobotImageInput(waist_image=[_make_img()], image=[_make_img()]),
        )
        d2 = time.time() - t2

        # Async bootstrap step blocks by design to ensure first runnable instruction.
        self.assertGreaterEqual(d1, 0.7)
        self.assertLess(d2, 0.2)
        self.assertEqual(manager._get_task(state.task_id).runtime_state, TaskStateEnum.PLANNER_RUNNING)

        t3 = time.time()
        manager.add_step_and_maybe_refine_robot(
            state.task_id,
            RobotImageInput(waist_image=[_make_img()], image=[_make_img()]),
        )
        d3 = time.time() - t3
        self.assertLess(d3, 0.2)
        self.assertEqual(manager._get_task(state.task_id).runtime_state, TaskStateEnum.PLANNER_RUNNING)

        # Wait for async planner to finish, then send one more step.
        time.sleep(1.0)
        manager.add_step_and_maybe_refine_robot(
            state.task_id,
            RobotImageInput(waist_image=[_make_img()], image=[_make_img()]),
        )
        self.assertEqual(manager._get_task(state.task_id).runtime_state, TaskStateEnum.OBSERVING)

        # create + bootstrap + one async refine; second "done" while running should not add another refine.
        self.assertEqual(len(planner.calls), 3)

    def test_user_instruction_is_blocking_even_in_async_mode(self):
        manager = ServerTaskManager()
        planner = FakePlannerAgent(non_initial_sleep_s=0.7)
        observer = FakeObserverAgent(statuses=["not_done"])
        manager.set_agents(planner, observer)

        cfg = TaskConfig(use_observer=False, planner_execution_mode="async", observer_window_size=4, use_memory=False)
        state = manager.create_task(
            global_instruction="initial",
            initial_robot_input=RobotImageInput(waist_image=_make_img(), image=_make_img()),
            config=cfg,
        )

        manager.add_step_and_maybe_refine_robot(
            state.task_id,
            RobotImageInput(waist_image=[_make_img()], image=[_make_img()]),
        )

        t = time.time()
        updated = manager.refine_with_user_instruction(state.task_id, "new user instruction")
        dt = time.time() - t

        # User instruction path is blocking; threshold keeps test robust across environments.
        self.assertGreaterEqual(dt, 0.65)
        self.assertEqual(updated.global_instruction, "new user instruction")
        self.assertIn(updated.runtime_state, (TaskStateEnum.OBSERVING, TaskStateEnum.DONE))
        self.assertEqual(len(planner.calls), 3)

    def test_user_instruction_after_done_extends_completed_plan(self):
        manager = ServerTaskManager()
        planner = FakePlannerAgent()
        observer = FakeObserverAgent(statuses=["not_done"])
        manager.set_agents(planner, observer)

        cfg = TaskConfig(use_observer=False, planner_execution_mode="sync", observer_window_size=4, use_memory=False)
        state = manager.create_task(
            global_instruction="initial",
            initial_robot_input=RobotImageInput(waist_image=_make_img(), image=_make_img()),
            config=cfg,
        )
        manager.add_step_and_maybe_refine_robot(
            state.task_id,
            RobotImageInput(waist_image=[_make_img()], image=[_make_img()]),
        )

        state.is_done = True
        state.runtime_state = TaskStateEnum.DONE
        state.plan_list = "[done] everything finished"
        state.current_subtask_description = None

        updated = manager.refine_with_user_instruction(state.task_id, "do one more thing")

        self.assertEqual(updated.global_instruction, "do one more thing")
        self.assertFalse(updated.extra.get("extend_from_done", False))
        self.assertEqual(planner.calls[-1]["initial_plan_list"], "[done] everything finished")
        self.assertIn("do one more thing", planner.calls[-1]["user_instruction"])
        self.assertIn("Keep the completed plan as history", planner.calls[-1]["user_instruction"])
        self.assertIn("----- Past Plan History -----", planner.calls[-1]["user_instruction"])
        self.assertIn("[done] everything finished", planner.calls[-1]["user_instruction"])
        self.assertEqual(planner.calls[-1]["image_count"], 1)

    def test_pending_only_plan_promotes_first_pending_to_current(self):
        manager = ServerTaskManager()
        planner = FakePendingOnlyPlannerAgent()
        observer = FakeObserverAgent(statuses=["not_done"])
        manager.set_agents(planner, observer)

        cfg = TaskConfig(use_observer=False, planner_execution_mode="sync", observer_window_size=4, use_memory=False)
        state = manager.create_task(
            global_instruction="initial",
            initial_robot_input=RobotImageInput(waist_image=_make_img(), image=_make_img()),
            config=cfg,
        )

        self.assertFalse(state.is_done)
        self.assertEqual(state.runtime_state, TaskStateEnum.OBSERVING)
        self.assertTrue(state.plan_list.startswith("[current] pick up the toy croissant"))
        self.assertEqual(
            state.current_subtask_description,
            "pick up the toy croissant from the left plate and place it in the box",
        )

    def test_user_instruction_after_done_merges_past_history_into_plan(self):
        manager = ServerTaskManager()
        planner = FakeSingleNewPlanPlannerAgent()
        observer = FakeObserverAgent(statuses=["not_done"])
        manager.set_agents(planner, observer)

        cfg = TaskConfig(use_observer=False, planner_execution_mode="sync", observer_window_size=4, use_memory=False)
        state = manager.create_task(
            global_instruction="initial",
            initial_robot_input=RobotImageInput(waist_image=_make_img(), image=_make_img()),
            config=cfg,
        )

        state.is_done = True
        state.runtime_state = TaskStateEnum.DONE
        state.plan_list = "[done] pick up the toy croissant from the left plate and place it in the box"
        state.current_subtask_description = None

        updated = manager.refine_with_user_instruction(state.task_id, "reset the toys")

        self.assertIn("----- Past Plan History -----", updated.plan_list)
        self.assertIn("----- Current Active Plan -----", updated.plan_list)
        self.assertIn("[done] pick up the toy croissant from the left plate and place it in the box", updated.plan_list)
        self.assertIn("[current] pick up the toy croissant from the box and place it on the left plate", updated.plan_list)
        self.assertEqual(
            updated.current_subtask_description,
            "pick up the toy croissant from the box and place it on the left plate",
        )


if __name__ == "__main__":
    unittest.main()
