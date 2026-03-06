import unittest

from agent.multitag_planner import PlannerAgent


class _DummyBaseClient:
    def encode_image_to_data_url(self, _):
        return None


class _DummyVLM:
    def __init__(self):
        self.base_client = _DummyBaseClient()

    def chat(self, messages, max_tokens):
        return "<summary></summary><memory_operations></memory_operations><plan_list></plan_list>"


class PlannerImageIndexTest(unittest.TestCase):
    def test_resolve_uses_current_turn_images_first(self):
        agent = PlannerAgent(vlm=_DummyVLM(), memory=None, prompt_name="multitag_planner")
        agent.current_input_image_paths = ["input_a.png", "input_b.png"]
        agent.current_turn_image_paths = ["turn_x.png", "turn_y.png"]
        self.assertEqual(agent._resolve_image_paths("2"), ["turn_y.png"])

    def test_resolve_falls_back_to_input_images(self):
        agent = PlannerAgent(vlm=_DummyVLM(), memory=None, prompt_name="multitag_planner")
        agent.current_input_image_paths = ["input_a.png", "input_b.png"]
        agent.current_turn_image_paths = []
        self.assertEqual(agent._resolve_image_paths("1,2"), ["input_a.png", "input_b.png"])


if __name__ == "__main__":
    unittest.main()
