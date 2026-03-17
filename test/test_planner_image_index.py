import unittest
from unittest.mock import patch
import json
import tempfile

from agent.multitag_planner import PlannerAgent


class _DummyBaseClient:
    def encode_image_to_data_url(self, _):
        return None


class _DummyVLM:
    def __init__(self):
        self.base_client = _DummyBaseClient()

    def chat(self, messages, max_tokens):
        return "<summary></summary><memory_operations></memory_operations><plan_list></plan_list>"


class _CapturingBaseClient:
    def encode_image_to_data_url(self, path):
        return {"type": "image_url", "image_url": {"url": path}}


class _CapturingVLM:
    def __init__(self):
        self.base_client = _CapturingBaseClient()
        self.last_messages = None

    def chat(self, messages, max_tokens):
        self.last_messages = messages
        return "<summary></summary><memory_operations></memory_operations><plan_list></plan_list>"


class PlannerImageIndexTest(unittest.TestCase):
    def test_resolve_prefers_turn1_input_images(self):
        agent = PlannerAgent(vlm=_DummyVLM(), memory=None, prompt_name="multitag_planner")
        agent.current_input_image_paths = ["input_a.png", "input_b.png"]
        agent.current_turn_image_paths = ["turn_x.png", "turn_y.png"]
        self.assertEqual(agent._resolve_image_paths("2"), ["input_b.png"])

    def test_resolve_falls_back_to_current_turn_images_when_turn1_missing(self):
        agent = PlannerAgent(vlm=_DummyVLM(), memory=None, prompt_name="multitag_planner")
        agent.current_input_image_paths = []
        agent.current_turn_image_paths = ["turn_x.png", "turn_y.png"]
        self.assertEqual(agent._resolve_image_paths("1,2"), ["turn_x.png", "turn_y.png"])

    def test_resolve_representative_uses_latest_valid_turn1_frame(self):
        agent = PlannerAgent(vlm=_DummyVLM(), memory=None, prompt_name="multitag_planner")
        agent.current_input_image_paths = ["input_a.png", "input_b.png", "input_c.png"]
        self.assertEqual(agent._resolve_representative_image_path("1,3"), "input_c.png")

    def test_turn2_prompt_keeps_image_path_bound_to_turn1_frames(self):
        agent = PlannerAgent(vlm=_DummyVLM(), memory=None, prompt_name="multitag_planner")
        agent.current_input_image_paths = ["input_a.png", "input_b.png"]

        prompt = agent._build_user_prompt_turn_n(
            turn_num=2,
            query_results_text="--- QUERY: toy_bread ---",
            attachment_count=1,
        )

        self.assertIn("historical memory evidence for reading only", prompt)
        self.assertIn("original execution frames from TURN 1", prompt)
        self.assertIn("historical records from the past", prompt)
        self.assertNotIn('image_path="1..1"', prompt)

    def test_turn1_prompt_marks_turn1_images_as_current_state(self):
        agent = PlannerAgent(vlm=_DummyVLM(), memory=None, prompt_name="multitag_planner")
        with patch("agent.multitag_planner.time.time", return_value=123.0):
            prompt = agent._build_user_prompt_turn1(
                instruction="move the bread",
                plan_list="",
                image_count=2,
                current_timestamp_text="1970-01-01 00:02:03",
            )

        self.assertIn("TURN 1 images below are the current world state", prompt)
        self.assertIn("authoritative source", prompt)
        self.assertIn("CURRENT TIME: 1970-01-01 00:02:03", prompt)
        self.assertIn("choose exactly one frame index", prompt)
        self.assertIn("frame's local timestamp", prompt)
        self.assertNotIn('image_path="1" or image_path="2,3"', prompt)

    def test_turn1_step_adds_frame_timestamp_anchors(self):
        vlm = _CapturingVLM()
        agent = PlannerAgent(vlm=vlm, memory=None, prompt_name="multitag_planner")

        agent.step(
            user_instruction="move the bread",
            image_paths=["frame_1773650828637.png"],
            current_plan_list="",
            turn_num=1,
        )

        user_message = [msg for msg in vlm.last_messages if msg["role"] == "user"][-1]
        user_content = user_message["content"]
        text_blocks = [item["text"] for item in user_content if item["type"] == "text"]
        self.assertIn("[CURRENT FRAME 1 | Timestamp=2026-03-16 16:47:08]", text_blocks)

    def test_turn1_uses_one_current_time_and_index_only_frame_anchors(self):
        vlm = _CapturingVLM()
        agent = PlannerAgent(vlm=vlm, memory=None, prompt_name="multitag_planner")

        with patch("agent.multitag_planner.infer_frame_timestamp", side_effect=[123.0, 100.0, 123.0]):
            agent.step(
                user_instruction="move the bread",
                image_paths=["a.png", "b.png"],
                current_plan_list="",
                turn_num=1,
            )

        user_message = [msg for msg in vlm.last_messages if msg["role"] == "user"][-1]
        user_content = user_message["content"]
        frame_texts = [item["text"] for item in user_content if item["type"] == "text" and item["text"].startswith("[CURRENT FRAME")]
        general_text = [item["text"] for item in user_content if item["type"] == "text" and item["text"].startswith("=== TURN 1 ===")][0]
        self.assertEqual(len(frame_texts), 2)
        self.assertEqual(frame_texts, ["[CURRENT FRAME 1 | Timestamp=1970-01-01 08:01:40]", "[CURRENT FRAME 2 | Timestamp=1970-01-01 08:02:03]"])
        self.assertIn("CURRENT TIME: 1970-01-01 08:02:03", general_text)

    def test_export_conversation_preserves_image_meta(self):
        agent = PlannerAgent(vlm=_DummyVLM(), memory=None, prompt_name="multitag_planner")
        agent.messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                        "meta": {
                            "source_path": "frame.png",
                            "frame_index": 1,
                            "memory_updated_at": "2026-03-16 16:47:19",
                        },
                    },
                ],
            },
        ]

        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            agent.export_conversation_json(tmp.name, drop_images=True)
            with open(tmp.name, "r", encoding="utf-8") as f:
                data = json.load(f)

        image_item = data[1]["content"][1]
        self.assertEqual(image_item["meta"]["source_path"], "frame.png")
        self.assertEqual(image_item["meta"]["frame_index"], 1)
        self.assertEqual(image_item["meta"]["memory_updated_at"], "2026-03-16 16:47:19")


if __name__ == "__main__":
    unittest.main()
