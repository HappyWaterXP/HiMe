import unittest

from agent.multitag_planner import PlannerAgent
from extractor import MemoryOperation


class _DummyBaseClient:
    def encode_image_to_data_url(self, _):
        return None


class _DummyVLM:
    def __init__(self):
        self.base_client = _DummyBaseClient()

    def chat(self, messages, max_tokens):
        return "<summary></summary><memory_operations></memory_operations><plan_list></plan_list>"


class _DummyRecord:
    def __init__(self, rid, tags, data_type="text", text="x", image_path=None):
        self.id = rid
        self.tags = tags
        self.data_type = data_type
        self.text = text
        self.image_path = image_path


class _DummyMemory:
    def __init__(self):
        self.query_calls = 0
        self.query_contents = []
        self.create_calls = 0
        self.update_calls = 0
        self.delete_calls = 0
        self._next_id = 1

    def query(self, content, top_k):
        self.query_calls += 1
        self.query_contents.append(content)
        return [_DummyRecord(1, ["tag"])], {1: 0.9}

    def create(self, tags, text=None, image_path=None, **kwargs):
        self.create_calls += 1
        rec = _DummyRecord(self._next_id, tags or [], text=text or "", image_path=image_path)
        self._next_id += 1
        return rec

    def update(self, **kwargs):
        self.update_calls += 1
        return _DummyRecord(kwargs.get("rec_id", 0), ["updated"])

    def delete(self, rec_id):
        self.delete_calls += 1
        return True


class PlannerMemoryPolicyTest(unittest.TestCase):
    def test_query_create_only_blocks_update_delete(self):
        memory = _DummyMemory()
        planner = PlannerAgent(
            vlm=_DummyVLM(),
            memory=memory,
            prompt_name="multitag_planner",
            memory_op_policy="query_create_only",
        )
        ops = [
            MemoryOperation(type="QUERY", id=None, obj_name=None, text=None, reason="", raw_xml="", query="a"),
            MemoryOperation(type="CREATE", id=None, obj_name=None, text="t", reason="", raw_xml="", tags=["x"]),
            MemoryOperation(type="UPDATE", id="1", obj_name=None, text="u", reason="", raw_xml=""),
            MemoryOperation(type="DELETE", id="1", obj_name=None, text=None, reason="", raw_xml=""),
        ]
        planner._apply_memory_operations(ops)

        self.assertEqual(memory.query_calls, 1)
        self.assertEqual(memory.create_calls, 1)
        self.assertEqual(memory.update_calls, 0)
        self.assertEqual(memory.delete_calls, 0)

    def test_query_splits_comma_separated_terms(self):
        memory = _DummyMemory()
        planner = PlannerAgent(
            vlm=_DummyVLM(),
            memory=memory,
            prompt_name="multitag_planner",
        )
        ops = [
            MemoryOperation(type="QUERY", id=None, obj_name=None, text=None, reason="", raw_xml="", query="toy_bread, box, left_plate"),
        ]

        planner._apply_memory_operations(ops)

        self.assertEqual(memory.query_calls, 3)
        self.assertEqual(memory.query_contents, ["toy_bread", "box", "left_plate"])

    def test_query_results_are_deduplicated_across_terms(self):
        memory = _DummyMemory()
        planner = PlannerAgent(
            vlm=_DummyVLM(),
            memory=memory,
            prompt_name="multitag_planner",
        )
        ops = [
            MemoryOperation(type="QUERY", id=None, obj_name=None, text=None, reason="", raw_xml="", query="toy_bread,box"),
        ]

        result_text = planner._apply_memory_operations(ops)

        self.assertEqual(result_text.count("Record ID=1"), 1)
        self.assertIn("--- QUERY: toy_bread ---", result_text)
        self.assertIn("--- QUERY: box ---", result_text)
        self.assertIn("UpdatedAt:", result_text)
        self.assertIn("All matching records were already shown above", result_text)


if __name__ == "__main__":
    unittest.main()
