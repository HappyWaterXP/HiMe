import unittest

from extractor import parse_planner_output


class ExtractorSingleSubtaskTest(unittest.TestCase):
    def test_maps_single_subtask_to_current_plan_line(self):
        xml = """
<summary>ok</summary>
<memory_operations></memory_operations>
<subtask>inspect the left box</subtask>
<is_complete>no</is_complete>
"""
        parsed = parse_planner_output(xml)
        self.assertEqual(parsed["plan_list"], "[current] inspect the left box")

    def test_maps_single_subtask_to_done_plan_line(self):
        xml = """
<summary>done</summary>
<memory_operations></memory_operations>
<subtask>pick up red cube and place it to right box</subtask>
<is_complete>yes</is_complete>
"""
        parsed = parse_planner_output(xml)
        self.assertEqual(
            parsed["plan_list"],
            "[done] pick up red cube and place it to right box",
        )


if __name__ == "__main__":
    unittest.main()
