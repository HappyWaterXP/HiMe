import unittest

from extractor import parse_planner_output
from agent.multitag_planner import parse_query_terms


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

    def test_normalizes_space_tags_to_underscored_tags(self):
        xml = """
<summary>ok</summary>
<memory_operations>
  <operation>
    <type>CREATE</type>
    <tags>toy bread, left plate, [box]</tags>
    <text>example</text>
  </operation>
</memory_operations>
<plan_list></plan_list>
"""
        parsed = parse_planner_output(xml)
        self.assertEqual(
            parsed["memory_operations"][0].tags,
            ["toy_bread", "left_plate", "box"],
        )

    def test_normalizes_query_terms_to_underscored_tags(self):
        self.assertEqual(
            parse_query_terms('toy bread, left plate, "box"'),
            ["toy_bread", "left_plate", "box"],
        )


if __name__ == "__main__":
    unittest.main()
