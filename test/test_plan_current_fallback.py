import unittest

from extractor import ensure_plan_has_current


class PlanCurrentFallbackTest(unittest.TestCase):
    def test_promotes_first_pending_when_no_current_exists(self):
        plan = (
            "[pending] task a\n"
            "[pending] task b\n"
            "[done] task c"
        )
        normalized = ensure_plan_has_current(plan)
        self.assertEqual(
            normalized,
            "[current] task a\n[pending] task b\n[done] task c",
        )

    def test_keeps_existing_current_unchanged(self):
        plan = "[done] task a\n[current] task b\n[pending] task c"
        self.assertEqual(ensure_plan_has_current(plan), plan)


if __name__ == "__main__":
    unittest.main()
