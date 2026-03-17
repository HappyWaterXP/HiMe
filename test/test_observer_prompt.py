import unittest
from unittest.mock import patch

from agent.observer import ObserverAgent


class _DummyObserverVLM:
    pass


class ObserverPromptTest(unittest.TestCase):
    def test_custom_prompt_name_is_loaded(self):
        with patch("agent.observer.load_prompt", return_value="observer prompt body") as mocked:
            agent = ObserverAgent(vlm=_DummyObserverVLM(), prompt_name="task3_obs")

        mocked.assert_called_once_with("task3_obs")
        self.assertEqual(agent.system_prompt, "observer prompt body")


if __name__ == "__main__":
    unittest.main()
