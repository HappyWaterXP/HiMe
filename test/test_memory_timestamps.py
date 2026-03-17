import tempfile
import unittest

from memory.encoder import ZeroEncoder
from memory.multitag_recorder import MultiTagMemory


class MemoryTimestampTest(unittest.TestCase):
    def test_create_update_and_resume_preserve_updated_at(self):
        memory = MultiTagMemory(encoder=ZeroEncoder())
        memory._now_ts = lambda: 100.0

        rec = memory.create(
            tags=["toy_bread", "table"],
            data_type="image",
            text="The toy bread is on the table.",
            image_path="frame_1.png",
        )
        self.assertEqual(rec.updated_at, 100.0)

        memory._now_ts = lambda: 200.0
        updated = memory.update(rec_id=rec.id, text="The toy bread is in the box.")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.updated_at, 200.0)

        light = memory.all_light()
        self.assertEqual(light[0]["updated_at"], 200.0)
        self.assertEqual(light[0]["updated_at_readable"], memory.format_timestamp(200.0))

        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            memory.save_to_json(tmp.name)
            restored = MultiTagMemory.resume_from_json(tmp.name, ZeroEncoder())

        restored_rec = restored.get(rec.id)
        self.assertIsNotNone(restored_rec)
        self.assertEqual(restored_rec.updated_at, 200.0)


if __name__ == "__main__":
    unittest.main()
