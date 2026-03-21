import unittest

from memory.multitag_recorder import MultiTagMemory


class _ZeroEncoder:
    def encode_obj(self, _):
        return [0.0, 0.0]

    def encode_text(self, _):
        return [0.0, 0.0]


class MemoryFifoTest(unittest.TestCase):
    def test_create_prunes_oldest_ids_when_capacity_exceeded(self):
        memory = MultiTagMemory(encoder=_ZeroEncoder(), max_records=3)

        for idx in range(4):
            memory.create(
                tags=[f"tag_{idx}"],
                data_type="text",
                text=f"text_{idx}",
            )

        ids = [rec.id for rec in memory.all()]
        self.assertEqual(ids, [2, 3, 4])
        self.assertIsNone(memory.get(1))

    def test_set_max_records_prunes_existing_store(self):
        memory = MultiTagMemory(encoder=_ZeroEncoder())
        for idx in range(5):
            memory.create(
                tags=[f"tag_{idx}"],
                data_type="text",
                text=f"text_{idx}",
            )

        memory.set_max_records(2)

        ids = [rec.id for rec in memory.all()]
        self.assertEqual(ids, [4, 5])


if __name__ == "__main__":
    unittest.main()
