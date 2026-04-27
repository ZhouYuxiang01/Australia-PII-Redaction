import unittest

import eval_deployed_batch_dataset as batch_eval


class BatchEvalTests(unittest.TestCase):
    def test_batches_preserve_order(self):
        rows = [{"id": f"row-{index}"} for index in range(5)]

        batches = list(batch_eval.chunk_rows(rows, batch_size=2))

        self.assertEqual([[row["id"] for row in batch] for batch in batches], [["row-0", "row-1"], ["row-2", "row-3"], ["row-4"]])

    def test_completed_rows_are_skipped_before_batching(self):
        rows = [{"id": "done"}, {"id": "todo-1"}, {"id": "todo-2"}]
        completed = {"done": {"id": "done"}}

        pending = batch_eval.pending_rows(rows, completed)

        self.assertEqual([row["id"] for row in pending], ["todo-1", "todo-2"])


if __name__ == "__main__":
    unittest.main()
