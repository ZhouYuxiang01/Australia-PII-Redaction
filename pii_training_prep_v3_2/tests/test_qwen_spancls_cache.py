import tempfile
import unittest
from pathlib import Path

import torch

from pii_prep.qwen_spancls_cache import (
    CacheWriter,
    build_cache_record,
    pool_span_embeddings,
)


class QwenSpanClsCacheTests(unittest.TestCase):
    def test_pool_span_embeddings_returns_mean_first_last(self):
        hidden = torch.tensor(
            [
                [[1.0, 1.0], [3.0, 5.0], [7.0, 9.0], [11.0, 13.0]],
                [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0], [14.0, 16.0]],
            ]
        )

        mean, first, last = pool_span_embeddings(hidden, [(1, 3), (0, 2)])

        self.assertTrue(torch.equal(mean, torch.tensor([[5.0, 7.0], [4.0, 6.0]])))
        self.assertTrue(torch.equal(first, torch.tensor([[3.0, 5.0], [2.0, 4.0]])))
        self.assertTrue(torch.equal(last, torch.tensor([[7.0, 9.0], [6.0, 8.0]])))

    def test_cache_writer_flushes_chunks_and_skips_completed_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = CacheWriter(root / "cache.pt", split="train", chunk_size=2, dtype=torch.float16)
            rows = [
                {"id": "a", "split": "train", "source": "src", "start": 0, "end": 1, "value": "x", "top_type": "NON_PII", "target_distribution": {"NON_PII": 1.0}, "training_weight": 1.0},
                {"id": "b", "split": "train", "source": "src", "start": 1, "end": 2, "value": "y", "top_type": "FIRST_NAME", "target_distribution": {"FIRST_NAME": 1.0}, "training_weight": 0.5},
            ]
            embeddings = torch.ones((2, 4))

            writer.append(rows, embeddings, embeddings + 1, embeddings + 2)
            writer.close()
            cache = torch.load(root / "cache.pt", map_location="cpu", weights_only=False)

            self.assertEqual(cache["split"], "train")
            self.assertEqual(len(cache["records"]), 2)
            self.assertEqual(cache["mean_embeddings"].shape, (2, 4))
            self.assertEqual(cache["mean_embeddings"].dtype, torch.float16)
            self.assertTrue((root / "cache.chunks" / "chunk_000000.pt").exists())

            resumed = CacheWriter(root / "cache.pt", split="train", chunk_size=2, dtype=torch.float16)
            self.assertEqual(resumed.completed_count(), 2)

    def test_build_cache_record_preserves_required_metadata(self):
        row = {"id": "abc", "split": "dev", "source": "s", "start": 5, "end": 9, "value": "test", "top_type": "EMAIL", "target_distribution": {"EMAIL": 1.0}, "training_weight": 0.7}

        record = build_cache_record(row)

        self.assertEqual(record["example_id"], "abc")
        self.assertEqual(record["split"], "dev")
        self.assertEqual(record["training_weight"], 0.7)


if __name__ == "__main__":
    unittest.main()
