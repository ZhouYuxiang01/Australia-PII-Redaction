import unittest

import torch

from pii_prep.qwen_spancls_heads import (
    build_head,
    build_targets,
    classification_metrics,
    fit_temperature,
    parse_weight_overrides,
    select_features,
    soft_cross_entropy,
)


class QwenSpanClsHeadTests(unittest.TestCase):
    def test_select_features_supports_required_experiments(self):
        cache = {
            "mean_embeddings": torch.ones((2, 4), dtype=torch.float16),
            "first_embeddings": torch.ones((2, 4), dtype=torch.float16) * 2,
            "last_embeddings": torch.ones((2, 4), dtype=torch.float16) * 3,
        }

        self.assertEqual(select_features(cache, "mean_linear").shape, (2, 4))
        self.assertEqual(select_features(cache, "first_linear")[0, 0].item(), 2)
        self.assertEqual(select_features(cache, "last_linear")[0, 0].item(), 3)
        self.assertEqual(select_features(cache, "concat_mlp").shape, (2, 12))

    def test_build_head_outputs_80_logits(self):
        for experiment, input_dim in [("mean_linear", 4), ("first_linear", 4), ("last_linear", 4), ("concat_mlp", 12)]:
            head = build_head(experiment, input_dim=input_dim, num_labels=80)
            logits = head(torch.randn(3, input_dim))
            self.assertEqual(list(logits.shape), [3, 80])

    def test_weighted_soft_cross_entropy_uses_soft_targets(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
        weights = torch.tensor([1.0, 0.5])

        loss = soft_cross_entropy(logits, targets, weights)

        self.assertGreater(float(loss), 0.0)

    def test_build_targets_can_boost_hard_negative_sources_and_non_pii_label(self):
        labels = ["PERSON", "NON_PII"]
        records = [
            {
                "source": "candidate_level_negative",
                "top_type": "NON_PII",
                "target_distribution": {"NON_PII": 1.0},
                "training_weight": 0.5,
            },
            {
                "source": "sonnet_high_conf",
                "top_type": "PERSON",
                "target_distribution": {"PERSON": 0.95, "NON_PII": 0.05},
                "training_weight": 0.8,
            },
        ]

        _targets, weights, labels_outside = build_targets(
            records,
            labels,
            source_weight_overrides={"candidate_level_negative": 3.0},
            label_weight_overrides={"NON_PII": 2.0},
        )

        self.assertEqual(labels_outside, {})
        self.assertAlmostEqual(float(weights[0]), 3.0)
        self.assertAlmostEqual(float(weights[1]), 0.8)

    def test_parse_weight_overrides(self):
        self.assertEqual(parse_weight_overrides("candidate_level_negative=3,NON_PII=2.5"), {"candidate_level_negative": 3.0, "NON_PII": 2.5})
        self.assertEqual(parse_weight_overrides(""), {})

    def test_temperature_scaling_returns_positive_temperature(self):
        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [2.0, 1.0]])
        targets = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]])
        weights = torch.ones(3)

        temperature = fit_temperature(logits, targets, weights, max_iter=20)

        self.assertGreater(temperature, 0.0)

    def test_metrics_include_per_label_accuracy_and_confusion_pairs(self):
        labels = ["A", "B", "NON_PII"]
        logits = torch.tensor([[5.0, 1.0, 0.0], [3.0, 4.0, 0.0], [0.0, 1.0, 5.0]])
        targets = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rows = [{"source": "x"}, {"source": "x"}, {"source": "y"}]

        metrics = classification_metrics(logits, targets, labels, rows)

        self.assertIn("per_label_top1_accuracy", metrics)
        self.assertIn("A", metrics["per_label_top1_accuracy"])
        self.assertEqual(metrics["confusion_top_pairs"][0]["gold"], "A")
        self.assertEqual(metrics["confusion_top_pairs"][0]["predicted"], "B")


if __name__ == "__main__":
    unittest.main()
