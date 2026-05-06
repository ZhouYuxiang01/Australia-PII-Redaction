import unittest

import torch
import torch.nn as nn

from pii_prep.qwen_spancls_smoke import (
    FrozenQwenSpanClassifier,
    char_span_to_token_span,
    soft_cross_entropy,
    trainable_parameter_report,
)


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.embedding = nn.Embedding(10, hidden_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, use_cache=False, return_dict=True):
        class Output:
            pass

        out = Output()
        out.last_hidden_state = self.embedding(input_ids)
        return out


class QwenSpanClsSmokeTests(unittest.TestCase):
    def test_char_span_to_token_span_maps_overlapping_offsets(self):
        offsets = [(0, 0), (0, 4), (5, 8), (9, 12)]

        self.assertEqual(char_span_to_token_span(offsets, 5, 12), (2, 4))
        self.assertIsNone(char_span_to_token_span(offsets, 20, 22))

    def test_frozen_backbone_outputs_80_logits_and_only_head_trainable(self):
        model = FrozenQwenSpanClassifier(DummyBackbone(hidden_size=4), hidden_size=4, num_labels=80)
        logits = model(torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]), [(0, 3)])
        report = trainable_parameter_report(model)

        self.assertEqual(list(logits.shape), [1, 80])
        self.assertEqual(report["qwen_trainable_parameter_count"], 0)
        self.assertTrue(report["only_classification_head_trainable"])

    def test_soft_cross_entropy_accepts_soft_targets_and_weights(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        weights = torch.tensor([1.0, 0.5])
        loss = soft_cross_entropy(logits, targets, weights)

        self.assertGreater(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
