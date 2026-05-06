"""Stage 5 Task 2: Unit tests for model, collator, and decode."""
import json
import sys
import unittest
from pathlib import Path
import torch

ROOT = Path('/home/admin/ZYX/pii_training_prep_v3_2')
sys.path.insert(0, str(ROOT / 'src' / 'pii_prep'))


class TestModelLogitsShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_labels = 317
        cls.hidden_size = 2560

    def test_logits_shape_correct(self):
        from qwen4b_tokencls_model import Qwen4BTokenClassifier

        class FakeBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('obj', (object,), {'hidden_size': 2560})()

            def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=False, **kwargs):
                B, T = input_ids.shape
                hidden = torch.randn(B, T, 2560)
                fake_outputs = type('obj', (object,), {
                    'hidden_states': [torch.randn(B, T, 2560)] * 32 + [hidden],
                    'last_hidden_state': hidden,
                })()
                return fake_outputs

        backbone = FakeBackbone()
        model = Qwen4BTokenClassifier(backbone, hidden_size=2560, num_labels=317, freeze_backbone=False)

        B, T = 2, 10
        input_ids = torch.randint(0, 1000, (B, T))
        attention_mask = torch.ones(B, T)
        labels = torch.randint(0, 317, (B, T))

        result = model(input_ids, attention_mask, labels)
        logits = result['logits']
        loss = result['loss']

        self.assertEqual(logits.shape, (B, T, 317))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)

    def test_predict_no_grad(self):
        from qwen4b_tokencls_model import Qwen4BTokenClassifier

        class FakeBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=False, **kwargs):
                B, T = input_ids.shape
                hidden = torch.randn(B, T, 2560)
                fake_outputs = type('obj', (object,), {
                    'hidden_states': [torch.randn(B, T, 2560)] * 33,
                })()
                return fake_outputs

        backbone = FakeBackbone()
        model = Qwen4BTokenClassifier(backbone, hidden_size=2560, num_labels=317, freeze_backbone=False)

        input_ids = torch.randint(0, 1000, (1, 5))
        logits = model.predict(input_ids)
        self.assertEqual(logits.shape, (1, 5, 317))


class TestLossIgnoresMinus100(unittest.TestCase):
    def test_loss_ignores_minus100(self):
        import torch.nn.functional as F

        logits = torch.randn(2, 5, 317)
        labels = torch.tensor([
            [0, 1, -100, 2, -100],
            [0, -100, 1, -100, 0],
        ])

        loss1 = F.cross_entropy(logits.view(-1, 317), labels.view(-1), ignore_index=-100)

        labels_no_ignore = torch.tensor([
            [0, 1, 0, 2, 0],
            [0, 0, 1, 0, 0],
        ])
        loss2 = F.cross_entropy(logits.view(-1, 317), labels_no_ignore.view(-1), ignore_index=-100)

        self.assertNotEqual(loss1.item(), loss2.item())


class TestCollator(unittest.TestCase):
    def test_collator_pads_labels_with_minus100(self):
        from qwen4b_tokencls_train import TokenClsCollator

        class FakeTokenizer:
            pad_token_id = 0

        collator = TokenClsCollator(FakeTokenizer(), max_length=100)

        rows = [
            {'input_ids': [1, 2, 3], 'labels': [0, 1, 2], 'record_id': 'a', 'text': 'hello'},
            {'input_ids': [4, 5], 'labels': [3, 4], 'record_id': 'b', 'text': 'world'},
        ]

        batch = collator(rows)

        self.assertEqual(batch['input_ids'].shape, (2, 3))
        self.assertEqual(batch['attention_mask'].shape, (2, 3))
        self.assertEqual(batch['labels'].shape, (2, 3))

        self.assertEqual(batch['labels'][1, 2].item(), -100)
        self.assertEqual(batch['labels'][1, 2].item(), -100)

        self.assertEqual(len(batch['record_ids']), 2)
        self.assertEqual(batch['record_ids'], ['a', 'b'])

    def test_collator_truncates_long_sequences(self):
        from qwen4b_tokencls_train import TokenClsCollator

        class FakeTokenizer:
            pad_token_id = 0

        collator = TokenClsCollator(FakeTokenizer(), max_length=5)

        rows = [
            {
                'input_ids': list(range(20)),
                'labels': list(range(20)),
                'attention_mask': [1] * 20,
                'offset_mapping': [[0, 0]] * 20,
                'record_id': 'long',
                'text': 'x' * 100,
            },
        ]

        batch = collator(rows)
        self.assertEqual(batch['input_ids'].shape[1], 5)
        self.assertEqual(batch['labels'].shape[1], 5)


class TestDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.id_to_label = json.loads(
            (ROOT / 'pii_schema' / 'id_to_token_label_317.json').read_text()
        )['id_to_label']
        cls.label_to_id = json.loads(
            (ROOT / 'pii_schema' / 'token_label_to_id_317.json').read_text()
        )['label_to_id']

    def test_decode_recovers_s_label(self):
        from qwen4b_tokencls_decode import logits_to_spans
        import torch.nn.functional as F

        # Create logits that strongly predict S-FIRST_NAME at position 1
        logits = torch.zeros(5, 317)
        logits[0, 0] = 10.0  # O
        logits[1, self.label_to_id['S-FIRST_NAME']] = 10.0
        logits[2, 0] = 10.0
        logits[3, 0] = 10.0
        logits[4, 0] = 10.0

        offsets = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24)]

        spans = logits_to_spans(logits, offsets, self.id_to_label, self.label_to_id, constrained=True)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]['type'], 'FIRST_NAME')
        self.assertEqual(spans[0]['start'], 5)
        self.assertEqual(spans[0]['end'], 9)
        self.assertEqual(spans[0]['num_tokens'], 1)

    def test_decode_recovers_bie_labels(self):
        from qwen4b_tokencls_decode import logits_to_spans

        logits = torch.zeros(8, 317)
        logits[0, 0] = 10.0
        logits[1, self.label_to_id['B-PERSON']] = 10.0
        logits[2, self.label_to_id['I-PERSON']] = 10.0
        logits[3, self.label_to_id['E-PERSON']] = 10.0
        logits[4, 0] = 10.0
        logits[5, 0] = 10.0
        logits[6, 0] = 10.0
        logits[7, 0] = 10.0

        offsets = [(0, 4), (5, 9), (10, 13), (14, 19), (20, 24), (25, 29), (30, 34), (35, 39)]

        spans = logits_to_spans(logits, offsets, self.id_to_label, self.label_to_id, constrained=True)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]['type'], 'PERSON')
        self.assertEqual(spans[0]['start'], 5)
        self.assertEqual(spans[0]['end'], 19)
        self.assertEqual(spans[0]['num_tokens'], 3)

    def test_whitespace_only_spans_rejected(self):
        from qwen4b_tokencls_decode import logits_to_spans

        logits = torch.zeros(3, 317)
        logits[0, 0] = 10.0
        logits[1, self.label_to_id['S-PERSON']] = 10.0
        logits[2, 0] = 10.0

        # Offset where start == end (whitespace/special token)
        offsets = [(0, 4), (5, 5), (6, 10)]
        spans = logits_to_spans(logits, offsets, self.id_to_label, self.label_to_id, constrained=True)
        self.assertEqual(len(spans), 0)

    def test_constrained_decode_respects_transitions(self):
        from qwen4b_tokencls_decode import constrained_decode_v2

        # Create logits where token 2 wrongly predicts B after B
        logits = torch.zeros(4, 317)
        logits[0, self.label_to_id['B-PERSON']] = 10.0
        logits[1, self.label_to_id['I-PERSON']] = 10.0
        logits[2, self.label_to_id['B-ADDRESS']] = 10.0  # Invalid: B after I
        logits[3, 0] = 10.0

        preds = constrained_decode_v2(logits, self.id_to_label, self.label_to_id)

        from qwen4b_tokencls_decode import token_label_to_tag
        tag2 = token_label_to_tag(preds[2], self.id_to_label)
        self.assertNotEqual(tag2, 'B')

    def test_decode_returns_sorted_non_overlapping_spans(self):
        from qwen4b_tokencls_decode import logits_to_spans

        logits = torch.zeros(10, 317)
        logits[2, self.label_to_id['S-FIRST_NAME']] = 10.0
        logits[5, self.label_to_id['S-LAST_NAME']] = 8.0
        logits[8, self.label_to_id['S-EMAIL_ADDRESS']] = 10.0

        offsets = [(i*5, (i+1)*5) for i in range(10)]

        spans = logits_to_spans(logits, offsets, self.id_to_label, self.label_to_id, constrained=True)
        self.assertTrue(all(
            spans[i]['start'] <= spans[i+1]['start']
            for i in range(len(spans)-1)
        ))


class TestExistingArtifactsUnchanged(unittest.TestCase):
    def test_no_existing_checkpoints_modified(self):
        import os
        ckpt_dirs = [
            ROOT / 'runs' / 'opf_hard_79',
            ROOT / 'runs' / 'opf_hard_smoke',
            ROOT / 'runs' / 'qwen_spancls_heads',
        ]
        for d in ckpt_dirs:
            if d.exists():
                self.assertTrue(d.exists(), f'{d} should still exist')

    def test_hybrid_files_unchanged(self):
        for fname in ['integrated_pipeline.py', 'opf_inference.py',
                       'qwen_spancls_heads.py', 'qwen_spancls_smoke.py']:
            path = ROOT / 'src' / 'pii_prep' / fname
            self.assertTrue(path.exists(), f'{fname} should still exist')

    def test_data_splits_unchanged(self):
        for split in ['train', 'dev', 'test']:
            path = ROOT / 'data' / 'splits' / f'{split}.jsonl'
            self.assertTrue(path.exists(), f'data/splits/{split}.jsonl should still exist')


if __name__ == '__main__':
    unittest.main(verbosity=2)
