"""Stage 5 Task 1: Unit tests for token label space and dataset builder."""
import json
import sys
import unittest
from pathlib import Path
from collections import Counter

# Ensure we can find project modules
ROOT = Path('/home/admin/ZYX/pii_training_prep_v3_2')
sys.path.insert(0, str(ROOT / 'scripts'))


class TestTokenLabelSpace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.label_space = json.loads(
            (ROOT / 'pii_schema' / 'token_label_space_317.json').read_text()
        )
        cls.label_to_id = json.loads(
            (ROOT / 'pii_schema' / 'token_label_to_id_317.json').read_text()
        )['label_to_id']
        cls.id_to_label = json.loads(
            (ROOT / 'pii_schema' / 'id_to_token_label_317.json').read_text()
        )['id_to_label']

    def test_exactly_317_labels(self):
        self.assertEqual(len(self.label_space['token_labels']), 317)

    def test_o_label_exists_and_is_first(self):
        self.assertIn('O', self.label_space['token_labels'])
        self.assertEqual(self.label_space['token_labels'][0], 'O')
        self.assertEqual(self.label_space['o_label_id'], 0)
        self.assertEqual(self.label_to_id['O'], 0)
        self.assertEqual(self.id_to_label['0'], 'O')

    def test_no_non_pii_bioes_labels(self):
        for label in self.label_space['token_labels']:
            if label != 'O':
                self.assertNotIn('NON_PII', label,
                    f'Label {label} should not contain NON_PII')
        bioes_labels = [l for l in self.label_space['token_labels'] if l != 'O']
        for label in bioes_labels:
            prefix = label[:2]
            self.assertIn(prefix, ['B-', 'I-', 'E-', 'S-'],
                f'Label {label} must start with B-/I-/E-/S-')

    def test_label_ids_contiguous(self):
        ids = sorted(int(k) for k in self.id_to_label.keys())
        self.assertEqual(ids, list(range(317)))

    def test_ignore_index_is_negative_100(self):
        self.assertEqual(self.label_space['ignore_index'], -100)

    def test_79_base_pii_labels(self):
        bioes_labels = [l for l in self.label_space['token_labels'] if l != 'O']
        base_labels = set(l[2:] for l in bioes_labels)
        self.assertEqual(len(base_labels), 79)

    def test_deterministic_ordering(self):
        labels = self.label_space['token_labels']
        self.assertEqual(labels[0], 'O')
        first_five = labels[1:5]
        self.assertTrue(all(l.startswith(('B-','I-','E-','S-')) for l in first_five))
        last_label = labels[-1]
        self.assertTrue(last_label.startswith('S-'))


class TestBIOESConstruction(unittest.TestCase):
    def setUp(self):
        self.label_to_id = json.loads(
            (ROOT / 'pii_schema' / 'token_label_to_id_317.json').read_text()
        )['label_to_id']

    def test_single_token_span_becomes_s_label(self):
        token_offsets = [(0, 5), (5, 10), (10, 15)]
        spans = [{'start': 0, 'end': 5, 'top_type': 'FIRST_NAME', 'value': 'John'}]
        labels, included, errors = self._assign_labels(token_offsets, spans)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(included), 1)
        self.assertEqual(labels[0], self.label_to_id['S-FIRST_NAME'])
        self.assertEqual(labels[1], self.label_to_id['O'])
        self.assertEqual(labels[2], self.label_to_id['O'])

    def test_multi_token_span_becomes_bie_labels(self):
        token_offsets = [(0, 2), (2, 6), (6, 10), (10, 15)]
        spans = [{'start': 0, 'end': 10, 'top_type': 'PERSON', 'value': 'John Smith'}]
        labels, included, errors = self._assign_labels(token_offsets, spans)
        self.assertEqual(len(errors), 0)
        self.assertEqual(labels[0], self.label_to_id['B-PERSON'])
        self.assertEqual(labels[1], self.label_to_id['I-PERSON'])
        self.assertEqual(labels[2], self.label_to_id['E-PERSON'])
        self.assertEqual(labels[3], self.label_to_id['O'])

    def test_non_pii_tokens_get_o_label(self):
        token_offsets = [(0, 5), (5, 10)]
        spans = []
        labels, included, errors = self._assign_labels(token_offsets, spans)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(included), 0)
        self.assertEqual(labels[0], self.label_to_id['O'])
        self.assertEqual(labels[1], self.label_to_id['O'])

    def test_whitespace_only_span_rejected(self):
        token_offsets = [(0, 3), (3, 5), (5, 8)]
        spans = [{'start': 3, 'end': 5, 'top_type': 'PERSON', 'value': '  '}]
        labels, included, errors = self._assign_labels(token_offsets, spans)
        self.assertTrue(len(errors) > 0)
        self.assertEqual(errors[0]['reason'], 'whitespace_only_value')
        self.assertTrue(all(l == self.label_to_id['O'] for l in labels))

    def test_span_no_token_overlap_reported(self):
        token_offsets = [(0, 5), (5, 10)]
        spans = [{'start': 100, 'end': 105, 'top_type': 'EMAIL_ADDRESS', 'value': 'a@b.com'}]
        labels, included, errors = self._assign_labels(token_offsets, spans)
        self.assertTrue(len(errors) > 0)
        self.assertEqual(errors[0]['reason'], 'no_tokens_overlap_span')

    def test_missing_top_type_reported(self):
        token_offsets = [(0, 5)]
        spans = [{'start': 0, 'end': 5, 'value': 'test'}]
        labels, included, errors = self._assign_labels(token_offsets, spans)
        self.assertTrue(len(errors) > 0)
        self.assertEqual(errors[0]['reason'], 'missing_top_type')

    def _assign_labels(self, token_offsets, spans):
        from build_qwen4b_tokencls_dataset import assign_bioes_labels
        return assign_bioes_labels(
            token_offsets, spans,
            self.label_to_id,
            o_id=self.label_to_id['O'],
            ignore_id=-100,
        )


class TestOverlapResolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classification_weights = {}
        canonical = json.loads((ROOT / 'pii_schema' / 'canonical_labels_79.json').read_text())
        for label in canonical:
            cls.classification_weights[label] = 'Protected'
        # Make some Highly Protected
        for label in ['DATE_OF_BIRTH', 'BANK_ACCOUNT_NUMBER', 'MEDICARE_NUMBER',
                       'AU_TFN', 'CRIMINAL_RECORDS', 'STUDENT_ID', 'PASSPORT_NUMBER']:
            cls.classification_weights[label] = 'Highly Protected'

    def test_no_overlap_no_conflicts(self):
        spans = [
            {'start': 0, 'end': 5, 'top_type': 'FIRST_NAME', 'training_weight': 0.8},
            {'start': 10, 'end': 15, 'top_type': 'LAST_NAME', 'training_weight': 0.8},
        ]
        resolved, conflicts = self._resolve(spans)
        self.assertEqual(len(resolved), 2)
        self.assertEqual(len(conflicts), 0)

    def test_highly_protected_wins_over_protected(self):
        spans = [
            {'start': 0, 'end': 10, 'top_type': 'FIRST_NAME', 'training_weight': 0.8},
            {'start': 5, 'end': 15, 'top_type': 'DATE_OF_BIRTH', 'training_weight': 0.8},
        ]
        resolved, conflicts = self._resolve(spans)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]['top_type'], 'DATE_OF_BIRTH')
        self.assertEqual(len(conflicts), 1)

    def test_higher_confidence_wins_same_classification(self):
        spans = [
            {'start': 0, 'end': 10, 'top_type': 'FIRST_NAME', 'training_weight': 0.5},
            {'start': 5, 'end': 15, 'top_type': 'LAST_NAME', 'training_weight': 0.9},
        ]
        resolved, conflicts = self._resolve(spans)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]['top_type'], 'LAST_NAME')

    def test_longer_span_wins_same_confidence(self):
        spans = [
            {'start': 0, 'end': 5, 'top_type': 'FIRST_NAME', 'training_weight': 0.8},
            {'start': 3, 'end': 15, 'top_type': 'LAST_NAME', 'training_weight': 0.8},
        ]
        resolved, conflicts = self._resolve(spans)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]['top_type'], 'LAST_NAME')

    def test_earlier_start_wins_same_length(self):
        spans = [
            {'start': 0, 'end': 10, 'top_type': 'FIRST_NAME', 'training_weight': 0.8},
            {'start': 5, 'end': 15, 'top_type': 'LAST_NAME', 'training_weight': 0.8},
        ]
        resolved, conflicts = self._resolve(spans)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]['top_type'], 'FIRST_NAME')

    def test_deterministic_tiebreak_by_name(self):
        spans = [
            {'start': 0, 'end': 10, 'top_type': 'ZZZ_LABEL', 'training_weight': 0.8},
            {'start': 0, 'end': 10, 'top_type': 'AAA_LABEL', 'training_weight': 0.8},
        ]
        resolved, conflicts = self._resolve(spans)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(resolved[0]['top_type'], 'AAA_LABEL')

    def _resolve(self, spans):
        from build_qwen4b_tokencls_dataset import resolve_overlaps
        return resolve_overlaps(spans, self.classification_weights)


class TestDatasetOutputValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.label_to_id = json.loads(
            (ROOT / 'pii_schema' / 'token_label_to_id_317.json').read_text()
        )['label_to_id']
        cls.id_to_label = json.loads(
            (ROOT / 'pii_schema' / 'id_to_token_label_317.json').read_text()
        )['id_to_label']
        cls.max_label_id = max(int(k) for k in cls.id_to_label.keys())

    def _check_file(self, split):
        path = ROOT / 'data' / 'train' / f'qwen4b_tokencls_{split}.jsonl'
        self.assertTrue(path.exists(), f'{split} file missing')
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = row.get('record_id', f'line_{line_num}')

                input_ids = row['input_ids']
                labels = row['labels']
                attn_mask = row.get('attention_mask', [1]*len(input_ids))
                offsets = row.get('offset_mapping', [])

                # Mandatory fields
                self.assertIn('record_id', row, f'{split}:{line_num} missing record_id')
                self.assertIn('split', row, f'{split}:{line_num} missing split')
                self.assertIn('input_ids', row, f'{split}:{line_num} missing input_ids')
                self.assertIn('labels', row, f'{split}:{line_num} missing labels')
                self.assertIn('original_char_start', row)
                self.assertIn('original_char_end', row)

                # Length consistency
                n = len(input_ids)
                self.assertEqual(len(labels), n,
                    f'{split}:{line_num} [{row_id}] labels len {len(labels)} != input_ids len {n}')
                self.assertEqual(len(attn_mask), n,
                    f'{split}:{line_num} [{row_id}] attn_mask len mismatch')
                self.assertEqual(len(offsets), n,
                    f'{split}:{line_num} [{row_id}] offset_mapping len mismatch')

                # Label validity
                for j, lid in enumerate(labels):
                    self.assertNotEqual(lid, None,
                        f'{split}:{line_num}:{j} [{row_id}] None label')
                    if lid != -100:
                        self.assertGreaterEqual(lid, 0,
                            f'{split}:{line_num}:{j} [{row_id}] negative label {lid}')
                        self.assertLessEqual(lid, self.max_label_id,
                            f'{split}:{line_num}:{j} [{row_id}] label {lid} > max {self.max_label_id}')

                # Special tokens use -100 or O
                for j, (ts, te) in enumerate(offsets):
                    if ts == te:
                        self.assertIn(labels[j], [-100, 0],
                            f'{split}:{line_num}:{j} [{row_id}] special token has label {labels[j]}')

                # Only check first 100 rows per file for performance
                if line_num >= 100:
                    break

    def test_train_file_valid(self):
        self._check_file('train')

    def test_dev_file_valid(self):
        self._check_file('dev')

    def test_test_file_valid(self):
        self._check_file('test')

    def test_all_labels_in_valid_range(self):
        for split in ['train', 'dev', 'test']:
            path = ROOT / 'data' / 'train' / f'qwen4b_tokencls_{split}.jsonl'
            with open(path) as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for lid in row['labels']:
                        if lid != -100:
                            label_str = self.id_to_label.get(str(lid), '')
                            self.assertNotIn('NON_PII', label_str,
                                f'{split}:{line_num} NON_PII in BIOES label {label_str}')
                    if line_num >= 50:
                        break

    def test_special_tokens_padding_are_ignore(self):
        for split in ['train', 'dev', 'test']:
            path = ROOT / 'data' / 'train' / f'qwen4b_tokencls_{split}.jsonl'
            with open(path) as f:
                for line_num, line in enumerate(f, 1):
                    if not line_num % 2000 != 0:
                        continue
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    attn_mask = row.get('attention_mask', [1]*len(row['input_ids']))
                    labels = row['labels']
                    for j, am in enumerate(attn_mask):
                        if am == 0:
                            self.assertEqual(labels[j], -100,
                                f'{split}:{line_num}:{j} padding token not -100')
                    if line_num >= 10000:
                        break
                if line_num > 10000:
                    break


if __name__ == '__main__':
    unittest.main(verbosity=2)
