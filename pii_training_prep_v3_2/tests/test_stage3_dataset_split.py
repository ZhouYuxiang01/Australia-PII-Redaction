import unittest

from pii_prep.stage3_dataset_split import (
    build_audit_v2,
    build_opf_hard_records,
    build_qwen_spancls_examples,
    group_key,
    split_records,
    validate_opf_records,
    validate_qwen_examples,
)


class Stage3DatasetSplitTests(unittest.TestCase):
    def test_group_key_keeps_stage2_self_consistency_together(self):
        a = {"id": "STAGE2-STAGE2-FULL-BASE-0001-SC1", "metadata": {}, "text": "x", "spans": []}
        b = {"id": "STAGE2-STAGE2-FULL-BASE-0001-SC3", "metadata": {}, "text": "x", "spans": []}

        self.assertEqual(group_key(a), group_key(b))

    def test_group_key_keeps_hard_negative_self_consistency_together(self):
        base = {
            "metadata": {
                "subtype": "candidate_level_negative",
                "ambiguity_group": "hard_negative_bank_name",
                "context_type": "hard_negative_context",
            },
            "text": 'Bank name written as "Southern Mutual"; acct: 0088 1992 44.',
            "spans": [
                {
                    "value": "Southern Mutual",
                    "format_candidates": ["BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_INFORMATION", "NON_PII"],
                }
            ],
        }
        a = {**base, "id": "STAGE2-STAGE2-HARDNEG-BASE-0001-SC1"}
        b = {**base, "id": "STAGE2-STAGE2-HARDNEG-BASE-0001-SC3"}

        self.assertEqual(group_key(a), group_key(b))

    def test_build_audit_v2_splits_record_and_span_sources(self):
        records = [
            {"id": "doc", "metadata": {"subtype": "doc_level"}, "spans": []},
            {
                "id": "s",
                "metadata": {"source_type": "email"},
                "spans": [{"source": "sonnet_high_conf", "top_type": "PERSON", "type_distribution": {"PERSON": 1.0}}],
            },
        ]

        audit = build_audit_v2(records, {"source_distribution": {"old": 1}})

        self.assertNotIn("source_distribution", audit)
        self.assertEqual(audit["source_record_distribution"]["document_level_negative"], 1)
        self.assertEqual(audit["source_span_distribution"]["sonnet_high_conf"], 1)

    def test_spancls_and_opf_outputs_validate(self):
        labels = ["PERSON", "NON_PII"]
        records = [
            {
                "id": "r1",
                "text": "Name: Mia",
                "metadata": {"source_type": "qwen_5way_ranking"},
                "spans": [
                    {
                        "start": 6,
                        "end": 9,
                        "value": "Mia",
                        "type_distribution": {"PERSON": 0.9, "NON_PII": 0.1},
                        "top_type": "PERSON",
                        "source": "qwen_5way_ranking",
                        "training_weight": 0.5,
                    }
                ],
            },
            {
                "id": "r2",
                "text": "Order 123",
                "metadata": {"subtype": "candidate_level_negative"},
                "spans": [
                    {
                        "start": 6,
                        "end": 9,
                        "value": "123",
                        "type_distribution": {"PERSON": 0.1, "NON_PII": 0.9},
                        "top_type": "NON_PII",
                        "source": "qwen_5way_ranking",
                        "training_weight": 0.5,
                    }
                ],
            },
        ]

        qwen = build_qwen_spancls_examples(records, "train", labels)
        opf = build_opf_hard_records(records, "train")

        self.assertEqual(len(qwen), 2)
        self.assertEqual(validate_qwen_examples(qwen, set(labels))["validation_error_count"], 0)
        self.assertEqual(len(opf[1]["spans"]), 0)
        self.assertEqual(validate_opf_records(opf, set(labels))["validation_error_count"], 0)

    def test_split_records_keeps_group_members_together(self):
        records = [
            {"id": "AU-PII-00001", "metadata": {}, "spans": []},
            {"id": "AU-PII-00001-HN-01", "metadata": {}, "spans": []},
        ]

        splits = split_records(records)
        locations = [split for split, rows in splits.items() for row in rows if row["id"].startswith("AU-PII-00001")]

        self.assertEqual(len(set(locations)), 1)


if __name__ == "__main__":
    unittest.main()
