import json
import tempfile
import unittest
from pathlib import Path

from pii_prep.stage2_augmented_merge import merge_augmented_dataset, merge_project


class Stage2AugmentedMergeTests(unittest.TestCase):
    def test_merge_preserves_sources_and_validates_records(self):
        training_labels = {"EMAIL_ADDRESS", "FIRST_NAME", "NON_PII"}
        stage1 = [
            {
                "id": "s1",
                "text": "Email alex@example.com",
                "metadata": {"source_type": "email"},
                "spans": [
                    {
                        "start": 6,
                        "end": 22,
                        "value": "alex@example.com",
                        "type_distribution": {"EMAIL_ADDRESS": 0.95, "NON_PII": 0.05},
                        "top_type": "EMAIL_ADDRESS",
                        "source": "sonnet_high_conf",
                        "training_weight": 0.8,
                        "format_candidates": ["EMAIL_ADDRESS", "NON_PII"],
                    }
                ],
            },
            {"id": "d1", "text": "Order #123 shipped.", "metadata": {"subtype": "doc_level"}, "spans": []},
        ]
        stage2 = [
            {
                "id": "p1",
                "context_type": "strong_positive_context",
                "span_value": "Mia",
                "context": "First name: Mia",
                "candidate_labels": ["FIRST_NAME", "NON_PII"],
                "type_distribution": {"FIRST_NAME": 0.9, "NON_PII": 0.1},
                "top_type": "FIRST_NAME",
                "verdicts": {"FIRST_NAME": "strong_for", "NON_PII": "weak_against"},
            }
        ]

        merged, audit, distribution, warnings = merge_augmented_dataset(stage1, stage2, training_labels)

        self.assertEqual(len(merged), 3)
        self.assertEqual(audit["validation_error_count"], 0)
        self.assertEqual(audit["labels_outside_training_space"], {})
        self.assertEqual(audit["source_distribution"]["sonnet_high_conf"], 1)
        self.assertEqual(audit["source_distribution"]["qwen_5way_ranking"], 1)
        self.assertEqual(audit["source_distribution"]["document_level_negative"], 1)
        self.assertEqual(distribution["per_label_count"]["FIRST_NAME"], 1)
        self.assertEqual(warnings["student_training_started"], False)

    def test_duplicate_record_ids_fail_validation(self):
        training_labels = {"NON_PII"}
        records = [
            {"id": "dup", "text": "a", "metadata": {}, "spans": []},
            {"id": "dup", "text": "b", "metadata": {}, "spans": []},
        ]

        _merged, audit, _distribution, _warnings = merge_augmented_dataset(records, [], training_labels)

        self.assertEqual(audit["duplicate_record_id_count"], 1)
        self.assertGreater(audit["validation_error_count"], 0)

    def test_hard_negative_teacher_rows_merge_as_candidate_level_negatives(self):
        training_labels = {"BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_INFORMATION", "NON_PII"}
        stage2 = [
            {
                "id": "hn1",
                "context_type": "hard_negative_context",
                "span_value": "Southern Mutual",
                "context": 'Bank name written as "Southern Mutual"; acct: 0088 1992 44.',
                "candidate_labels": ["BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_INFORMATION", "NON_PII"],
                "type_distribution": {"BANK_ACCOUNT_NUMBER": 0.05, "BANK_ACCOUNT_INFORMATION": 0.10, "NON_PII": 0.85},
                "top_type": "NON_PII",
                "verdicts": {
                    "BANK_ACCOUNT_NUMBER": "strong_against",
                    "BANK_ACCOUNT_INFORMATION": "weak_against",
                    "NON_PII": "strong_for",
                },
            }
        ]

        merged, audit, _distribution, _warnings = merge_augmented_dataset([], stage2, training_labels)

        self.assertEqual(audit["validation_error_count"], 0)
        self.assertEqual(audit["source_distribution"]["candidate_level_negative"], 1)
        self.assertEqual(merged[0]["metadata"]["subtype"], "candidate_level_negative")
        self.assertGreaterEqual(merged[0]["spans"][0]["training_weight"], 0.8)

    def test_merge_project_includes_optional_hard_negative_teacher_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pii_schema").mkdir()
            (root / "data" / "processed").mkdir(parents=True)
            (root / "data" / "generated").mkdir(parents=True)
            (root / "pii_schema" / "training_label_space_80.json").write_text(
                json.dumps(["BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_INFORMATION", "FIRST_NAME", "NON_PII"]),
                encoding="utf-8",
            )
            (root / "data" / "processed" / "stage1_v3_2_canonical.jsonl").write_text(
                json.dumps({"id": "s1", "text": "No spans here.", "metadata": {}, "spans": []}) + "\n",
                encoding="utf-8",
            )
            (root / "data" / "generated" / "stage2_full_teacher_converted.jsonl").write_text(
                json.dumps(
                    {
                        "id": "pos1",
                        "context_type": "strong_positive_context",
                        "span_value": "Mia",
                        "context": "First name: Mia",
                        "candidate_labels": ["FIRST_NAME", "NON_PII"],
                        "type_distribution": {"FIRST_NAME": 0.9, "NON_PII": 0.1},
                        "top_type": "FIRST_NAME",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "data" / "generated" / "stage2_hard_negative_teacher_converted.jsonl").write_text(
                json.dumps(
                    {
                        "id": "hn1",
                        "context_type": "hard_negative_context",
                        "span_value": "Southern Mutual",
                        "context": 'Bank name written as "Southern Mutual"; acct: 0088 1992 44.',
                        "candidate_labels": ["BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_INFORMATION", "NON_PII"],
                        "type_distribution": {
                            "BANK_ACCOUNT_NUMBER": 0.05,
                            "BANK_ACCOUNT_INFORMATION": 0.10,
                            "NON_PII": 0.85,
                        },
                        "top_type": "NON_PII",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            audit = merge_project(root)
            rows = [
                json.loads(line)
                for line in (root / "data" / "processed" / "stage2_v3_2_augmented.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

            self.assertEqual(audit["stage2_record_count"], 2)
            self.assertEqual(len(rows), 3)
            self.assertEqual(audit["source_distribution"]["candidate_level_negative"], 1)


if __name__ == "__main__":
    unittest.main()
