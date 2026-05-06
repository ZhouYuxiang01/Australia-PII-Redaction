import unittest

from pii_prep.stage2_augmented_merge import merge_augmented_dataset


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


if __name__ == "__main__":
    unittest.main()
