import unittest

from pii_prep.stage3a_model_selection import (
    choose_best_model,
    high_entropy_mask,
    leakage_summary,
    normalize_text,
)


class Stage3AModelSelectionTests(unittest.TestCase):
    def test_choose_best_model_uses_dev_metrics_only_with_tie_breakers(self):
        reports = {
            "a": {"metrics": {"after_temperature": {"dev": {"nll": 0.20, "ece": 0.01, "top3_accuracy": 1.0}, "test": {"nll": 0.01}}}, "temperature": 1.0},
            "b": {"metrics": {"after_temperature": {"dev": {"nll": 0.10, "ece": 0.09, "top3_accuracy": 0.9}, "test": {"nll": 0.99}}}, "temperature": 2.0},
            "c": {"metrics": {"after_temperature": {"dev": {"nll": 0.10, "ece": 0.05, "top3_accuracy": 0.8}, "test": {"nll": 0.50}}}, "temperature": 3.0},
        }

        selected = choose_best_model(reports, run_dir_name="custom_heads")

        self.assertEqual(selected["selected_model"], "c")
        self.assertEqual(selected["selected_checkpoint"], "runs/custom_heads/c/head.pt")
        self.assertEqual(selected["selection_sort_key"], [0.1, 0.05, -0.8])

    def test_choose_best_model_can_prioritize_hard_negative_dev_recall(self):
        reports = {
            "low_nll_bad_negatives": {
                "metrics": {
                    "after_temperature": {
                        "dev": {
                            "nll": 0.05,
                            "ece": 0.01,
                            "top3_accuracy": 1.0,
                            "non_pii_accuracy": 0.20,
                            "per_source_accuracy": {"candidate_level_negative": 0.10},
                        }
                    }
                },
                "temperature": 1.0,
            },
            "higher_nll_good_negatives": {
                "metrics": {
                    "after_temperature": {
                        "dev": {
                            "nll": 0.10,
                            "ece": 0.03,
                            "top3_accuracy": 0.98,
                            "non_pii_accuracy": 0.95,
                            "per_source_accuracy": {"candidate_level_negative": 0.90},
                        }
                    }
                },
                "temperature": 1.0,
            },
        }

        selected = choose_best_model(reports, selection_strategy="hard_negative_aware")

        self.assertEqual(selected["selected_model"], "higher_nll_good_negatives")
        self.assertEqual(selected["selection_strategy"], "hard_negative_aware")
        self.assertEqual(selected["selection_sort_key"][0], 0.1)

    def test_leakage_summary_reports_cross_split_duplicates(self):
        rows = {
            "train": [{"id": "a::span-0", "record_id": "a", "text": "Hello  World", "start": 0, "end": 5, "value": "Hello"}],
            "dev": [{"id": "b::span-0", "record_id": "b", "text": "hello world", "start": 0, "end": 5, "value": "Hello"}],
            "test": [{"id": "a::span-1", "record_id": "a", "text": "Other", "start": 1, "end": 3, "value": "th"}],
        }

        report = leakage_summary(rows)

        self.assertEqual(report["normalized_text_overlap_count"], 1)
        self.assertEqual(report["duplicate_record_id_cross_split_count"], 1)
        self.assertTrue(report["severe_leakage_detected"])
        self.assertEqual(normalize_text("Hello  World"), "hello world")

    def test_high_entropy_mask_marks_large_entropy_values(self):
        mask = high_entropy_mask([0.1, 1.0, 2.0, 3.0], quantile=0.75)

        self.assertEqual(mask, [False, False, False, True])


if __name__ == "__main__":
    unittest.main()
