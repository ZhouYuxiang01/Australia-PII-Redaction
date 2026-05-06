import csv
import json
import tempfile
import unittest
from pathlib import Path

from pii_prep.reconcile_taxonomy import reconcile_project, remap_stage1_jsonl


class ReconcileTaxonomyTests(unittest.TestCase):
    def test_reconciliation_uses_csv_only_labels_and_reports_zero_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            data_raw = root / "data" / "raw"
            data_processed = root / "data" / "processed"
            reports = root / "reports"
            docs.mkdir()
            data_raw.mkdir(parents=True)
            data_processed.mkdir(parents=True)
            reports.mkdir()

            with (docs / "Data Sensitivity.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["Name", "Note", "Data Classification", "Category Type"],
                )
                writer.writeheader()
                for name in ["Bank Account Number", "Mobile phone", "First Name"]:
                    writer.writerow(
                        {
                            "Name": name,
                            "Note": "",
                            "Data Classification": "Protected",
                            "Category Type": "Test",
                        }
                    )
                writer.writerow({"Name": "", "Note": "", "Data Classification": "", "Category Type": ""})

            (data_raw / "au_pii_19000_final.json").write_text(
                json.dumps(
                    {
                        "pii_types": ["BSB", "AU_PHONE"],
                        "records": [
                            {
                                "positive_sample": {
                                    "labels": [
                                        {"type": "BSB"},
                                        {"type": "AU_PHONE"},
                                    ]
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            (data_processed / "stage1_v3_2.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "one",
                                "text": "BSB 123456",
                                "spans": [
                                    {
                                        "start": 4,
                                        "end": 10,
                                        "value": "123456",
                                        "top_type": "BSB",
                                        "type_distribution": {"BSB": 0.95, "NON_PII": 0.05},
                                        "format_candidates": ["BSB", "NON_PII"],
                                        "training_weight": 0.8,
                                    }
                                ],
                            }
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = reconcile_project(root, enforce_expected_counts=False)

            self.assertEqual(report["csv_row_count"], 4)
            self.assertEqual(report["csv_effective_label_count"], 3)
            self.assertEqual(report["raw_only_unmapped"], [])
            self.assertEqual(report["csv_only_zero_example"], ["FIRST_NAME"])
            self.assertEqual(report["final_training_class_count"], 4)
            self.assertEqual(json.loads((root / "pii_schema" / "canonical_labels_79.json").read_text()), ["BANK_ACCOUNT_NUMBER", "MOBILE", "FIRST_NAME"])
            remapped = json.loads((root / "data" / "processed" / "stage1_v3_2_canonical.jsonl").read_text().strip())
            self.assertEqual(remapped["spans"][0]["top_type"], "BANK_ACCOUNT_NUMBER")
            self.assertEqual(remapped["spans"][0]["type_distribution"], {"BANK_ACCOUNT_NUMBER": 0.95, "NON_PII": 0.05})

    def test_unmapped_raw_label_fails_before_writing_canonical_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "data" / "raw").mkdir(parents=True)
            (root / "data" / "processed").mkdir(parents=True)
            with (root / "docs" / "Data Sensitivity.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Name", "Note", "Data Classification", "Category Type"])
                writer.writeheader()
                writer.writerow({"Name": "Full Name", "Note": "", "Data Classification": "Protected", "Category Type": "Test"})
            (root / "data" / "raw" / "au_pii_19000_final.json").write_text(
                json.dumps({"pii_types": ["UNKNOWN_RAW"], "records": []}),
                encoding="utf-8",
            )
            (root / "data" / "processed" / "stage1_v3_2.jsonl").write_text("", encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                reconcile_project(root, enforce_expected_counts=False)

        self.assertIn("Unmapped raw labels", str(ctx.exception))

    def test_remap_stage1_rejects_labels_outside_training_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "stage1.jsonl"
            dst = Path(tmp) / "canonical.jsonl"
            src.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "text": "abc",
                        "spans": [
                            {
                                "start": 0,
                                "end": 3,
                                "value": "abc",
                                "top_type": "RAW_ONLY",
                                "type_distribution": {"RAW_ONLY": 1.0},
                                "format_candidates": ["RAW_ONLY"],
                                "training_weight": 1.0,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            audit = remap_stage1_jsonl(
                src,
                dst,
                alias_map={},
                training_labels={"PERSON", "NON_PII"},
            )

        self.assertEqual(audit["validation_error_count"], 3)
        self.assertFalse(dst.exists())
