import csv
import json
import tempfile
import unittest
from pathlib import Path

from pii_prep.build_distribution_dataset import build_records


class BuildDistributionDatasetTests(unittest.TestCase):
    def test_positive_labels_become_smoothed_distributions_and_negatives_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.json"
            taxonomy_path = tmp_path / "taxonomy.csv"

            raw_path.write_text(
                json.dumps(
                    {
                        "version": "test",
                        "pii_types": ["IP_ADDRESS"],
                        "records": [
                            {
                                "id": "AU-PII-00001",
                                "positive_sample": {
                                    "text": "Login from 242.30.143.150 was reviewed.",
                                    "labels": [
                                        {
                                            "start": 11,
                                            "end": 25,
                                            "type": "IP_ADDRESS",
                                            "value": "242.30.143.150",
                                            "confidence": 0.883,
                                        }
                                    ],
                                },
                                "input": {"metadata": {"source_type": "email", "language": "en-AU"}},
                                "hard_negatives": [
                                    {"text": "Order #123456 shipped today.", "labels": []}
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with taxonomy_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["Name", "Note", "Data Classification", "Category Type"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Name": "IP Address",
                        "Note": "IPv4 or IPv6",
                        "Data Classification": "Protected",
                        "Category Type": "Technical",
                    }
                )

            records, audit = build_records(raw_path, taxonomy_path, include_hard_negatives=True)

        self.assertEqual(len(records), 2)
        positive = records[0]
        span = positive["spans"][0]
        self.assertEqual(span["type_distribution"], {"IP_ADDRESS": 0.95, "NON_PII": 0.05})
        self.assertEqual(span["top_type"], "IP_ADDRESS")
        self.assertEqual(span["source"], "sonnet_high_conf")
        self.assertEqual(span["training_weight"], 1.0)
        self.assertTrue(span["rule_verified"])
        self.assertIn("IP_ADDRESS", span["format_candidates"])
        self.assertEqual(records[1]["metadata"]["data_category"], "D")
        self.assertEqual(records[1]["metadata"]["subtype"], "doc_level")
        self.assertEqual(records[1]["spans"], [])
        self.assertEqual(audit["input_records"], 1)
        self.assertEqual(audit["span_count"], 1)


if __name__ == "__main__":
    unittest.main()
