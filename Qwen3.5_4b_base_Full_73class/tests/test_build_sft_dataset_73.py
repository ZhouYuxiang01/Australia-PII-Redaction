import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_sft_dataset_73 as builder


class BuildSftDataset73Tests(unittest.TestCase):
    def write_taxonomy(self, directory: Path) -> Path:
        taxonomy = directory / "taxonomy.yaml"
        taxonomy.write_text(
            "\n".join(
                [
                    "version: test",
                    "classes:",
                    "- code: EMAIL",
                    "  source_types:",
                    "  - EMAIL_ADDRESS",
                    "  - WORK_EMAIL",
                    "- code: PHONE",
                    "  source_types:",
                    "  - AU_PHONE",
                    "  - WORK_PHONE",
                    "- code: VEHICLE_ID",
                    "  source_types:",
                    "  - NUMBER_PLATE",
                    "  - VEHICLE_REGO",
                    "- code: GEOLOCATION_INFORMATION",
                    "  source_types:",
                    "  - GEOLOCATION_INFORMATION",
                    "- code: LATITUDE",
                    "  source_types:",
                    "  - LATITUDE",
                    "- code: LONGITUDE",
                    "  source_types:",
                    "  - LONGITUDE",
                    "- code: PERSON",
                    "  source_types:",
                    "  - PERSON",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return taxonomy

    def test_load_taxonomy_mapping_preserves_class_order_and_source_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            taxonomy = self.write_taxonomy(Path(tmp))

            mapping = builder.load_taxonomy_mapping(taxonomy)

        self.assertEqual(
            mapping.class_labels,
            [
                "EMAIL",
                "PHONE",
                "VEHICLE_ID",
                "GEOLOCATION_INFORMATION",
                "LATITUDE",
                "LONGITUDE",
                "PERSON",
            ],
        )
        self.assertEqual(mapping.source_to_class["WORK_EMAIL"], "EMAIL")
        self.assertEqual(mapping.source_to_class["AU_PHONE"], "PHONE")
        self.assertEqual(mapping.source_to_class["VEHICLE_REGO"], "VEHICLE_ID")

    def test_map_labels_deduplicates_source_synonyms_after_mapping(self):
        mapping = builder.TaxonomyMapping(
            class_labels=["VEHICLE_ID"],
            source_to_class={"NUMBER_PLATE": "VEHICLE_ID", "VEHICLE_REGO": "VEHICLE_ID"},
        )
        text = "Plate O385UM is on file."
        labels = [
            {"start": 6, "end": 12, "type": "NUMBER_PLATE", "value": "O385UM"},
            {"start": 6, "end": 12, "type": "VEHICLE_REGO", "value": "O385UM"},
        ]

        spans, audit = builder.map_source_labels(text, labels, mapping)

        self.assertEqual(
            spans,
            [{"start": 6, "end": 12, "type": "VEHICLE_ID", "value": "O385UM"}],
        )
        self.assertEqual(audit["deduped_after_mapping"], 1)
        self.assertEqual(audit["dropped"], [])

    def test_map_labels_keeps_overlapping_canonical_spans_for_json_output(self):
        mapping = builder.TaxonomyMapping(
            class_labels=["GEOLOCATION_INFORMATION", "LATITUDE", "LONGITUDE"],
            source_to_class={
                "GEOLOCATION_INFORMATION": "GEOLOCATION_INFORMATION",
                "LATITUDE": "LATITUDE",
                "LONGITUDE": "LONGITUDE",
            },
        )
        text = "Coords -17.24473, 144.665352 confirmed."
        labels = [
            {"start": 7, "end": 28, "type": "GEOLOCATION_INFORMATION", "value": "-17.24473, 144.665352"},
            {"start": 7, "end": 16, "type": "LATITUDE", "value": "-17.24473"},
            {"start": 18, "end": 28, "type": "LONGITUDE", "value": "144.665352"},
        ]

        spans, audit = builder.map_source_labels(text, labels, mapping)

        self.assertEqual([span["type"] for span in spans], ["GEOLOCATION_INFORMATION", "LATITUDE", "LONGITUDE"])
        self.assertEqual(audit["overlap_count"], 2)
        assistant = builder.build_assistant_json(spans)
        decoded = json.loads(assistant)
        self.assertEqual(len(decoded["spans"]), 3)

    def test_build_records_writes_json_span_messages_and_split_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            taxonomy = self.write_taxonomy(root)
            raw_path = root / "raw.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "version": "fixture",
                        "pii_types": ["PERSON", "EMAIL_ADDRESS", "WORK_EMAIL"],
                        "total_records": 2,
                        "records": [
                            {
                                "id": "R1",
                                "positive_sample": {
                                    "text": "Email alice@example.edu.au for Alice.",
                                    "labels": [
                                        {
                                            "start": 6,
                                            "end": 26,
                                            "type": "EMAIL_ADDRESS",
                                            "value": "alice@example.edu.au",
                                        },
                                        {"start": 31, "end": 36, "type": "PERSON", "value": "Alice"},
                                    ],
                                },
                                "hard_negatives": [{"text": "Ticket 12345 is internal.", "labels": []}],
                            },
                            {
                                "id": "R2",
                                "positive_sample": {
                                    "text": "Work email bob@uni.edu.au.",
                                    "labels": [
                                        {"start": 11, "end": 25, "type": "WORK_EMAIL", "value": "bob@uni.edu.au"}
                                    ],
                                },
                                "hard_negatives": [{"text": "Reference ABC-999.", "labels": []}],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            rows, meta, audit = builder.build_dataset_rows(
                raw_path=raw_path,
                taxonomy_path=taxonomy,
                seed=7,
                train_ratio=0.5,
                dev_ratio=0.5,
                test_ratio=0.0,
                train_negatives_per_record=1,
                keep_all_eval_negatives=True,
            )

        self.assertEqual(meta["class_count"], 7)
        self.assertEqual(meta["record_count"], 2)
        self.assertEqual(len(rows["train"]), 2)
        self.assertEqual(len(rows["dev"]), 2)
        self.assertEqual(len(rows["test"]), 0)
        self.assertEqual(audit["dropped_span_count"], 0)
        positive = next(row for split in rows.values() for row in split if row["used_labels"])
        self.assertEqual([m["role"] for m in positive["messages"]], ["system", "user", "assistant"])
        assistant_json = json.loads(positive["messages"][2]["content"])
        self.assertIn("spans", assistant_json)
        self.assertIn("EMAIL", {span["type"] for span in assistant_json["spans"]})


if __name__ == "__main__":
    unittest.main()
