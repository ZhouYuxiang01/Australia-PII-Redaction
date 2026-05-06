import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from pii_prep.stage2_generation import (
    ZERO_EXAMPLE_LABELS,
    generate_stage2_artifacts,
    validate_seed_records,
)


class Stage2GenerationTests(unittest.TestCase):
    def test_stage2_outputs_zero_examples_ambiguous_prompts_and_valid_seed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pii_schema").mkdir()
            (root / "data" / "generated").mkdir(parents=True)
            (root / "reports").mkdir()

            labels = sorted(
                set(
                    ZERO_EXAMPLE_LABELS
                    + [
                        "BANK_ACCOUNT_NUMBER",
                        "MOBILE",
                        "STUDENT_ID",
                        "EMPLOYEE_NUMBER",
                        "AU_TFN",
                        "DATE_OF_BIRTH",
                        "EMAIL_ADDRESS",
                        "PERSON",
                        "ADDRESS",
                        "LATITUDE",
                        "LONGITUDE",
                        "NUMBER_PLATE",
                        "VEHICLE_REGO",
                        "PAYMENT_CARD_NUMBER",
                        "MEDICARE_NUMBER",
                        "IHI",
                        "SOCIAL_MEDIA_ACCOUNT",
                        "SOCIAL_MEDIA_ID",
                        "USERNAME",
                    ]
                )
            )
            training_labels = labels + ["NON_PII"]
            (root / "pii_schema" / "canonical_labels_79.json").write_text(json.dumps(labels), encoding="utf-8")
            (root / "pii_schema" / "training_label_space_80.json").write_text(json.dumps(training_labels), encoding="utf-8")
            (root / "pii_schema" / "label_aliases_v3_2.json").write_text("{}", encoding="utf-8")
            (root / "data" / "processed").mkdir(parents=True)
            (root / "data" / "processed" / "stage1_v3_2_canonical.jsonl").write_text("", encoding="utf-8")
            (root / "reports" / "stage1_canonical_audit.json").write_text(
                json.dumps({"validation_error_count": 0}),
                encoding="utf-8",
            )

            plan = generate_stage2_artifacts(
                root,
                zero_examples_per_label=20,
                prompt_sample_size=20,
                enforce_expected_counts=False,
            )

            seed_path = root / "data" / "generated" / "stage2_seed_examples.jsonl"
            prompts_path = root / "data" / "generated" / "stage2_teacher_prompts_sample.jsonl"
            generation_plan_path = root / "reports" / "stage2_generation_plan.json"
            zero_plan_path = root / "reports" / "zero_example_label_plan.json"

            self.assertTrue(seed_path.exists())
            self.assertTrue(prompts_path.exists())
            self.assertTrue(generation_plan_path.exists())
            self.assertTrue(zero_plan_path.exists())

            seed_records = [json.loads(line) for line in seed_path.read_text(encoding="utf-8").splitlines()]
            counts = Counter(
                span["top_type"]
                for record in seed_records
                for span in record["spans"]
                if record["metadata"]["source"] == "synthetic_zero_example"
            )
            for label in ZERO_EXAMPLE_LABELS:
                self.assertGreaterEqual(counts[label], 20)

            errors = validate_seed_records(seed_records, set(training_labels))
            self.assertEqual(errors, [])
            self.assertEqual(plan["teacher_prompt_sample_count"], 20)

            candidate_negatives = [
                record for record in seed_records if record["metadata"].get("subtype") == "candidate_level"
            ]
            self.assertGreater(len(candidate_negatives), 0)
            for record in candidate_negatives:
                self.assertGreater(record["spans"][0]["type_distribution"]["NON_PII"], 0.5)

            document_negatives = [
                record for record in seed_records if record["metadata"].get("subtype") == "doc_level"
            ]
            self.assertGreater(len(document_negatives), 0)
            self.assertTrue(all(record["spans"] == [] for record in document_negatives))

            prompts = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(prompts), 20)
            self.assertEqual(
                {"bare_span", "weak_context", "strong_positive_context", "reverse_negative_context"},
                {prompt["context_type"] for prompt in prompts},
            )
            for prompt in prompts:
                self.assertIn("strong_for", prompt["prompt"])
                self.assertIn("weak_against", prompt["prompt"])
                self.assertIn("Output JSON only", prompt["prompt"])


if __name__ == "__main__":
    unittest.main()
