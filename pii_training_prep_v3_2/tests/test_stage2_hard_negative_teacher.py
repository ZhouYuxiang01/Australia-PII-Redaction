import json
import tempfile
import unittest
import asyncio
from pathlib import Path

from pii_prep.stage2_hard_negative_teacher import (
    build_hard_negative_teacher_prompts,
    generate_hard_negative_teacher_artifacts,
    run_hard_negative_local_project,
    run_hard_negative_project_async,
    run_local_transformers_outputs,
)


class Stage2HardNegativeTeacherTests(unittest.TestCase):
    def test_builds_candidate_level_hard_negative_prompts_without_extra_labels(self):
        labels = {
            "BANK_ACCOUNT_NUMBER",
            "BANK_ACCOUNT_INFORMATION",
            "COURSE_CODE",
            "CREDIT_CARD_EXPIRY",
            "DATE_OF_BIRTH",
            "DEVICE_ID",
            "EMAIL_ADDRESS",
            "HASHED_PAYMENT_CARD_NUMBER",
            "HOME_PHONE",
            "IP_ADDRESS",
            "MOBILE",
            "NUMBER_PLATE",
            "PAYMENT_CARD_NUMBER",
            "STUDENT_ID",
            "UAC_ID",
            "USERNAME",
            "VEHICLE_REGO",
            "WEBSITE_HISTORY",
            "WORK_EMAIL",
            "WORK_PHONE",
            "NON_PII",
        }

        prompts, report = build_hard_negative_teacher_prompts(labels, examples_per_scenario=1, self_consistency=2)

        self.assertGreaterEqual(len(prompts), 10)
        self.assertEqual(report["context_type"], "hard_negative_context")
        self.assertEqual(report["self_consistency"], 2)
        self.assertNotIn("GENDER", report["candidate_label_counts"])
        self.assertTrue(any(prompt["span_value"] == "Southern Mutual" for prompt in prompts))
        self.assertTrue(any(prompt["span_value"] == "192.168.1.1" for prompt in prompts))
        for prompt in prompts:
            self.assertEqual(prompt["context_type"], "hard_negative_context")
            self.assertEqual(prompt["expected_top_type"], "NON_PII")
            self.assertEqual(prompt["teacher_model_path"], "/home/admin/model/Qwen3.6-27B")
            self.assertIn("NON_PII", prompt["candidate_labels"])
            self.assertNotIn("GENDER", prompt["candidate_labels"])
            self.assertIn("strong_against", prompt["prompt"])
            self.assertIn("hard-negative", prompt["prompt"])

    def test_writes_prompt_artifacts_for_teacher_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pii_schema").mkdir()
            (root / "pii_schema" / "training_label_space_80.json").write_text(
                json.dumps(["BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_INFORMATION", "NON_PII"]),
                encoding="utf-8",
            )

            report = generate_hard_negative_teacher_artifacts(root, examples_per_scenario=1, self_consistency=1)

            prompt_path = root / "data" / "generated" / "stage2_hard_negative_teacher_prompts.jsonl"
            report_path = root / "reports" / "stage2_hard_negative_teacher_plan.json"
            self.assertTrue(prompt_path.exists())
            self.assertTrue(report_path.exists())
            rows = [json.loads(line) for line in prompt_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["teacher_prompt_count"], len(rows))
            allowed = {"BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_INFORMATION", "NON_PII"}
            self.assertTrue(all(set(row["candidate_labels"]) <= allowed for row in rows))
            self.assertTrue(
                any(row["candidate_labels"] == ["BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_INFORMATION", "NON_PII"] for row in rows)
            )

    def test_run_hard_negative_project_writes_converted_rows_with_fake_sender(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pii_schema").mkdir()
            (root / "pii_schema" / "training_label_space_80.json").write_text(
                json.dumps(["BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_INFORMATION", "NON_PII"]),
                encoding="utf-8",
            )

            async def fake_sender(prompt, payload, endpoint):
                verdicts = {label: "strong_against" for label in prompt["candidate_labels"]}
                verdicts["NON_PII"] = "strong_for"
                return json.dumps({"verdicts": verdicts})

            report = asyncio.run(
                run_hard_negative_project_async(
                    root,
                    base_url="http://localhost:8000/v1",
                    model_name="test-27b",
                    concurrency=2,
                    max_tokens=64,
                    timeout_seconds=5,
                    max_retries=0,
                    examples_per_scenario=1,
                    self_consistency=1,
                    sender=fake_sender,
                )
            )

            converted_path = root / "data" / "generated" / "stage2_hard_negative_teacher_converted.jsonl"
            quality_path = root / "reports" / "stage2_hard_negative_teacher_quality_report.json"
            self.assertTrue(converted_path.exists())
            self.assertTrue(quality_path.exists())
            rows = [json.loads(line) for line in converted_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["converted_count"], len(rows))
            self.assertTrue(all(row["top_type"] == "NON_PII" for row in rows))

    def test_local_transformers_runner_writes_raw_rows_with_fake_model(self):
        prompts, _report = build_hard_negative_teacher_prompts(
            {"BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_INFORMATION", "NON_PII"},
            examples_per_scenario=1,
            self_consistency=1,
        )

        class FakeRunner:
            def generate(self, prompt):
                self.last_prompt = prompt
                return '{"verdicts":{"BANK_ACCOUNT_NUMBER":"strong_against","BANK_ACCOUNT_INFORMATION":"weak_against","NON_PII":"strong_for"}}'

        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw.jsonl"
            rows = run_local_transformers_outputs(
                prompts[:1],
                raw_path=raw_path,
                model_path="/fake/qwen3.6-27b",
                max_new_tokens=64,
                runner_factory=lambda model_path, max_new_tokens: FakeRunner(),
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[0]["teacher_model_path"], "/fake/qwen3.6-27b")
            self.assertTrue(raw_path.exists())

    def test_run_hard_negative_local_project_converts_fake_model_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pii_schema").mkdir()
            (root / "pii_schema" / "training_label_space_80.json").write_text(
                json.dumps(["BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_INFORMATION", "NON_PII"]),
                encoding="utf-8",
            )

            class FakeRunner:
                def generate(self, prompt):
                    return '{"verdicts":{"BANK_ACCOUNT_NUMBER":"strong_against","BANK_ACCOUNT_INFORMATION":"weak_against","NON_PII":"strong_for"}}'

            report = run_hard_negative_local_project(
                root,
                model_name="/fake/qwen3.6-27b",
                examples_per_scenario=1,
                self_consistency=1,
                runner_factory=lambda model_path, max_new_tokens: FakeRunner(),
            )

            self.assertGreater(report["converted_count"], 0)
            self.assertEqual(report["backend"], "local_transformers")
            converted_path = root / "data" / "generated" / "stage2_hard_negative_teacher_converted.jsonl"
            rows = [json.loads(line) for line in converted_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["top_type"] == "NON_PII" for row in rows))


if __name__ == "__main__":
    unittest.main()
