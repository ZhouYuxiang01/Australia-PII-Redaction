import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from pii_prep.stage2_full_teacher import (
    generate_full_teacher_prompts,
    run_full_teacher_async,
)


class Stage2FullTeacherTests(unittest.TestCase):
    def setUp(self):
        self.training_labels = {
            "ADDRESS",
            "AU_TFN",
            "BANK_ACCOUNT_INFORMATION",
            "BANK_ACCOUNT_NUMBER",
            "DATE_OF_BIRTH",
            "EMAIL_ADDRESS",
            "EMPLOYEE_NUMBER",
            "FIRST_NAME",
            "GEOLOCATION_INFORMATION",
            "HOME_PHONE",
            "LAST_NAME",
            "LATITUDE",
            "LONGITUDE",
            "MEDICARE_EXPIRY",
            "MOBILE",
            "NON_PII",
            "NUMBER_PLATE",
            "PASSPORT_EXPIRY",
            "PASSPORT_START_DATE",
            "PAYMENT_CARD_NUMBER",
            "PERSON",
            "SOCIAL_MEDIA_ACCOUNT",
            "SOCIAL_MEDIA_ID",
            "STUDENT_ID",
            "USERNAME",
            "VEHICLE_REGO",
            "WORK_EMAIL",
            "WORK_PHONE",
        }

    def test_generate_full_teacher_prompts_uses_self_consistency(self):
        prompts, plan = generate_full_teacher_prompts(self.training_labels, base_example_count=2000, self_consistency=3)

        self.assertEqual(len(prompts), 6000)
        self.assertEqual(plan["base_example_count"], 2000)
        self.assertEqual(plan["self_consistency"], 3)
        self.assertEqual(plan["teacher_prompt_count"], 6000)
        self.assertEqual(plan["prompt_coverage"]["zero_example_labels"]["FIRST_NAME"], 300)
        self.assertEqual({1, 2, 3}, {prompt["self_consistency_index"] for prompt in prompts})

    def test_run_full_teacher_async_skips_completed_and_writes_progress(self):
        prompts, _plan = generate_full_teacher_prompts(self.training_labels, base_example_count=2000, self_consistency=3)
        prompts = prompts[:4]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.jsonl"
            progress_path = tmp_path / "progress.json"
            raw_path.write_text(json.dumps({"id": prompts[0]["id"], "status": "ok"}) + "\n", encoding="utf-8")

            async def fake_sender(prompt, payload, endpoint):
                verdicts = {label: "weak_against" for label in prompt["candidate_labels"]}
                verdicts["NON_PII"] = "strong_for"
                return json.dumps({"verdicts": verdicts})

            new_rows, runtime = asyncio.run(
                run_full_teacher_async(
                    prompts,
                    raw_path=raw_path,
                    progress_path=progress_path,
                    base_url="http://localhost:8000/v1",
                    model_name="test-model",
                    concurrency=2,
                    max_tokens=32,
                    timeout_seconds=5,
                    max_retries=1,
                    progress_interval=2,
                    sender=fake_sender,
                )
            )

            self.assertEqual(len(new_rows), 3)
            self.assertEqual(runtime["skipped_completed"], 1)
            self.assertTrue(progress_path.exists())
            self.assertEqual(sum(1 for line in raw_path.read_text().splitlines() if line.strip()), 4)


if __name__ == "__main__":
    unittest.main()
