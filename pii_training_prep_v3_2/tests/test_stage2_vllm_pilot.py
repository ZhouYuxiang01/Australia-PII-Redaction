import asyncio
import json
import unittest

from pii_prep.stage2_vllm_pilot import (
    generate_pilot_prompts,
    run_vllm_concurrent_outputs,
    score_raw_rows,
    summarize_prompt_coverage,
)


class Stage2VllmPilotTests(unittest.TestCase):
    def setUp(self):
        self.training_labels = {
            "ADDRESS",
            "AU_TFN",
            "BANK_ACCOUNT_INFORMATION",
            "BANK_ACCOUNT_NUMBER",
            "CREDIT_CARD_EXPIRY",
            "DATE_OF_BIRTH",
            "EMAIL_ADDRESS",
            "EMPLOYEE_NUMBER",
            "FIRST_NAME",
            "GEOLOCATION_INFORMATION",
            "HOME_PHONE",
            "IHI",
            "LAST_NAME",
            "LATITUDE",
            "LONGITUDE",
            "MEDICARE_EXPIRY",
            "MEDICARE_NUMBER",
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

    def test_generate_pilot_prompts_covers_required_categories(self):
        prompts = generate_pilot_prompts(self.training_labels, total=200)
        coverage = summarize_prompt_coverage(prompts)

        self.assertEqual(len(prompts), 200)
        for context_type in [
            "bare_span",
            "weak_context",
            "strong_positive_context",
            "reverse_negative_context",
        ]:
            self.assertGreater(coverage["context_types"].get(context_type, 0), 0)
        for label in ["BANK_ACCOUNT_INFORMATION", "FIRST_NAME", "HOME_PHONE", "LAST_NAME"]:
            self.assertEqual(coverage["zero_example_labels"].get(label), 20)
        self.assertGreaterEqual(coverage["candidate_level_ambiguous_negative_count"], 1)
        self.assertTrue(all("NON_PII" in prompt["candidate_labels"] for prompt in prompts))
        self.assertEqual(set(prompt["id"] for prompt in prompts), {f"STAGE2-VLLM-PILOT-{i:03d}" for i in range(1, 201)})

    def test_bare_full_date_prompt_uses_compatible_date_candidates(self):
        prompts = generate_pilot_prompts(self.training_labels, total=200)
        date_prompt = next(prompt for prompt in prompts if prompt["ambiguity_group"] == "date" and prompt["context_type"] == "bare_span")

        self.assertIn("DATE_OF_BIRTH", date_prompt["candidate_labels"])
        self.assertIn("PASSPORT_EXPIRY", date_prompt["candidate_labels"])
        self.assertIn("PASSPORT_START_DATE", date_prompt["candidate_labels"])
        self.assertIn("MEDICARE_EXPIRY", date_prompt["candidate_labels"])
        self.assertIn("NON_PII", date_prompt["candidate_labels"])
        self.assertNotIn("CREDIT_CARD_EXPIRY", date_prompt["candidate_labels"])

    def test_concurrent_outputs_score_with_fake_sender(self):
        prompts = generate_pilot_prompts(self.training_labels, total=40)[:6]

        async def fake_sender(prompt, payload, endpoint):
            verdicts = {label: "neutral" for label in prompt["candidate_labels"]}
            verdicts[prompt["candidate_labels"][0]] = "strong_for"
            if "NON_PII" in verdicts and prompt.get("pilot_category") == "candidate_level_ambiguous_negative":
                verdicts["NON_PII"] = "strong_for"
            return json.dumps({"verdicts": verdicts})

        raw_rows, wall_time = asyncio.run(
            run_vllm_concurrent_outputs(
                prompts,
                concurrency=4,
                model_name="test-model",
                max_tokens=64,
                sender=fake_sender,
            )
        )
        converted, metrics, errors = score_raw_rows(
            raw_rows,
            self.training_labels,
            wall_time_seconds=wall_time,
            concurrency=4,
        )

        self.assertEqual(len(raw_rows), 6)
        self.assertEqual(len(converted), 6)
        self.assertEqual(metrics["valid_json_outputs"], 6)
        self.assertEqual(metrics["validation_error_count"], 0)
        self.assertEqual(metrics["labels_outside_training_space"], {})
        self.assertEqual(metrics["timeout_count"], 0)
        self.assertEqual(metrics["retry_count"], 0)
        self.assertGreater(metrics["requests_per_minute"], 0)
        self.assertEqual(errors["malformed_json_count"], 0)


if __name__ == "__main__":
    unittest.main()
