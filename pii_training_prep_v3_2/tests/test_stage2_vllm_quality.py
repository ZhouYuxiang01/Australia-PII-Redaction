import unittest

from pii_prep.stage2_vllm_quality import analyze_quality


class Stage2VllmQualityTests(unittest.TestCase):
    def test_quality_report_summarizes_contexts_and_warnings(self):
        converted_rows = [
            {
                "id": "bare-date",
                "context_type": "bare_span",
                "span_value": "04/05/1998",
                "candidate_labels": ["DATE_OF_BIRTH", "NON_PII"],
                "top_type": "DATE_OF_BIRTH",
                "type_distribution": {"DATE_OF_BIRTH": 0.8, "NON_PII": 0.2},
            },
            {
                "id": "weak",
                "context_type": "weak_context",
                "span_value": "123456",
                "candidate_labels": ["STUDENT_ID", "NON_PII"],
                "top_type": "STUDENT_ID",
                "type_distribution": {"STUDENT_ID": 0.9, "NON_PII": 0.1},
            },
            {
                "id": "strong-low",
                "context_type": "strong_positive_context",
                "span_value": "Mia",
                "candidate_labels": ["FIRST_NAME", "PERSON", "NON_PII"],
                "top_type": "FIRST_NAME",
                "type_distribution": {"FIRST_NAME": 0.49, "PERSON": 0.31, "NON_PII": 0.2},
            },
            {
                "id": "reverse-fail",
                "context_type": "reverse_negative_context",
                "span_value": "ABC123",
                "candidate_labels": ["VEHICLE_REGO", "NON_PII"],
                "top_type": "VEHICLE_REGO",
                "type_distribution": {"VEHICLE_REGO": 0.75, "NON_PII": 0.25},
            },
            {
                "id": "zero-bank",
                "context_type": "strong_positive_context",
                "ambiguity_group": "zero_example_bank_account_information",
                "span_value": "bank account information",
                "candidate_labels": ["BANK_ACCOUNT_INFORMATION", "NON_PII"],
                "top_type": "BANK_ACCOUNT_INFORMATION",
                "type_distribution": {"BANK_ACCOUNT_INFORMATION": 0.75, "NON_PII": 0.25},
            },
            {
                "id": "hard-negative-fail",
                "context_type": "hard_negative_context",
                "span_value": "192.168.1.1",
                "candidate_labels": ["IP_ADDRESS", "NON_PII"],
                "top_type": "IP_ADDRESS",
                "type_distribution": {"IP_ADDRESS": 0.7, "NON_PII": 0.3},
            },
        ]

        report, warnings = analyze_quality(converted_rows)

        self.assertEqual(report["record_count"], 6)
        self.assertEqual(report["malformed_converted_record_count"], 0)
        self.assertEqual(report["by_context_type"]["bare_span"]["top_probability"]["max"], 0.8)
        self.assertEqual(report["by_context_type"]["hard_negative"]["count"], 1)
        self.assertEqual(report["warning_counts"]["bare_span_overconfident_count"], 1)
        self.assertEqual(report["warning_counts"]["bare_date_dob_overconfident_count"], 1)
        self.assertEqual(report["warning_counts"]["weak_context_overconfident_count"], 1)
        self.assertEqual(report["warning_counts"]["reverse_negative_non_pii_failure_count"], 1)
        self.assertEqual(report["warning_counts"]["hard_negative_non_pii_failure_count"], 1)
        self.assertEqual(report["warning_counts"]["strong_positive_not_confident_count"], 1)
        self.assertEqual(report["zero_example_label_coverage"]["BANK_ACCOUNT_INFORMATION"]["count"], 1)
        self.assertEqual(warnings["bare_date_dob_overconfident"][0]["id"], "bare-date")
        self.assertIn("VEHICLE_REGO", report["label_distribution"])


if __name__ == "__main__":
    unittest.main()
