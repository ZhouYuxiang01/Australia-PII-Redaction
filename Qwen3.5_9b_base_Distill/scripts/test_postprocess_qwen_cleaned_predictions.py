import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from postprocess_qwen_cleaned_predictions import evaluate, normalize_cleaned_pair, postprocess_row


class LoraPostprocessTests(unittest.TestCase):
    def test_normalize_cleaned_pair_applies_format_rules_to_gold_and_predictions(self):
        self.assertEqual(normalize_cleaned_pair("AU_TFN", "TFN: 832 109 111"), "832109111")
        self.assertEqual(normalize_cleaned_pair("AU_PHONE", "(0473) 607-078"), "0473607078")
        self.assertEqual(
            normalize_cleaned_pair("ADDRESS", "Level 7, 34 King Street, Alice Springs NT 0870"),
            "level 7 34 king street alice springs nt 0870",
        )
        self.assertEqual(normalize_cleaned_pair("STUDENT_ID", "Student ID: SID: 570032552"), "570032552")
        self.assertEqual(normalize_cleaned_pair("STUDENT_ID", "student id 438161142"), "438161142")
        self.assertEqual(normalize_cleaned_pair("DATE_OF_BIRTH", "November 04"), "november 4")

    def test_postprocess_keeps_next_of_kin_phone_when_model_predicts_it(self):
        row = {
            "text": "Record for Ingrid Garcia. Next of kin: Anh Hassan (spouse) (0467 419 919).",
            "pred_spans": [
                {"type": "PERSON", "start": 11, "end": 24, "value": "Ingrid Garcia"},
                {"type": "AU_PHONE", "start": 59, "end": 73, "value": "0467 419 919"},
            ],
        }

        pairs = postprocess_row(row, add_date_variants=True, collapse_work_contact=True, add_encoded_emails=True)

        self.assertEqual(pairs, {("PERSON", "ingrid garcia"), ("AU_PHONE", "0467419919")})

    def test_postprocess_preserves_complete_month_date(self):
        row = {
            "text": "born September 13, 1966",
            "pred_spans": [
                {"type": "DATE_OF_BIRTH", "start": 5, "end": 23, "value": "September 13, 1966"},
            ],
        }

        pairs = postprocess_row(row, add_date_variants=True, collapse_work_contact=True, add_encoded_emails=True)

        self.assertEqual(pairs, {("DATE_OF_BIRTH", "september 13, 1966")})

    def test_postprocess_preserves_work_contact_types(self):
        row = {
            "text": "Work email zoe.campbell@unsw.edu.au and work phone 0772196968.",
            "pred_spans": [
                {"type": "WORK_EMAIL", "start": 11, "end": 38, "value": "zoe.campbell@unsw.edu.au"},
                {"type": "WORK_PHONE", "start": 54, "end": 64, "value": "0772196968"},
            ],
        }

        pairs = postprocess_row(row, add_date_variants=True, collapse_work_contact=True, add_encoded_emails=True)

        self.assertEqual(
            pairs,
            {
                ("WORK_EMAIL", "zoe.campbell@unsw.edu.au"),
                ("WORK_PHONE", "0772196968"),
            },
        )

    def test_postprocess_keeps_standalone_driver_licence_state(self):
        row = {
            "text": "driver licence ACT 1640717",
            "pred_spans": [
                {"type": "AU_DRIVERS_LICENCE", "start": 15, "end": 18, "value": "ACT"},
                {"type": "AU_DRIVERS_LICENCE", "start": 19, "end": 26, "value": "1640717"},
            ],
        }

        pairs = postprocess_row(row, add_date_variants=True, collapse_work_contact=True, add_encoded_emails=True)

        self.assertEqual(pairs, {("AU_DRIVERS_LICENCE", "act"), ("AU_DRIVERS_LICENCE", "1640717")})

    def test_postprocess_does_not_add_unpredicted_encoded_email(self):
        row = {
            "text": "Callback URL contains alice%40example.edu.au but the model predicted nothing.",
            "pred_spans": [],
        }

        pairs = postprocess_row(row, add_date_variants=True, collapse_work_contact=True, add_encoded_emails=True)

        self.assertEqual(pairs, set())

    def test_postprocess_keeps_model_person_value(self):
        row = {
            "text": "Record for O. Okonkwo (full name: Olivia Okonkwo).",
            "pred_spans": [
                {"type": "PERSON", "start": 11, "end": 21, "value": "O. Okonkwo"},
            ],
        }

        pairs = postprocess_row(row, add_date_variants=True, collapse_work_contact=True, add_encoded_emails=True)

        self.assertEqual(pairs, {("PERSON", "o. okonkwo")})

    def test_evaluator_normalizes_gold_and_prediction_pairs(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            predictions = tmp_path / "predictions.jsonl"
            summary = tmp_path / "summary.json"
            predictions_out = tmp_path / "predictions_out.jsonl"
            json_out = tmp_path / "summary_out.json"
            md_out = tmp_path / "summary_out.md"
            predictions.write_text(
                (
                    '{"id":"x","difficulty":"EASY","text":"TFN: 832 109 111",'
                    '"gt_pairs":[["AU_TFN","832 109 111"]],'
                    '"pred_pairs":[["AU_TFN","TFN: 832 109 111"]],'
                    '"pred_spans":[{"type":"AU_TFN","start":0,"end":16,"value":"TFN: 832 109 111"}],'
                    '"fp_pairs":[["AU_TFN","TFN: 832 109 111"]],'
                    '"fn_pairs":[["AU_TFN","832 109 111"]]}'
                )
                + "\n",
                encoding="utf-8",
            )
            summary.write_text('{"trained_types":["AU_TFN"],"overall":{"precision":0,"recall":0,"f1":0,"sample_exact_acc":0,"tp":0,"fp":1,"fn":1}}', encoding="utf-8")

            payload = evaluate(
                Namespace(
                    predictions=predictions,
                    summary=summary,
                    predictions_out=predictions_out,
                    json_out=json_out,
                    md_out=md_out,
                    add_date_variants=True,
                    collapse_work_contact=True,
                    add_encoded_emails=True,
                )
            )

        self.assertEqual(payload["overall"]["tp"], 1)
        self.assertEqual(payload["overall"]["fp"], 0)
        self.assertEqual(payload["overall"]["fn"], 0)


if __name__ == "__main__":
    unittest.main()
