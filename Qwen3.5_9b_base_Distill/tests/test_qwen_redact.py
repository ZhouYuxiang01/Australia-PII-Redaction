from __future__ import annotations

import json
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qwen_redact import (  # noqa: E402
    apply_policy,
    build_response,
    load_json,
    parse_annotated_output,
    redact_text,
    repair_offsets_to_input,
    safe_postprocess_spans,
)


POLICY = load_json(ROOT / "configs" / "policies" / "qwen-safe-default-v1.json")


class QwenRedactTests(unittest.TestCase):
    def test_parse_annotated_output_round_trips_offsets(self) -> None:
        output = 'Name: <pii type="PERSON">Alice Wong</pii> TFN: <pii type="AU_TFN">123 456 789</pii>'
        plain, spans = parse_annotated_output(output)
        self.assertEqual(plain, "Name: Alice Wong TFN: 123 456 789")
        self.assertEqual([(s.type, s.start, s.end, plain[s.start : s.end]) for s in spans], [
            ("PERSON", 6, 16, "Alice Wong"),
            ("AU_TFN", 22, 33, "123 456 789"),
        ])

    def test_parse_annotated_output_accepts_unquoted_type(self) -> None:
        output = "Name: <pii type=PERSON>Alice Wong</pii>"
        plain, spans = parse_annotated_output(output)
        self.assertEqual(plain, "Name: Alice Wong")
        self.assertEqual(spans[0].type, "PERSON")
        self.assertEqual((spans[0].start, spans[0].end), (6, 16))

    def test_parse_annotated_output_canonicalizes_short_model_types(self) -> None:
        output = 'Name: <pii type="NAME">Alice Wong</pii> TFN: <pii type="TFN">123 456 789</pii>'
        plain, spans = parse_annotated_output(output)
        self.assertEqual(plain, "Name: Alice Wong TFN: 123 456 789")
        self.assertEqual([span.type for span in spans], ["PERSON", "AU_TFN"])
        self.assertEqual(spans[0].postprocess, ["type_alias:NAME->PERSON"])
        self.assertEqual(spans[1].postprocess, ["type_alias:TFN->AU_TFN"])

    def test_safe_postprocess_strips_known_prefixes(self) -> None:
        text = "DOB: 25/06/1969. TFN: 832 109 111. ID: 405997905"
        output = '<pii type="DATE_OF_BIRTH">DOB: 25/06/1969</pii>. <pii type="AU_TFN">TFN: 832 109 111</pii>. <pii type="STUDENT_ID">ID: 405997905</pii>'
        plain, spans = parse_annotated_output(output)
        self.assertEqual(plain, text)
        processed, warnings = safe_postprocess_spans(text, spans, POLICY)
        self.assertEqual(warnings, [])
        by_type = {span.type: span for span in processed}
        self.assertEqual(by_type["DATE_OF_BIRTH"].value, "25/06/1969")
        self.assertEqual(text[by_type["DATE_OF_BIRTH"].start : by_type["DATE_OF_BIRTH"].end], "25/06/1969")
        self.assertEqual(by_type["AU_TFN"].value, "832 109 111")
        self.assertEqual(by_type["STUDENT_ID"].value, "405997905")

    def test_safe_postprocess_keeps_student_id_prefix_when_model_does_not_include_generic_id(self) -> None:
        text = "Student ID 465726169"
        output = '<pii type="STUDENT_ID">Student ID 465726169</pii>'
        _, spans = parse_annotated_output(output)
        processed, _ = safe_postprocess_spans(text, spans, POLICY)
        self.assertEqual(processed[0].value, "Student ID 465726169")
        self.assertEqual(processed[0].start, 0)

    def test_collapses_generic_work_phone_context(self) -> None:
        text = "Phone: 0412146668"
        output = 'Phone: <pii type="WORK_PHONE">0412146668</pii>'
        _, spans = parse_annotated_output(output)
        processed, _ = safe_postprocess_spans(text, spans, POLICY)
        self.assertEqual(processed[0].type, "AU_PHONE")

    def test_adds_url_encoded_email_span(self) -> None:
        text = "Reset: https://example/reset?user=mei-hassan%40gmail.com"
        output = text
        _, spans = parse_annotated_output(output)
        processed, _ = safe_postprocess_spans(text, spans, POLICY)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].type, "EMAIL_ADDRESS")
        self.assertEqual(text[processed[0].start : processed[0].end], "mei-hassan%40gmail.com")

    def test_repair_offsets_on_roundtrip_mismatch_when_value_unique(self) -> None:
        text = "Name: Alice Wong"
        generated = "Record: <pii type=\"PERSON\">Alice Wong</pii>"
        plain, spans = parse_annotated_output(generated)
        repaired, warnings, did_repair = repair_offsets_to_input(text, plain, spans)
        self.assertTrue(did_repair)
        self.assertIn("round_trip_mismatch_offsets_repaired_when_unique", warnings)
        self.assertEqual(repaired[0].start, 6)
        self.assertEqual(repaired[0].end, 16)

    def test_redact_text_replace_and_schema_payload(self) -> None:
        text = "Name: Alice Wong TFN: 123 456 789"
        output = 'Name: <pii type="PERSON">Alice Wong</pii> TFN: <pii type="AU_TFN">123 456 789</pii>'
        _, spans = parse_annotated_output(output)
        spans, _ = safe_postprocess_spans(text, spans, POLICY)
        spans = apply_policy(spans, POLICY)
        self.assertEqual(redact_text(text, spans), "Name: [PERSON] TFN: [AU_TFN]")
        payload = build_response(text=text, spans=spans, policy=POLICY, raw_offset_mapping_applied=False, warnings=[])
        self.assertEqual(payload["redacted_text"], "Name: [PERSON] TFN: [AU_TFN]")
        self.assertEqual(payload["spans"][0]["confidence"], None)
        self.assertEqual(payload["metadata"]["policy_id"], "qwen-safe-default-v1")
        self.assertIn("confidence_uncalibrated_null", payload["warnings"])

    def test_cli_writes_json(self) -> None:
        text = "Name: Alice Wong"
        annotated = 'Name: <pii type="PERSON">Alice Wong</pii>'
        tmp_root = ROOT / ".tmp-tests"
        tmp_root.mkdir(exist_ok=True)
        out = tmp_root / "qwen_redact_cli_out.json"
        out.unlink(missing_ok=True)
        try:
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "qwen_redact.py"),
                    "--text",
                    text,
                    "--annotated-output",
                    annotated,
                    "--policy",
                    str(ROOT / "configs" / "policies" / "qwen-safe-default-v1.json"),
                    "--json-out",
                    str(out),
                ],
                check=True,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["redacted_text"], "Name: [PERSON]")
        finally:
            out.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
