import unittest
from pathlib import Path


FRONTEND_PATH = Path(__file__).resolve().parent / "static" / "redaction_demo.html"


class RedactionDemoFrontendTests(unittest.TestCase):
    def setUp(self):
        self.html = FRONTEND_PATH.read_text(encoding="utf-8")

    def test_metrics_cards_are_removed_from_demo_page(self):
        self.assertNotIn("/api/metrics-summary", self.html)
        self.assertNotIn("Random-1000", self.html)
        self.assertNotIn("Precision", self.html)
        self.assertNotIn("Recall", self.html)

    def test_run_model_has_thinking_state(self):
        self.assertIn("thinking...", self.html)
        self.assertIn("setBusy(true)", self.html)

    def test_redacted_display_uses_star_masking(self):
        self.assertIn("function maskSensitiveText", self.html)
        self.assertIn("redactedEl.textContent = maskSensitiveText", self.html)
        self.assertNotIn("redactedEl.textContent = payload.redacted_text", self.html)

    def test_file_upload_controls_call_redact_file_endpoint(self):
        self.assertIn('id="file-input"', self.html)
        self.assertIn("OCR Text", self.html)
        self.assertIn("FormData", self.html)
        self.assertIn('fetch("/api/redact-file"', self.html)
        self.assertIn("function renderFilePayload", self.html)

    def test_demo_example_buttons_are_removed(self):
        self.assertNotIn('id="examples"', self.html)
        self.assertNotIn("/api/examples", self.html)
        self.assertNotIn("selectedExample", self.html)
        self.assertNotIn("example_id", self.html)
        self.assertNotIn("Student Support", self.html)
        self.assertNotIn("Payroll Update", self.html)
        self.assertNotIn("Hard Negative", self.html)

    def test_textarea_has_max_length_and_counter(self):
        self.assertIn("MAX_TEXT_CHARS = 3000", self.html)
        self.assertIn('maxlength="3000"', self.html)
        self.assertIn('id="char-count"', self.html)
        self.assertIn("updateCharCount", self.html)


if __name__ == "__main__":
    unittest.main()
