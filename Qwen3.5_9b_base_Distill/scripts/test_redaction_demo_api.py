import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from redaction_demo_api import LiveModelBackend, create_app


class FakeLiveBackend:
    def __init__(self) -> None:
        self.loaded = True
        self.max_concurrent_generate = 4
        self.calls: list[str] = []

    def generate(self, text: str, target_labels: list[str]) -> str:
        self.calls.append(text)
        return (
            text.replace("Alice Nguyen", '<pii type="PERSON">Alice Nguyen</pii>')
            .replace("alice.nguyen@example.edu.au", '<pii type="EMAIL_ADDRESS">alice.nguyen@example.edu.au</pii>')
            .replace("Olivia Okonkwo", '<pii type="PERSON">Olivia Okonkwo</pii>')
            .replace("832 109 111", '<pii type="AU_TFN">832 109 111</pii>')
        )


def write_policy(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "policy_id": "demo-test-policy",
                "model_version": "demo-test-model",
                "taxonomy_version": "demo-test-taxonomy",
                "schema_version": "redaction-output-v1",
                "redaction_mode": "replace_with_tag",
                "default_action": "AUTO_REDACT",
                "confidence": {"calibrated": False, "default_value": None},
                "type_actions": {
                    "PERSON": "AUTO_REDACT",
                    "EMAIL_ADDRESS": "AUTO_REDACT",
                    "AU_TFN": "AUTO_REDACT",
                },
                "postprocess": {
                    "strip_known_prefixes": True,
                    "collapse_generic_work_contacts": True,
                    "add_url_encoded_emails": True,
                },
            }
        ),
        encoding="utf-8",
    )


class RedactionDemoApiTests(unittest.TestCase):
    def make_client(self, backend: FakeLiveBackend | None = None) -> tuple[TestClient, FakeLiveBackend]:
        self.tmp = TemporaryDirectory()
        policy_path = Path(self.tmp.name) / "policy.json"
        write_policy(policy_path)
        backend = backend or FakeLiveBackend()
        return TestClient(create_app(mode="live", policy_path=policy_path, live_backend=backend)), backend

    def tearDown(self) -> None:
        tmp = getattr(self, "tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def test_health_reports_live_backend(self):
        client, _ = self.make_client()

        response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["mode"], "live")
        self.assertIs(payload["model_loaded"], True)
        self.assertEqual(payload["max_concurrent_generate"], 4)

    def test_live_backend_defaults_to_four_concurrent_generations(self):
        backend = LiveModelBackend(base_model=Path("/tmp/base"), adapter_dir=Path("/tmp/adapter"))

        self.assertEqual(backend.max_concurrent_generate, 4)

    def test_examples_are_input_text_only(self):
        client, _ = self.make_client()

        response = client.get("/api/examples")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(len(payload["examples"]), 2)
        self.assertTrue({"id", "title", "text"}.issubset(payload["examples"][0]))
        self.assertNotIn("annotated_output", payload["examples"][0])

    def test_redact_text_uses_live_backend_and_returns_schema(self):
        backend = FakeLiveBackend()
        client, _ = self.make_client(backend)

        response = client.post(
            "/api/redact",
            json={"text": "Please contact Alice Nguyen at alice.nguyen@example.edu.au."},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["demo"]["backend"], "live_model")
        self.assertEqual(backend.calls, ["Please contact Alice Nguyen at alice.nguyen@example.edu.au."])
        self.assertEqual(payload["redacted_text"], "Please contact [PERSON] at [EMAIL_ADDRESS].")
        self.assertEqual(payload["metadata"]["schema_version"], "redaction-output-v1")
        self.assertEqual(payload["warnings"], ["confidence_uncalibrated_null"])
        self.assertEqual(
            [(span["type"], span["start"], span["end"]) for span in payload["spans"]],
            [("PERSON", 15, 27), ("EMAIL_ADDRESS", 31, 58)],
        )

    def test_example_id_uses_live_backend_not_cached_output(self):
        backend = FakeLiveBackend()
        client, _ = self.make_client(backend)
        example_id = client.get("/api/examples").json()["examples"][0]["id"]

        response = client.post("/api/redact", json={"example_id": example_id})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["demo"]["backend"], "live_model")
        self.assertEqual(len(backend.calls), 1)
        self.assertIn("Olivia Okonkwo", backend.calls[0])
        self.assertIn("[PERSON]", payload["redacted_text"])
        self.assertIn("[AU_TFN]", payload["redacted_text"])

    def test_redact_requires_text_or_example_id(self):
        client, _ = self.make_client()

        response = client.post("/api/redact", json={})

        self.assertEqual(response.status_code, 400)

    def test_redact_rejects_text_over_max_length(self):
        backend = FakeLiveBackend()
        client, _ = self.make_client(backend)

        response = client.post("/api/redact", json={"text": "x" * 3001})

        self.assertEqual(response.status_code, 413)
        self.assertEqual(backend.calls, [])
        self.assertIn("3000", response.json()["detail"])

    def test_redact_file_uses_uploaded_text_and_returns_ocr_metadata(self):
        backend = FakeLiveBackend()
        client, _ = self.make_client(backend)
        text = "Please contact Alice Nguyen at alice.nguyen@example.edu.au."

        response = client.post(
            "/api/redact-file",
            files={"file": ("note.txt", text.encode("utf-8"), "text/plain")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(backend.calls, [text])
        self.assertEqual(payload["file"]["name"], "note.txt")
        self.assertEqual(payload["file"]["kind"], "text")
        self.assertEqual(payload["ocr"]["text_source"], "text_upload")
        self.assertEqual(payload["ocr_text"], text)
        self.assertEqual(payload["input_text"], text)
        self.assertEqual(payload["redacted_text"], "Please contact [PERSON] at [EMAIL_ADDRESS].")
        self.assertEqual(
            [(span["type"], span["start"], span["end"]) for span in payload["spans"]],
            [("PERSON", 15, 27), ("EMAIL_ADDRESS", 31, 58)],
        )

    def test_redact_file_rejects_unsupported_upload_type(self):
        client, _ = self.make_client()

        response = client.post(
            "/api/redact-file",
            files={"file": ("archive.zip", b"not supported", "application/zip")},
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
