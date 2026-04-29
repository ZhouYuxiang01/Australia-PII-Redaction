"""FastAPI server. Model-agnostic — pick a backend via env / create_app args.

Environment variables:
  WRAPPER_BACKEND_CONFIG : path to backends/<backend>.json (required)
  WRAPPER_POLICY_CONFIG  : path to policies/<policy>.json  (required)
  WRAPPER_FRONTEND       : path to static/redaction_demo.html  (optional)
  WRAPPER_MAX_UPLOAD_BYTES (default 25 MB)
  WRAPPER_MAX_PDF_OCR_PAGES (default 8)
  WRAPPER_MAX_FILE_TEXT_CHARS (default 12000)
  WRAPPER_MAX_TEXT_CHARS (default 3000)

Endpoints:
  GET  /                 -> static demo page
  GET  /api/health       -> service + backend info
  GET  /api/examples     -> demo examples
  POST /api/redact       -> redact a text string
  POST /api/redact-file  -> redact a file (text/image/pdf with OCR)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..backends import RedactionBackend, build_backend_from_path
from ..core.normalize import normalize_text
from ..core.policy import apply_policy, build_response, load_json
from ..core.postprocess import safe_postprocess_spans
from ..ocr import OcrError, extract_upload_text


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRONTEND = PACKAGE_ROOT / "static" / "redaction_demo.html"

DEFAULT_DEMO_EXAMPLES: list[dict[str, str]] = [
    {
        "id": "student-support",
        "title": "Student Support",
        "text": (
            "Student record: Olivia Okonkwo, DOB November 04, 1999, email "
            "olivia.okonkwo@example.edu.au, TFN 832 109 111."
        ),
    },
    {
        "id": "payroll-update",
        "title": "Payroll Update",
        "text": (
            "Payroll update for Marco Kowalski. BSB 062-001, account 123456789, "
            "salary $118,000, work email marco.kowalski@uni.example.edu.au."
        ),
    },
    {
        "id": "hard-negative",
        "title": "Hard Negative",
        "text": "Invoice reference 532799124 was sent to the warehouse queue.",
    },
]


class RedactRequest(BaseModel):
    text: str | None = Field(default=None, description="Input text to redact.")
    example_id: str | None = Field(default=None, description="Demo example id.")


def _load_examples(path: Path | None) -> list[dict[str, str]]:
    if path is None or not Path(path).exists():
        return list(DEFAULT_DEMO_EXAMPLES)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_payload(
    *,
    text: str,
    backend: RedactionBackend,
    policy: dict[str, Any],
    started: float,
    backend_label: str,
) -> dict[str, Any]:
    text = normalize_text(text)
    spans, diag = backend.detect_spans(text)
    spans, post_warnings = safe_postprocess_spans(text, spans, policy)
    spans = apply_policy(spans, policy)
    payload = build_response(
        text=text, spans=spans, policy=policy,
        raw_offset_mapping_applied=bool(diag.get("raw_offset_mapping_applied", False)),
        warnings=[*diag.get("warnings", []), *post_warnings],
        extra_metadata={
            "backend_name": backend.name,
            "backend_route": backend_label,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "input_length": len(text),
        },
    )
    return payload


def create_app(
    *,
    backend: RedactionBackend | None = None,
    backend_config_path: str | Path | None = None,
    policy_path: str | Path | None = None,
    frontend_path: str | Path | None = None,
    examples_path: str | Path | None = None,
) -> FastAPI:
    backend_config_path = backend_config_path or os.getenv("WRAPPER_BACKEND_CONFIG")
    policy_path = policy_path or os.getenv("WRAPPER_POLICY_CONFIG")
    frontend_path = Path(frontend_path or os.getenv("WRAPPER_FRONTEND") or DEFAULT_FRONTEND)
    examples_path = examples_path or os.getenv("WRAPPER_EXAMPLES")

    if backend is None:
        if backend_config_path is None:
            raise RuntimeError(
                "No backend supplied. Set WRAPPER_BACKEND_CONFIG to a JSON file "
                "or pass backend / backend_config_path to create_app()."
            )
        backend = build_backend_from_path(backend_config_path)

    if policy_path is None:
        raise RuntimeError(
            "No policy supplied. Set WRAPPER_POLICY_CONFIG to a JSON file "
            "or pass policy_path to create_app()."
        )
    policy = load_json(policy_path)

    examples = _load_examples(Path(examples_path) if examples_path else None)
    examples_lookup = {e["id"]: e for e in examples}

    max_upload_bytes = int(os.getenv("WRAPPER_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
    max_pdf_ocr_pages = int(os.getenv("WRAPPER_MAX_PDF_OCR_PAGES", "8"))
    max_file_text_chars = int(os.getenv("WRAPPER_MAX_FILE_TEXT_CHARS", "12000"))
    max_text_chars = int(os.getenv("WRAPPER_MAX_TEXT_CHARS", "3000"))

    app = FastAPI(title="PII Redaction Wrapper", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        if not frontend_path.exists():
            raise HTTPException(status_code=404, detail="Frontend file not found")
        return frontend_path.read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": backend.info(),
            "policy_id": policy.get("policy_id"),
            "model_version": policy.get("model_version") or backend.model_version,
            "schema_version": policy.get("schema_version", "redaction-output-v1"),
        }

    @app.get("/api/examples")
    def list_examples() -> dict[str, Any]:
        return {"examples": [{"id": e["id"], "title": e["title"], "text": e["text"]} for e in examples]}

    @app.post("/api/redact")
    @app.post("/redact")
    def redact(request: RedactRequest) -> dict[str, Any]:
        started = time.perf_counter()
        example = examples_lookup.get(request.example_id) if request.example_id else None
        text = request.text or (example["text"] if example else None)
        if not text:
            raise HTTPException(status_code=400, detail="Provide text or example_id")
        if len(text) > max_text_chars:
            raise HTTPException(status_code=413, detail=f"Text input exceeds {max_text_chars} character limit")
        return _build_payload(
            text=text, backend=backend, policy=policy,
            started=started, backend_label="text_input",
        )

    @app.post("/api/redact-file")
    async def redact_file(file: UploadFile = File(...)) -> dict[str, Any]:
        started = time.perf_counter()
        data = await file.read()
        try:
            extracted = extract_upload_text(
                filename=file.filename or "upload",
                content_type=file.content_type,
                data=data,
                max_upload_bytes=max_upload_bytes,
                max_pdf_ocr_pages=max_pdf_ocr_pages,
                max_file_text_chars=max_file_text_chars,
            )
        except OcrError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        text = extracted["text"]
        payload = _build_payload(
            text=text, backend=backend, policy=policy,
            started=started, backend_label="file_upload",
        )
        payload["metadata"]["file"] = {
            "name": file.filename or "upload",
            "content_type": file.content_type,
            "kind": extracted["kind"],
            "size_bytes": len(data),
        }
        payload["metadata"]["ocr"] = {
            "text_source": extracted["text_source"],
            "pages": extracted["pages"],
            "warnings": extracted["warnings"],
        }
        if extracted["warnings"]:
            payload["warnings"] = [*payload.get("warnings", []), *extracted["warnings"]]
        return payload

    return app


# uvicorn entrypoint
app = None  # populated lazily by `uvicorn redaction.api.server:get_app`


def get_app() -> FastAPI:
    """uvicorn factory entrypoint."""
    global app
    if app is None:
        app = create_app()
    return app
