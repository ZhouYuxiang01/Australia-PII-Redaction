#!/usr/bin/env python3
"""FastAPI live demo for Qwen LoRA PII redaction."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from qwen_redact import (
    apply_policy,
    build_response,
    load_json,
    normalize_text,
    parse_annotated_output,
    repair_offsets_to_input,
    safe_postprocess_spans,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = PROJECT_ROOT / "configs" / "policies" / "qwen-safe-default-v1.json"
DEFAULT_BASE_MODEL = Path("/home/admin/model/Qwen3.5-9B-Base")
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "qwen3_5_9b_base_lora_tagged_28_fastretry"
DEFAULT_FRONTEND = SCRIPT_DIR / "static" / "redaction_demo.html"
DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_PDF_OCR_PAGES = 8
DEFAULT_MAX_FILE_TEXT_CHARS = 12000
DEFAULT_MAX_TEXT_CHARS = 3000
DEFAULT_MAX_CONCURRENT_GENERATE = 4

TEXT_SUFFIXES = {".txt", ".text", ".md", ".csv", ".tsv", ".json", ".log"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


DEMO_EXAMPLES: list[dict[str, str]] = [
    {
        "id": "student-support",
        "title": "Student Support",
        "text": (
            "Student record: Olivia Okonkwo, DOB November 04, 1999, email "
            "olivia.okonkwo@example.edu.au, TFN 832 109 111."
        ),
        "annotated_output": (
            "Student record: <pii type=\"PERSON\">Olivia Okonkwo</pii>, DOB "
            "<pii type=\"DATE_OF_BIRTH\">November 04, 1999</pii>, email "
            "<pii type=\"EMAIL_ADDRESS\">olivia.okonkwo@example.edu.au</pii>, TFN "
            "<pii type=\"AU_TFN\">832 109 111</pii>."
        ),
    },
    {
        "id": "payroll-update",
        "title": "Payroll Update",
        "text": (
            "Payroll update for Marco Kowalski. BSB 062-001, account 123456789, "
            "salary $118,000, work email marco.kowalski@uni.example.edu.au."
        ),
        "annotated_output": (
            "Payroll update for <pii type=\"PERSON\">Marco Kowalski</pii>. BSB "
            "<pii type=\"BSB\">062-001</pii>, account "
            "<pii type=\"AU_BANK_ACCOUNT\">123456789</pii>, salary "
            "<pii type=\"SALARY\">$118,000</pii>, work email "
            "<pii type=\"WORK_EMAIL\">marco.kowalski@uni.example.edu.au</pii>."
        ),
    },
    {
        "id": "hard-negative",
        "title": "Hard Negative",
        "text": "Invoice reference 532799124 was sent to the warehouse queue.",
        "annotated_output": "Invoice reference 532799124 was sent to the warehouse queue.",
    },
]


class RedactRequest(BaseModel):
    text: str | None = Field(default=None, description="Input text to redact.")
    example_id: str | None = Field(default=None, description="Demo example id; the example text is sent through the live model.")


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def run_command(command: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Required OCR command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"OCR command timed out: {command[0]}") from exc


def upload_kind(filename: str, content_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    content_type = content_type or ""
    if suffix == ".pdf" or content_type == "application/pdf":
        return "pdf"
    if suffix in IMAGE_SUFFIXES or content_type.startswith("image/"):
        return "image"
    if suffix in TEXT_SUFFIXES or content_type.startswith("text/"):
        return "text"
    return "unsupported"


def decode_text_upload(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_text_with_tesseract(path: Path) -> str:
    if not command_exists("tesseract"):
        raise HTTPException(status_code=500, detail="Tesseract OCR is not installed on the server")
    result = run_command(["tesseract", str(path), "stdout", "-l", "eng"], timeout=180.0)
    if result.returncode != 0:
        detail = result.stderr.strip() or "Tesseract OCR failed"
        raise HTTPException(status_code=422, detail=detail[:500])
    return result.stdout


def extract_pdf_text_layer(path: Path) -> str:
    if not command_exists("pdftotext"):
        return ""
    result = run_command(["pdftotext", "-layout", str(path), "-"], timeout=60.0)
    if result.returncode != 0:
        return ""
    return result.stdout


def extract_pdf_text_by_ocr(path: Path, *, max_pages: int) -> tuple[str, int]:
    if not command_exists("pdftoppm"):
        raise HTTPException(status_code=500, detail="pdftoppm is not installed on the server")
    with tempfile.TemporaryDirectory(prefix="redact_pdf_ocr_") as tmpdir:
        prefix = Path(tmpdir) / "page"
        result = run_command(
            ["pdftoppm", "-r", "200", "-png", "-f", "1", "-l", str(max_pages), str(path), str(prefix)],
            timeout=180.0,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "PDF rendering failed"
            raise HTTPException(status_code=422, detail=detail[:500])
        page_paths = sorted(Path(tmpdir).glob("page-*.png"))
        page_texts = [extract_text_with_tesseract(page_path).strip() for page_path in page_paths]
        return "\n\n".join(text for text in page_texts if text), len(page_paths)


def extract_upload_text(
    *,
    filename: str,
    content_type: str | None,
    data: bytes,
    max_upload_bytes: int,
    max_pdf_ocr_pages: int,
    max_file_text_chars: int,
) -> dict[str, Any]:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {max_upload_bytes} byte upload limit")

    kind = upload_kind(filename, content_type)
    warnings: list[str] = []
    pages = 0
    text_source = ""
    if kind == "text":
        text = decode_text_upload(data)
        text_source = "text_upload"
    elif kind in {"image", "pdf"}:
        suffix = Path(filename or "").suffix.lower() or (".pdf" if kind == "pdf" else ".png")
        with tempfile.NamedTemporaryFile(prefix="redact_upload_", suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            if kind == "image":
                text = extract_text_with_tesseract(tmp_path)
                text_source = "image_ocr"
                pages = 1
            else:
                text = extract_pdf_text_layer(tmp_path)
                if text.strip():
                    text_source = "pdf_text_layer"
                else:
                    text, pages = extract_pdf_text_by_ocr(tmp_path, max_pages=max_pdf_ocr_pages)
                    text_source = "pdf_ocr"
                    if pages >= max_pdf_ocr_pages:
                        warnings.append(f"pdf_ocr_limited_to_first_{max_pdf_ocr_pages}_pages")
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a PDF, image, or plain text file.")

    text = normalize_text(text)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No readable text found in uploaded file")
    if len(text) > max_file_text_chars:
        text = text[:max_file_text_chars]
        warnings.append(f"text_truncated_to_{max_file_text_chars}_characters")
    return {
        "text": text,
        "kind": kind,
        "text_source": text_source,
        "pages": pages,
        "warnings": warnings,
    }


class LiveModelBackend:
    def __init__(
        self,
        *,
        base_model: Path,
        adapter_dir: Path,
        max_new_tokens: int = 512,
        max_concurrent_generate: int = DEFAULT_MAX_CONCURRENT_GENERATE,
    ) -> None:
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.max_new_tokens = max_new_tokens
        self.max_concurrent_generate = max(1, int(max_concurrent_generate))
        self._load_lock = threading.Lock()
        self._generate_semaphore = threading.BoundedSemaphore(self.max_concurrent_generate)
        self._loaded = False
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir, trust_remote_code=True, use_fast=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            base = AutoModelForCausalLM.from_pretrained(
                str(self.base_model),
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base, self.adapter_dir)
            model.eval()
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._loaded = True

    def generate(self, text: str, target_labels: list[str]) -> str:
        self.load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        system_prompt = (
            "You are an Australian PII redaction system. Return the input text with each supported PII span wrapped as "
            "<pii type=\"TYPE\">exact text</pii>. Preserve every character. Do not explain.\n"
            "Supported types:\n- " + "\n- ".join(target_labels)
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
        encoded = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if isinstance(encoded, self._torch.Tensor):
            encoded = {"input_ids": encoded}
        target_device = getattr(self._model, "device", None) or next(self._model.parameters()).device
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[-1]
        with self._generate_semaphore, self._torch.no_grad():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=[self._tokenizer.eos_token_id, self._tokenizer.convert_tokens_to_ids("<|im_end|>")],
            )
        generated_ids = output_ids[0][prompt_length:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def example_public_payload(example: dict[str, str]) -> dict[str, str]:
    return {"id": example["id"], "title": example["title"], "text": example["text"]}


def build_redaction_payload(
    *,
    text: str,
    annotated_output: str,
    policy: dict[str, Any],
    backend: str,
    started: float,
) -> dict[str, Any]:
    text = normalize_text(text)
    parsed_text, spans = parse_annotated_output(annotated_output)
    parsed_text = normalize_text(parsed_text)
    spans, repair_warnings, repaired = repair_offsets_to_input(text, parsed_text, spans)
    spans, post_warnings = safe_postprocess_spans(text, spans, policy)
    spans = apply_policy(spans, policy)
    payload = build_response(
        text=text,
        spans=spans,
        policy=policy,
        raw_offset_mapping_applied=repaired,
        warnings=[*repair_warnings, *post_warnings],
    )
    payload["input_text"] = text
    payload["model_output"] = annotated_output
    payload["demo"] = {"backend": backend, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
    return payload


def load_metrics_summary(project_root: Path) -> dict[str, Any]:
    candidates = {
        "cleaned_200_optimized": project_root
        / "scripts"
        / "eval_cleaned_200_report_9b"
        / "postprocessed_summary_optimized.json",
        "random1000_seed42": project_root
        / "outputs"
        / "qwen3_5_9b_base_lora_tagged_28_fastretry"
        / "processed_test_summary_optimized_random1000_seed42_20260422.json",
    }
    payload: dict[str, Any] = {}
    for name, path in candidates.items():
        if not path.exists():
            payload[name] = {"available": False, "path": str(path)}
            continue
        data = load_json(path)
        payload[name] = {
            "available": True,
            "path": str(path),
            "overall": data.get("overall") or data.get("postprocessed_value_level"),
            "rows_completed": data.get("rows_completed"),
            "rows_total": data.get("rows_total"),
        }
    return payload


def create_app(
    *,
    mode: str | None = None,
    policy_path: Path | None = None,
    base_model: Path | None = None,
    adapter_dir: Path | None = None,
    frontend_path: Path | None = None,
    live_backend: Any | None = None,
) -> FastAPI:
    mode = mode or os.getenv("DEMO_MODEL_MODE", "live")
    policy_path = policy_path or Path(os.getenv("DEMO_POLICY_PATH", str(DEFAULT_POLICY_PATH)))
    base_model = base_model or Path(os.getenv("DEMO_BASE_MODEL", str(DEFAULT_BASE_MODEL)))
    adapter_dir = adapter_dir or Path(os.getenv("DEMO_ADAPTER_DIR", str(DEFAULT_ADAPTER_DIR)))
    frontend_path = frontend_path or DEFAULT_FRONTEND
    policy = load_json(policy_path)
    target_labels = list(policy.get("type_actions", {}).keys())
    max_upload_bytes = int(os.getenv("DEMO_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)))
    max_pdf_ocr_pages = int(os.getenv("DEMO_MAX_PDF_OCR_PAGES", str(DEFAULT_MAX_PDF_OCR_PAGES)))
    max_file_text_chars = int(os.getenv("DEMO_MAX_FILE_TEXT_CHARS", str(DEFAULT_MAX_FILE_TEXT_CHARS)))
    max_text_chars = int(os.getenv("DEMO_MAX_TEXT_CHARS", str(DEFAULT_MAX_TEXT_CHARS)))
    max_concurrent_generate = int(os.getenv("DEMO_MAX_CONCURRENT_GENERATE", str(DEFAULT_MAX_CONCURRENT_GENERATE)))
    if live_backend is None:
        live_backend = LiveModelBackend(
            base_model=base_model,
            adapter_dir=adapter_dir,
            max_new_tokens=int(os.getenv("DEMO_MAX_NEW_TOKENS", "512")),
            max_concurrent_generate=max_concurrent_generate,
        )

    app = FastAPI(title="PII Redaction Demo", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        if not frontend_path.exists():
            raise HTTPException(status_code=404, detail="Frontend file not found")
        return frontend_path.read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": mode,
            "model_loaded": bool(getattr(live_backend, "loaded", False)),
            "max_concurrent_generate": int(getattr(live_backend, "max_concurrent_generate", max_concurrent_generate)),
            "policy_id": policy.get("policy_id"),
            "model_version": policy.get("model_version"),
        }

    @app.get("/api/examples")
    def examples() -> dict[str, Any]:
        return {"examples": [example_public_payload(item) for item in DEMO_EXAMPLES]}

    @app.get("/api/metrics-summary")
    def metrics_summary() -> dict[str, Any]:
        return load_metrics_summary(PROJECT_ROOT)

    @app.post("/api/redact")
    @app.post("/redact")
    def redact(request: RedactRequest) -> dict[str, Any]:
        started = time.perf_counter()
        example = next((item for item in DEMO_EXAMPLES if item["id"] == request.example_id), None)
        text = request.text or (example["text"] if example else None)
        if not text:
            raise HTTPException(status_code=400, detail="Provide text or example_id")
        if len(text) > max_text_chars:
            raise HTTPException(status_code=413, detail=f"Text input exceeds {max_text_chars} character limit")

        if mode != "live":
            raise HTTPException(status_code=503, detail="Live model is disabled; choose a cached example or restart the API in live mode")

        annotated_output = live_backend.generate(text, target_labels)
        return build_redaction_payload(
            text=text,
            annotated_output=annotated_output,
            policy=policy,
            backend="live_model",
            started=started,
        )

    @app.post("/api/redact-file")
    async def redact_file(file: UploadFile = File(...)) -> dict[str, Any]:
        started = time.perf_counter()
        data = await file.read()
        extracted = extract_upload_text(
            filename=file.filename or "upload",
            content_type=file.content_type,
            data=data,
            max_upload_bytes=max_upload_bytes,
            max_pdf_ocr_pages=max_pdf_ocr_pages,
            max_file_text_chars=max_file_text_chars,
        )
        text = extracted["text"]

        if mode != "live":
            raise HTTPException(status_code=503, detail="Live model is disabled; restart the API in live mode")

        annotated_output = live_backend.generate(text, target_labels)
        payload = build_redaction_payload(
            text=text,
            annotated_output=annotated_output,
            policy=policy,
            backend="live_model_file_upload",
            started=started,
        )
        payload["ocr_text"] = text
        payload["file"] = {
            "name": file.filename or "upload",
            "content_type": file.content_type,
            "kind": extracted["kind"],
            "size_bytes": len(data),
        }
        payload["ocr"] = {
            "text_source": extracted["text_source"],
            "pages": extracted["pages"],
            "warnings": extracted["warnings"],
        }
        if extracted["warnings"]:
            payload["warnings"] = [*payload.get("warnings", []), *extracted["warnings"]]
        return payload

    return app


app = create_app()
