"""Qwen-VL transcription + plain-text extraction for uploaded files.

Pulled out of the original Qwen demo server so any backend can use it.

Supported inputs:
  - text: txt / md / csv / tsv / json / log
  - image: png / jpg / jpeg / tif / tiff / bmp / webp (Qwen-VL transcription)
  - pdf: pdftotext text layer first, fall back to per-page Qwen-VL transcription
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from ..core.normalize import normalize_text


TEXT_SUFFIXES = {".txt", ".text", ".md", ".csv", ".tsv", ".json", ".log"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

_QWEN_VL_LOCK = threading.Lock()
_QWEN_VL_MODEL: Any | None = None
_QWEN_VL_PROCESSOR: Any | None = None


class OcrError(RuntimeError):
    """Raised when OCR or PDF extraction fails. Carries an HTTP status hint."""

    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def _run_command(command: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise OcrError(f"Required command not found: {command[0]}", status_code=500) from exc
    except subprocess.TimeoutExpired as exc:
        raise OcrError(f"Command timed out: {command[0]}", status_code=504) from exc


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


def _qwen_vl_model_path() -> str:
    return os.getenv("WRAPPER_QWEN_VL_MODEL", "/home/admin/model/Qwen3.5-9B-Base")


def _load_qwen_vl() -> tuple[Any, Any]:
    global _QWEN_VL_MODEL, _QWEN_VL_PROCESSOR
    if _QWEN_VL_MODEL is not None and _QWEN_VL_PROCESSOR is not None:
        return _QWEN_VL_MODEL, _QWEN_VL_PROCESSOR

    with _QWEN_VL_LOCK:
        if _QWEN_VL_MODEL is not None and _QWEN_VL_PROCESSOR is not None:
            return _QWEN_VL_MODEL, _QWEN_VL_PROCESSOR

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise OcrError(f"Qwen-VL OCR dependencies are unavailable: {exc}", status_code=500) from exc

        model_path = _qwen_vl_model_path()
        dtype_name = os.getenv("WRAPPER_QWEN_VL_DTYPE", "bfloat16").strip().lower()
        dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} else torch.float16
        device = os.getenv("WRAPPER_QWEN_VL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        attn_impl = (
            os.getenv("WRAPPER_QWEN_VL_ATTN_IMPLEMENTATION")
            or os.getenv("REDACTION_QWEN_ATTN_IMPLEMENTATION")
            or ""
        ).strip()
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        except (TypeError, ValueError):
            kwargs.pop("attn_implementation", None)
            model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        model.to(device)
        model.eval()

        _QWEN_VL_MODEL = model
        _QWEN_VL_PROCESSOR = processor
        return model, processor


def extract_text_from_image_via_qwen_vl(path: Path) -> str:
    try:
        import torch
        from PIL import Image, ImageOps
    except Exception as exc:
        raise OcrError(f"Qwen-VL OCR image dependencies are unavailable: {exc}", status_code=500) from exc

    model, processor = _load_qwen_vl()
    prompt = (
        "Transcribe all visible text in this image. Preserve line breaks and reading order. "
        "Return only the transcribed text; do not explain or add labels."
    )
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        if getattr(processor, "chat_template", None):
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"<|vision_start|><|image_pad|><|vision_end|>\n{prompt}\n"
        inputs = processor(text=[text], images=[image], return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    max_new_tokens = int(os.getenv("WRAPPER_QWEN_VL_MAX_NEW_TOKENS", "2048"))
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_len = int(inputs["input_ids"].shape[-1])
    generated = generated[:, prompt_len:]
    output = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output.strip()


def _clean_ocr_text(text: str) -> str:
    """Conservative cleanup for transcribed output.

    Goal: reduce random glyph noise from scans while keeping short, meaningful
    document fields such as IDs and abbreviations.
    """
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            cleaned_lines.append("")
            continue

        alnum = sum(ch.isalnum() for ch in line)
        weird = len(re.findall(r"[^\w\s,./:@&()\-]", line))
        line_len = len(line)

        # Drop lines dominated by noisy symbols with almost no readable tokens.
        if line_len >= 10 and alnum <= 3 and weird >= 3:
            continue
        if line_len >= 14 and (weird / max(line_len, 1)) > 0.35 and alnum <= 5:
            continue

        cleaned_lines.append(line)

    compact = "\n".join(cleaned_lines)
    # Avoid tall blocks of empty lines while preserving paragraph breaks.
    return re.sub(r"\n{3,}", "\n\n", compact).strip()


def extract_text_from_image(path: Path) -> str:
    return extract_text_from_image_via_qwen_vl(path)


def extract_text_from_image_with_provider(path: Path, *, warnings: list[str]) -> tuple[str, str]:
    text = extract_text_from_image(path)
    if text.strip():
        warnings.append("qwen_vl_provider")
    return text, "qwen_vl_ocr"


def extract_pdf_text_layer(path: Path) -> str:
    if not command_exists("pdftotext"):
        return ""
    r = _run_command(["pdftotext", "-layout", str(path), "-"], timeout=60.0)
    if r.returncode != 0:
        return ""
    return r.stdout


def extract_pdf_via_ocr(path: Path, *, max_pages: int, warnings: list[str] | None = None) -> tuple[str, int]:
    if not command_exists("pdftoppm"):
        raise OcrError("pdftoppm is not installed on the server", status_code=500)
    with tempfile.TemporaryDirectory(prefix="redact_pdf_ocr_") as tmpdir:
        prefix = Path(tmpdir) / "page"
        r = _run_command(
            ["pdftoppm", "-r", "200", "-png",
             "-f", "1", "-l", str(max_pages), str(path), str(prefix)],
            timeout=180.0,
        )
        if r.returncode != 0:
            raise OcrError((r.stderr or "PDF rendering failed").strip()[:500], status_code=422)
        pages = sorted(Path(tmpdir).glob("page-*.png"))
        page_texts = []
        page_warnings = warnings if warnings is not None else []
        for p in pages:
            text, _ = extract_text_from_image_with_provider(p, warnings=page_warnings)
            page_texts.append(text.strip())
        return "\n\n".join(t for t in page_texts if t), len(pages)


def extract_upload_text(
    *,
    filename: str,
    content_type: str | None,
    data: bytes,
    max_upload_bytes: int = 25 * 1024 * 1024,
    max_pdf_ocr_pages: int = 8,
    max_file_text_chars: int = 12000,
) -> dict[str, Any]:
    if not data:
        raise OcrError("Uploaded file is empty", status_code=400)
    if len(data) > max_upload_bytes:
        raise OcrError(f"File exceeds {max_upload_bytes} byte upload limit", status_code=413)

    kind = upload_kind(filename, content_type)
    warnings: list[str] = []
    pages = 0
    text_source = ""
    if kind == "text":
        text = decode_text_upload(data)
        text_source = "text_upload"
    elif kind in {"image", "pdf"}:
        suffix = Path(filename or "").suffix.lower() or (".pdf" if kind == "pdf" else ".png")
        text = ""

        if not text:
            with tempfile.NamedTemporaryFile(prefix="redact_upload_", suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            try:
                if kind == "image":
                    text, text_source = extract_text_from_image_with_provider(tmp_path, warnings=warnings)
                    pages = 1
                else:
                    text = extract_pdf_text_layer(tmp_path)
                    if text.strip():
                        text_source = "pdf_text_layer"
                    else:
                        text, pages = extract_pdf_via_ocr(
                            tmp_path, max_pages=max_pdf_ocr_pages, warnings=warnings
                        )
                        text_source = "pdf_ocr"
                        if pages >= max_pdf_ocr_pages:
                            warnings.append(f"pdf_ocr_limited_to_first_{max_pdf_ocr_pages}_pages")
            finally:
                tmp_path.unlink(missing_ok=True)
    else:
        raise OcrError(
            "Unsupported file type. Upload a PDF, image, or plain text file.",
            status_code=400,
        )

    text = normalize_text(text)
    if kind in {"image", "pdf"}:
        cleaned = _clean_ocr_text(text)
        if cleaned and cleaned != text:
            text = cleaned
            warnings.append("ocr_text_cleaned")
    if not text.strip():
        raise OcrError("No readable text found in uploaded file", status_code=422)
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
