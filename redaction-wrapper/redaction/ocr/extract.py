"""OCR + plain-text extraction for uploaded files.

Pulled out of the original Qwen demo server so any backend can use it.

Supported inputs:
  - text: txt / md / csv / tsv / json / log
  - image: png / jpg / jpeg / tif / tiff / bmp / webp (Tesseract)
  - pdf: pdftotext text layer first, fall back to per-page Tesseract OCR
"""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any

from ..core.normalize import normalize_text


TEXT_SUFFIXES = {".txt", ".text", ".md", ".csv", ".tsv", ".json", ".log"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


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


def _preprocess_image(path: Path) -> Path | None:
    try:
        from PIL import Image, ImageFilter, ImageOps
    except ImportError:
        return None
    with Image.open(path) as image:
        gray = ImageOps.grayscale(image)
        enhanced = ImageOps.autocontrast(gray)
        denoised = enhanced.filter(ImageFilter.MedianFilter(size=3))
        upscaled = denoised.resize(
            (denoised.width * 2, denoised.height * 2), Image.Resampling.LANCZOS
        )
        sharpened = upscaled.filter(ImageFilter.SHARPEN)
        with tempfile.NamedTemporaryFile(prefix="redact_ocr_pre_", suffix=".png", delete=False) as tmp:
            out_path = Path(tmp.name)
        sharpened.save(out_path, format="PNG", dpi=(300, 300))
        return out_path


def _score_text(text: str) -> float:
    alnum = sum(c.isalnum() for c in text)
    weird = len(re.findall(r"[^\w\s,./:@&()\-]", text))
    short = sum(1 for line in text.splitlines() if 0 < len(line.strip()) <= 2)
    useful = sum(1 for line in text.splitlines() if len(line.strip()) >= 4)
    return float(alnum * 2 + useful * 5 - weird * 6 - short * 4)


def _tesseract_cmd(path: Path, *, psm: int) -> list[str]:
    return [
        "tesseract", str(path), "stdout",
        "-l", "eng", "--psm", str(psm),
        "-c", "preserve_interword_spaces=1",
    ]


def _run_tesseract(path: Path, *, psm: int) -> tuple[str, float]:
    if not command_exists("tesseract"):
        raise OcrError("Tesseract OCR is not installed on the server", status_code=500)
    result = _run_command(_tesseract_cmd(path, psm=psm), timeout=180.0)
    if result.returncode != 0:
        detail = result.stderr.strip() or "Tesseract OCR failed"
        raise OcrError(detail[:500], status_code=422)
    text = result.stdout.strip()
    # OCR confidence boost
    tsv = _run_command([*_tesseract_cmd(path, psm=psm), "tsv"], timeout=180.0)
    conf_score = 0.0
    if tsv.returncode == 0 and tsv.stdout.strip():
        confidences: list[float] = []
        tokens = 0
        reader = csv.DictReader(StringIO(tsv.stdout), delimiter="\t")
        for row in reader:
            token = (row.get("text") or "").strip()
            if not token:
                continue
            try:
                c = float((row.get("conf") or "").strip())
            except ValueError:
                continue
            if c >= 0:
                confidences.append(c)
                tokens += 1
        if confidences:
            conf_score = sum(confidences) / len(confidences) + min(tokens, 40) * 0.5
    return text, conf_score + _score_text(text)


def extract_text_from_image(path: Path) -> str:
    candidates: list[tuple[str, float]] = []
    pre = _preprocess_image(path)
    paths = [path] + ([pre] if pre is not None else [])
    try:
        for p in paths:
            for psm in (6, 4):
                text, score = _run_tesseract(p, psm=psm)
                if text:
                    candidates.append((text, score))
    finally:
        if pre is not None:
            pre.unlink(missing_ok=True)
    if not candidates:
        return ""
    return max(candidates, key=lambda x: x[1])[0]


def extract_pdf_text_layer(path: Path) -> str:
    if not command_exists("pdftotext"):
        return ""
    r = _run_command(["pdftotext", "-layout", str(path), "-"], timeout=60.0)
    if r.returncode != 0:
        return ""
    return r.stdout


def extract_pdf_via_ocr(path: Path, *, max_pages: int) -> tuple[str, int]:
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
        page_texts = [extract_text_from_image(p).strip() for p in pages]
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
        with tempfile.NamedTemporaryFile(prefix="redact_upload_", suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            if kind == "image":
                text = extract_text_from_image(tmp_path)
                text_source = "image_ocr"
                pages = 1
            else:
                text = extract_pdf_text_layer(tmp_path)
                if text.strip():
                    text_source = "pdf_text_layer"
                else:
                    text, pages = extract_pdf_via_ocr(tmp_path, max_pages=max_pdf_ocr_pages)
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
