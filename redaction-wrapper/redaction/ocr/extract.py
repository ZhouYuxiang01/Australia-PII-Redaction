"""OCR + plain-text extraction for uploaded files.

Pulled out of the original Qwen demo server so any backend can use it.

Supported inputs:
  - text: txt / md / csv / tsv / json / log
  - image: png / jpg / jpeg / tif / tiff / bmp / webp (RapidOCR / PaddleOCR / Tesseract)
  - pdf: pdftotext text layer first, fall back to per-page OCR
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any

from ..core.normalize import normalize_text


TEXT_SUFFIXES = {".txt", ".text", ".md", ".csv", ".tsv", ".json", ".log"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
OCR_PROVIDERS = {"auto", "rapidocr", "paddle", "tesseract"}


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


def _ocr_provider() -> str:
    provider = os.getenv("WRAPPER_OCR_PROVIDER", "auto").strip().lower()
    if provider not in OCR_PROVIDERS:
        raise OcrError(
            f"Unsupported WRAPPER_OCR_PROVIDER={provider!r}. "
            f"Expected one of: {', '.join(sorted(OCR_PROVIDERS))}",
            status_code=500,
        )
    return provider


def _save_temp_image(image: Any, *, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(prefix="redact_ocr_pre_", suffix=suffix, delete=False) as tmp:
        out_path = Path(tmp.name)
    image.save(out_path, format="PNG", dpi=(300, 300))
    return out_path


def _preprocess_image(path: Path) -> list[Path]:
    try:
        from PIL import Image, ImageFilter, ImageOps
    except ImportError:
        return []
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        gray = ImageOps.grayscale(image)
        enhanced = ImageOps.autocontrast(gray)
        denoised = enhanced.filter(ImageFilter.MedianFilter(size=3))
        scale = 3 if max(denoised.size) < 2200 else 2
        upscaled = denoised.resize((denoised.width * scale, denoised.height * scale), Image.Resampling.LANCZOS)
        sharpened = upscaled.filter(ImageFilter.SHARPEN)

        # ID cards and screenshots often have patterned backgrounds. A high
        # contrast binary candidate helps Tesseract focus on dark glyphs.
        thresholded = sharpened.point(lambda px: 255 if px > 178 else 0)

        return [
            _save_temp_image(sharpened, suffix=".png"),
            _save_temp_image(thresholded, suffix=".png"),
        ]


def extract_text_from_image_via_rapidocr(path: Path) -> str:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:
        raise OcrError(
            "RapidOCR is not installed. Install `rapidocr-onnxruntime`, "
            "or set WRAPPER_OCR_PROVIDER=auto/tesseract.",
            status_code=500,
        ) from exc

    ocr = RapidOCR()
    result, _ = ocr(str(path))
    if not result:
        return ""
    min_conf = float(os.getenv("RAPIDOCR_MIN_CONFIDENCE", "0.30"))
    lines: list[str] = []
    for item in result:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text = str(item[1] or "").strip()
        try:
            conf = float(item[2]) if len(item) > 2 else 0.0
        except (TypeError, ValueError):
            conf = 0.0
        if text and conf >= min_conf:
            lines.append(text)
    return "\n".join(lines)


def extract_text_from_image_via_paddle(path: Path) -> str:
    env = os.environ.copy()
    env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    result = subprocess.run(
        [sys.executable, "-m", "redaction.ocr.paddle_runner", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=float(os.getenv("PADDLEOCR_TIMEOUT_SECONDS", "240")),
        env=env,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "PaddleOCR failed").strip()
        raise OcrError(f"PaddleOCR failed in worker process: {detail[-800:]}", status_code=500)
    try:
        import json
        payload = json.loads(result.stdout)
    except Exception as exc:
        raise OcrError(f"PaddleOCR returned invalid JSON: {result.stdout[:500]}", status_code=500) from exc
    return str(payload.get("text") or "")


def _score_text(text: str) -> float:
    alnum = sum(c.isalnum() for c in text)
    weird = len(re.findall(r"[^\w\s,./:@&()\-]", text))
    short = sum(1 for line in text.splitlines() if 0 < len(line.strip()) <= 2)
    useful = sum(1 for line in text.splitlines() if len(line.strip()) >= 4)
    return float(alnum * 2 + useful * 5 - weird * 6 - short * 4)


def _clean_ocr_text(text: str) -> str:
    """Conservative cleanup for OCR output.

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


def _tesseract_cmd(path: Path, *, psm: int) -> list[str]:
    return [
        "tesseract", str(path), "stdout",
        "-l", "eng", "--oem", "1", "--psm", str(psm),
        "-c", "preserve_interword_spaces=1",
        "-c", "tessedit_do_invert=1",
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
    preprocessed = _preprocess_image(path)
    paths = [path] + preprocessed
    try:
        for p in paths:
            for psm in (6, 4, 11, 12):
                text, score = _run_tesseract(p, psm=psm)
                if text:
                    candidates.append((text, score))
    finally:
        for pre in preprocessed:
            pre.unlink(missing_ok=True)
    if not candidates:
        return ""
    return max(candidates, key=lambda x: x[1])[0]


def extract_text_from_image_with_provider(path: Path, *, warnings: list[str]) -> tuple[str, str]:
    provider = _ocr_provider()
    if provider in {"auto", "rapidocr"}:
        try:
            text = extract_text_from_image_via_rapidocr(path)
            if text.strip():
                warnings.append("rapidocr_provider")
                return text, "rapid_ocr"
            warnings.append("rapidocr_returned_empty")
        except OcrError as exc:
            if provider == "rapidocr":
                raise
            warnings.append(f"rapidocr_failed_fallback: {str(exc)[:160]}")
    if provider == "rapidocr":
        return "", "rapid_ocr"
    if provider == "paddle":
        try:
            text = extract_text_from_image_via_paddle(path)
            if text.strip():
                warnings.append("paddleocr_provider")
                return text, "paddle_ocr"
            warnings.append("paddleocr_returned_empty")
        except OcrError as exc:
            raise
    if provider == "paddle":
        return "", "paddle_ocr"
    text = extract_text_from_image(path)
    return text, "image_ocr"


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
