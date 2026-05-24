from __future__ import annotations

import os
import tempfile
from pathlib import Path

from redaction.ocr import extract as ocr_extract


def test_images_are_transcribed_with_qwen_vl() -> None:
    original_qwen = getattr(ocr_extract, "extract_text_from_image_via_qwen_vl", None)
    try:
        def fake_qwen(path: Path) -> str:
            assert path.name == "sample.png"
            return "Name: Mia Tran"

        ocr_extract.extract_text_from_image_via_qwen_vl = fake_qwen
        text, source = ocr_extract.extract_text_from_image_with_provider(
            Path("/tmp/sample.png"),
            warnings=[],
        )
    finally:
        if original_qwen is None:
            delattr(ocr_extract, "extract_text_from_image_via_qwen_vl")
        else:
            ocr_extract.extract_text_from_image_via_qwen_vl = original_qwen

    assert text == "Name: Mia Tran"
    assert source == "qwen_vl_ocr"


def test_uploaded_images_are_marked_as_qwen_vl_source() -> None:
    old_provider = os.environ.get("WRAPPER_OCR_PROVIDER")
    original_qwen = getattr(ocr_extract, "extract_text_from_image_via_qwen_vl", None)
    try:
        os.environ["WRAPPER_OCR_PROVIDER"] = "legacy_provider_should_be_ignored"
        ocr_extract.extract_text_from_image_via_qwen_vl = lambda path: "UAC no. 221 904 778"
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            payload = ocr_extract.extract_upload_text(
                filename="sample.png",
                content_type="image/png",
                data=Path(tmp.name).read_bytes() or b"not-empty",
            )
    finally:
        if old_provider is None:
            os.environ.pop("WRAPPER_OCR_PROVIDER", None)
        else:
            os.environ["WRAPPER_OCR_PROVIDER"] = old_provider
        if original_qwen is None:
            delattr(ocr_extract, "extract_text_from_image_via_qwen_vl")
        else:
            ocr_extract.extract_text_from_image_via_qwen_vl = original_qwen

    assert payload["text"] == "UAC no. 221 904 778"
    assert payload["text_source"] == "qwen_vl_ocr"
    assert "qwen_vl_provider" in payload["warnings"]


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
