"""Subprocess runner for PaddleOCR.

Paddle's native inference runtime can segfault on some platforms. Running it in
a short-lived worker process keeps the FastAPI server alive and lets the caller
fall back to Tesseract when configured with WRAPPER_OCR_PROVIDER=auto.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any


def _iter_paddle_items(result: Any) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            text = node.get("text") or node.get("rec_text") or node.get("transcription")
            score = node.get("score") or node.get("confidence") or node.get("rec_score") or 0.0
            if text:
                try:
                    conf = float(score)
                except (TypeError, ValueError):
                    conf = 0.0
                items.append((str(text), conf))
            for value in node.values():
                if isinstance(value, (list, tuple, dict)):
                    visit(value)
            return
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[1], (list, tuple)) and node[1]:
                maybe_text = node[1][0]
                maybe_score = node[1][1] if len(node[1]) > 1 else 0.0
                if isinstance(maybe_text, str):
                    try:
                        conf = float(maybe_score)
                    except (TypeError, ValueError):
                        conf = 0.0
                    items.append((maybe_text, conf))
                    return
            for child in node:
                visit(child)

    visit(result)
    return items


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(json.dumps({"error": "missing image path"}))
        return 2
    image_path = Path(argv[0])

    from paddleocr import PaddleOCR

    lang = os.getenv("PADDLEOCR_LANG", "en")
    use_angle_cls = os.getenv("PADDLEOCR_USE_ANGLE_CLS", "false").lower() in {"1", "true", "yes"}
    min_conf = float(os.getenv("PADDLEOCR_MIN_CONFIDENCE", "0.30"))

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        try:
            ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)
        except (TypeError, ValueError):
            ocr = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=use_angle_cls,
                text_rec_score_thresh=min_conf,
            )
        try:
            if hasattr(ocr, "ocr"):
                result = ocr.ocr(str(image_path), cls=use_angle_cls)
            else:
                result = ocr.predict(str(image_path))
        except TypeError:
            result = ocr.ocr(str(image_path))

    items = _iter_paddle_items(result)
    lines = [text.strip() for text, conf in items if text.strip() and conf >= min_conf]
    if not lines and items:
        lines = [text.strip() for text, _ in items if text.strip()]
    print(json.dumps({"text": "\n".join(lines)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
