#!/usr/bin/env python3
"""CLI entry point — single-shot redact via any backend.

Examples:
    python scripts/cli.py \\
        --backend configs/backends/opf-v3.json \\
        --policy configs/policies/opf-v3-default-v1.json \\
        --text "Call Alice on 0421 909 121"

    python scripts/cli.py \\
        --backend configs/backends/qwen-9b-lora.json \\
        --policy configs/policies/qwen-9b-lora-default-v1.json \\
        --text-file path/to/input.txt --json-out out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `redaction` importable when running from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redaction.backends import build_backend_from_path
from redaction.core import (apply_policy, build_response, load_json,
                             normalize_text, safe_postprocess_spans)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, type=Path)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--indent", type=int, default=2)
    args = ap.parse_args()

    if args.text is None and args.text_file is None:
        ap.error("provide --text or --text-file")
    raw = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")

    backend = build_backend_from_path(args.backend)
    policy = load_json(args.policy)

    text = normalize_text(raw)
    spans, diag = backend.detect_spans(text)
    spans, post_warnings = safe_postprocess_spans(text, spans, policy)
    spans = apply_policy(spans, policy)
    payload = build_response(
        text=text, spans=spans, policy=policy,
        raw_offset_mapping_applied=bool(diag.get("raw_offset_mapping_applied", False)),
        warnings=[*diag.get("warnings", []), *post_warnings],
        extra_metadata={"input_length": len(text)},
    )

    out = json.dumps(payload, ensure_ascii=False, indent=args.indent)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(out + "\n", encoding="utf-8")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
