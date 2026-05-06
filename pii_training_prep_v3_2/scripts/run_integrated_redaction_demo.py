#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Integrated PII redaction demo pipeline")
    parser.add_argument("--root", default="/home/admin/ZYX/pii_training_prep_v3_2")
    parser.add_argument("--text", help="Text to redact")
    parser.add_argument("--model-path", default="/home/admin/model/Qwen3.5-9B-Base")
    parser.add_argument("--qwen-experiment", default="last_linear")
    parser.add_argument("--opf-run", default="opf_hard_79")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--redact-threshold", type=float, default=0.60)
    parser.add_argument("--review-threshold", type=float, default=0.25)
    parser.add_argument("--csv-taxonomy", default="docs/Data Sensitivity.csv")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Qwen classifier (OPF only)")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    reports_dir = root / "reports"

    sys.path.insert(0, str(root / "src" / "pii_prep"))
    from qwen_spancls_inference import QwenSpanClassifier
    from opf_inference import OPFDetector
    from integrated_pipeline import (
        PolicyLayer,
        run_integrated_pipeline,
        build_redaction_output,
    )

    start_time = time.time()

    smoke_examples = [
        {
            "id": "smoke_dob",
            "text": "My date of birth is 04/05/1998 and I need to update my records.",
            "expected_pii": ["DATE_OF_BIRTH"],
        },
        {
            "id": "smoke_ambiguous_date",
            "text": "The deadline for submission is 15/06/2025 please confirm.",
            "expected_pii": ["DATE_OF_BIRTH"],
            "note": "ambiguous_date_without_dob_context",
        },
        {
            "id": "smoke_bsb_account",
            "text": "Please transfer funds to BSB 062-000 account 12345678 for payment.",
            "expected_pii": ["BANK_ACCOUNT_NUMBER"],
        },
        {
            "id": "smoke_email",
            "text": "You can reach me at alex.johnson@example.com for further details.",
            "expected_pii": ["EMAIL_ADDRESS"],
        },
        {
            "id": "smoke_phone",
            "text": "My mobile is 0412 345 678 and home phone is 02 9876 5432.",
            "expected_pii": ["MOBILE", "HOME_PHONE"],
        },
        {
            "id": "smoke_address",
            "text": "I live at 42 Wallaby Way, Sydney NSW 2000, Australia.",
            "expected_pii": ["ADDRESS"],
        },
        {
            "id": "smoke_order_ref_negative",
            "text": "Your order number is ORD-987654 and will ship on Monday.",
            "expected_pii": [],
            "note": "negative_example_no_pii",
        },
        {
            "id": "smoke_gender",
            "text": "The applicant identifies as male and prefers he/him pronouns.",
            "expected_pii": ["GENDER", "PRONOUN"],
        },
        {
            "id": "smoke_salary",
            "text": "My current salary is $85,000 per annum plus superannuation.",
            "expected_pii": ["SALARY"],
        },
        {
            "id": "smoke_medicare",
            "text": "Medicare number is 2123 45678 1 and expiry is 01/2026.",
            "expected_pii": ["MEDICARE_NUMBER", "MEDICARE_EXPIRY"],
        },
    ]

    if args.text:
        smoke_examples = [{"id": "cli_input", "text": args.text, "expected_pii": []}]

    csv_taxonomy_path = str(root / args.csv_taxonomy) if Path(root / args.csv_taxonomy).exists() else None

    policy = PolicyLayer(
        csv_path=csv_taxonomy_path,
        redact_threshold=args.redact_threshold,
        review_threshold=args.review_threshold,
    )

    print("Loading Qwen span classifier...", flush=True)
    time_qwen_start = time.time()
    qwen_cls = QwenSpanClassifier.from_project_root(
        root,
        model_path=args.model_path,
        experiment=args.qwen_experiment,
        device=args.device,
        dtype=args.dtype,
    )
    time_qwen = time.time() - time_qwen_start
    print(f"  Loaded in {time_qwen:.2f}s | labels={qwen_cls.num_labels} temperature={qwen_cls.temperature}", flush=True)

    print("Loading OPF detector...", flush=True)
    time_opf_start = time.time()
    opf_det = OPFDetector.from_project_root(root, run_name=args.opf_run, device=args.device)
    time_opf = time.time() - time_opf_start
    print(f"  Loaded in {time_opf:.2f}s", flush=True)

    all_examples = []
    policy_decisions = []

    print(f"\nProcessing {len(smoke_examples)} examples...", flush=True)
    for i, example in enumerate(smoke_examples):
        text = example["text"]
        ex_id = example["id"]
        print(f"\n[{i+1}/{len(smoke_examples)}] {ex_id}", flush=True)

        qwen_enabled = not args.skip_qwen

        if qwen_enabled:
            result = run_integrated_pipeline(
                text, qwen_cls, opf_det, policy
            )
        else:
            opf_result = opf_det.detect_spans(text)
            result = {
                "text": text,
                "stage": "opf_only",
                "spans": opf_result.get("candidate_spans", []),
                "summary": opf_result.get("summary", {}),
            }

        redact_output = build_redaction_output(text, result)

        example_record = {
            "id": ex_id,
            "expected_pii": example.get("expected_pii", []),
            "note": example.get("note", ""),
            "text": text,
            "redacted_text": redact_output["redacted_text"],
            "detected_spans": [
                {
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "value": s.get("value"),
                    "opf_type": s.get("opf_top_type", ""),
                    "qwen_top_type": s.get("top_type"),
                    "qwen_top_probability": s.get("top_probability"),
                    "decision": s.get("decision", ""),
                    "risk_score": s.get("risk_score"),
                }
                for s in result.get("spans", [])
            ],
            "summary": {
                "span_count": redact_output["span_count"],
                "redact_count": redact_output["redact_count"],
                "review_count": redact_output["review_count"],
                "ignore_count": redact_output["ignore_count"],
            },
            "original_text": text,
        }
        all_examples.append(example_record)

        for s in result.get("spans", []):
            if s.get("decision") in ("redact", "review"):
                policy_decisions.append({
                    "example_id": ex_id,
                    "text_snippet": text[max(0, s.get("start", 0) - 20):s.get("end", 0) + 20],
                    "span": {
                        "start": s.get("start"),
                        "end": s.get("end"),
                        "value": s.get("value"),
                        "top_type": s.get("top_type"),
                        "top_probability": s.get("top_probability"),
                        "risk_score": s.get("risk_score"),
                        "decision": s.get("decision"),
                    },
                })

        if redact_output["redact_count"] > 0:
            print(f"  REDACTED: {redact_output['redacted_text'][:200]}", flush=True)
        else:
            print(f"  No redactions needed", flush=True)

    elapsed = time.time() - start_time

    smoke_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pipeline": "stage4_integration_smoke",
        "configuration": {
            "qwen_model": args.model_path,
            "qwen_experiment": args.qwen_experiment,
            "qwen_labels": qwen_cls.num_labels,
            "qwen_temperature": qwen_cls.temperature,
            "opf_run": args.opf_run,
            "opf_checkpoint": str(root / "runs" / args.opf_run),
            "redact_threshold": args.redact_threshold,
            "review_threshold": args.review_threshold,
            "risk_weights_loaded": csv_taxonomy_path is not None,
            "qwen_enabled": not args.skip_qwen,
            "device": args.device,
            "dtype": args.dtype,
        },
        "timing": {
            "qwen_load_s": round(time_qwen, 2),
            "opf_load_s": round(time_opf, 2),
            "total_elapsed_s": round(elapsed, 2),
        },
        "example_count": len(smoke_examples),
        "redact_count": sum(ex["summary"]["redact_count"] for ex in all_examples),
        "review_count": sum(ex["summary"]["review_count"] for ex in all_examples),
        "ignore_count": sum(ex["summary"]["ignore_count"] for ex in all_examples),
        "examples": all_examples,
    }

    write_json(reports_dir / "stage4_integration_smoke_report.json", smoke_report)
    write_jsonl(reports_dir / "stage4_integration_examples.jsonl", all_examples)
    write_jsonl(reports_dir / "stage4_policy_decision_examples.jsonl", policy_decisions)

    print(f"\n\n{'='*60}")
    print(f"Integration smoke test complete in {elapsed:.2f}s")
    print(f"  Examples: {len(smoke_examples)}")
    print(f"  Redacted: {smoke_report['redact_count']} spans")
    print(f"  Review:   {smoke_report['review_count']} spans")
    print(f"  Ignored:  {smoke_report['ignore_count']} spans")
    print(f"\nReports written:")
    print(f"  {reports_dir / 'stage4_integration_smoke_report.json'}")
    print(f"  {reports_dir / 'stage4_integration_examples.jsonl'}")
    print(f"  {reports_dir / 'stage4_policy_decision_examples.jsonl'}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
