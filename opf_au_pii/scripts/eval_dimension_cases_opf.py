#!/usr/bin/env python3
"""Batch-evaluate dimension test cases against a local OPF/privacy-filter checkpoint.

Input CSV columns:
  dimension,tester,text,expected_label,expected_text

By default, matching is label-only: a case passes when the expected label
appears in any detected span. Use `--match-mode exact` when the detected span
text must also match `expected_text`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVACY_FILTER_ROOT = REPO_ROOT / "privacy-filter"
sys.path.insert(0, str(PRIVACY_FILTER_ROOT))

from opf import OPF  # noqa: E402


STATUS_PASS = "通过"
STATUS_MINOR = "有问题，但是出现的概率在30%以下"
STATUS_URGENT = "急需改进，case出现问题的概率在30%以上"


@dataclass(frozen=True)
class Case:
    case_id: str
    dimension: str
    tester: str
    text: str
    expected_label: str
    expected_text: str


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _read_cases(path: Path) -> list[Case]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"dimension", "text", "expected_label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV missing columns: {', '.join(sorted(missing))}")

        cases: list[Case] = []
        for idx, row in enumerate(reader, start=1):
            if row.get(None):
                raise ValueError(
                    f"Row {idx} has extra CSV columns. If the text contains commas, "
                    "wrap the text field in double quotes."
                )
            text = _clean(row.get("text"))
            expected_label = _clean(row.get("expected_label")).upper()
            if not text and not expected_label:
                continue
            if not text or not expected_label:
                raise ValueError(f"Row {idx} must include both text and expected_label")
            cases.append(
                Case(
                    case_id=_clean(row.get("case_id")) or str(idx),
                    dimension=_clean(row.get("dimension")),
                    tester=_clean(row.get("tester")),
                    text=text,
                    expected_label=expected_label,
                    expected_text=_clean(row.get("expected_text")),
                )
            )
    return cases


def _span_to_dict(span: Any) -> dict[str, Any]:
    return {
        "label": span.label,
        "start": span.start,
        "end": span.end,
        "text": span.text,
        "placeholder": span.placeholder,
    }


def _matches(case: Case, span: dict[str, Any], *, match_mode: str) -> bool:
    if _clean(span.get("label")).upper() != case.expected_label:
        return False
    if match_mode == "label":
        return True
    if not case.expected_text:
        return True
    return _norm(_clean(span.get("text"))) == _norm(case.expected_text)


def _evaluate_case(case: Case, redactor: OPF, *, match_mode: str) -> dict[str, Any]:
    result = redactor.redact(case.text)
    if isinstance(result, str):
        raise TypeError("Expected structured OPF result, got text-only output")

    spans = [_span_to_dict(span) for span in result.detected_spans]
    passed = any(_matches(case, span, match_mode=match_mode) for span in spans)
    return {
        "case_id": case.case_id,
        "dimension": case.dimension,
        "tester": case.tester,
        "text": case.text,
        "expected_label": case.expected_label,
        "expected_text": case.expected_text,
        "match_mode": match_mode,
        "passed": "YES" if passed else "NO",
        "detected_labels": ", ".join(sorted({_clean(span.get("label")) for span in spans})),
        "detected_spans_json": json.dumps(spans, ensure_ascii=False),
        "redacted_text": result.redacted_text,
        "warning": result.warning or "",
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dimension[row["dimension"]].append(row)

    summary = []
    for dimension, items in sorted(by_dimension.items()):
        total = len(items)
        failed = sum(1 for item in items if item["passed"] != "YES")
        failure_rate = failed / total if total else 0.0
        if failed == 0:
            status = STATUS_PASS
        elif failure_rate < 0.30:
            status = STATUS_MINOR
        else:
            status = STATUS_URGENT
        summary.append(
            {
                "dimension": dimension,
                "tester": ", ".join(sorted({item["tester"] for item in items if item["tester"]})),
                "total_cases": total,
                "failed_cases": failed,
                "failure_rate": f"{failure_rate:.1%}",
                "测试结果说明": status,
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path, help="CSV exported from the OneDrive sheet.")
    parser.add_argument(
        "--checkpoint",
        default=REPO_ROOT / "runs/final/opf_73class_v3_full/checkpoint",
        type=Path,
        help="Local OPF/privacy-filter checkpoint directory.",
    )
    parser.add_argument("--out-dir", default=REPO_ROOT / "outputs/dimension_eval_opf", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--decode-mode", default="viterbi", choices=("viterbi", "argmax"))
    parser.add_argument(
        "--match-mode",
        default="label",
        choices=("label", "exact"),
        help=(
            "label: pass when expected_label is detected anywhere. "
            "exact: also require detected span text to match expected_text."
        ),
    )
    args = parser.parse_args()

    cases = _read_cases(args.cases)
    if not cases:
        raise SystemExit("No cases found.")

    redactor = OPF(
        model=args.checkpoint,
        device=args.device,
        output_mode="typed",
        decode_mode=args.decode_mode,
        output_text_only=False,
    )

    rows = [_evaluate_case(case, redactor, match_mode=args.match_mode) for case in cases]
    summary = _summarize(rows)

    detail_path = args.out_dir / "case_results.csv"
    summary_path = args.out_dir / "dimension_summary.csv"
    _write_csv(
        detail_path,
        rows,
        [
            "case_id",
            "dimension",
            "tester",
            "text",
            "expected_label",
            "expected_text",
            "match_mode",
            "passed",
            "detected_labels",
            "detected_spans_json",
            "redacted_text",
            "warning",
        ],
    )
    _write_csv(
        summary_path,
        summary,
        ["dimension", "tester", "total_cases", "failed_cases", "failure_rate", "测试结果说明"],
    )
    print(f"Wrote detail results: {detail_path}")
    print(f"Wrote dimension summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
