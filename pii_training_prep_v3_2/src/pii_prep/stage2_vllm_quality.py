from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from pii_prep.stage2_generation import ZERO_EXAMPLE_LABELS
from pii_prep.stage2_teacher_dryrun import load_jsonl, validate_converted_rows, write_json


CONTEXT_ALIASES = {
    "bare_span": "bare_span",
    "weak_context": "weak_context",
    "strong_positive_context": "strong_positive",
    "strong_positive": "strong_positive",
    "reverse_negative_context": "reverse_negative",
    "reverse_negative": "reverse_negative",
    "hard_negative_context": "hard_negative",
    "hard_negative": "hard_negative",
}

WARNING_NAMES = [
    "bare_span_overconfident",
    "bare_date_dob_overconfident",
    "weak_context_overconfident",
    "reverse_negative_non_pii_failure",
    "hard_negative_non_pii_failure",
    "strong_positive_not_confident",
]


def analyze_quality(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    malformed = validate_converted_records(rows)
    by_context_rows: dict[str, list[dict[str, Any]]] = {
        "bare_span": [],
        "weak_context": [],
        "strong_positive": [],
        "reverse_negative": [],
        "hard_negative": [],
    }
    warnings: dict[str, list[dict[str, Any]]] = {name: [] for name in WARNING_NAMES}

    top_type_counts: dict[str, int] = {}
    label_stats: dict[str, dict[str, float]] = {}
    for row in rows:
        context = normalize_context_type(str(row.get("context_type", "")))
        if context in by_context_rows:
            by_context_rows[context].append(row)
        top_type = str(row.get("top_type", ""))
        top_type_counts[top_type] = top_type_counts.get(top_type, 0) + 1
        for label, probability in row.get("type_distribution", {}).items():
            stats = label_stats.setdefault(str(label), {"candidate_count": 0, "top_type_count": 0, "probability_sum": 0.0})
            stats["candidate_count"] += 1
            stats["probability_sum"] += float(probability)
            if label == top_type:
                stats["top_type_count"] += 1
        collect_warnings(row, context, warnings)

    report = {
        "record_count": len(rows),
        "malformed_converted_record_count": len(malformed),
        "malformed_converted_records": malformed,
        "by_context_type": {
            context: summarize_context(context_rows)
            for context, context_rows in by_context_rows.items()
        },
        "top_type_counts": dict(sorted(top_type_counts.items())),
        "label_distribution": summarize_label_distribution(label_stats),
        "zero_example_label_coverage": zero_example_label_coverage(rows),
        "warning_counts": {
            f"{name}_count": len(items)
            for name, items in warnings.items()
        },
    }
    return report, warnings


def validate_converted_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    malformed: list[dict[str, Any]] = []
    for row in rows:
        errors: list[str] = []
        dist = row.get("type_distribution")
        if not isinstance(dist, dict) or not dist:
            errors.append("missing type_distribution")
        else:
            try:
                probabilities = [float(value) for value in dist.values()]
            except Exception:
                probabilities = []
                errors.append("non-numeric distribution probability")
            if probabilities and abs(sum(probabilities) - 1.0) > 1e-6:
                errors.append("distribution does not sum to 1")
            top_type = row.get("top_type")
            if top_type not in dist:
                errors.append("top_type not present in type_distribution")
            elif probabilities and top_type != max(dist, key=dist.get):
                errors.append("top_type is not argmax")
        if not row.get("id"):
            errors.append("missing id")
        if normalize_context_type(str(row.get("context_type", ""))) not in CONTEXT_ALIASES.values():
            errors.append("unknown context_type")
        if errors:
            malformed.append({"id": row.get("id"), "errors": errors})
    return malformed


def normalize_context_type(context_type: str) -> str:
    return CONTEXT_ALIASES.get(context_type, context_type)


def summarize_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top_probabilities = [top_probability(row) for row in rows]
    entropies = [entropy(row.get("type_distribution", {})) for row in rows]
    non_pii_probabilities = [float(row.get("type_distribution", {}).get("NON_PII", 0.0)) for row in rows]
    top_counts: dict[str, int] = {}
    label_stats: dict[str, dict[str, float]] = {}
    for row in rows:
        top_type = str(row.get("top_type", ""))
        top_counts[top_type] = top_counts.get(top_type, 0) + 1
        for label, probability in row.get("type_distribution", {}).items():
            stats = label_stats.setdefault(str(label), {"candidate_count": 0, "top_type_count": 0, "probability_sum": 0.0})
            stats["candidate_count"] += 1
            stats["probability_sum"] += float(probability)
            if label == top_type:
                stats["top_type_count"] += 1
    return {
        "count": len(rows),
        "top_probability": numeric_summary(top_probabilities, include_max=True),
        "entropy": numeric_summary(entropies, include_max=False),
        "non_pii_probability_mean": round(mean(non_pii_probabilities), 6),
        "top_type_counts": dict(sorted(top_counts.items())),
        "label_distribution": summarize_label_distribution(label_stats),
    }


def summarize_label_distribution(label_stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for label, stats in sorted(label_stats.items()):
        candidate_count = int(stats["candidate_count"])
        summary[label] = {
            "candidate_count": candidate_count,
            "top_type_count": int(stats["top_type_count"]),
            "probability_sum": round(float(stats["probability_sum"]), 6),
            "probability_mean": round(float(stats["probability_sum"]) / candidate_count, 6) if candidate_count else 0.0,
        }
    return summary


def zero_example_label_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for label in ZERO_EXAMPLE_LABELS:
        expected_group = f"zero_example_{label.lower()}"
        label_rows = [
            row
            for row in rows
            if label in row.get("candidate_labels", [])
            and (str(row.get("ambiguity_group", "")) == expected_group or row.get("target_label") == label)
        ]
        coverage[label] = {
            "count": len(label_rows),
            "top_type_count": sum(1 for row in label_rows if row.get("top_type") == label),
            "mean_probability": round(mean([float(row.get("type_distribution", {}).get(label, 0.0)) for row in label_rows]), 6),
        }
    return coverage


def collect_warnings(row: dict[str, Any], context: str, warnings: dict[str, list[dict[str, Any]]]) -> None:
    dist = row.get("type_distribution", {})
    top_prob = top_probability(row)
    if context == "bare_span" and top_prob > 0.70:
        warnings["bare_span_overconfident"].append(warning_example(row, "bare span top_probability > 0.70"))
    if (
        context == "bare_span"
        and is_date_like(str(row.get("span_value", "")))
        and float(dist.get("DATE_OF_BIRTH", 0.0)) > 0.60
    ):
        warnings["bare_date_dob_overconfident"].append(warning_example(row, "bare date-like span has DATE_OF_BIRTH probability > 0.60"))
    if context == "weak_context" and top_prob > 0.80:
        warnings["weak_context_overconfident"].append(warning_example(row, "weak context top_probability > 0.80"))
    if context == "reverse_negative" and row.get("top_type") != "NON_PII":
        warnings["reverse_negative_non_pii_failure"].append(warning_example(row, "reverse/negative context did not rank NON_PII first"))
    if context == "hard_negative" and row.get("top_type") != "NON_PII":
        warnings["hard_negative_non_pii_failure"].append(warning_example(row, "hard-negative context did not rank NON_PII first"))
    if context == "strong_positive" and top_prob < 0.50:
        warnings["strong_positive_not_confident"].append(warning_example(row, "strong positive context top_probability < 0.50"))


def warning_example(row: dict[str, Any], reason: str) -> dict[str, Any]:
    dist = row.get("type_distribution", {})
    top_type = row.get("top_type")
    return {
        "id": row.get("id"),
        "reason": reason,
        "context_type": normalize_context_type(str(row.get("context_type", ""))),
        "ambiguity_group": row.get("ambiguity_group"),
        "span_value": row.get("span_value"),
        "context": row.get("context"),
        "top_type": top_type,
        "top_probability": round(float(dist.get(top_type, 0.0)), 6) if top_type else 0.0,
        "non_pii_probability": round(float(dist.get("NON_PII", 0.0)), 6),
        "date_of_birth_probability": round(float(dist.get("DATE_OF_BIRTH", 0.0)), 6),
        "type_distribution": dist,
    }


def top_probability(row: dict[str, Any]) -> float:
    dist = row.get("type_distribution", {})
    if not dist:
        return 0.0
    top_type = row.get("top_type")
    if top_type in dist:
        return float(dist[top_type])
    return float(max(dist.values()))


def entropy(distribution: dict[str, Any]) -> float:
    total = 0.0
    for value in distribution.values():
        probability = float(value)
        if probability > 0:
            total -= probability * math.log2(probability)
    return round(total, 6)


def numeric_summary(values: list[float], *, include_max: bool) -> dict[str, float]:
    summary = {
        "mean": round(mean(values), 6),
        "median": round(statistics.median(values), 6) if values else 0.0,
    }
    if include_max:
        summary["max"] = round(max(values), 6) if values else 0.0
    return summary


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def is_date_like(value: str) -> bool:
    parts = value.strip().replace("-", "/").split("/")
    return len(parts) == 3 and all(part.isdigit() for part in parts) and len(parts[-1]) in {2, 4}


def build_quality_reports(
    root: Path | str = ".",
    *,
    converted_path: Path | None = None,
    raw_path: Path | None = None,
    pilot_report_path: Path | None = None,
    output_suffix: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root)
    converted_path = converted_path or root / "data" / "generated" / "stage2_vllm_pilot_200_converted.jsonl"
    raw_path = raw_path or root / "data" / "generated" / "stage2_vllm_pilot_200_raw.jsonl"
    pilot_report_path = pilot_report_path or root / "reports" / "stage2_vllm_concurrency_pilot_report.json"
    quality_report_path = root / "reports" / f"stage2_vllm_pilot_quality_report{output_suffix}.json"
    warning_examples_path = root / "reports" / f"stage2_vllm_pilot_warning_examples{output_suffix}.json"

    converted_rows = load_jsonl(converted_path)
    raw_rows = load_jsonl(raw_path)
    pilot_report = json.loads(pilot_report_path.read_text(encoding="utf-8"))
    report, warnings = analyze_quality(converted_rows)
    training_path = root / "pii_schema" / "training_label_space_80.json"
    if training_path.exists():
        training_labels = set(json.loads(training_path.read_text(encoding="utf-8")))
        report["conversion_validation"] = validate_converted_rows(converted_rows, training_labels)
    report["acceptance_criteria"] = {
        "no_malformed_converted_records": report["malformed_converted_record_count"] == 0,
        "bare_date_dob_overconfident_count_is_zero": report["warning_counts"]["bare_date_dob_overconfident_count"] == 0,
        "reverse_negative_non_pii_failure_count": report["warning_counts"]["reverse_negative_non_pii_failure_count"],
        "strong_positive_not_confident_count": report["warning_counts"]["strong_positive_not_confident_count"],
    }
    report["inputs"] = {
        "converted_path": str(converted_path),
        "raw_path": str(raw_path),
        "pilot_report_path": str(pilot_report_path),
        "raw_record_count": len(raw_rows),
        "pilot_best_concurrency": pilot_report.get("best_concurrency"),
    }
    report["teacher_calls_executed_by_quality_analysis"] = 0
    report["merged_into_training_dataset"] = False
    warning_payload = {
        "warning_examples": warnings,
        "warning_counts": report["warning_counts"],
    }
    write_json(quality_report_path, report)
    write_json(warning_examples_path, warning_payload)
    return report, warning_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--converted-path")
    parser.add_argument("--raw-path")
    parser.add_argument("--pilot-report-path")
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args(argv)
    report, _warnings = build_quality_reports(
        args.root,
        converted_path=Path(args.converted_path) if args.converted_path else None,
        raw_path=Path(args.raw_path) if args.raw_path else None,
        pilot_report_path=Path(args.pilot_report_path) if args.pilot_report_path else None,
        output_suffix=args.output_suffix,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["malformed_converted_record_count"] != 0:
        raise SystemExit("malformed converted records found")
    if report["warning_counts"]["bare_date_dob_overconfident_count"] != 0:
        raise SystemExit("bare date DOB overconfidence found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
