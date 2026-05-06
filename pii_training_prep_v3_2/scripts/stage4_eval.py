"""Stage 4 structured evaluation: OPF-only, Qwen-only, Hybrid, Policy sweep, Smoke investigation."""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def load_records(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def task_a_opf_only_eval(root: Path, reports_dir: Path) -> dict[str, Any]:
    opf_eval = read_json(root / "reports" / "stage3b_opf_hard_test_eval.json")
    metrics = opf_eval.get("metrics", {})
    summary = opf_eval.get("summary", {})

    detection = {
        "precision": metrics.get("detection.precision"),
        "recall": metrics.get("detection.recall"),
        "f1": metrics.get("detection.f1"),
        "f2": metrics.get("detection.f2"),
        "span_precision": metrics.get("detection.span.precision"),
        "span_recall": metrics.get("detection.span.recall"),
        "span_f1": metrics.get("detection.span.f1"),
        "span_f2": metrics.get("detection.span.f2"),
    }

    per_label = {}
    for key, val in metrics.items():
        if key.startswith("by_class.") and ".span.f1" in key:
            label = key[len("by_class."):].split(".span.f1")[0]
            per_label[label] = {
                "precision": metrics.get(f"by_class.{label}.span.precision"),
                "recall": metrics.get(f"by_class.{label}.span.recall"),
                "f1": val,
                "f2": metrics.get(f"by_class.{label}.span.f2"),
            }

    missed_labels = [l for l, m in per_label.items() if m["f1"] is not None and m["f1"] < 0.50]
    low_recall = [l for l, m in per_label.items() if m.get("recall") is not None and m["recall"] < 0.50]

    error_examples = []
    test_records = load_records(root / "data" / "train" / "opf_test_opf_format.jsonl")
    predictions = load_records(root / "reports" / "stage3b_opf_hard_test_predictions.jsonl")

    pred_map = {}
    for p in predictions:
        pid = p.get("example_id") or p.get("id", "")
        pred_map[pid] = p

    for rec in test_records[:500]:
        rec_id = rec.get("id", "")
        gold_spans = rec.get("spans", {})
        if isinstance(gold_spans, list):
            gold_labels = set(s.get("label", "") for s in gold_spans)
        else:
            gold_labels = set(gold_spans.keys())
        pred = pred_map.get(rec_id, {})
        pred_spans = pred.get("predicted_spans", [])
        pred_labels = set(s.get("label", "") for s in pred_spans)
        missing = gold_labels - pred_labels
        if missing:
            error_examples.append({
                "id": rec_id,
                "text_snippet": rec.get("text", "")[:200],
                "gold_labels": sorted(gold_labels),
                "predicted_labels": sorted(pred_labels),
                "missing": sorted(missing),
            })

    report = {
        "task": "A_opf_only_eval",
        "checkpoint": "runs/opf_hard_79",
        "test_examples": summary.get("examples"),
        "test_tokens": summary.get("tokens"),
        "loss": summary.get("loss"),
        "token_accuracy": summary.get("token_accuracy"),
        "detection": detection,
        "per_label_span_f1": per_label,
        "missed_labels_f1_below_050": missed_labels,
        "low_recall_labels": low_recall,
    }
    write_json(reports_dir / "stage4_opf_only_eval.json", report)
    write_jsonl(reports_dir / "stage4_opf_only_error_examples.jsonl", error_examples)

    print(f"  Task A: {len(per_label)} labels, {len(missed_labels)} missed (F1<0.5), {len(error_examples)} error examples")
    return report


def task_b_qwen_head_only_eval(root: Path, reports_dir: Path) -> dict[str, Any]:
    head_eval = read_json(root / "reports" / "stage3a_head_eval_last_linear.json")
    metrics = head_eval.get("metrics", {})
    after_temp = metrics.get("after_temperature", {})

    test_metrics = after_temp.get("test", {})
    dev_metrics = after_temp.get("dev", {})

    per_label = test_metrics.get("per_label_top1_accuracy", {})
    confusion = after_temp.get("test", {}).get("confusion_top_pairs", [])

    low_accuracy = {l: a for l, a in per_label.items() if a < 0.80}
    non_pii_acc = test_metrics.get("non_pii_accuracy")

    high_entropy = []
    test_records = load_records(root / "data" / "train" / "qwen_spancls_test.jsonl")
    for rec in test_records[:200]:
        dist = rec.get("target_distribution", {})
        nz = sum(1 for v in dist.values() if float(v) > 0.01)
        if nz >= 3:
            high_entropy.append({
                "id": rec.get("id", ""),
                "value": rec.get("value", ""),
                "top_type": rec.get("top_type", ""),
                "nonzero_labels": nz,
                "text_snippet": rec.get("text", "")[:150],
            })

    report = {
        "task": "B_qwen_head_only_eval",
        "experiment": "last_linear",
        "temperature": head_eval.get("temperature", 1.0),
        "test": {
            "top1_accuracy": test_metrics.get("top1_accuracy"),
            "top3_accuracy": test_metrics.get("top3_accuracy"),
            "nll": test_metrics.get("nll"),
            "brier_score": test_metrics.get("brier_score"),
            "ece": test_metrics.get("ece"),
            "non_pii_accuracy": non_pii_acc,
            "example_count": test_metrics.get("example_count"),
        },
        "dev": {
            "top1_accuracy": dev_metrics.get("top1_accuracy"),
            "nll": dev_metrics.get("nll"),
            "ece": dev_metrics.get("ece"),
        },
        "per_label_accuracy": per_label,
        "low_accuracy_labels_below_080": low_accuracy,
        "confusion_top20": confusion[:20],
        "high_entropy_examples": high_entropy[:30],
    }
    write_json(reports_dir / "stage4_qwen_head_only_eval.json", report)
    write_jsonl(reports_dir / "stage4_qwen_head_only_error_examples.jsonl", [
        {"label": l, "accuracy": a, "note": "accuracy below 0.80"}
        for l, a in sorted(low_accuracy.items(), key=lambda x: x[1])
    ])

    print(f"  Task B: top1={test_metrics.get('top1_accuracy')}, NON_PII_acc={non_pii_acc}, low_accuracy_labels={len(low_accuracy)}")
    return report


def task_c_hybrid_eval(root: Path, reports_dir: Path, sample_size: int = 500) -> dict[str, Any]:
    from opf_inference import OPFDetector
    from integrated_pipeline import PolicyLayer, run_integrated_pipeline

    opf = OPFDetector.from_project_root(root)
    policy = PolicyLayer(csv_path=str(root / "docs" / "Data Sensitivity.csv"))

    random.seed(42)
    test_records = load_records(root / "data" / "train" / "opf_test_opf_format.jsonl")
    if sample_size < len(test_records):
        test_records = random.sample(test_records, sample_size)

    opf_recall = Counter()
    qwen_correct = Counter()
    qwen_total = Counter()
    decision_counts = Counter()
    fn_examples = []
    fp_reduced_examples = []
    error_examples = []

    for idx, rec in enumerate(test_records):
        if idx % 100 == 0:
            print(f"    hybrid: {idx}/{len(test_records)}", flush=True)

        text = rec.get("text", "")
        gold_spans_raw = rec.get("spans", {})
        if isinstance(gold_spans_raw, dict):
            gold_labels_set = set(gold_spans_raw.keys())
            gold_spans_list = []
            for label, positions in gold_spans_raw.items():
                for pos in positions:
                    gold_spans_list.append({"label": label, "start": pos[0], "end": pos[1]})
        else:
            gold_labels_set = set(s.get("label", "") for s in gold_spans_raw)
            gold_spans_list = list(gold_spans_raw)

        if not gold_labels_set:
            continue

        opf_result = opf.detect_spans(text)
        opf_spans = opf_result.get("candidate_spans", [])
        opf_labels = set(s.get("opf_top_type", "") for s in opf_spans)
        opf_label_set = set(opf_labels)

        found = gold_labels_set & opf_label_set
        missed = gold_labels_set - opf_label_set
        opf_recall["found"] += len(found)
        opf_recall["total"] += len(gold_labels_set)
        opf_recall["missed"] += len(missed)

        if missed:
            fn_examples.append({
                "id": rec.get("id", ""),
                "text_snippet": text[:200],
                "gold_labels": sorted(gold_labels_set),
                "opf_labels": sorted(opf_label_set),
                "missed_by_opf": sorted(missed),
            })

        result = run_integrated_pipeline(text, None, opf, policy)
        spans = result.get("spans", [])
        for s in spans:
            decision = s.get("decision", "")
            decision_counts[decision] += 1

        for s in spans:
            span_label = s.get("opf_top_type") or s.get("top_type", "")
            if span_label in gold_labels_set:
                qwen_correct[span_label] += 1
            qwen_total[span_label] += 1

        if len(opf_spans) > len(spans):
            fp_reduced_examples.append({
                "id": rec.get("id", ""),
                "text_snippet": text[:200],
                "opf_span_count": len(opf_spans),
                "pipeline_span_count": len(spans),
                "reduction": len(opf_spans) - len(spans),
            })

        if idx >= len(test_records) - 1:
            break

    total_found = opf_recall["found"]
    total_gold = opf_recall["total"]
    opf_label_recall = total_found / max(1, total_gold)

    per_label_qwen_acc = {}
    for label in qwen_total:
        per_label_qwen_acc[label] = qwen_correct[label] / max(1, qwen_total[label])

    report = {
        "task": "C_hybrid_eval",
        "sample_size": len(test_records),
        "opf_candidate_recall": round(opf_label_recall, 4),
        "opf_found": total_found,
        "opf_missed": opf_recall["missed"],
        "gold_total_labels": total_gold,
        "qwen_rescoring_accuracy_per_label": per_label_qwen_acc,
        "final_decision_distribution": dict(decision_counts),
        "false_negative_examples": fn_examples[:30],
        "fp_reduced_by_non_pii": fp_reduced_examples[:30],
    }
    write_json(reports_dir / "stage4_hybrid_eval.json", report)
    write_jsonl(reports_dir / "stage4_hybrid_error_examples.jsonl", fn_examples[:50])

    print(f"  Task C: OPF recall={opf_label_recall:.4f}, Qwen rescore labels={len(per_label_qwen_acc)}, decisions={dict(decision_counts)}")
    return report


def task_d_policy_sweep(root: Path, reports_dir: Path, sample_size: int = 500) -> dict[str, Any]:
    from opf_inference import OPFDetector
    from integrated_pipeline import PolicyLayer, run_integrated_pipeline

    opf = OPFDetector.from_project_root(root)

    random.seed(42)
    dev_records = load_records(root / "data" / "train" / "opf_dev_opf_format.jsonl")
    if sample_size < len(dev_records):
        dev_records = random.sample(dev_records, sample_size)

    redact_thresholds = [0.30, 0.40, 0.50, 0.60]
    review_thresholds = [0.15, 0.20, 0.25, 0.30]

    print(f"    sweeping {len(redact_thresholds)}x{len(review_thresholds)} thresholds on {len(dev_records)} dev examples...", flush=True)

    sweep_results = []
    for rt in redact_thresholds:
        for rvt in review_thresholds:
            policy = PolicyLayer(
                csv_path=str(root / "docs" / "Data Sensitivity.csv"),
                redact_threshold=rt,
                review_threshold=rvt,
            )
            redact_correct = 0
            redact_total = 0
            review_correct = 0
            review_total = 0
            ignore_fn = 0
            ignore_total = 0
            total_spans = 0

            for idx, rec in enumerate(dev_records):
                text = rec.get("text", "")
                gold_spans_raw = rec.get("spans", {})
                if isinstance(gold_spans_raw, dict):
                    gold_labels_set = set(gold_spans_raw.keys())
                else:
                    gold_labels_set = set(s.get("label", "") for s in gold_spans_raw)

                if not gold_labels_set:
                    continue

                opf_result = opf.detect_spans(text)
                result = run_integrated_pipeline(text, None, opf, policy)
                spans = result.get("spans", [])

                for s in spans:
                    span_label = s.get("opf_top_type") or s.get("top_type", "")
                    decision = s.get("decision", "")
                    total_spans += 1

                    is_gold = span_label in gold_labels_set

                    if decision == "redact":
                        redact_total += 1
                        if is_gold:
                            redact_correct += 1
                    elif decision == "review":
                        review_total += 1
                        if is_gold:
                            review_correct += 1
                    elif decision == "ignore":
                        ignore_total += 1
                        if is_gold:
                            ignore_fn += 1

            sweep_results.append({
                "redact_threshold": rt,
                "review_threshold": rvt,
                "redact_precision": round(redact_correct / max(1, redact_total), 4),
                "review_recall": round(review_correct / max(1, review_total), 4) if review_total else 0,
                "ignore_false_negative_rate": round(ignore_fn / max(1, ignore_total), 4) if ignore_total else 0,
                "redact_total": redact_total,
                "review_total": review_total,
                "ignore_total": ignore_total,
                "total_spans": total_spans,
            })

    sweep_results.sort(key=lambda x: (x["redact_precision"], x["review_recall"]), reverse=True)
    best = sweep_results[0] if sweep_results else None

    recommended = {
        "best_by_redact_precision": best,
        "default": {"redact_threshold": 0.60, "review_threshold": 0.25},
        "lenient": {"redact_threshold": 0.40, "review_threshold": 0.20},
        "strict": {"redact_threshold": 0.60, "review_threshold": 0.30},
        "note": "Use lenient for high recall, strict for high precision. Default balances both.",
    }

    sweep_report = {
        "task": "D_policy_threshold_sweep",
        "dev_sample_size": len(dev_records),
        "thresholds_tested": len(sweep_results),
        "results": sweep_results,
    }
    write_json(reports_dir / "stage4_policy_threshold_sweep_dev.json", sweep_report)
    write_json(reports_dir / "stage4_policy_recommended_thresholds.json", recommended)

    print(f"  Task D: {len(sweep_results)} combinations tested, best: R={best['redact_threshold']}/r={best['review_threshold']} P={best['redact_precision']}")
    return recommended


def task_e_smoke_miss_investigation(root: Path, reports_dir: Path) -> dict[str, Any]:
    from opf_inference import OPFDetector

    opf = OPFDetector.from_project_root(root)
    canonical_labels = set(json.loads((root / "pii_schema" / "canonical_labels_79.json").read_text(encoding="utf-8")))
    opf_labels = set(json.loads((root / "pii_schema" / "opf_label_space_79.json").read_text(encoding="utf-8"))["span_class_names"])

    investigations = []

    test_cases = [
        {
            "id": "ambiguous_date",
            "text": "The deadline for submission is 15/06/2025 please confirm.",
            "expected_label": "DATE_OF_BIRTH",
        },
        {
            "id": "bsb_account",
            "text": "Please transfer funds to BSB 062-000 account 12345678 for payment.",
            "expected_label": "BANK_ACCOUNT_NUMBER",
        },
        {
            "id": "medicare",
            "text": "Medicare number is 2123 45678 1 and expiry is 01/2026.",
            "expected_label": "MEDICARE_NUMBER",
        },
        {
            "id": "gender_male",
            "text": "The applicant identifies as male and prefers he/him pronouns.",
            "expected_label": "GENDER",
        },
    ]

    for tc in test_cases:
        label = tc["expected_label"]
        inv = {
            "id": tc["id"],
            "text": tc["text"],
            "expected_label": label,
            "label_in_canonical_79": label in canonical_labels,
            "label_in_opf_label_space": label in opf_labels,
            "label_in_opf_span_classes": label in opf_labels,
        }

        opf_result = opf.detect_spans(tc["text"])
        opf_spans = opf_result.get("candidate_spans", [])
        inv["opf_detected"] = len(opf_spans) > 0
        inv["opf_span_count"] = len(opf_spans)
        inv["opf_spans"] = [
            {"start": s["start"], "end": s["end"], "label": s.get("opf_top_type", ""), "value": s["value"]}
            for s in opf_spans
        ]

        label_in_result = any(s.get("opf_top_type") == label for s in opf_spans)
        inv["expected_label_detected"] = label_in_result

        train_path = root / "data" / "train" / "opf_train_opf_format.jsonl"
        count = 0
        for rec in iter_jsonl(train_path):
            spans = rec.get("spans", {})
            if isinstance(spans, dict) and label in spans:
                count += len(spans[label])
            elif isinstance(spans, list):
                count += sum(1 for s in spans if s.get("label") == label)
            if count >= 100:
                break
        inv["training_examples_with_label"] = min(count, 100)
        inv["has_sufficient_training"] = count >= 10

        if not label_in_result:
            if not inv["has_sufficient_training"]:
                inv["root_cause"] = f"insufficient_training_data (found ~{count} examples)"
            elif label in opf_labels:
                inv["root_cause"] = "opf_model_failure_label_in_space"
            else:
                inv["root_cause"] = "label_not_in_opf_model_label_space"

        investigations.append(inv)

    report = {
        "task": "E_smoke_miss_investigation",
        "canonical_labels_count": len(canonical_labels),
        "opf_label_space_count": len(opf_labels),
        "investigations": investigations,
        "summary": {
            "labels_in_opf_space": all(inv["label_in_opf_label_space"] for inv in investigations),
            "detected_any": sum(1 for inv in investigations if inv["opf_detected"]),
            "detected_expected": sum(1 for inv in investigations if inv["expected_label_detected"]),
            "total_tested": len(investigations),
        },
    }
    write_json(reports_dir / "stage4_smoke_miss_investigation.json", report)

    print(f"  Task E: {len(investigations)} cases investigated, {report['summary']['detected_expected']}/{report['summary']['total_tested']} expected labels detected")
    return report


def main() -> int:
    root = Path("/home/admin/ZYX/pii_training_prep_v3_2")
    reports_dir = root / "reports"

    sys.path.insert(0, str(root / "src" / "pii_prep"))

    print("=" * 60)
    print("Stage 4 Structured Evaluation")
    print("=" * 60)

    t0 = time.time()

    print("\nTask A: OPF-only evaluation...")
    task_a_opf_only_eval(root, reports_dir)

    print("\nTask B: Qwen-head-only evaluation...")
    task_b_qwen_head_only_eval(root, reports_dir)

    print("\nTask C: Hybrid evaluation (500 samples)...")
    task_c_hybrid_eval(root, reports_dir, sample_size=500)

    print("\nTask D: Policy threshold sweep (500 dev samples)...")
    task_d_policy_sweep(root, reports_dir, sample_size=500)

    print("\nTask E: Smoke miss investigation...")
    task_e_smoke_miss_investigation(root, reports_dir)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All evaluations complete in {elapsed:.1f}s")
    print(f"Reports in: {reports_dir}")
    for f in sorted(reports_dir.glob("stage4_*")):
        print(f"  {f.name} ({f.stat().st_size} bytes)")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
