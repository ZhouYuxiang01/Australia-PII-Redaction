#!/usr/bin/env python3
"""Search OPF Viterbi transition biases on a held-out dev subset.

This script does not train or modify the checkpoint by default. It evaluates a
small, explicit candidate set and writes the best OPF-compatible calibration
artifact to the output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPF = Path("/home/admin/miniconda3/envs/opf/bin/opf")
DEFAULT_GOLD = ROOT / "data" / "processed" / "data_opf_v2b" / "dev.jsonl"
DEFAULT_CHECKPOINT = ROOT / "runs" / "final" / "opf_73class_v3_full" / "checkpoint"
DEFAULT_TAXONOMY = ROOT / "configs" / "taxonomy_v1.1.1.yaml"
DEFAULT_OUT_DIR = (
    ROOT / "runs" / "final" / "opf_73class_v3_full" / "viterbi_calibration_search"
)
EVAL_SCRIPT = ROOT / "scripts" / "eval_char_spans_v2.py"

BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    biases: dict[str, float]


def zero_biases() -> dict[str, float]:
    return {key: 0.0 for key in BIAS_KEYS}


def make_candidate(name: str, **overrides: float) -> Candidate:
    biases = zero_biases()
    for key, value in overrides.items():
        if key not in biases:
            raise KeyError(f"unknown Viterbi bias key: {key}")
        biases[key] = float(value)
    return Candidate(name=name, biases=biases)


def default_candidates() -> list[Candidate]:
    return [
        make_candidate("baseline_zero"),
        make_candidate(
            "recall_start_005",
            transition_bias_background_stay=-0.02,
            transition_bias_background_to_start=0.05,
        ),
        make_candidate(
            "recall_start_010",
            transition_bias_background_stay=-0.04,
            transition_bias_background_to_start=0.10,
        ),
        make_candidate(
            "precision_start_005",
            transition_bias_background_stay=0.02,
            transition_bias_background_to_start=-0.05,
        ),
        make_candidate(
            "precision_start_010",
            transition_bias_background_stay=0.04,
            transition_bias_background_to_start=-0.10,
        ),
        make_candidate(
            "longer_spans",
            transition_bias_inside_to_continue=0.05,
            transition_bias_inside_to_end=-0.03,
        ),
        make_candidate(
            "shorter_spans",
            transition_bias_inside_to_continue=-0.05,
            transition_bias_inside_to_end=0.03,
        ),
        make_candidate(
            "adjacent_spans",
            transition_bias_end_to_start=0.04,
        ),
        make_candidate(
            "conservative_end",
            transition_bias_end_to_background=0.03,
            transition_bias_end_to_start=-0.03,
        ),
        make_candidate(
            "recall_lenient_combo",
            transition_bias_background_stay=-0.04,
            transition_bias_background_to_start=0.08,
            transition_bias_inside_to_continue=0.03,
        ),
        make_candidate(
            "precision_combo",
            transition_bias_background_stay=0.04,
            transition_bias_background_to_start=-0.08,
            transition_bias_inside_to_continue=-0.02,
            transition_bias_end_to_background=0.03,
        ),
    ]


def wide_candidates() -> list[Candidate]:
    return [
        make_candidate("baseline_zero"),
        make_candidate(
            "wide_recall_start_025",
            transition_bias_background_stay=-0.10,
            transition_bias_background_to_start=0.25,
        ),
        make_candidate(
            "wide_recall_start_050",
            transition_bias_background_stay=-0.20,
            transition_bias_background_to_start=0.50,
        ),
        make_candidate(
            "wide_precision_start_025",
            transition_bias_background_stay=0.10,
            transition_bias_background_to_start=-0.25,
        ),
        make_candidate(
            "wide_precision_start_050",
            transition_bias_background_stay=0.20,
            transition_bias_background_to_start=-0.50,
        ),
        make_candidate(
            "wide_longer_spans",
            transition_bias_inside_to_continue=0.25,
            transition_bias_inside_to_end=-0.15,
        ),
        make_candidate(
            "wide_shorter_spans",
            transition_bias_inside_to_continue=-0.25,
            transition_bias_inside_to_end=0.15,
        ),
        make_candidate(
            "wide_recall_span_combo",
            transition_bias_background_stay=-0.12,
            transition_bias_background_to_start=0.30,
            transition_bias_inside_to_continue=0.15,
            transition_bias_end_to_start=0.10,
        ),
        make_candidate(
            "wide_precision_short_combo",
            transition_bias_background_stay=0.12,
            transition_bias_background_to_start=-0.30,
            transition_bias_inside_to_continue=-0.10,
            transition_bias_inside_to_end=0.10,
            transition_bias_end_to_background=0.10,
        ),
    ]


def candidate_subset(name: str) -> list[Candidate]:
    candidates = default_candidates()
    if name == "all":
        return candidates
    if name == "quick":
        return candidates[:5]
    if name == "wide":
        return wide_candidates()
    raise ValueError(name)


def write_calibration_artifact(path: Path, biases: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "operating_points": {
            "default": {
                "biases": {key: float(biases[key]) for key in BIAS_KEYS},
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sample_jsonl(src: Path, dst: Path, sample_size: int, seed: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sample_size > 0 and sample_size < len(lines):
        rng = random.Random(seed)
        selected = sorted(rng.sample(range(len(lines)), sample_size))
        lines = [lines[i] for i in selected]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def run(cmd: list[str], *, cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_char_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    typed = metrics["typed"]
    partial = typed["partial"]
    return {
        "typed_precision": typed["precision"],
        "typed_recall": typed["recall"],
        "typed_f1": typed["f1"],
        "typed_tp": typed["tp"],
        "typed_fp": typed["fp"],
        "typed_fn": typed["fn"],
        "typed_partial_precision": partial["precision"],
        "typed_partial_recall": partial["recall"],
        "typed_partial_f1": partial["f1"],
        "over_redaction_cost": typed["over_redaction_cost"],
        "under_redaction_cost": typed["under_redaction_cost"],
    }


def select_best(rows: list[dict[str, Any]], objective: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("no candidate rows to rank")
    if objective not in rows[0]:
        raise KeyError(f"objective {objective!r} not present in candidate rows")
    return max(
        rows,
        key=lambda row: (
            float(row[objective]),
            float(row.get("typed_precision", 0.0)),
            float(row.get("typed_recall", 0.0)),
            -float(row.get("under_redaction_cost", 0.0)),
        ),
    )


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    metric_fields = [
        "candidate",
        "typed_precision",
        "typed_recall",
        "typed_f1",
        "typed_partial_f1",
        "typed_tp",
        "typed_fp",
        "typed_fn",
        "over_redaction_cost",
        "under_redaction_cost",
    ]
    fields = metric_fields + list(BIAS_KEYS)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_candidate(
    candidate: Candidate,
    *,
    gold_path: Path,
    out_dir: Path,
    opf_bin: Path,
    checkpoint: Path,
    taxonomy: Path,
    device: str,
    n_ctx: int | None,
    window_batch_size: int | None,
) -> dict[str, Any]:
    candidate_dir = out_dir / "candidates" / candidate.name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = candidate_dir / "viterbi_calibration.json"
    predictions_path = candidate_dir / "predictions.jsonl"
    opf_metrics_path = candidate_dir / "opf_metrics.json"
    char_metrics_path = candidate_dir / "char_metrics.json"

    write_calibration_artifact(calibration_path, candidate.biases)

    eval_cmd = [
        str(opf_bin),
        "eval",
        str(gold_path),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--decode-mode",
        "viterbi",
        "--viterbi-calibration-path",
        str(calibration_path),
        "--predictions-out",
        str(predictions_path),
        "--metrics-out",
        str(opf_metrics_path),
    ]
    if n_ctx is not None:
        eval_cmd.extend(["--n-ctx", str(n_ctx)])
    if window_batch_size is not None:
        eval_cmd.extend(["--window-batch-size", str(window_batch_size)])
    run(eval_cmd, cwd=ROOT)

    run(
        [
            "python3",
            str(EVAL_SCRIPT),
            "--gold",
            str(gold_path),
            "--pred",
            str(predictions_path),
            "--taxonomy",
            str(taxonomy),
            "--out",
            str(char_metrics_path),
        ],
        cwd=ROOT,
    )

    row = {
        "candidate": candidate.name,
        **flatten_char_metrics(load_json(char_metrics_path)),
        **candidate.biases,
        "calibration_path": str(calibration_path),
        "predictions_path": str(predictions_path),
        "opf_metrics_path": str(opf_metrics_path),
        "char_metrics_path": str(char_metrics_path),
    }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--opf-bin", type=Path, default=DEFAULT_OPF)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-set", choices=("quick", "all", "wide"), default="quick")
    parser.add_argument(
        "--objective",
        choices=("typed_f1", "typed_partial_f1", "typed_recall", "typed_precision"),
        default="typed_f1",
    )
    parser.add_argument("--n-ctx", type=int, default=None)
    parser.add_argument("--window-batch-size", type=int, default=None)
    parser.add_argument(
        "--install",
        action="store_true",
        help="also write the best artifact to <checkpoint>/viterbi_calibration.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.out_dir / f"dev_sample_{args.sample_size}_seed{args.seed}.jsonl"
    sample_count = sample_jsonl(args.gold, sample_path, args.sample_size, args.seed)

    rows: list[dict[str, Any]] = []
    for candidate in candidate_subset(args.candidate_set):
        rows.append(
            run_candidate(
                candidate,
                gold_path=sample_path,
                out_dir=args.out_dir,
                opf_bin=args.opf_bin,
                checkpoint=args.checkpoint,
                taxonomy=args.taxonomy,
                device=args.device,
                n_ctx=args.n_ctx,
                window_batch_size=args.window_batch_size,
            )
        )
        write_summary_csv(args.out_dir / "candidate_summary.csv", rows)
        (args.out_dir / "candidate_results.json").write_text(
            json.dumps(rows, indent=2) + "\n",
            encoding="utf-8",
        )

    best = select_best(rows, args.objective)
    best_biases = {key: float(best[key]) for key in BIAS_KEYS}
    best_artifact = args.out_dir / "best_viterbi_calibration.json"
    write_calibration_artifact(best_artifact, best_biases)

    report = {
        "objective": args.objective,
        "sample_path": str(sample_path),
        "sample_count": sample_count,
        "candidate_set": args.candidate_set,
        "best": best,
        "best_artifact": str(best_artifact),
    }
    (args.out_dir / "best_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.install:
        install_path = args.checkpoint / "viterbi_calibration.json"
        write_calibration_artifact(install_path, best_biases)
        report["installed_artifact"] = str(install_path)
        (args.out_dir / "best_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "best_candidate": best["candidate"],
                "objective": args.objective,
                "objective_value": best[args.objective],
                "best_artifact": str(best_artifact),
                "summary_csv": str(args.out_dir / "candidate_summary.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
