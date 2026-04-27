#!/usr/bin/env python3
"""
run_opf_pipeline.py
===================

Official-OPF launcher for the AU PII 73-class Privacy Filter experiment.

This script deliberately delegates model training/evaluation to the OPF CLI
(`opf train` / `opf eval`) instead of re-implementing training through
Transformers. That keeps OPF's custom model stack, BIOES label-space handling,
checkpoint format, and constrained decoding path aligned with the official repo.

Expected project files
----------------------
- au_pii_19000_final.json                 # raw generated source data
- taxonomy_v1.1.1.yaml                    # K=73 taxonomy
- custom_label_space_73.v1.1.1.json       # ["O"] + 73 span labels
- prepare_dataset_v2.py                   # raw JSON -> OPF JSONL splits
- validate_taxonomy.py                    # taxonomy / label-space / data checks
- eval_char_spans.py                      # optional char-span metrics

Typical workflow
----------------
1) Smoke test with fresh dataset preparation:

   python run_opf_pipeline.py \
     --raw-json ./au_pii_19000_final.json \
     --taxonomy ./taxonomy_v1.1.1.yaml \
     --label-space ./custom_label_space_73.v1.1.1.json \
     --data-dir ./data_opf \
     --run-dir ./runs/opf_73_smoke \
     --smoke

2) Full training after smoke succeeds:

   python run_opf_pipeline.py \
     --data-dir ./data_opf \
     --taxonomy ./taxonomy_v1.1.1.yaml \
     --label-space ./custom_label_space_73.v1.1.1.json \
     --run-dir ./runs/opf_73class_v1 \
     --train \
     --eval-on-test \
     --char-eval

3) Pass extra OPF training flags after `--train-extra`, for example:

   python run_opf_pipeline.py ... --train --train-extra -- --epochs 3 --device cuda

Notes
-----
OPF CLI flags can vary slightly by installed version. This launcher checks
`opf train --help` and `opf eval --help` for common optional flags before adding
them. If OPF rejects any command, inspect the saved logs under --run-dir/logs.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pip install pyyaml") from exc


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def as_cmd(cmd_text: str) -> list[str]:
    parts = shlex.split(cmd_text)
    if not parts:
        raise ValueError("empty command")
    return parts


def quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def run_logged(cmd: Sequence[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    """Run command, stream stdout/stderr to terminal and log file, fail on nonzero."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cmd] {quote_cmd(cmd)}")
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"$ {quote_cmd(cmd)}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            list(map(str, cmd)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        rc = proc.wait()
        logf.write(f"\n[exit_code] {rc}\n")
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def run_capture(cmd: Sequence[str], timeout: int = 30) -> tuple[int, str]:
    try:
        res = subprocess.run(
            list(map(str, cmd)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return res.returncode, res.stdout or ""
    except Exception as exc:  # noqa: BLE001
        return 999, str(exc)


def copy_if_exists(src: Path, dst_dir: Path) -> None:
    if src.is_file():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)


# ---------------------------------------------------------------------------
# Taxonomy / label-space checks
# ---------------------------------------------------------------------------


def load_taxonomy(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def taxonomy_classes(path: Path) -> list[str]:
    doc = load_taxonomy(path)
    classes = [entry["code"] for entry in doc.get("classes", [])]
    if not classes:
        raise ValueError(f"No classes found in taxonomy: {path}")
    if len(classes) != len(set(classes)):
        seen: set[str] = set()
        dupes: list[str] = []
        for c in classes:
            if c in seen:
                dupes.append(c)
            seen.add(c)
        raise ValueError(f"Duplicate class codes in taxonomy: {sorted(set(dupes))}")
    return classes


def label_space_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("span_class_names")
    if not isinstance(names, list) or not names:
        raise ValueError("label-space JSON must contain non-empty span_class_names")
    return [str(x) for x in names]


def validate_label_space(taxonomy: Path, label_space: Path) -> None:
    classes = taxonomy_classes(taxonomy)
    names = label_space_names(label_space)
    expected = ["O"] + classes
    if names != expected:
        raise ValueError(
            "label space does not match taxonomy.\n"
            f"Expected {len(expected)} labels: {expected[:8]} ...\n"
            f"Got      {len(names)} labels: {names[:8]} ..."
        )
    print(f"[ok] taxonomy classes={len(classes)} span_labels={len(names)-1} token_labels={1 + 4 * len(classes)}")


def check_jsonl_schema(path: Path, *, max_rows: int = 20) -> int:
    """Validate first rows have example_id/text and spans or label."""
    ensure_file(path, path.name)
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "text" not in row:
                raise ValueError(f"{path}: row missing `text`")
            if "spans" not in row and "label" not in row:
                raise ValueError(f"{path}: row missing `spans` or `label`")
            if "example_id" not in row:
                print(f"[warn] {path}: row has no example_id; predictions alignment may be harder")
            n += 1
            if n >= max_rows:
                break
    if n == 0:
        raise ValueError(f"{path}: no JSONL records found")
    print(f"[ok] checked {n} rows in {path}")
    return n


def make_jsonl_prefix(src: Path, dst: Path, n: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open("r", encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            fo.write(line)
            count += 1
            if count >= n:
                break
    if count == 0:
        raise ValueError(f"No rows copied from {src}")
    return count


# ---------------------------------------------------------------------------
# OPF command helpers
# ---------------------------------------------------------------------------


def opf_help(opf_cmd: list[str], subcommand: str) -> str:
    rc, text = run_capture(opf_cmd + [subcommand, "--help"], timeout=60)
    if rc != 0:
        print(f"[warn] could not inspect `{quote_cmd(opf_cmd + [subcommand, '--help'])}`:\n{text[:1000]}")
        return ""
    return text


def supports_flag(help_text: str, flag: str) -> bool:
    return flag in help_text


def build_train_cmd(
    *,
    opf_cmd: list[str],
    train_path: Path,
    dev_path: Path,
    label_space: Path,
    output_dir: Path,
    checkpoint: str | None,
    help_text: str,
    extra: list[str],
) -> list[str]:
    cmd = opf_cmd + [
        "train",
        str(train_path),
        "--validation-dataset", str(dev_path),
        "--label-space-json", str(label_space),
        "--output-dir", str(output_dir),
    ]
    if checkpoint:
        if supports_flag(help_text, "--checkpoint"):
            cmd += ["--checkpoint", checkpoint]
        else:
            print("[warn] `opf train --help` does not show --checkpoint; not adding it")
    cmd += extra
    return cmd


def build_eval_cmd(
    *,
    opf_cmd: list[str],
    test_path: Path,
    checkpoint_dir: Path,
    label_space: Path,
    pred_out: Path,
    help_text: str,
    extra: list[str],
) -> list[str]:
    cmd = opf_cmd + ["eval", str(test_path)]
    if supports_flag(help_text, "--checkpoint"):
        cmd += ["--checkpoint", str(checkpoint_dir)]
    else:
        print("[warn] `opf eval --help` does not show --checkpoint; relying on OPF default checkpoint")
    if supports_flag(help_text, "--label-space-json"):
        cmd += ["--label-space-json", str(label_space)]
    if supports_flag(help_text, "--predictions-out"):
        cmd += ["--predictions-out", str(pred_out)]
    else:
        print("[warn] `opf eval --help` does not show --predictions-out; prediction JSONL may not be created")
    cmd += extra
    return cmd


def split_extra(extra: list[str] | None) -> list[str]:
    if not extra:
        return []
    if extra and extra[0] == "--":
        return extra[1:]
    return extra


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--raw-json", default=None, help="raw au_pii_19000_final.json; if set, run prepare_dataset_v2.py")
    p.add_argument("--data-dir", default="data_opf", help="OPF JSONL directory containing train/dev/test.jsonl")
    p.add_argument("--taxonomy", default="taxonomy_v1.1.1.yaml")
    p.add_argument("--label-space", default="custom_label_space_73.v1.1.1.json")
    p.add_argument("--prepare-script", default="prepare_dataset_v2.py")
    p.add_argument("--validate-script", default="validate_taxonomy.py")
    p.add_argument("--char-eval-script", default="eval_char_spans.py")
    p.add_argument("--run-dir", default="runs/opf_73class_v1")
    p.add_argument("--opf-cmd", default="opf", help="OPF command, e.g. `opf` or `python -m opf`")
    p.add_argument("--checkpoint", default=None, help="optional base checkpoint path if your OPF train supports --checkpoint")

    p.add_argument("--prepare-format", choices=("dict", "list"), default="dict")
    p.add_argument("--auto-salt-trials", type=int, default=50)
    p.add_argument("--allow-offset-drops", action="store_true")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="run small train using train/dev prefixes")
    mode.add_argument("--train", action="store_true", help="run full training")
    mode.add_argument("--prepare-only", action="store_true", help="only validate/prepare data; do not train")
    mode.add_argument("--eval-only", action="store_true", help="only run eval on existing checkpoint/run-dir")

    p.add_argument("--smoke-train-rows", type=int, default=100)
    p.add_argument("--smoke-dev-rows", type=int, default=50)
    p.add_argument("--eval-on-test", action="store_true", help="run `opf eval` on test.jsonl after training")
    p.add_argument("--char-eval", action="store_true", help="run eval_char_spans.py if predictions JSONL exists")
    p.add_argument("--predictions-out", default=None, help="prediction JSONL path; default run-dir/test_predictions.jsonl")
    p.add_argument("--dry-run", action="store_true", help="print commands/checks without executing OPF train/eval")

    p.add_argument("--train-extra", nargs=argparse.REMAINDER, help="extra args for `opf train`, put after `--train-extra --`")
    p.add_argument("--eval-extra", nargs=argparse.REMAINDER, help="extra args for `opf eval`, put after `--eval-extra --`")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    taxonomy = Path(args.taxonomy).resolve()
    label_space = Path(args.label_space).resolve()
    data_dir = Path(args.data_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ensure_file(taxonomy, "taxonomy")
    ensure_file(label_space, "label space")
    validate_label_space(taxonomy, label_space)

    # Preserve the exact taxonomy/label-space used for this run.
    copy_if_exists(taxonomy, artifacts_dir)
    copy_if_exists(label_space, artifacts_dir)

    opf_cmd = as_cmd(args.opf_cmd)
    train_help = opf_help(opf_cmd, "train")
    eval_help = opf_help(opf_cmd, "eval")

    run_config = vars(args).copy()
    run_config["timestamp"] = now_stamp()
    run_config["opf_cmd_resolved"] = opf_cmd
    run_config["train_help_seen"] = bool(train_help)
    run_config["eval_help_seen"] = bool(eval_help)
    (run_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Optional full taxonomy/data validation.
    validate_script = Path(args.validate_script).resolve()
    if validate_script.is_file():
        validate_cmd = [sys.executable, str(validate_script), "--taxonomy", str(taxonomy), "--label-space", str(label_space), "--report", str(run_dir / "taxonomy_validation_report.json")]
        if args.raw_json:
            validate_cmd += ["--data", str(Path(args.raw_json).resolve())]
        if not args.dry_run:
            run_logged(validate_cmd, logs_dir / "00_validate_taxonomy.log")
        else:
            print(f"[dry-run] {quote_cmd(validate_cmd)}")
    else:
        print(f"[warn] validate script not found: {validate_script}")

    # Optional dataset preparation.
    if args.raw_json:
        raw_json = Path(args.raw_json).resolve()
        prepare_script = Path(args.prepare_script).resolve()
        ensure_file(raw_json, "raw JSON")
        ensure_file(prepare_script, "prepare script")
        prepare_cmd = [
            sys.executable, str(prepare_script),
            "--src", str(raw_json),
            "--taxonomy", str(taxonomy),
            "--out", str(data_dir),
            "--format", args.prepare_format,
            "--auto-salt-trials", str(args.auto_salt_trials),
        ]
        if args.allow_offset_drops:
            prepare_cmd.append("--allow-offset-drops")
        if not args.dry_run:
            run_logged(prepare_cmd, logs_dir / "01_prepare_dataset.log")
        else:
            print(f"[dry-run] {quote_cmd(prepare_cmd)}")

    # Check prepared data.
    if not args.raw_json and not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}. Provide --raw-json or create data first.")
    ensure_dir(data_dir, "data directory")
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"
    test_path = data_dir / "test.jsonl"
    check_jsonl_schema(train_path)
    check_jsonl_schema(dev_path)
    if test_path.exists():
        check_jsonl_schema(test_path)
    else:
        print(f"[warn] test split not found: {test_path}")

    if args.prepare_only:
        print("[done] prepare-only complete")
        return 0

    # Default to smoke if the user did not explicitly select a training/eval mode.
    # This prevents accidental full training from a bare command invocation.
    if not (args.smoke or args.train or args.prepare_only or args.eval_only):
        print("[info] no mode selected; defaulting to --smoke for safety")
        args.smoke = True

    # Smoke prefixes if requested.
    actual_train = train_path
    actual_dev = dev_path
    output_checkpoint_dir = run_dir / "checkpoint"
    if args.smoke:
        smoke_dir = run_dir / "smoke_data"
        actual_train = smoke_dir / "train_smoke.jsonl"
        actual_dev = smoke_dir / "dev_smoke.jsonl"
        if not args.dry_run:
            tn = make_jsonl_prefix(train_path, actual_train, args.smoke_train_rows)
            dn = make_jsonl_prefix(dev_path, actual_dev, args.smoke_dev_rows)
            print(f"[ok] smoke subsets train={tn} dev={dn}")
        else:
            print(f"[dry-run] would write smoke subsets to {smoke_dir}")
        output_checkpoint_dir = run_dir / "smoke_checkpoint"

    # Train unless eval-only.
    train_extra = split_extra(args.train_extra)
    if not args.eval_only:
        train_cmd = build_train_cmd(
            opf_cmd=opf_cmd,
            train_path=actual_train,
            dev_path=actual_dev,
            label_space=label_space,
            output_dir=output_checkpoint_dir,
            checkpoint=args.checkpoint,
            help_text=train_help,
            extra=train_extra,
        )
        if not args.dry_run:
            run_logged(train_cmd, logs_dir / "02_opf_train.log")
        else:
            print(f"[dry-run] {quote_cmd(train_cmd)}")

    # Eval if requested. In eval-only mode, use run-dir/checkpoint unless --checkpoint points elsewhere.
    should_eval = args.eval_on_test or args.eval_only
    pred_out = Path(args.predictions_out).resolve() if args.predictions_out else run_dir / "test_predictions.jsonl"
    if should_eval:
        ensure_file(test_path, "test.jsonl")
        checkpoint_for_eval = Path(args.checkpoint).resolve() if args.eval_only and args.checkpoint else output_checkpoint_dir
        eval_extra = split_extra(args.eval_extra)
        eval_cmd = build_eval_cmd(
            opf_cmd=opf_cmd,
            test_path=test_path,
            checkpoint_dir=checkpoint_for_eval,
            label_space=label_space,
            pred_out=pred_out,
            help_text=eval_help,
            extra=eval_extra,
        )
        if not args.dry_run:
            run_logged(eval_cmd, logs_dir / "03_opf_eval.log")
        else:
            print(f"[dry-run] {quote_cmd(eval_cmd)}")

    # Optional char-span eval if predictions are available.
    if args.char_eval:
        char_eval_script = Path(args.char_eval_script).resolve()
        if not char_eval_script.is_file():
            print(f"[warn] char eval script not found: {char_eval_script}")
        elif not pred_out.is_file():
            print(f"[warn] predictions file not found, skipping char eval: {pred_out}")
        else:
            char_cmd = [
                sys.executable, str(char_eval_script),
                "--gold", str(test_path),
                "--pred", str(pred_out),
                "--taxonomy", str(taxonomy),
                "--out", str(run_dir / "char_eval_metrics.json"),
            ]
            if not args.dry_run:
                run_logged(char_cmd, logs_dir / "04_char_eval.log")
            else:
                print(f"[dry-run] {quote_cmd(char_cmd)}")

    print(f"[done] run directory: {run_dir}")
    print(f"[done] logs: {logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
