from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from pii_prep.stage2_generation import (
    AMBIGUITY_GROUPS,
    CONTEXT_TYPES,
    ZERO_EXAMPLE_LABELS,
    build_5way_prompt,
    format_compatible_candidates,
    generate_document_level_negatives,
    validate_prompt_records,
    zero_example_text,
)
from pii_prep.stage2_teacher_dryrun import (
    build_error_report,
    build_json_only_prompt,
    build_report,
    convert_raw_outputs,
    extract_json_object,
    validate_converted_rows,
    write_json,
    write_jsonl,
)
from pii_prep.stage2_vllm_quality import analyze_quality


AsyncSender = Callable[[dict[str, Any], dict[str, Any], str], Awaitable[str]]


def generate_full_teacher_prompts(
    training_labels: set[str],
    *,
    base_example_count: int = 2000,
    self_consistency: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if "NON_PII" not in training_labels:
        raise ValueError("training label space must include NON_PII")
    base_prompts: list[dict[str, Any]] = []
    base_prompts.extend(_repeated_ambiguity_prompts(training_labels, repeats_per_context=10))
    base_prompts.extend(_zero_example_full_prompts(training_labels, examples_per_label=100))
    remaining = base_example_count - len(base_prompts)
    base_prompts.extend(_candidate_negative_full_prompts(training_labels, remaining))
    if len(base_prompts) != base_example_count:
        raise ValueError(f"expected {base_example_count} base prompts, got {len(base_prompts)}")

    prompts: list[dict[str, Any]] = []
    for base_index, base in enumerate(base_prompts, 1):
        base_id = f"STAGE2-FULL-BASE-{base_index:04d}"
        for consistency_index in range(1, self_consistency + 1):
            prompt = dict(base)
            prompt["base_prompt_id"] = base_id
            prompt["self_consistency_index"] = consistency_index
            prompt["id"] = f"{base_id}-SC{consistency_index}"
            prompt["prompt"] = build_5way_prompt(prompt["span_value"], prompt["context"], prompt["candidate_labels"])
            prompt["teacher_model_path"] = "/home/admin/model/qwen3.5-27b"
            prompt["teacher_call_status"] = "not_run"
            prompts.append(prompt)

    errors = validate_prompt_records(prompts, training_labels)
    if errors:
        raise ValueError("full teacher prompt validation failed: " + "; ".join(errors[:5]))
    plan = {
        "base_example_count": len(base_prompts),
        "self_consistency": self_consistency,
        "teacher_prompt_count": len(prompts),
        "document_level_negative_count": len(generate_document_level_negatives()),
        "prompt_coverage": summarize_prompt_coverage(prompts),
    }
    return prompts, plan


def _repeated_ambiguity_prompts(training_labels: set[str], repeats_per_context: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for repeat in range(repeats_per_context):
        for group in AMBIGUITY_GROUPS:
            candidates = format_compatible_candidates(group["span"], group["candidates"], training_labels)
            if len(candidates) <= 1:
                continue
            for context_type in CONTEXT_TYPES:
                prompts.append(
                    {
                        "id": "",
                        "ambiguity_group": group["name"],
                        "context_type": context_type,
                        "pilot_category": context_type,
                        "generation_bucket": "ambiguity_context",
                        "repeat_index": repeat + 1,
                        "span_value": group["span"],
                        "context": _context_variant(group, context_type, repeat),
                        "candidate_labels": candidates,
                    }
                )
    return prompts


def _context_variant(group: dict[str, Any], context_type: str, repeat: int) -> str:
    context = group["contexts"][context_type]
    if context_type == "strong_positive_context":
        explicit = {
            "name": "The field label is FIRST_NAME. Value: Mia.",
            "coordinates": "The field label is LATITUDE. Value: -33.8688.",
            "social_account": "The field label is SOCIAL_MEDIA_ACCOUNT. Value: alex_chen91.",
        }
        context = explicit.get(group["name"], context)
    if repeat == 0:
        return context
    prefixes = {
        "bare_span": "",
        "weak_context": "Record note. ",
        "strong_positive_context": "Confirmed field. ",
        "reverse_negative_context": "Non-production context. ",
    }
    return f"{prefixes.get(context_type, '')}{context}".strip()


def _zero_example_full_prompts(training_labels: set[str], examples_per_label: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for label in ZERO_EXAMPLE_LABELS:
        if label not in training_labels:
            raise ValueError(f"zero-example label not in training space: {label}")
        for index in range(examples_per_label):
            context, span_value = zero_example_text(label, index)
            prompts.append(
                {
                    "id": "",
                    "ambiguity_group": f"zero_example_{label.lower()}",
                    "context_type": "strong_positive_context",
                    "pilot_category": "zero_example_label",
                    "generation_bucket": "zero_example_label",
                    "target_label": label,
                    "span_value": span_value,
                    "context": context,
                    "candidate_labels": [label, "NON_PII"],
                }
            )
    return prompts


def _candidate_negative_full_prompts(training_labels: set[str], target_count: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    if target_count <= 0:
        return prompts
    groups = [
        group
        for group in AMBIGUITY_GROUPS
        if len(format_compatible_candidates(group["span"], group["candidates"], training_labels)) > 1
    ]
    index = 0
    while len(prompts) < target_count:
        group = groups[index % len(groups)]
        candidates = format_compatible_candidates(group["span"], group["candidates"], training_labels)
        prompts.append(
            {
                "id": "",
                "ambiguity_group": group["name"],
                "context_type": "reverse_negative_context",
                "pilot_category": "candidate_level_ambiguous_negative",
                "generation_bucket": "candidate_level_ambiguous_negative",
                "repeat_index": index // len(groups) + 1,
                "span_value": group["span"],
                "context": _negative_context_variant(group, index),
                "candidate_labels": candidates,
            }
        )
        index += 1
    return prompts


def _negative_context_variant(group: dict[str, Any], index: int) -> str:
    context = group["contexts"]["reverse_negative_context"]
    prefixes = [
        "Documentation example: ",
        "Sandbox fixture: ",
        "Reference ticket: ",
        "Build log: ",
        "Fake tutorial value: ",
    ]
    return prefixes[index % len(prefixes)] + context


def summarize_prompt_coverage(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    contexts: dict[str, int] = {}
    buckets: dict[str, int] = {}
    zero_labels: dict[str, int] = {}
    base_ids = set()
    for prompt in prompts:
        contexts[prompt["context_type"]] = contexts.get(prompt["context_type"], 0) + 1
        bucket = str(prompt.get("generation_bucket", prompt.get("pilot_category", "unknown")))
        buckets[bucket] = buckets.get(bucket, 0) + 1
        base_ids.add(prompt["base_prompt_id"])
        if bucket == "zero_example_label":
            target = str(prompt.get("target_label"))
            zero_labels[target] = zero_labels.get(target, 0) + 1
    return {
        "context_types": contexts,
        "generation_buckets": buckets,
        "zero_example_labels": zero_labels,
        "base_prompt_count": len(base_ids),
    }


def completed_prompt_ids(raw_path: Path) -> set[str]:
    if not raw_path.exists():
        return set()
    completed: set[str] = set()
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok":
            completed.add(str(row.get("id")))
    return completed


async def run_full_teacher_async(
    prompts: list[dict[str, Any]],
    *,
    raw_path: Path,
    progress_path: Path,
    base_url: str,
    model_name: str,
    concurrency: int,
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
    progress_interval: int = 100,
    sender: AsyncSender | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_prompt_ids(raw_path)
    pending = [prompt for prompt in prompts if prompt["id"] not in completed]
    endpoint = base_url.rstrip("/") + "/chat/completions"
    settings = {
        "backend": "vllm_openai",
        "base_url": base_url,
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "enable_thinking": False,
    }
    client_cm = None
    client = None
    if sender is None:
        import httpx

        client_cm = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=10.0))
        client = await client_cm.__aenter__()
    semaphore = asyncio.Semaphore(concurrency)
    started = time.time()
    new_rows: list[dict[str, Any]] = []
    try:
        for offset in range(0, len(pending), progress_interval):
            batch = pending[offset : offset + progress_interval]
            rows = await asyncio.gather(
                *(
                    _call_one_prompt(
                        prompt,
                        semaphore=semaphore,
                        endpoint=endpoint,
                        model_name=model_name,
                        max_tokens=max_tokens,
                        settings=settings,
                        max_retries=max_retries,
                        sender=sender,
                        client=client,
                    )
                    for prompt in batch
                )
            )
            new_rows.extend(rows)
            with raw_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            write_json(
                progress_path,
                {
                    "total_prompts": len(prompts),
                    "completed_before_run": len(completed),
                    "completed_this_run": len(new_rows),
                    "remaining_after_last_save": max(0, len(pending) - len(new_rows)),
                    "progress_saved_every": progress_interval,
                    "last_saved_at": time.time(),
                    "merged_into_training_dataset": False,
                    "student_training_started": False,
                },
            )
    finally:
        if client_cm is not None:
            await client_cm.__aexit__(None, None, None)
    return new_rows, {
        "wall_time_seconds": round(time.time() - started, 3),
        "completed_before_run": len(completed),
        "attempted_this_run": len(new_rows),
        "skipped_completed": len(completed),
    }


async def _call_one_prompt(
    prompt: dict[str, Any],
    *,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    model_name: str,
    max_tokens: int,
    settings: dict[str, Any],
    max_retries: int,
    sender: AsyncSender | None,
    client: Any,
) -> dict[str, Any]:
    async with semaphore:
        started = time.time()
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a strict JSON generator for PII ambiguity verdicts. Return JSON only."},
                {"role": "user", "content": build_json_only_prompt(prompt)},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        retry_count = 0
        last_error = ""
        output_text = ""
        status = "error"
        for attempt in range(max_retries + 1):
            try:
                if sender is not None:
                    output_text = await sender(prompt, payload, endpoint)
                else:
                    response = await client.post(endpoint, json=payload, headers={"Authorization": "Bearer EMPTY"})
                    response.raise_for_status()
                    output_text = response.json()["choices"][0]["message"]["content"].strip()
                extract_json_object(output_text)
                status = "ok"
                last_error = ""
                break
            except Exception as exc:  # pragma: no cover - network exception classes vary
                last_error = repr(exc)
                status = "timeout" if "timeout" in exc.__class__.__name__.lower() else "malformed_or_error"
                if attempt < max_retries:
                    retry_count += 1
                    await asyncio.sleep(min(2.0, 0.25 * (attempt + 1)))
                    continue
        return {
            "id": prompt["id"],
            "prompt": prompt,
            "output_text": output_text if status == "ok" else "",
            "status": status,
            "error": last_error if status != "ok" else None,
            "elapsed_seconds": round(time.time() - started, 3),
            "teacher_model_path": model_name,
            "generation_settings": settings,
            "backend": "vllm_openai",
            "retry_count": retry_count,
        }


def load_all_raw_rows(raw_path: Path) -> list[dict[str, Any]]:
    if not raw_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = str(row.get("id"))
        if row_id in seen:
            continue
        seen.add(row_id)
        rows.append(row)
    return rows


def write_quality_reports(
    *,
    root: Path,
    converted_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    generation_report: dict[str, Any],
) -> dict[str, Any]:
    quality, warnings = analyze_quality(converted_rows)
    quality["conversion_validation"] = {
        "validation_error_count": generation_report["validation_error_count"],
        "labels_outside_training_space": generation_report["labels_outside_training_space"],
    }
    quality["inputs"] = {
        "converted_path": "data/generated/stage2_full_teacher_converted.jsonl",
        "raw_path": "data/generated/stage2_full_teacher_raw.jsonl",
        "raw_record_count": len(raw_rows),
    }
    quality["merged_into_training_dataset"] = False
    quality["student_training_started"] = False
    write_json(root / "reports" / "stage2_full_teacher_quality_report.json", quality)
    write_json(
        root / "reports" / "stage2_full_teacher_warning_examples.json",
        {"warning_examples": warnings, "warning_counts": quality["warning_counts"]},
    )
    return quality


async def run_full_project_async(
    root: Path | str = ".",
    *,
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "qwen3.5-27b",
    concurrency: int = 16,
    max_tokens: int = 96,
    timeout_seconds: float = 240.0,
    max_retries: int = 1,
    base_example_count: int = 2000,
    self_consistency: int = 3,
    sender: AsyncSender | None = None,
) -> dict[str, Any]:
    root = Path(root)
    generated_dir = root / "data" / "generated"
    reports_dir = root / "reports"
    generated_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    training_labels = set(json.loads((root / "pii_schema" / "training_label_space_80.json").read_text(encoding="utf-8")))

    prompts, prompt_plan = generate_full_teacher_prompts(
        training_labels,
        base_example_count=base_example_count,
        self_consistency=self_consistency,
    )
    prompt_path = generated_dir / "stage2_full_teacher_prompts.jsonl"
    raw_path = generated_dir / "stage2_full_teacher_raw.jsonl"
    converted_path = generated_dir / "stage2_full_teacher_converted.jsonl"
    generation_report_path = reports_dir / "stage2_full_teacher_generation_report.json"
    progress_path = reports_dir / "stage2_full_teacher_progress.json"
    errors_path = reports_dir / "stage2_full_teacher_errors.json"
    write_jsonl(prompt_path, prompts)

    new_rows, runtime = await run_full_teacher_async(
        prompts,
        raw_path=raw_path,
        progress_path=progress_path,
        base_url=base_url,
        model_name=model_name,
        concurrency=concurrency,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        sender=sender,
    )
    raw_rows = load_all_raw_rows(raw_path)
    converted_rows, conversion_errors = convert_raw_outputs(raw_rows, training_labels)
    validation = validate_converted_rows(converted_rows, training_labels)
    write_jsonl(converted_path, converted_rows)
    generation_report = build_report(
        raw_rows,
        converted_rows,
        conversion_errors,
        validation,
        backend="vllm_openai",
        wall_time_seconds=runtime["wall_time_seconds"],
    )
    generation_report.update(prompt_plan)
    generation_report.update(runtime)
    generation_report["student_training_started"] = False
    generation_report["merged_into_training_dataset"] = False
    generation_report["full_6000_teacher_calls_executed"] = len(raw_rows) >= 6000
    generation_report["resumable"] = True
    generation_report["raw_output_written_incrementally"] = True
    generation_report["progress_saved_every_prompts"] = 100
    generation_report["new_rows_this_run"] = len(new_rows)
    generation_report["expected_teacher_calls"] = len(prompts)
    generation_report["teacher_calls_completed_total"] = len(raw_rows)
    generation_report["full_teacher_calls_completed"] = len(raw_rows) == len(prompts)
    write_json(generation_report_path, generation_report)
    quality = write_quality_reports(
        root=root,
        converted_rows=converted_rows,
        raw_rows=raw_rows,
        generation_report=generation_report,
    )
    error_report = build_error_report(conversion_errors, validation, converted_rows)
    error_report["raw_status_counts"] = _counts(row.get("status") for row in raw_rows)
    error_report["retry_count"] = sum(int(row.get("retry_count", 0)) for row in raw_rows)
    error_report["quality_warning_counts"] = quality["warning_counts"]
    error_report["merged_into_training_dataset"] = False
    error_report["student_training_started"] = False
    write_json(errors_path, error_report)
    return generation_report


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def run_full_project(root: Path | str = ".", **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(run_full_project_async(root, **kwargs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--base-example-count", type=int, default=2000)
    parser.add_argument("--self-consistency", type=int, default=3)
    args = parser.parse_args(argv)
    report = run_full_project(
        args.root,
        base_url=args.base_url,
        model_name=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        base_example_count=args.base_example_count,
        self_consistency=args.self_consistency,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
