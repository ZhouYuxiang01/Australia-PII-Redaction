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
    filter_candidates,
    format_compatible_candidates,
    validate_prompt_records,
    zero_example_text,
)
from pii_prep.stage2_teacher_dryrun import (
    build_error_report,
    build_json_only_prompt,
    convert_raw_outputs,
    validate_converted_rows,
    write_json,
    write_jsonl,
)


AsyncSender = Callable[[dict[str, Any], dict[str, Any], str], Awaitable[str]]


def generate_pilot_prompts(training_labels: set[str], total: int = 200) -> list[dict[str, Any]]:
    if total < 40:
        raise ValueError("pilot total must be at least 40 to cover all ambiguity contexts")
    if "NON_PII" not in training_labels:
        raise ValueError("training label space must include NON_PII")

    prompts: list[dict[str, Any]] = []
    prompts.extend(_ambiguity_context_prompts(training_labels))
    prompts.extend(_zero_example_prompts(training_labels, examples_per_label=20))
    prompts.extend(_candidate_negative_prompts(training_labels, target_count=max(0, total - len(prompts))))

    cursor = 0
    base_prompts = list(prompts)
    while len(prompts) < total:
        source = dict(base_prompts[cursor % len(base_prompts)])
        source["id"] = ""
        source["pilot_repeat_index"] = cursor + 1
        prompts.append(source)
        cursor += 1

    selected = prompts[:total]
    for index, prompt in enumerate(selected, 1):
        prompt["id"] = f"STAGE2-VLLM-PILOT-{index:03d}"
        prompt["prompt"] = build_5way_prompt(prompt["span_value"], prompt["context"], prompt["candidate_labels"])
        prompt["teacher_call_status"] = "not_run"

    errors = validate_prompt_records(selected, training_labels)
    if errors:
        raise ValueError("pilot prompt validation failed: " + "; ".join(errors[:5]))
    return selected


def _ambiguity_context_prompts(training_labels: set[str]) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
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
                    "span_value": group["span"],
                    "context": group["contexts"][context_type],
                    "candidate_labels": candidates,
                    "teacher_model_path": "/home/admin/model/qwen3.5-27b",
                }
            )
    return prompts


def _zero_example_prompts(training_labels: set[str], examples_per_label: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for label in ZERO_EXAMPLE_LABELS:
        if label not in training_labels:
            raise ValueError(f"zero-example label not in training space: {label}")
        candidates = [label, "NON_PII"]
        for index in range(examples_per_label):
            context, span_value = zero_example_text(label, index)
            prompts.append(
                {
                    "id": "",
                    "ambiguity_group": f"zero_example_{label.lower()}",
                    "context_type": "strong_positive_context",
                    "pilot_category": "zero_example_label",
                    "target_label": label,
                    "span_value": span_value,
                    "context": context,
                    "candidate_labels": candidates,
                    "teacher_model_path": "/home/admin/model/qwen3.5-27b",
                }
            )
    return prompts


def _candidate_negative_prompts(training_labels: set[str], target_count: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    if target_count <= 0:
        return prompts
    usable_groups = [
        group
        for group in AMBIGUITY_GROUPS
        if len(format_compatible_candidates(group["span"], group["candidates"], training_labels)) > 1
    ]
    if not usable_groups:
        raise ValueError("no ambiguity groups have usable candidate labels")

    index = 0
    while len(prompts) < target_count:
        group = usable_groups[index % len(usable_groups)]
        candidates = format_compatible_candidates(group["span"], group["candidates"], training_labels)
        prompts.append(
            {
                "id": "",
                "ambiguity_group": group["name"],
                "context_type": "reverse_negative_context",
                "pilot_category": "candidate_level_ambiguous_negative",
                "span_value": group["span"],
                "context": group["contexts"]["reverse_negative_context"],
                "candidate_labels": candidates,
                "teacher_model_path": "/home/admin/model/qwen3.5-27b",
            }
        )
        index += 1
    return prompts


async def run_vllm_concurrent_outputs(
    prompts: list[dict[str, Any]],
    *,
    concurrency: int,
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "/home/admin/model/qwen3.5-27b",
    max_tokens: int = 256,
    timeout_seconds: float = 120.0,
    max_retries: int = 1,
    sender: AsyncSender | None = None,
) -> tuple[list[dict[str, Any]], float]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    settings = {
        "backend": "vllm_openai_async",
        "base_url": base_url,
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
    }
    semaphore = asyncio.Semaphore(concurrency)

    client_cm = None
    client = None
    if sender is None:
        import httpx

        client_cm = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=10.0))
        client = await client_cm.__aenter__()

    async def call_one(prompt: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            started = time.time()
            payload = _chat_completion_payload(prompt, model_name, max_tokens)
            retries = 0
            last_error = ""
            for attempt in range(max_retries + 1):
                try:
                    if sender is not None:
                        output_text = await sender(prompt, payload, endpoint)
                    else:
                        assert client is not None
                        response = await client.post(
                            endpoint,
                            json=payload,
                            headers={"Authorization": "Bearer EMPTY"},
                        )
                        response.raise_for_status()
                        response_payload = response.json()
                        output_text = response_payload["choices"][0]["message"]["content"].strip()
                    return {
                        "id": prompt["id"],
                        "prompt": prompt,
                        "output_text": output_text,
                        "status": "ok",
                        "elapsed_seconds": round(time.time() - started, 3),
                        "teacher_model_path": model_name,
                        "generation_settings": settings,
                        "backend": "vllm_openai_async",
                        "retry_count": retries,
                    }
                except Exception as exc:  # pragma: no cover - exact network exceptions vary
                    last_error = repr(exc)
                    if _is_timeout_error(exc):
                        error_type = "timeout"
                    else:
                        error_type = "error"
                    if attempt < max_retries:
                        retries += 1
                        await asyncio.sleep(min(2.0, 0.25 * (attempt + 1)))
                        continue
                    return {
                        "id": prompt["id"],
                        "prompt": prompt,
                        "output_text": "",
                        "status": error_type,
                        "error": last_error,
                        "elapsed_seconds": round(time.time() - started, 3),
                        "teacher_model_path": model_name,
                        "generation_settings": settings,
                        "backend": "vllm_openai_async",
                        "retry_count": retries,
                    }

    started_all = time.time()
    try:
        rows = await asyncio.gather(*(call_one(prompt) for prompt in prompts))
    finally:
        if client_cm is not None:
            await client_cm.__aexit__(None, None, None)
    return rows, time.time() - started_all


def _chat_completion_payload(prompt: dict[str, Any], model_name: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict JSON generator for PII ambiguity verdicts. Return JSON only.",
            },
            {"role": "user", "content": build_json_only_prompt(prompt)},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _is_timeout_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    return "timeout" in name or "timedout" in name


def score_raw_rows(
    raw_rows: list[dict[str, Any]],
    training_labels: set[str],
    *,
    wall_time_seconds: float,
    concurrency: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    converted_rows, conversion_errors = convert_raw_outputs(raw_rows, training_labels)
    validation = validate_converted_rows(converted_rows, training_labels)
    errors = build_error_report(conversion_errors, validation, converted_rows)
    attempted = len(raw_rows)
    metrics = {
        "concurrency": concurrency,
        "prompts_attempted": attempted,
        "total_wall_time_seconds": round(wall_time_seconds, 3),
        "average_seconds_per_prompt": round(wall_time_seconds / attempted, 3) if attempted else 0.0,
        "requests_per_minute": round((attempted / wall_time_seconds) * 60, 3) if wall_time_seconds > 0 else 0.0,
        "raw_ok_count": sum(1 for row in raw_rows if row.get("status") == "ok"),
        "valid_json_outputs": len(converted_rows),
        "malformed_json_count": errors["malformed_json_count"],
        "missing_verdict_count": errors["missing_verdict_count"],
        "unexpected_label_count": errors["unexpected_label_count"],
        "validation_error_count": validation["validation_error_count"],
        "labels_outside_training_space": validation["labels_outside_training_space"],
        "timeout_count": sum(1 for row in raw_rows if row.get("status") == "timeout"),
        "retry_count": sum(int(row.get("retry_count", 0)) for row in raw_rows),
        "conversion_error_count": len(conversion_errors),
        "overconfident_distribution_count": errors["overconfident_distribution_count"],
    }
    return converted_rows, metrics, errors


def choose_best_concurrency(level_metrics: list[dict[str, Any]]) -> int:
    acceptable = [
        metric
        for metric in level_metrics
        if metric["valid_json_outputs"] >= 190
        and metric["validation_error_count"] == 0
        and metric["labels_outside_training_space"] == {}
    ]
    candidates = acceptable or level_metrics
    best = sorted(
        candidates,
        key=lambda item: (
            -int(item["valid_json_outputs"]),
            int(item["validation_error_count"]),
            int(item["timeout_count"]),
            float(item["total_wall_time_seconds"]),
        ),
    )[0]
    return int(best["concurrency"])


async def run_pilot_project_async(
    root: Path | str = ".",
    *,
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "/home/admin/model/qwen3.5-27b",
    total: int = 200,
    concurrency_levels: list[int] | None = None,
    max_tokens: int = 256,
    timeout_seconds: float = 120.0,
    max_retries: int = 1,
    output_suffix: str = "",
) -> dict[str, Any]:
    root = Path(root)
    concurrency_levels = concurrency_levels or [1, 4, 8, 16]
    training_path = root / "pii_schema" / "training_label_space_80.json"
    training_labels = set(json.loads(training_path.read_text(encoding="utf-8")))
    prompts = generate_pilot_prompts(training_labels, total=total)

    per_level: dict[str, dict[str, Any]] = {}
    raw_by_level: dict[int, list[dict[str, Any]]] = {}
    converted_by_level: dict[int, list[dict[str, Any]]] = {}
    errors_by_level: dict[int, dict[str, Any]] = {}
    for concurrency in concurrency_levels:
        raw_rows, wall_time_seconds = await run_vllm_concurrent_outputs(
            prompts,
            concurrency=concurrency,
            base_url=base_url,
            model_name=model_name,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        converted_rows, metrics, errors = score_raw_rows(
            raw_rows,
            training_labels,
            wall_time_seconds=wall_time_seconds,
            concurrency=concurrency,
        )
        per_level[str(concurrency)] = metrics
        raw_by_level[concurrency] = raw_rows
        converted_by_level[concurrency] = converted_rows
        errors_by_level[concurrency] = errors

    best_concurrency = choose_best_concurrency(list(per_level.values()))
    best_metrics = per_level[str(best_concurrency)]
    best_raw = raw_by_level[best_concurrency]
    best_converted = converted_by_level[best_concurrency]
    best_errors = errors_by_level[best_concurrency]

    raw_path = root / "data" / "generated" / f"stage2_vllm_pilot_200_raw{output_suffix}.jsonl"
    converted_path = root / "data" / "generated" / f"stage2_vllm_pilot_200_converted{output_suffix}.jsonl"
    report_path = root / "reports" / f"stage2_vllm_concurrency_pilot_report{output_suffix}.json"
    errors_path = root / "reports" / f"stage2_vllm_pilot_errors{output_suffix}.json"

    report = {
        "stage": "2.2",
        "backend": "vllm_openai_async",
        "base_url": base_url,
        "model": model_name,
        "prompts_attempted": total,
        "pilot_prompt_count": len(prompts),
        "concurrency_levels": concurrency_levels,
        "per_concurrency": per_level,
        "best_concurrency": best_concurrency,
        "best_metrics": best_metrics,
        "prompt_coverage": summarize_prompt_coverage(prompts),
        "teacher_calls_executed": total * len(concurrency_levels),
        "full_6000_teacher_calls_executed": False,
        "merged_into_training_dataset": False,
        "output_suffix": output_suffix,
        "output_selection": {
            "raw_jsonl": str(raw_path),
            "converted_jsonl": str(converted_path),
            "selected_from_concurrency": best_concurrency,
        },
    }
    write_jsonl(raw_path, best_raw)
    write_jsonl(converted_path, best_converted)
    write_json(report_path, report)
    write_json(
        errors_path,
        {
            "selected_concurrency": best_concurrency,
            "selected_errors": best_errors,
            "per_concurrency_error_summary": {
                str(level): {
                    "malformed_json_count": errors["malformed_json_count"],
                    "missing_verdict_count": errors["missing_verdict_count"],
                    "unexpected_label_count": errors["unexpected_label_count"],
                    "overconfident_distribution_count": errors["overconfident_distribution_count"],
                }
                for level, errors in errors_by_level.items()
            },
        },
    )
    return report


def summarize_prompt_coverage(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, int] = {}
    contexts: dict[str, int] = {}
    zero_labels: dict[str, int] = {}
    for prompt in prompts:
        category = str(prompt.get("pilot_category"))
        context_type = str(prompt.get("context_type"))
        categories[category] = categories.get(category, 0) + 1
        contexts[context_type] = contexts.get(context_type, 0) + 1
        if category == "zero_example_label":
            target = str(prompt.get("target_label"))
            zero_labels[target] = zero_labels.get(target, 0) + 1
    return {
        "pilot_categories": categories,
        "context_types": contexts,
        "zero_example_labels": zero_labels,
        "candidate_level_ambiguous_negative_count": categories.get("candidate_level_ambiguous_negative", 0),
    }


def run_pilot_project(root: Path | str = ".", **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(run_pilot_project_async(root, **kwargs))


def _parse_concurrency_levels(value: str) -> list[int]:
    levels = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not levels:
        raise argparse.ArgumentTypeError("at least one concurrency level is required")
    if any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError("concurrency levels must be >= 1")
    return levels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="/home/admin/model/qwen3.5-27b")
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--concurrency-levels", type=_parse_concurrency_levels, default=[1, 4, 8, 16])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args(argv)
    report = run_pilot_project(
        args.root,
        base_url=args.base_url,
        model_name=args.model,
        total=args.total,
        concurrency_levels=args.concurrency_levels,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        output_suffix=args.output_suffix,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    best = report["best_metrics"]
    if report["prompts_attempted"] != 200:
        raise SystemExit("expected exactly 200 prompts attempted")
    if best["valid_json_outputs"] < 190:
        raise SystemExit("fewer than 190 valid JSON outputs for best concurrency")
    if best["validation_error_count"] != 0:
        raise SystemExit("conversion validation failed for best concurrency")
    if best["labels_outside_training_space"]:
        raise SystemExit("labels outside training space found for best concurrency")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
