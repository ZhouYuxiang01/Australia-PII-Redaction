from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


VERDICT_MULTIPLIERS = {
    "strong_for": 3.0,
    "weak_for": 1.5,
    "neutral": 1.0,
    "weak_against": 0.5,
    "strong_against": 0.1,
}

DATE_LABELS = {
    "DATE_OF_BIRTH",
    "PASSPORT_EXPIRY",
    "PASSPORT_START_DATE",
    "MEDICARE_EXPIRY",
    "CREDIT_CARD_EXPIRY",
}

NEGATIVE_CONTEXT_MARKERS = {
    "order",
    "invoice",
    "reference",
    "ticket",
    "case",
    "build",
    "documentation",
    "placeholder",
    "version",
    "simulated",
    "sandbox",
    "test token",
    "tutorial",
    "fake",
    "public",
    "guide",
    "handbook",
    "template",
    "main line",
    "reception",
    "course code",
    "router",
    "internal",
    "demo",
    "asset",
    "invoice",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_json_only_prompt(prompt_record: dict[str, Any]) -> str:
    candidates = prompt_record["candidate_labels"]
    candidate_lines = "\n".join(f'- "{candidate}"' for candidate in candidates)
    return (
        "Return valid JSON only. Do not include markdown, explanation, comments, or extra text.\n"
        "Use exactly this schema: {\"verdicts\":{\"LABEL\":\"strong_for|weak_for|neutral|weak_against|strong_against\"}}\n"
        "Every candidate label listed below must appear exactly once in verdicts.\n"
        "Do not add labels that are not listed.\n\n"
        f"Span: {json.dumps(prompt_record['span_value'], ensure_ascii=False)}\n"
        f"Context: {json.dumps(prompt_record['context'], ensure_ascii=False)}\n\n"
        f"Context type: {prompt_record.get('context_type', 'unknown')}\n\n"
        "Candidate labels:\n"
        f"{candidate_lines}\n\n"
        "Context-specific rules:\n"
        "- bare_span: Do not use strong_for unless the format uniquely identifies one label.\n"
        "- bare_span: Date-like bare spans must remain ambiguous across date-related labels and NON_PII.\n"
        "- weak_context: Avoid strong_for unless the field name explicitly identifies the label.\n"
        "- strong_positive_context: strong_for is allowed when the context clearly identifies the label.\n"
        "- reverse_negative_context: NON_PII should usually be strong_for.\n"
        "- reverse_negative_context: PII labels that only match by format should usually be weak_against or strong_against.\n\n"
        "- hard_negative_context: This is a hard-negative candidate; choose NON_PII when the span is a public, placeholder, template, demo, internal, course, asset, invoice, or organisation-level value rather than an individual's PII.\n"
        "- hard_negative_context: PII labels that only match by surface format should usually be weak_against or strong_against.\n\n"
        "Verdict meanings:\n"
        "- strong_for: Context strongly suggests this type\n"
        "- weak_for: Context mildly suggests this type\n"
        "- neutral: No evidence either way\n"
        "- weak_against: Context mildly contradicts this type\n"
        "- strong_against: Context strongly contradicts this type\n\n"
        "JSON only:"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("no JSON object found")
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("JSON output is not an object")
    return obj


def verdicts_to_distribution(
    verdicts: dict[str, str],
    candidate_labels: list[str],
    prompt_record: dict[str, Any] | None = None,
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for label in candidate_labels:
        verdict = verdicts.get(label, "neutral")
        if verdict not in VERDICT_MULTIPLIERS:
            raise ValueError(f"invalid verdict for {label}: {verdict}")
        raw[label] = VERDICT_MULTIPLIERS[verdict]
    raw = calibrate_raw_scores(raw, verdicts, candidate_labels, prompt_record or {})
    dist = normalize_scores(raw)
    dist = cap_bare_date_of_birth(dist, prompt_record or {})
    return normalize_scores(dist)


def calibrate_raw_scores(
    raw: dict[str, float],
    verdicts: dict[str, str],
    candidate_labels: list[str],
    prompt_record: dict[str, Any],
) -> dict[str, float]:
    calibrated = dict(raw)
    context_type = normalize_context_type(str(prompt_record.get("context_type", "")))
    if context_type == "bare_span" and not prompt_record.get("rule_verified_unique", False):
        for label in candidate_labels:
            if label != "NON_PII" and verdicts.get(label) == "strong_for":
                calibrated[label] = VERDICT_MULTIPLIERS["weak_for"]
    if context_type in {"reverse_negative", "hard_negative"}:
        if "NON_PII" in calibrated:
            calibrated["NON_PII"] = max(calibrated["NON_PII"], VERDICT_MULTIPLIERS["strong_for"])
        for label in candidate_labels:
            if label != "NON_PII" and verdicts.get(label) == "strong_for" and not prompt_record.get("rule_verified_unique", False):
                calibrated[label] = VERDICT_MULTIPLIERS["weak_against"]
            if context_type == "hard_negative" and label != "NON_PII" and verdicts.get(label) == "weak_for" and not prompt_record.get("rule_verified_unique", False):
                calibrated[label] = VERDICT_MULTIPLIERS["weak_against"]
        context = str(prompt_record.get("context", "")).lower()
        if "NON_PII" in calibrated and any(marker in context for marker in NEGATIVE_CONTEXT_MARKERS):
            calibrated["NON_PII"] = VERDICT_MULTIPLIERS["strong_for"]
    return calibrated


def normalize_context_type(context_type: str) -> str:
    if context_type == "strong_positive_context":
        return "strong_positive"
    if context_type == "reverse_negative_context":
        return "reverse_negative"
    if context_type == "hard_negative_context":
        return "hard_negative"
    return context_type


def normalize_scores(raw: dict[str, float]) -> dict[str, float]:
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("verdict multipliers sum to zero")
    dist = {label: round(value / total, 6) for label, value in raw.items()}
    drift = round(1.0 - sum(dist.values()), 6)
    if dist:
        first = next(iter(dist))
        dist[first] = round(dist[first] + drift, 6)
    return dist


def cap_bare_date_of_birth(dist: dict[str, float], prompt_record: dict[str, Any]) -> dict[str, float]:
    if normalize_context_type(str(prompt_record.get("context_type", ""))) != "bare_span":
        return dist
    if not is_date_like(str(prompt_record.get("span_value", ""))):
        return dist
    dob_probability = float(dist.get("DATE_OF_BIRTH", 0.0))
    if dob_probability <= 0.60:
        return dist
    capped = dict(dist)
    excess = dob_probability - 0.60
    capped["DATE_OF_BIRTH"] = 0.60
    recipients = [
        label
        for label in capped
        if label != "DATE_OF_BIRTH" and (label in DATE_LABELS or label == "NON_PII")
    ]
    if not recipients:
        return capped
    share = excess / len(recipients)
    for label in recipients:
        capped[label] = capped[label] + share
    return capped


def is_date_like(value: str) -> bool:
    text = value.strip()
    month_names = (
        "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        "jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    )
    return bool(
        re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", text)
        or re.fullmatch(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}", text)
        or re.fullmatch(rf"(?i)(?:{month_names})\s+\d{{1,2}},?\s+\d{{4}}", text)
    )


def convert_raw_output(raw: dict[str, Any], training_labels: set[str]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    prompt = raw.get("prompt", {})
    prompt_id = str(raw.get("id", prompt.get("id", "unknown")))
    candidate_labels = [str(label) for label in prompt.get("candidate_labels", [])]
    if raw.get("status") != "ok":
        return None, [f"{prompt_id}: raw status is {raw.get('status')}"]
    for label in candidate_labels:
        if label not in training_labels:
            errors.append(f"{prompt_id}: candidate label outside training space: {label}")
    try:
        parsed = extract_json_object(str(raw.get("output_text", "")))
    except Exception as exc:
        return None, [f"{prompt_id}: malformed JSON: {exc}"]
    verdicts = parsed.get("verdicts")
    if not isinstance(verdicts, dict):
        return None, [f"{prompt_id}: missing verdicts object"]
    verdicts = {str(label): str(verdict) for label, verdict in verdicts.items()}
    unexpected = sorted(set(verdicts) - set(candidate_labels))
    missing = sorted(set(candidate_labels) - set(verdicts))
    for label in unexpected:
        errors.append(f"{prompt_id}: unexpected verdict label: {label}")
    for label in missing:
        errors.append(f"{prompt_id}: missing verdict label: {label}")
    for label, verdict in verdicts.items():
        if verdict not in VERDICT_MULTIPLIERS:
            errors.append(f"{prompt_id}: invalid verdict {verdict!r} for {label}")
        if label not in training_labels:
            errors.append(f"{prompt_id}: verdict label outside training space: {label}")
    if errors:
        return None, errors
    distribution = verdicts_to_distribution(verdicts, candidate_labels, prompt)
    top_type = max(distribution, key=distribution.get)
    max_probability = distribution[top_type]
    warnings = []
    if max_probability >= 0.85 and prompt.get("context_type") in {"bare_span", "weak_context"}:
        warnings.append("possibly_overconfident_distribution")
    return {
        "id": prompt_id,
        "source_prompt_id": prompt_id,
        "ambiguity_group": prompt.get("ambiguity_group"),
        "context_type": prompt.get("context_type"),
        "span_value": prompt.get("span_value"),
        "context": prompt.get("context"),
        "candidate_labels": candidate_labels,
        "verdicts": verdicts,
        "type_distribution": distribution,
        "top_type": top_type,
        "warnings": warnings,
        "teacher_model_path": raw.get("teacher_model_path"),
        "generation_settings": raw.get("generation_settings", {}),
    }, []


def validate_converted_rows(rows: list[dict[str, Any]], training_labels: set[str]) -> dict[str, Any]:
    labels_outside: dict[str, int] = {}
    errors: list[str] = []
    for row in rows:
        labels = set(row.get("candidate_labels", [])) | set(row.get("verdicts", {})) | set(row.get("type_distribution", {})) | {row.get("top_type")}
        for label in sorted(label for label in labels if label not in training_labels):
            labels_outside[label] = labels_outside.get(label, 0) + 1
            errors.append(f"{row.get('id')}: label outside training space: {label}")
        dist = row.get("type_distribution", {})
        if abs(sum(float(v) for v in dist.values()) - 1.0) > 1e-6:
            errors.append(f"{row.get('id')}: distribution does not sum to 1")
        if dist and row.get("top_type") != max(dist, key=dist.get):
            errors.append(f"{row.get('id')}: top_type is not argmax")
        if "NON_PII" in row.get("candidate_labels", []) and "NON_PII" not in dist:
            errors.append(f"{row.get('id')}: NON_PII missing from distribution")
    return {
        "validation_error_count": len(errors),
        "validation_errors": errors,
        "labels_outside_training_space": labels_outside,
    }


def run_teacher_outputs(
    prompts: list[dict[str, Any]],
    model_path: str,
    *,
    max_new_tokens: int,
    load_in_4bit: bool = False,
    raw_output_path: Path | None = None,
    python_note: str = "transformers",
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import BitsAndBytesConfig
    except Exception:  # pragma: no cover - depends on runtime extras
        BitsAndBytesConfig = None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("BitsAndBytesConfig is not available for --load-in-4bit")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    settings = {
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "python_note": python_note,
        "load_in_4bit": load_in_4bit,
    }
    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text("", encoding="utf-8")
    for prompt in prompts:
        text = build_json_only_prompt(prompt)
        messages = [
            {"role": "system", "content": "You are a strict JSON generator for PII ambiguity verdicts."},
            {"role": "user", "content": text},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            model_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            model_input = text
        started = time.time()
        try:
            inputs = tokenizer([model_input], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = outputs[0][inputs["input_ids"].shape[1] :]
            output_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            row = {
                "id": prompt["id"],
                "prompt": prompt,
                "output_text": output_text,
                "status": "ok",
                "elapsed_seconds": round(time.time() - started, 3),
                "teacher_model_path": model_path,
                "generation_settings": settings,
            }
            rows.append(row)
        except Exception as exc:
            row = {
                "id": prompt["id"],
                "prompt": prompt,
                "output_text": "",
                "status": "error",
                "error": repr(exc),
                "elapsed_seconds": round(time.time() - started, 3),
                "teacher_model_path": model_path,
                "generation_settings": settings,
            }
            rows.append(row)
        if raw_output_path is not None:
            with raw_output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return rows


def run_vllm_openai_outputs(
    prompts: list[dict[str, Any]],
    *,
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "/home/admin/model/qwen3.5-27b",
    max_tokens: int = 256,
    raw_output_path: Path | None = None,
    timeout_seconds: float = 120.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    settings = {
        "backend": "vllm_openai",
        "base_url": base_url,
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    endpoint = base_url.rstrip("/") + "/chat/completions"
    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text("", encoding="utf-8")
    for prompt in prompts:
        started = time.time()
        payload = {
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
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            output_text = response_payload["choices"][0]["message"]["content"].strip()
            row = {
                "id": prompt["id"],
                "prompt": prompt,
                "output_text": output_text,
                "status": "ok",
                "elapsed_seconds": round(time.time() - started, 3),
                "teacher_model_path": model_name,
                "generation_settings": settings,
                "backend": "vllm_openai",
            }
        except Exception as exc:
            row = {
                "id": prompt["id"],
                "prompt": prompt,
                "output_text": "",
                "status": "error",
                "error": repr(exc),
                "elapsed_seconds": round(time.time() - started, 3),
                "teacher_model_path": model_name,
                "generation_settings": settings,
                "backend": "vllm_openai",
            }
        rows.append(row)
        if raw_output_path is not None:
            with raw_output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return rows


def convert_raw_outputs(raw_rows: list[dict[str, Any]], training_labels: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    converted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for raw in raw_rows:
        row, row_errors = convert_raw_output(raw, training_labels)
        if row is None:
            errors.append({"id": raw.get("id"), "errors": row_errors, "raw_status": raw.get("status"), "raw_error": raw.get("error")})
        else:
            converted.append(row)
    return converted, errors


def build_report(raw_rows: list[dict[str, Any]], converted_rows: list[dict[str, Any]], conversion_errors: list[dict[str, Any]], validation: dict[str, Any], *, backend: str = "transformers_local", wall_time_seconds: float | None = None) -> dict[str, Any]:
    warnings: dict[str, int] = {}
    for row in converted_rows:
        for warning in row.get("warnings", []):
            warnings[warning] = warnings.get(warning, 0) + 1
    valid_json_outputs = len(converted_rows)
    return {
        "backend": backend,
        "prompts_attempted": len(raw_rows),
        "raw_ok_count": sum(1 for row in raw_rows if row.get("status") == "ok"),
        "valid_json_outputs": valid_json_outputs,
        "converted_count": len(converted_rows),
        "conversion_error_count": len(conversion_errors),
        "validation_error_count": validation["validation_error_count"],
        "labels_outside_training_space": validation["labels_outside_training_space"],
        "warnings": warnings,
        "teacher_calls_executed": len(raw_rows),
        "full_6000_teacher_calls_executed": False,
        "merged_into_training_dataset": False,
        "total_wall_time_seconds": round(wall_time_seconds, 3) if wall_time_seconds is not None else round(sum(float(row.get("elapsed_seconds", 0.0)) for row in raw_rows), 3),
        "average_seconds_per_prompt": round(sum(float(row.get("elapsed_seconds", 0.0)) for row in raw_rows) / len(raw_rows), 3) if raw_rows else 0.0,
    }


def build_error_report(
    conversion_errors: list[dict[str, Any]],
    validation: dict[str, Any],
    converted_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    overconfident = [
        {
            "id": row.get("id"),
            "context_type": row.get("context_type"),
            "top_type": row.get("top_type"),
            "top_probability": row.get("type_distribution", {}).get(row.get("top_type")),
            "warnings": row.get("warnings", []),
        }
        for row in converted_rows
        if "possibly_overconfident_distribution" in row.get("warnings", [])
    ]
    return {
        "conversion_errors": conversion_errors,
        "validation_errors": validation["validation_errors"],
        "malformed_json_count": sum(1 for item in conversion_errors for e in item["errors"] if "malformed JSON" in e),
        "missing_verdict_count": sum(1 for item in conversion_errors for e in item["errors"] if "missing verdict" in e),
        "unexpected_label_count": sum(1 for item in conversion_errors for e in item["errors"] if "unexpected verdict label" in e),
        "overconfident_distribution_count": len(overconfident),
        "overconfident_distributions": overconfident,
    }


def dryrun_project(
    root: Path | str = ".",
    *,
    model_path: str = "/home/admin/model/qwen3.5-27b",
    limit: int = 20,
    max_new_tokens: int = 256,
    load_in_4bit: bool = False,
    backend: str = "transformers_local",
    base_url: str = "http://localhost:8000/v1",
    output_prefix: str = "stage2_teacher_dryrun_20",
) -> dict[str, Any]:
    root = Path(root)
    prompt_path = root / "data" / "generated" / "stage2_teacher_prompts_sample.jsonl"
    training_path = root / "pii_schema" / "training_label_space_80.json"
    raw_out_path = root / "data" / "generated" / f"{output_prefix}_raw.jsonl"
    converted_path = root / "data" / "generated" / f"{output_prefix}_converted.jsonl"
    report_path = root / "reports" / ("stage2_vllm_smoke_report.json" if backend == "vllm_openai" else "stage2_teacher_dryrun_report.json")
    errors_path = root / "reports" / ("stage2_vllm_output_errors.json" if backend == "vllm_openai" else "stage2_teacher_output_errors.json")

    prompts = load_jsonl(prompt_path)[:limit]
    training_labels = set(json.loads(training_path.read_text(encoding="utf-8")))
    started = time.time()
    if backend == "transformers_local":
        raw_rows = run_teacher_outputs(
            prompts,
            model_path,
            max_new_tokens=max_new_tokens,
            load_in_4bit=load_in_4bit,
            raw_output_path=raw_out_path,
        )
    elif backend == "vllm_openai":
        raw_rows = run_vllm_openai_outputs(
            prompts,
            base_url=base_url,
            model_name=model_path,
            max_tokens=max_new_tokens,
            raw_output_path=raw_out_path,
        )
    else:
        raise ValueError(f"unknown backend: {backend}")
    wall_time_seconds = time.time() - started
    converted_rows, conversion_errors = convert_raw_outputs(raw_rows, training_labels)
    validation = validate_converted_rows(converted_rows, training_labels)
    report = build_report(raw_rows, converted_rows, conversion_errors, validation, backend=backend, wall_time_seconds=wall_time_seconds)
    if backend == "vllm_openai":
        local_report_path = root / "reports" / "stage2_teacher_dryrun_report.json"
        if local_report_path.exists():
            local = json.loads(local_report_path.read_text(encoding="utf-8"))
            report["local_backend_comparison"] = {
                "local_total_wall_time_seconds": local.get("total_wall_time_seconds"),
                "local_average_seconds_per_prompt": local.get("average_seconds_per_prompt"),
                "vllm_total_wall_time_seconds": report["total_wall_time_seconds"],
                "vllm_average_seconds_per_prompt": report["average_seconds_per_prompt"],
                "vllm_faster_total_wall_time": (
                    report["total_wall_time_seconds"] < local["total_wall_time_seconds"]
                    if local.get("total_wall_time_seconds") is not None
                    else None
                ),
            }

    write_jsonl(raw_out_path, raw_rows)
    write_jsonl(converted_path, converted_rows)
    write_json(report_path, report)
    write_json(errors_path, build_error_report(conversion_errors, validation, converted_rows))
    return report


def convert_existing_project(
    root: Path | str = ".",
    *,
    raw_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    raw_out_path = raw_path or root / "data" / "generated" / "stage2_teacher_dryrun_20_raw.jsonl"
    training_path = root / "pii_schema" / "training_label_space_80.json"
    converted_path = root / "data" / "generated" / "stage2_teacher_dryrun_20_converted.jsonl"
    report_path = root / "reports" / "stage2_teacher_dryrun_report.json"
    errors_path = root / "reports" / "stage2_teacher_output_errors.json"
    raw_rows = load_jsonl(raw_out_path)
    training_labels = set(json.loads(training_path.read_text(encoding="utf-8")))
    converted_rows, conversion_errors = convert_raw_outputs(raw_rows, training_labels)
    validation = validate_converted_rows(converted_rows, training_labels)
    report = build_report(raw_rows, converted_rows, conversion_errors, validation)
    write_jsonl(converted_path, converted_rows)
    write_json(report_path, report)
    write_json(errors_path, build_error_report(conversion_errors, validation, converted_rows))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--model-path", default="/home/admin/model/qwen3.5-27b")
    parser.add_argument("--backend", choices=["transformers_local", "vllm_openai"], default="transformers_local")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--convert-existing-only", action="store_true")
    args = parser.parse_args(argv)
    if args.convert_existing_only:
        report = convert_existing_project(args.root)
    else:
        report = dryrun_project(
            args.root,
            model_path=args.model_path,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            backend=args.backend,
            base_url=args.base_url,
            output_prefix="stage2_vllm_dryrun_20" if args.backend == "vllm_openai" else "stage2_teacher_dryrun_20",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["prompts_attempted"] != 20:
        raise SystemExit("expected exactly 20 prompts attempted")
    if report["valid_json_outputs"] < 18:
        raise SystemExit("fewer than 18 valid JSON outputs")
    if report["validation_error_count"] != 0:
        raise SystemExit("conversion validation failed")
    if report["labels_outside_training_space"]:
        raise SystemExit("labels outside training space found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
