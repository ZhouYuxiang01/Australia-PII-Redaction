import json
from pathlib import Path

from pii_prep.stage2_teacher_dryrun import (
    build_error_report,
    build_report,
    convert_raw_outputs,
    load_jsonl,
    validate_converted_rows,
    write_json,
    write_jsonl,
)


def main() -> int:
    root = Path(".")
    raw_path = root / "data" / "generated" / "stage2_vllm_pilot_200_raw.jsonl"
    converted_path = root / "data" / "generated" / "stage2_vllm_pilot_200_converted_calibrated.jsonl"
    report_path = root / "reports" / "stage2_vllm_pilot_calibrated_conversion_report.json"
    errors_path = root / "reports" / "stage2_vllm_pilot_calibrated_conversion_errors.json"
    training_path = root / "pii_schema" / "training_label_space_80.json"

    raw_rows = load_jsonl(raw_path)
    training_labels = set(json.loads(training_path.read_text(encoding="utf-8")))
    converted_rows, conversion_errors = convert_raw_outputs(raw_rows, training_labels)
    validation = validate_converted_rows(converted_rows, training_labels)
    report = build_report(
        raw_rows,
        converted_rows,
        conversion_errors,
        validation,
        backend="vllm_openai_async_calibrated_offline",
    )
    report["source_teacher_call_count"] = report["teacher_calls_executed"]
    report["teacher_calls_executed"] = 0
    report["teacher_calls_executed_by_reconversion"] = 0
    report["source_raw_path"] = str(raw_path)
    report["merged_into_training_dataset"] = False

    write_jsonl(converted_path, converted_rows)
    write_json(report_path, report)
    write_json(errors_path, build_error_report(conversion_errors, validation, converted_rows))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["validation_error_count"] != 0:
        raise SystemExit("calibrated conversion validation failed")
    if report["labels_outside_training_space"]:
        raise SystemExit("labels outside training space found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
