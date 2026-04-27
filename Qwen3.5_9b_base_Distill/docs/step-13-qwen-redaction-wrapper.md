# Step 13: Qwen Safe Redaction Wrapper

This step turns the Qwen3.5 9B base + LoRA adapter route into a deployable redaction surface.

The official route remains the safe post-processing version only. Benchmark-only DOB splitting is intentionally excluded.

## Added Files

- `scripts/qwen_redact.py`
- `configs/policies/qwen-safe-default-v1.json`
- `tests/test_qwen_redact.py`

The existing schema was also extended:

- `schemas/redaction-output-v1.schema.json`

## Wrapper Behavior

`scripts/qwen_redact.py` supports two modes:

1. Parser/wrapper mode with an existing tagged model output.
2. Inference mode with `--base-model` and `--adapter-dir`, using `transformers` + `peft`.

The wrapper:

- normalizes input text as NFC;
- parses Qwen tagged output like `<pii type="PERSON">Alice Wong</pii>`;
- tolerates shell-stripped tags like `<pii type=PERSON>Alice Wong</pii>`;
- repairs offsets when the generated plain text differs from the input but the detected value is unique;
- applies the safe post-processing rules from the cleaned-200 evaluation;
- resolves overlaps deterministically;
- applies the safe policy;
- emits schema v1 JSON with spans and `redacted_text`.

## Policy

`configs/policies/qwen-safe-default-v1.json` records the deployable policy assumptions:

- `policy_id`: `qwen-safe-default-v1`
- `model_version`: `qwen3.5-9b-base-lora-tagged-28-fastretry`
- `taxonomy_version`: `qwen-pii-27-safe-v1`
- confidence is currently uncalibrated and emitted as `null`;
- benchmark-only DOB variant expansion is disabled;
- overlap resolution keeps the longer span, then the earlier span.

Most PII types are `AUTO_REDACT`. `SALARY` is marked `REVIEW`, but the current redacted preview still masks review spans so that the preview does not leak sensitive content.

## Example

```powershell
python scripts\qwen_redact.py `
  --text "Name: Alice Wong TFN: 123 456 789" `
  --annotated-output 'Name: <pii type="PERSON">Alice Wong</pii> TFN: <pii type="AU_TFN">123 456 789</pii>' `
  --policy configs\policies\qwen-safe-default-v1.json `
  --json-out outputs\qwen3.5-9b-adapter-cleaned-200\wrapper_demo.json
```

Expected redacted text:

```text
Name: [PERSON] TFN: [AU_TFN]
```

For raw model output captured in a file, prefer:

```powershell
python scripts\qwen_redact.py `
  --text-file input.txt `
  --annotated-output-file model_output.txt `
  --json-out redaction.json
```

## Validation

Local checks run on 2026-04-22:

```text
python -m unittest discover -s tests -v
python -m py_compile scripts\qwen_redact.py scripts\postprocess_qwen_cleaned_predictions.py scripts\analyze_qwen_cleaned_errors.py
python -c "... JSON parse check ..."
```

Results:

- 9 wrapper tests passed.
- Python compile check passed.
- Policy and schema JSON parse checks passed.
- Full JSON Schema validation was not run locally because `jsonschema` is not installed in this environment.

## Remaining Work

- Run the inference path on the remote server with the real base model and LoRA adapter.
- Add JSON Schema validation in CI or install-time test dependencies.
- Add latency and VRAM measurements on the target 24GB GPU.
- Add batch JSONL input/output if the deployment harness needs bulk redaction.
- Calibrate confidence or keep `confidence: null` explicitly documented in the model card.
