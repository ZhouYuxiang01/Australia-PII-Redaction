# Step 12: Deployable Redaction Gap Checklist

This checklist maps the current project state against the requirements in `deployable-redaction-model.pdf`.

The project should keep the Qwen3.5 9B base + LoRA adapter as the high-accuracy route and keep only the safe post-processing result as the official cleaned-200 result:

| Version | Precision | Recall | F1 | Sample Exact | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 9B adapter | 0.9247 | 0.8976 | 0.9110 | 0.6900 | 798 | 65 | 91 |
| Safe post-processing | 0.9446 | 0.9201 | 0.9322 | 0.7200 | 818 | 48 | 71 |

Benchmark-only DOB splitting is not part of the official result.

## Already Covered

- Taxonomy draft exists for the ModernBERT line:
  - `configs/taxonomy/au-pii-v1.json`
  - `docs/step-01-taxonomy-and-schema.md`
- Offset-based BIO dataset conversion exists for the 20-class ModernBERT line.
- ModernBERT baseline training and span-level evaluation are documented.
- Qwen 9B adapter cleaned-200 value-level baseline and error analysis are documented.
- Safe Qwen output post-processing exists:
  - `scripts/postprocess_qwen_cleaned_predictions.py`
  - `outputs/qwen3.5-9b-adapter-cleaned-200/postprocessed_safe_summary.md`
- Hard-negative/TRAP behavior has been checked on cleaned 200:
  - Qwen 9B adapter + safe post-processing keeps TRAP FP at `0`.

## High-Priority Missing Items

### 1. Deployable Inference Wrapper

The PDF asks for a wrapper that produces redaction-ready spans and deterministic redacted text.

Add a production-facing script or package entry point:

```text
scripts/qwen_redact.py
```

Required behavior:

- input: raw text or JSONL;
- run Qwen adapter inference;
- parse tagged-text output;
- apply safe post-processing;
- return schema v1 spans;
- generate deterministic masked text;
- write JSON output.

This should become the main demo/integration surface.

### 2. Stable Output Schema For The Qwen Route

The existing schema is oriented around span output, but the Qwen route needs an explicit final schema:

```json
{
  "schema_version": "redaction-output-v1",
  "model_version": "qwen3.5-9b-base-lora-tagged-28-fastretry",
  "taxonomy_version": "qwen-pii-27-or-au-pii-v1",
  "policy_id": "safe-default-v1",
  "text_sha256": "...",
  "spans": [
    {
      "start": 0,
      "end": 10,
      "type": "PERSON",
      "value": "Alice Wong",
      "confidence": null,
      "source": "model",
      "postprocess": []
    }
  ],
  "redacted_text": "...",
  "warnings": []
}
```

The main gap is `confidence`. The current Qwen generated output does not expose calibrated span confidence. For now, use `null` and document that confidence calibration is future work, or implement a proxy confidence later from generation scores.

### 3. Deterministic Redaction Function

Add a tested function that applies spans to text without changing offsets during processing:

- replace mode: `[PERSON]`, `[AU_TFN]`, etc.;
- mask mode: preserve length with fixed character;
- remove mode: remove values;
- handles overlapping spans by deterministic priority;
- validates `text[start:end] == value` when value is present;
- logs skipped invalid spans.

This is central to the PDF because the deliverable is a redaction model, not only a detector.

### 4. Policy Configuration

The PDF explicitly asks for `block vs review` operating modes and versioned policies.

Add:

```text
configs/policies/safe-default-v1.json
configs/policies/review-high-recall-v1.json
```

Even before confidence calibration, the policy should define:

- type groups;
- action per type: redact / review / allow;
- high-risk types: TFN, passport, licence, Medicare, IHI, payment card;
- collapsed type mappings if using 20-class deployment;
- post-processing rules enabled.

### 5. Unit And Regression Tests

Add tests for:

- parser correctness for Qwen tagged text;
- safe post-processing prefix stripping;
- URL-encoded email recovery;
- work/general email and phone collapsing;
- deterministic redaction and offset correctness;
- hard-negative/TRAP no-redaction cases;
- schema validation.

Recommended initial files:

```text
tests/test_qwen_postprocess.py
tests/test_redaction_wrapper.py
tests/test_schema_validation.py
```

### 6. Calibration And Thresholds

The PDF asks for calibration, reliability curves, ECE, and block/review operating points.

Current state:

- ModernBERT has token probabilities.
- Qwen route does not yet have span confidence.

For this project, add a documented limitation and an implementation plan:

- short term: deterministic policy by type with no confidence;
- medium term: collect generation/token logprob or verifier score;
- final report: state that Qwen safe output is high-accuracy but not yet calibrated.

### 7. Robustness Evaluation Harness

Add a script that runs fixed robustness suites:

- punctuation/spacing/casing changes;
- newline/table-like text;
- near-miss numbers;
- URL-encoded identifiers;
- hard negatives.

Output JSON + Markdown reports. This is separate from cleaned 200 and should be reproducible.

### 8. Latency And Memory Benchmark

The PDF requires one-24GB-GPU deployability evidence.

Add a benchmark script on the remote server:

```text
scripts/benchmark_qwen_inference.py
```

Report:

- GPU model;
- VRAM used;
- batch size;
- max input/new tokens;
- docs/sec;
- p50/p95 latency;
- model loading time.

If 9B is too heavy for 24GB without quantization, document required quantization or offload settings.

### 9. Model Card And Handover README

Add:

```text
MODEL_CARD.md
README.md
docs/reproducibility-checklist.md
```

These should cover:

- model lineage;
- training data provenance;
- taxonomy and unsupported types;
- evaluation results;
- known failure modes;
- safe post-processing;
- privacy constraints;
- how to run inference/evaluation;
- deployment assumptions.

## Medium-Priority Items

- Dockerfile or deployment environment notes.
- FastAPI service wrapper if needed for demo.
- CI or at least local test command documentation.
- Teacher baseline comparison table, if teacher outputs are available.
- Dataset provenance logs for synthetic/templated data generation.
- Over-redaction and under-redaction cost metrics, especially weighted missed high-risk identifiers.

## Recommended Next Implementation Order

1. Build `scripts/qwen_redact.py` with parser, safe post-processing, schema output, and redacted text.
2. Add deterministic redaction unit tests.
3. Add `configs/policies/safe-default-v1.json`.
4. Add a compact README section showing one-command inference and output schema.
5. Add remote latency/VRAM benchmark for the 9B adapter.
6. Add robustness evaluation after the wrapper is stable.

This order turns the current strong evaluation result into a deployable redaction component, which is the main remaining gap against the PDF.
