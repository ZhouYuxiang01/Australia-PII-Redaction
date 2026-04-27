# Step 11: Qwen 9B Adapter Post-Processing

This step adds a standalone post-processing and rescoring script for the existing Qwen3.5 9B adapter cleaned-200 predictions.

No model training or inference was run. The script reads the saved predictions:

```text
outputs/qwen3.5-9b-adapter-cleaned-200/cleaned_200_predictions.jsonl
```

## Files Added Or Produced

- `scripts/postprocess_qwen_cleaned_predictions.py`
- `outputs/qwen3.5-9b-adapter-cleaned-200/postprocessed_safe_predictions.jsonl`
- `outputs/qwen3.5-9b-adapter-cleaned-200/postprocessed_safe_summary.json`
- `outputs/qwen3.5-9b-adapter-cleaned-200/postprocessed_safe_summary.md`

## Rules

The safe post-processing rules are:

- strip field-label prefixes from selected value types:
  - `DATE_OF_BIRTH`: `DOB`, `D.O.B`, `date of birth`, `born`
  - `AU_TFN`: `TFN`, `T.F.N`, `tax file number`
  - `STUDENT_ID`: leading generic `ID:`
  - `CENTRELINK_REFERENCE_NUMBER`: duplicated or bare leading `CRN`
- collapse work/general contact types by context:
  - `WORK_EMAIL -> EMAIL_ADDRESS` only when the local context is a generic email label;
  - `WORK_PHONE -> AU_PHONE` only when the local context is a generic phone label.
- add URL-decoded emails from text containing patterns such as `user%40domain.com`.

## Results

| Version | Precision | Recall | F1 | Sample Exact | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 9B adapter | 0.9247 | 0.8976 | 0.9110 | 0.6900 | 798 | 65 | 91 |
| Safe post-processing | 0.9446 | 0.9201 | 0.9322 | 0.7200 | 818 | 48 | 71 |

The safe rules improve F1 by `+0.0212` without relying on benchmark-specific date splitting.

## Remaining Weak Types

After the safe post-processing run:

| Type | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `AU_TFN` | 0.7541 | 0.7541 | 0.7541 | 46 | 15 | 15 |
| `DATE_OF_BIRTH` | 0.9027 | 0.8226 | 0.8608 | 102 | 11 | 22 |
| `STUDENT_ID` | 0.8936 | 0.8750 | 0.8842 | 42 | 5 | 6 |
| `PERSON` | 0.9486 | 0.8973 | 0.9222 | 166 | 9 | 19 |
| `AU_PHONE` | 0.9143 | 1.0000 | 0.9552 | 64 | 6 | 0 |

Most structured types reach perfect value-level F1 after safe post-processing, including passport, Medicare, bank, BSB, payment card, expiry, CVV, UAC, USI, Centrelink, and email.

## Recommendation

Use the safe post-processing rules as the production-oriented path.

Do not use benchmark-only DOB splitting as an official result. It changes one complete date value into multiple benchmark values, which is not the desired production redaction behavior.

The next high-value work is to improve `AU_TFN`, `DATE_OF_BIRTH`, and `PERSON` on a dev/calibration set. Do not tune those remaining rules directly on cleaned 200.
