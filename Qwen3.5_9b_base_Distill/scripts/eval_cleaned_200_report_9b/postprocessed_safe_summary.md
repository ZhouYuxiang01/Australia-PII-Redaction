# Qwen Cleaned-200 Postprocessed Evaluation

Source predictions: `outputs\qwen3.5-9b-adapter-cleaned-200\cleaned_200_predictions.jsonl`
Postprocessed predictions: `outputs\qwen3.5-9b-adapter-cleaned-200\postprocessed_safe_predictions.jsonl`

## Before vs After

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| `precision` | 0.9247 | 0.9446 | +0.0199 |
| `recall` | 0.8976 | 0.9201 | +0.0225 |
| `f1` | 0.9110 | 0.9322 | +0.0212 |
| `sample_exact_acc` | 0.6900 | 0.7200 | +0.0300 |
| `tp` | 798 | 818 | +20 |
| `fp` | 65 | 48 | -17 |
| `fn` | 91 | 71 | -20 |

## Rule Impact

- Fixed FP pairs: 32
- Fixed FN pairs: 35
- Rows improved: 29
- Rows worsened: 15

## Difficulty

| Difficulty | Sample Exact | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EASY` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 88 | 0 | 0 |
| `EXPERT` | 0.5800 | 0.9340 | 0.9138 | 0.9238 | 297 | 21 | 28 |
| `HARD` | 0.7000 | 0.9390 | 0.9059 | 0.9222 | 231 | 15 | 24 |
| `MEDIUM` | 0.6600 | 0.9439 | 0.9140 | 0.9287 | 202 | 12 | 19 |
| `TRAP` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |

## Weakest Remaining Types

| Type | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `AU_TFN` | 0.7541 | 0.7541 | 0.7541 | 46 | 15 | 15 |
| `DATE_OF_BIRTH` | 0.9027 | 0.8226 | 0.8608 | 102 | 11 | 22 |
| `STUDENT_ID` | 0.8936 | 0.8750 | 0.8842 | 42 | 5 | 6 |
| `PERSON` | 0.9486 | 0.8973 | 0.9222 | 166 | 9 | 19 |
| `AU_PHONE` | 0.9143 | 1.0000 | 0.9552 | 64 | 6 | 0 |
| `SALARY` | 1.0000 | 0.9231 | 0.9600 | 12 | 0 | 1 |
| `ADDRESS` | 0.9910 | 0.9322 | 0.9607 | 110 | 1 | 8 |
| `AU_DRIVERS_LICENCE` | 0.9524 | 1.0000 | 0.9756 | 20 | 1 | 0 |
| `AU_BANK_ACCOUNT` | 1.0000 | 1.0000 | 1.0000 | 12 | 0 | 0 |
| `AU_PASSPORT` | 1.0000 | 1.0000 | 1.0000 | 30 | 0 | 0 |
| `BSB` | 1.0000 | 1.0000 | 1.0000 | 12 | 0 | 0 |
| `CENTRELINK_REFERENCE_NUMBER` | 1.0000 | 1.0000 | 1.0000 | 10 | 0 | 0 |
