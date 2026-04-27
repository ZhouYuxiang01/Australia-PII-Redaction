# Qwen Cleaned-200 Postprocessed Evaluation

Source predictions: `outputs\qwen3.5-9b-adapter-cleaned-200\cleaned_200_predictions.jsonl`
Postprocessed predictions: `outputs\qwen3.5-9b-adapter-cleaned-200\postprocessed_predictions.jsonl`

## Before vs After

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| `precision` | 0.9247 | 0.9426 | +0.0179 |
| `recall` | 0.8976 | 0.9415 | +0.0439 |
| `f1` | 0.9110 | 0.9420 | +0.0311 |
| `sample_exact_acc` | 0.6900 | 0.7200 | +0.0300 |
| `tp` | 798 | 837 | +39 |
| `fp` | 65 | 51 | -14 |
| `fn` | 91 | 52 | -39 |

## Rule Impact

- Fixed FP pairs: 32
- Fixed FN pairs: 54
- Rows improved: 35
- Rows worsened: 13

## Difficulty

| Difficulty | Sample Exact | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EASY` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 88 | 0 | 0 |
| `EXPERT` | 0.5800 | 0.9329 | 0.9415 | 0.9372 | 306 | 22 | 19 |
| `HARD` | 0.7000 | 0.9405 | 0.9294 | 0.9349 | 237 | 15 | 18 |
| `MEDIUM` | 0.6600 | 0.9364 | 0.9321 | 0.9342 | 206 | 14 | 15 |
| `TRAP` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |

## Weakest Remaining Types

| Type | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `AU_TFN` | 0.7541 | 0.7541 | 0.7541 | 46 | 15 | 15 |
| `STUDENT_ID` | 0.8936 | 0.8750 | 0.8842 | 42 | 5 | 6 |
| `PERSON` | 0.9486 | 0.8973 | 0.9222 | 166 | 9 | 19 |
| `DATE_OF_BIRTH` | 0.8963 | 0.9758 | 0.9344 | 121 | 14 | 3 |
| `AU_PHONE` | 0.9143 | 1.0000 | 0.9552 | 64 | 6 | 0 |
| `SALARY` | 1.0000 | 0.9231 | 0.9600 | 12 | 0 | 1 |
| `ADDRESS` | 0.9910 | 0.9322 | 0.9607 | 110 | 1 | 8 |
| `AU_DRIVERS_LICENCE` | 0.9524 | 1.0000 | 0.9756 | 20 | 1 | 0 |
| `AU_BANK_ACCOUNT` | 1.0000 | 1.0000 | 1.0000 | 12 | 0 | 0 |
| `AU_PASSPORT` | 1.0000 | 1.0000 | 1.0000 | 30 | 0 | 0 |
| `BSB` | 1.0000 | 1.0000 | 1.0000 | 12 | 0 | 0 |
| `CENTRELINK_REFERENCE_NUMBER` | 1.0000 | 1.0000 | 1.0000 | 10 | 0 | 0 |
