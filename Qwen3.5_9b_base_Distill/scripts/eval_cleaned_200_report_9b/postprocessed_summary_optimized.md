# Qwen Cleaned-200 Postprocessed Evaluation

Source predictions: `eval_cleaned_200_report_9b/cleaned_200_predictions.jsonl`
Postprocessed predictions: `eval_cleaned_200_report_9b/postprocessed_predictions_optimized.jsonl`

## Before vs After

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| `precision` | 0.9247 | 0.9919 | +0.0673 |
| `recall` | 0.8976 | 0.9696 | +0.0720 |
| `f1` | 0.9110 | 0.9807 | +0.0697 |
| `sample_exact_acc` | 0.6900 | 0.8750 | +0.1850 |
| `tp` | 798 | 862 | +64 |
| `fp` | 65 | 7 | -58 |
| `fn` | 91 | 27 | -64 |

## Rule Impact

- Fixed FP pairs: 27
- Fixed FN pairs: 33
- Rows improved: 49
- Rows worsened: 0

## Difficulty

| Difficulty | Sample Exact | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EASY` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 88 | 0 | 0 |
| `EXPERT` | 0.9000 | 1.0000 | 0.9846 | 0.9922 | 320 | 0 | 5 |
| `HARD` | 0.8333 | 0.9798 | 0.9529 | 0.9662 | 243 | 5 | 12 |
| `MEDIUM` | 0.8000 | 0.9906 | 0.9548 | 0.9724 | 211 | 2 | 10 |
| `TRAP` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |

## Weakest Remaining Types

| Type | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `PERSON` | 0.9655 | 0.9081 | 0.9359 | 168 | 6 | 17 |
| `SALARY` | 1.0000 | 0.9231 | 0.9600 | 12 | 0 | 1 |
| `ADDRESS` | 0.9910 | 0.9322 | 0.9607 | 110 | 1 | 8 |
| `STUDENT_ID` | 1.0000 | 0.9792 | 0.9895 | 47 | 0 | 1 |
| `AU_BANK_ACCOUNT` | 1.0000 | 1.0000 | 1.0000 | 12 | 0 | 0 |
| `AU_DRIVERS_LICENCE` | 1.0000 | 1.0000 | 1.0000 | 20 | 0 | 0 |
| `AU_PASSPORT` | 1.0000 | 1.0000 | 1.0000 | 30 | 0 | 0 |
| `AU_PHONE` | 1.0000 | 1.0000 | 1.0000 | 64 | 0 | 0 |
| `AU_TFN` | 1.0000 | 1.0000 | 1.0000 | 61 | 0 | 0 |
| `BSB` | 1.0000 | 1.0000 | 1.0000 | 12 | 0 | 0 |
| `CENTRELINK_REFERENCE_NUMBER` | 1.0000 | 1.0000 | 1.0000 | 10 | 0 | 0 |
| `CREDIT_CARD_CVV` | 1.0000 | 1.0000 | 1.0000 | 8 | 0 | 0 |
