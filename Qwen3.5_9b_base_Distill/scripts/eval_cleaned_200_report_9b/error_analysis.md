# Qwen 9B Adapter Cleaned-200 Error Analysis

Predictions: `outputs\qwen3.5-9b-adapter-cleaned-200\cleaned_200_predictions.jsonl`
Summary: `outputs\qwen3.5-9b-adapter-cleaned-200\cleaned_200_value_level_summary.json`

## Overall

- Rows: 200
- Parse failures: 0
- Round-trip failures: 11
- Sample-level error rows: 62
- Rows with FP / FN / both: 50 / 60 / 48
- Value-level P/R/F1: 0.9247 / 0.8976 / 0.9110
- TP / FP / FN: 798 / 65 / 91

## Error Categories

### False Positives

| Category | Count |
| --- | ---: |
| `included_field_label_or_prefix` | 31 |
| `same_type_partial_or_joined_value` | 17 |
| `extra_prediction` | 11 |
| `wrong_type_same_value` | 6 |

### False Negatives

| Category | Count |
| --- | ---: |
| `field_label_prefix_mismatch` | 31 |
| `same_type_partial_or_joined_value` | 28 |
| `missed_no_related_prediction` | 26 |
| `wrong_type_same_value` | 6 |

## FP By Type

| Type | Count |
| --- | ---: |
| `DATE_OF_BIRTH` | 18 |
| `STUDENT_ID` | 17 |
| `PERSON` | 9 |
| `AU_PHONE` | 6 |
| `CENTRELINK_REFERENCE_NUMBER` | 4 |
| `AU_TFN` | 3 |
| `WORK_PHONE` | 3 |
| `WORK_EMAIL` | 3 |
| `AU_DRIVERS_LICENCE` | 1 |
| `ADDRESS` | 1 |

## FN By Type

| Type | Count |
| --- | ---: |
| `DATE_OF_BIRTH` | 29 |
| `PERSON` | 19 |
| `STUDENT_ID` | 18 |
| `ADDRESS` | 8 |
| `EMAIL_ADDRESS` | 6 |
| `CENTRELINK_REFERENCE_NUMBER` | 4 |
| `AU_TFN` | 3 |
| `AU_PHONE` | 3 |
| `SALARY` | 1 |

## Top Type-Category Pairs

| Side | Type | Category | Count |
| --- | --- | --- | ---: |
| FP | `STUDENT_ID` | `included_field_label_or_prefix` | 17 |
| FP | `DATE_OF_BIRTH` | `same_type_partial_or_joined_value` | 11 |
| FP | `DATE_OF_BIRTH` | `included_field_label_or_prefix` | 7 |
| FP | `PERSON` | `same_type_partial_or_joined_value` | 6 |
| FP | `AU_PHONE` | `extra_prediction` | 6 |
| FP | `CENTRELINK_REFERENCE_NUMBER` | `included_field_label_or_prefix` | 4 |
| FP | `AU_TFN` | `included_field_label_or_prefix` | 3 |
| FP | `WORK_PHONE` | `wrong_type_same_value` | 3 |
| FP | `WORK_EMAIL` | `wrong_type_same_value` | 3 |
| FP | `PERSON` | `extra_prediction` | 3 |
| FP | `AU_DRIVERS_LICENCE` | `extra_prediction` | 1 |
| FP | `ADDRESS` | `extra_prediction` | 1 |
| FN | `DATE_OF_BIRTH` | `same_type_partial_or_joined_value` | 22 |
| FN | `STUDENT_ID` | `field_label_prefix_mismatch` | 17 |
| FN | `PERSON` | `missed_no_related_prediction` | 13 |
| FN | `ADDRESS` | `missed_no_related_prediction` | 8 |
| FN | `DATE_OF_BIRTH` | `field_label_prefix_mismatch` | 7 |
| FN | `PERSON` | `same_type_partial_or_joined_value` | 6 |
| FN | `CENTRELINK_REFERENCE_NUMBER` | `field_label_prefix_mismatch` | 4 |
| FN | `AU_TFN` | `field_label_prefix_mismatch` | 3 |
| FN | `AU_PHONE` | `wrong_type_same_value` | 3 |
| FN | `EMAIL_ADDRESS` | `missed_no_related_prediction` | 3 |
| FN | `EMAIL_ADDRESS` | `wrong_type_same_value` | 3 |
| FN | `SALARY` | `missed_no_related_prediction` | 1 |

## Round-Trip Failures

| ID | Difficulty | Sample Exact | FP | FN |
| --- | --- | ---: | ---: | ---: |
| `TEST-022` | `EXPERT` | true | 0 | 0 |
| `TEST-058` | `EXPERT` | false | 0 | 1 |
| `TEST-088` | `MEDIUM` | true | 0 | 0 |
| `TEST-092` | `MEDIUM` | true | 0 | 0 |
| `TEST-112` | `HARD` | true | 0 | 0 |
| `TEST-138` | `MEDIUM` | true | 0 | 0 |
| `TEST-142` | `MEDIUM` | true | 0 | 0 |
| `TEST-145` | `MEDIUM` | true | 0 | 0 |
| `TEST-156` | `MEDIUM` | true | 0 | 0 |
| `TEST-168` | `EXPERT` | false | 0 | 1 |
| `TEST-196` | `MEDIUM` | true | 0 | 0 |

## Interpretation

The model has no parse failures and no TRAP false positives in the saved summary. Most remaining errors are value-boundary or label-prefix issues, not fundamental extraction failures.

The largest mechanically recoverable group is field-label inclusion: values such as `DOB: 25/06/1969`, `TFN: 832 109 111`, `ID: 405997905`, and `CRN 585 024 614V` should be normalized to the identifier value before scoring or final output.

The second major group is date value shape. The model often outputs a complete date such as `September 13, 1966` while the value-level ground truth stores `September 13` and `1966` separately. For redaction, the complete-date span is arguably preferable; for this benchmark, a value normalizer or controlled split would improve the score.

