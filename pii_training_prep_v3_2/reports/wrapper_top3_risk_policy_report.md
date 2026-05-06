# Top-3 Risk Policy — Smoke Report

## Formula

```
risk_score = (1 - top3_sum) * data_classification_weight(top_type)
```

## Decision Rules

1. fallback_full_input / regex → review
2. risk_score >= 0.25 → review
3. otherwise → redact

## Results

| Test | Type | Top3 Sum | Uncert | DC Weight | Risk | Decision |
|------|------|----------|--------|-----------|------|----------|
| bare_5102 | DRIVERS_LICENCE | 0.542 | 0.458 | 0.5 | 0.229 | **review** |
| student_id | STUDENT_ID | 0.765 | 0.235 | 0.5 | 0.1177 | **redact** |
| bank | BANK_ACCOUNT_NUMBER | 0.945 | 0.055 | 1.0 | 0.0549 | **redact** |
| bank | BANK_ACCOUNT_NUMBER | 0.993 | 0.007 | 1.0 | 0.0071 | **redact** |
| email | EMAIL | 0.728 | 0.272 | 0.5 | 0.1361 | **redact** |
| multi | PERSON | 0.66 | 0.34 | 0.5 | 0.1702 | **redact** |
| multi | PASSPORT_NUMBER | 0.545 | 0.455 | 1.0 | 0.4548 | **review** |
| multi | DATE_OF_BIRTH | 0.94 | 0.06 | 0.5 | 0.03 | **redact** |
