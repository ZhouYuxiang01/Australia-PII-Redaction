# Final UI Layout — Smoke Report

## Decision Rules

1. fallback_full_input / regex_candidate → review
2. raw_review_risk >= 0.15 → review
3. otherwise → redact

**display_risk = min(1.0, raw_review_risk / 0.15)**

## Results

| Test | Type | Prob | Raw Risk | Display Risk | Decision |
|------|------|------|----------|-------------|----------|
| student_id | STUDENT_ID | 0.751 | 0.098 | 0.65 | **redact** |
| bare_5102 | DRIVERS_LICENCE | 0.221 | 0.4169 | 1.0 | **review** |
| driver | MOBILE | 0.859 | 0.0102 | 0.07 | **redact** |
| passport | PASSPORT_NUMBER | 0.887 | 0.1002 | 0.67 | **redact** |
| dob | DATE_OF_BIRTH | 0.948 | 0.0249 | 0.17 | **redact** |
| order | PAYMENT_CARD_NUMBER | 0.958 | 0.0015 | 0.01 | **review** |
