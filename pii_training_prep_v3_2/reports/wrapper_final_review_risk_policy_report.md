# Final Review Risk Policy — Smoke Report

## Decision Rules

1. fallback_full_input → review
2. privacy < 0.20 → review
3. top_prob >= 0.80 AND privacy >= 0.50 → redact
4. review_risk >= 0.15 → review
5. otherwise → redact

**risk_score = (1 - top_probability) * privacy_score**

## Results

| Test | Type | Prob | Privacy | Risk | Decision | Reason |
|------|------|------|---------|------|----------|--------|
| bank | BANK_ACCOUNT_NUMBER | 0.945 | 0.945 | 0.052 | **redact** | high_confidence_redact |
| bank | BANK_ACCOUNT_NUMBER | 0.993 | 0.993 | 0.007 | **redact** | high_confidence_redact |
| driver | MOBILE | 0.859 | 0.072 | 0.01 | **review** | low_privacy_score_review |
| bare_5102 | DRIVERS_LICENCE | 0.221 | 0.535 | 0.417 | **review** | fallback_full_input |
| student_id | STUDENT_ID | 0.751 | 0.394 | 0.098 | **redact** | low_review_risk_redact |
| dob | DATE_OF_BIRTH | 0.942 | 0.473 | 0.028 | **redact** | low_review_risk_redact |
| multi | PERSON | 0.661 | 0.332 | 0.112 | **redact** | low_review_risk_redact |
| multi | PASSPORT_NUMBER | 0.535 | 0.539 | 0.25 | **review** | high_review_risk |
| multi | DATE_OF_BIRTH | 0.938 | 0.47 | 0.029 | **redact** | low_review_risk_redact |
| multi | IHI | 0.78 | 0.129 | 0.028 | **review** | low_privacy_score_review |
