# Stage 4 Evaluation: Hybrid OPF + Qwen4B Rescue v2

**Backend**: hybrid-opf-qwen4b
**Policy**: hybrid-80class-v2-4b (redact=0.60, review=0.20)
**Head**: last_linear (Qwen3.5-4B-Base, causal_lm, temp=1.046)
**Deterministic rescue**: enabled (6 rules)

## Overall Metrics

| Metric | Value |
|---|---|
| Sample size | 200 |
| Exact P | 0.7596 |
| Exact R | 0.8352 |
| Exact F1 | 0.7956 |
| Overlap P | 0.899 |
| Overlap R | 0.9885 |
| Overlap F1 | 0.9416 |
| p50 latency | 215.7ms |
| p95 latency | 339.6ms |

## Decision Distribution

| Decision | Count |
|---|---|
| REDACT | 277 |
| REVIEW | 8 |
| IGNORE | 2 |
| PII leaks | 8 |

## Per-Label Recall

| Label | Recall |
|---|---|
| ABORIGINALITY | 1.0 |
| ADDRESS | 1.0 |
| AU_TFN | 1.0 |
| BANK_ACCOUNT_NUMBER | 1.0 |
| CARING_RESPONSIBILITIES | 1.0 |
| CENTRELINK_REFERENCE_NUMBER | 1.0 |
| CONTRACT_TYPE | 1.0 |
| COOKIE_INFORMATION | 1.0 |
| CREDIT_CARD_EXPIRY | 1.0 |
| CRIMINAL_RECORDS | 1.0 |
| DATE_OF_BIRTH | 1.0 |
| DEVICE_ID | 1.0 |
| DISABILITY_OR_SPECIFIC_CONDITION | 1.0 |
| DRIVERS_LICENCE | 0.8 |
| EMAIL_ADDRESS | 1.0 |
| EMPLOYEE_NUMBER | 1.0 |
| EMPLOYMENT_INFORMATION | 1.0 |
| FACIAL_RECOGNITION | 1.0 |
| FINGERPRINT | 1.0 |
| GENDER | 0.0 |
| HASHED_PAYMENT_CARD_NUMBER | 1.0 |
| HOME_PHONE | 1.0 |
| IHI | 1.0 |
| IP_ADDRESS | 1.0 |
| LAST_NAME | 1.0 |
| LATITUDE | 1.0 |
| MARITAL_STATUS | 1.0 |
| MEDICAL_CERTIFICATE | 1.0 |
| MEDICAL_INFORMATION | 1.0 |
| MEDICARE_EXPIRY | 1.0 |
| MEDICARE_NUMBER | 1.0 |
| MILITARY_VETERAN_STATUS | 1.0 |
| MOBILE | 1.0 |
| NATIONALITY | 1.0 |
| NATIONAL_IDENTITY_CARD | 1.0 |
| NUMBER_PLATE | 1.0 |
| PASSPORT_EXPIRY | 1.0 |
| PASSPORT_NUMBER | 1.0 |
| PASSPORT_START_DATE | 1.0 |
| PAYMENT_CARD_NUMBER | 1.0 |
| PERSON | 1.0 |
| PERSONAL_DEBT | 1.0 |
| PRONOUN | 0.0 |
| RACIAL_ETHNIC_ORIGIN | 1.0 |
| RELIGION_BELIEF | 1.0 |
| SALARY | 1.0 |
| SANCTIONS | 1.0 |
| SCHOLARSHIP | 1.0 |
| SEXUAL_ORIENTATION | 1.0 |
| SIGNATURE | 1.0 |
| SOCIAL_MEDIA_HISTORY | 1.0 |
| SOCIAL_MEDIA_ID | 1.0 |
| SOCIO_ECONOMIC_STATUS | 1.0 |
| STUDENT_ID | 1.0 |
| SUBJECT_RESULTS | 1.0 |
| UAC_ID | 1.0 |
| USERNAME | 1.0 |
| USI | 1.0 |
| VEHICLE_REGO | 1.0 |
| WAM_SCORE | 1.0 |
| WEBSITE_HISTORY | 1.0 |

## Low Recall Labels (<0.5)

{'GENDER': 0.0, 'PRONOUN': 0.0}

## Known Limitations

- GENDER, PRONOUN: context-based labels with zero recall
- DRIVERS_LICENCE: ~0.80 recall
- Conservative on ID documents (PASSPORT, IHI, CRN)

## Configuration

- Backend: configs/backends/hybrid-opf-qwen4b.json
- Policy: configs/policies/hybrid-80class-v2-4b.json
- Model: Qwen3.5-4B-Base + last_linear head (causal_lm)
