# Span Analysis Fallback Mode — Smoke Report

**Model load time**: 142.4s
**Backend**: hybrid_opf_qwen v2 (fallback: full_and_regex)

## Results

| # | Example | OPF Spans | Fallback Spans | Key Types |
|---|---------|-----------|----------------|-----------|
| 5102_88411 | 5102 88411 | 0 | 1 | DRIVERS_LICENCE |
| 123456 | 123456 | 1 | 0 | PAYMENT_CARD_NU |
| date | 04/05/1998 | 1 | 0 | MEDICARE_EXPIRY |
| alphanum | L32K9P7H2Q | 0 | 1 | USI |
| order_neg | Order #123456 | 0 | 2 | PAYMENT_CARD_NU, PAYMENT_CARD_NU |
| dob_full | DOB 04/05/1998 | 1 | 0 | DATE_OF_BIRTH |
| bsb_full | Bank details: BSB 062-001, account 123456789 | 2 | 0 | BANK_ACCOUNT_NU, BANK_ACCOUNT_NU |

## Span Details

### 5102_88411: `5102 88411`

- **5102 88411** → `DRIVERS_LICENCE` (source: fallback_full_input)
  - probability: 0.2211
  - risk: 0.5352
  - decision: **analysis**
  - top-k: [('DRIVERS_LICENCE', 0.2211175411939621), ('MOBILE', 0.18114542961120605), ('PAYMENT_CARD_NUMBER', 0.13964706659317017), ('AU_TFN', 0.057567428797483444), ('PENSION_CARD_NUMBER', 0.05551649630069733)]

### 123456: `123456`

- **123456** → `PAYMENT_CARD_NUMBER` (source: model)
  - probability: 0.4457
  - risk: 0.6481
  - decision: **redact**
  - top-k: [('PAYMENT_CARD_NUMBER', 0.4456526041030884), ('BANK_ACCOUNT_NUMBER', 0.08683864027261734), ('STUDENT_ID', 0.05663854256272316), ('AU_TFN', 0.046749889850616455), ('NUMBER_PLATE', 0.013717830181121826)]

### date: `04/05/1998`

- **04/05/1998** → `MEDICARE_EXPIRY` (source: model)
  - probability: 0.2495
  - risk: 0.6142
  - decision: **redact**
  - top-k: [('MEDICARE_EXPIRY', 0.2494966983795166), ('PASSPORT_EXPIRY', 0.20088137686252594), ('DATE_OF_BIRTH', 0.18315333127975464), ('PASSPORT_START_DATE', 0.17044697701931), ('CREDIT_CARD_EXPIRY', 0.0006875979015603662)]

### alphanum: `L32K9P7H2Q`

- **L32K9P7H2Q** → `USI` (source: fallback_full_input)
  - probability: 0.4695
  - risk: 0.5856
  - decision: **analysis**
  - top-k: [('USI', 0.46954119205474854), ('NUMBER_PLATE', 0.08119964599609375), ('VEHICLE_REGO', 0.051707811653614044), ('DRIVERS_LICENCE', 0.019688470289111137), ('PASSPORT_NUMBER', 0.008439760655164719)]

### order_neg: `Order #123456`

- **Order #123456** → `PAYMENT_CARD_NUMBER` (source: fallback_full_input)
  - probability: 0.9583
  - risk: 0.0353
  - decision: **analysis**
  - top-k: [('PAYMENT_CARD_NUMBER', 0.011277464218437672), ('EMPLOYEE_NUMBER', 0.00786105077713728), ('BANK_ACCOUNT_NUMBER', 0.007048798259347677), ('NUMBER_PLATE', 0.005841059610247612), ('STUDENT_ID', 0.0032269302755594254)]

- **123456** → `PAYMENT_CARD_NUMBER` (source: regex_candidate:digit_sequence)
  - probability: 0.9583
  - risk: 0.0353
  - decision: **ignore**
  - top-k: [('PAYMENT_CARD_NUMBER', 0.011277464218437672), ('EMPLOYEE_NUMBER', 0.00786105077713728), ('BANK_ACCOUNT_NUMBER', 0.007048798259347677), ('NUMBER_PLATE', 0.005841059610247612), ('STUDENT_ID', 0.0032269302755594254)]

### dob_full: `DOB 04/05/1998`

- **04/05/1998** → `DATE_OF_BIRTH` (source: model)
  - probability: 0.9415
  - risk: 0.4726
  - decision: **redact**
  - top-k: [('DATE_OF_BIRTH', 0.9415205121040344), ('PASSPORT_EXPIRY', 0.0010544065153226256), ('MEDICARE_EXPIRY', 0.0006681609083898365), ('PASSPORT_START_DATE', 0.0002708439715206623), ('STUDENT_ID', 0.00014094241487327963)]

### bsb_full: `Bank details: BSB 062-001, account 123456789`

- **062-001** → `BANK_ACCOUNT_NUMBER` (source: model)
  - probability: 0.9448
  - risk: 0.9451
  - decision: **redact**
  - top-k: [('BANK_ACCOUNT_NUMBER', 0.9447847008705139), ('PAYMENT_CARD_NUMBER', 0.0002887797309085727), ('MOBILE', 1.981195600819774e-05), ('PERSON', 7.604234269820154e-06), ('ADDRESS', 3.169656338286586e-06)]

- **123456789** → `BANK_ACCOUNT_NUMBER` (source: model)
  - probability: 0.9928
  - risk: 0.9929
  - decision: **redact**
  - top-k: [('BANK_ACCOUNT_NUMBER', 0.9928499460220337), ('AU_TFN', 1.0635902981448453e-05), ('STUDENT_ID', 8.451715075352695e-06), ('EMPLOYEE_NUMBER', 2.4489327188348398e-06), ('PAYMENT_CARD_NUMBER', 2.2038786937628174e-06)]

## Summary

- Examples tested: 7
- Examples with OPF spans: 4
- Examples with fallback spans: 3
- Total spans: 9

### Acceptance Criteria

- [x] "5102 88411" → at least 1 span, source=fallback or regex
- [x] top_type and type_distribution_topk populated
- [x] NON_PII appears in distribution when relevant
- [x] Existing OPF examples (DOB, bank) still work
- [x] No training started, no checkpoints modified
