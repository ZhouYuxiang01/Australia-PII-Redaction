# Wrapper Hybrid Backend Smoke Test

**Date**: 2026-05-01 17:59
**Backend**: hybrid_opf_qwen
**Model load time**: 142.5s

## Results

| # | Example | Spans | Types | Decisions | Latency |
|---|---------|-------|-------|-----------|---------|
| DOB | Student DOB 14/09/2002. | 1 | DATE_OF_BIRTH | redact | 214.3ms |
| Email | Email amelia.chen@student.example.edu.au. | 1 | WORK_EMAIL | review | 142.9ms |
| Phone | Mobile 0412 345 678. | 1 | MOBILE | redact | 135.7ms |
| Bank | Bank details: BSB 062-001, account 123456789. | 2 | BANK_ACCOUNT_NUMBER, BANK_ACCOUNT_NUMBER | redact, redact | 146.0ms |
| Medicare | Medicare number is 2123 45678 1 and expiry is 01/2 | 0 |  |  | 23.6ms |
| Negative | Order #123456 shipped today. | 0 |  |  | 13.3ms |
| Gender/Salary | Gender: Male. Salary expectation: $120,000. | 1 | SALARY_WAGE_EXPECTAT | redact | 140.6ms |

## Span Details

### DOB: Student DOB 14/09/2002.

- `14/09/2002` → **DATE_OF_BIRTH** (OPF: DATE_OF_BIRTH)
  - Probability: 0.8755
  - Risk score: 0.4575
  - Decision: **redact**
  - Top-k: [('DATE_OF_BIRTH', 0.8755082488059998), ('MEDICARE_EXPIRY', 0.011576320976018906), ('STUDENT_ID', 0.007211369462311268)]

### Email: Email amelia.chen@student.example.edu.au.

- `amelia.chen@student.example.edu.au` → **WORK_EMAIL** (OPF: EMAIL_ADDRESS)
  - Probability: 0.3862
  - Risk score: 0.3652
  - Decision: **review**
  - Top-k: [('WORK_EMAIL', 0.38622450828552246), ('EMAIL_ADDRESS', 0.3405716121196747), ('STUDENT_ID', 0.0010726979235187173)]

### Phone: Mobile 0412 345 678.

- `0412 345 678.` → **MOBILE** (OPF: MOBILE)
  - Probability: 0.6509
  - Risk score: 0.4
  - Decision: **redact**
  - Top-k: [('MOBILE', 0.6509392857551575), ('HOME_PHONE', 0.13075460493564606), ('WORK_PHONE', 0.009174594655632973)]

### Bank: Bank details: BSB 062-001, account 123456789.

- `062-001` → **BANK_ACCOUNT_NUMBER** (OPF: BANK_ACCOUNT_NUMBER)
  - Probability: 0.9448
  - Risk score: 0.9451
  - Decision: **redact**
  - Top-k: [('BANK_ACCOUNT_NUMBER', 0.9447847008705139), ('PAYMENT_CARD_NUMBER', 0.0002887797309085727), ('MOBILE', 1.981195600819774e-05)]

- `123456789` → **BANK_ACCOUNT_NUMBER** (OPF: BANK_ACCOUNT_NUMBER)
  - Probability: 0.9928
  - Risk score: 0.9929
  - Decision: **redact**
  - Top-k: [('BANK_ACCOUNT_NUMBER', 0.9928499460220337), ('AU_TFN', 1.0635902981448453e-05), ('STUDENT_ID', 8.451715075352695e-06)]

### Medicare: Medicare number is 2123 45678 1 and expiry is 01/2026.

No spans detected.

### Negative: Order #123456 shipped today.

No spans detected.

### Gender/Salary: Gender: Male. Salary expectation: $120,000.

- `$120,000.` → **SALARY_WAGE_EXPECTATION** (OPF: SALARY)
  - Probability: 0.3129
  - Risk score: 0.4722
  - Decision: **redact**
  - Top-k: [('SALARY_WAGE_EXPECTATION', 0.23668517172336578), ('SCHOLARSHIP', 0.19158075749874115), ('SALARY', 0.11423264443874359)]

## Summary

- Total examples: 7
- Total spans detected: 6
- Examples with spans: 5
- Model load time: 142.5s

### Acceptance Criteria

- [x] Old backends still import
- [x] New backend imports
- [x] Spans returned on DOB/email/phone examples
- [x] Response includes top_type, opf_top_type, top_probability, risk_score, decision, type_distribution_topk
- [x] No training started
- [x] No checkpoints modified
