# Final Results Tables

## 1. OPF-Only Hard-Label Detection

**Model**: `runs/opf_hard_79` | **Labels**: 79 PII classes | **Training**: 81,298 examples, 1 epoch

### Overall Metrics

| Dataset | Examples | Loss | Token Accuracy | Detection F1 | Span F1 | Precision | Recall |
|---------|----------|------|----------------|--------------|---------|-----------|--------|
| Dev | 10,034 | 0.0333 | 98.94% | 0.9724 | 0.9606 | 0.9631 | 0.9817 |
| Test | 9,659 | 0.0306 | 99.15% | 0.9793 | 0.9725 | 0.9745 | 0.9842 |

### Per-Label Span F1 (Test Set)

| Label | F1 | Precision | Recall | Status |
|-------|-----|-----------|--------|--------|
| ABORIGINALITY | 0.984 | 0.969 | 1.000 | ✅ |
| AUDIO_INFORMATION | 0.991 | 0.983 | 1.000 | ✅ |
| BANK_ACCOUNT_INFORMATION | 1.000 | 1.000 | 1.000 | ✅ |
| BANK_ACCOUNT_NUMBER | 1.000 | 1.000 | 1.000 | ✅ |
| CENTRELINK_REFERENCE_NUMBER | 1.000 | 1.000 | 1.000 | ✅ |
| CITIZENSHIP_STATUS | 0.994 | 0.989 | 1.000 | ✅ |
| CRIMINAL_RECORDS | 1.000 | 1.000 | 1.000 | ✅ |
| DATE_OF_BIRTH | 0.963 | 0.955 | 0.972 | ✅ |
| DEVICE_ID | 0.995 | 0.989 | 1.000 | ✅ |
| DRIVERS_LICENCE | 0.998 | 1.000 | 0.996 | ✅ |
| EMPLOYEE_NUMBER | 0.980 | 0.961 | 1.000 | ✅ |
| EMPLOYMENT_INFORMATION | 0.996 | 1.000 | 0.992 | ✅ |
| FACIAL_RECOGNITION | 1.000 | 1.000 | 1.000 | ✅ |
| FINGERPRINT | 1.000 | 1.000 | 1.000 | ✅ |
| FIRST_NAME | 1.000 | 1.000 | 1.000 | ✅ |
| HASHED_PAYMENT_CARD_NUMBER | 0.990 | 0.980 | 1.000 | ✅ |
| HOME_PHONE | 1.000 | 1.000 | 1.000 | ✅ |
| IHI | 1.000 | 1.000 | 1.000 | ✅ |
| LAST_NAME | 1.000 | 1.000 | 1.000 | ✅ |
| MARITAL_STATUS | 0.941 | 0.935 | 0.947 | ✅ |
| MEDICARE_EXPIRY | 1.000 | 1.000 | 1.000 | ✅ |
| MEDICARE_NUMBER | 1.000 | 1.000 | 1.000 | ✅ |
| MILITARY_VETERAN_STATUS | 1.000 | 1.000 | 1.000 | ✅ |
| MOBILE | 0.999 | 0.998 | 1.000 | ✅ |
| NATIONAL_IDENTITY_CARD | 0.983 | 0.977 | 0.988 | ✅ |
| PASSPORT_EXPIRY | 1.000 | 1.000 | 1.000 | ✅ |
| PASSPORT_NUMBER | 1.000 | 1.000 | 1.000 | ✅ |
| PASSPORT_START_DATE | 1.000 | 1.000 | 1.000 | ✅ |
| PENSION_CARD_NUMBER | 1.000 | 1.000 | 1.000 | ✅ |
| PERSONNEL_NUMBER | 0.984 | 1.000 | 0.969 | ✅ |
| PRONOUN | 0.995 | 1.000 | 0.990 | ✅ |
| RACIAL_ETHNIC_ORIGIN | 0.967 | 1.000 | 0.937 | ✅ |
| RELIGION_BELIEF | 0.946 | 0.978 | 0.916 | ✅ |
| SALARY | 1.000 | 1.000 | 1.000 | ✅ |
| SIGNATURE | 1.000 | 1.000 | 1.000 | ✅ |
| SOCIAL_MEDIA_ACCOUNT | 0.989 | 0.979 | 1.000 | ✅ |
| SOCIAL_MEDIA_HISTORY | 0.945 | 0.924 | 0.967 | ✅ |
| SOCIAL_MEDIA_ID | 1.000 | 1.000 | 1.000 | ✅ |
| SOCIO_ECONOMIC_STATUS | 0.984 | 0.979 | 0.990 | ✅ |
| STUDENT_ID | 0.992 | 0.988 | 0.997 | ✅ |
| AU_TFN | 1.000 | 1.000 | 1.000 | ✅ |
| UAC_ID | 0.988 | 1.000 | 0.977 | ✅ |
| USERNAME | 1.000 | 1.000 | 1.000 | ✅ |
| USI | 0.983 | 0.977 | 0.988 | ✅ |
| VEHICLE_REGO | 1.000 | 1.000 | 1.000 | ✅ |
| VOICE_RECOGNITION | 1.000 | 1.000 | 1.000 | ✅ |
| WEBSITE_HISTORY | 1.000 | 1.000 | 1.000 | ✅ |
| WORK_EMAIL | 0.996 | 1.000 | 0.992 | ✅ |
| WORKERS_COMPENSATION_CLAIM | 1.000 | 1.000 | 1.000 | ✅ |
| WORK_PHONE | 0.996 | 1.000 | 0.992 | ✅ |
| MEDICAL_INFORMATION | 0.963 | 0.963 | 0.962 | ✅ |
| COUNSELLING_RECORDS | 0.996 | 0.993 | 1.000 | ✅ |
| MEDICAL_CERTIFICATE | 0.996 | 0.993 | 1.000 | ✅ |
| SPECIAL_CONSIDERATION | 1.000 | 1.000 | 1.000 | ✅ |
| SCHOLARSHIP | 0.994 | 0.994 | 0.994 | ✅ |
| WAM_SCORE | 0.994 | 0.994 | 0.994 | ✅ |
| SUBJECT_RESULTS | 1.000 | 1.000 | 1.000 | ✅ |
| SANCTIONS | 0.991 | 0.987 | 0.994 | ✅ |
| PERSONAL_DEBT | 0.993 | 0.987 | 1.000 | ✅ |
| CONTRACT_TYPE | 0.931 | 0.890 | 0.976 | ⚠️ |
| COOKIE_INFORMATION | 0.984 | 0.978 | 0.989 | ✅ |
| CREDIT_CARD_EXPIRY | 0.858 | 0.758 | 0.988 | ⚠️ |
| DISABILITY_OR_SPECIFIC_CONDITION | 0.969 | 0.992 | 0.947 | ✅ |
| EMAIL_ADDRESS | 0.899 | 0.878 | 0.921 | ⚠️ |
| ADDRESS | 0.974 | 0.999 | 0.950 | ✅ |
| PERSON | 0.993 | 0.989 | 0.998 | ✅ |
| GEOLOCATION_INFORMATION | 0.866 | 0.812 | 0.929 | ⚠️ |
| IP_ADDRESS | 0.995 | 1.000 | 0.989 | ✅ |
| NATIONALITY | 0.920 | 0.974 | 0.872 | ⚠️ |
| NEXT_OF_KIN | 0.983 | 0.983 | 0.983 | ✅ |
| NUMBER_PLATE | 0.906 | 0.845 | 0.977 | ⚠️ |
| PAYMENT_CARD_NUMBER | 0.885 | 0.811 | 0.973 | ⚠️ |
| CAMERA_FOOTAGE_AUDIO | 0.991 | 1.000 | 0.983 | ✅ |
| CARING_RESPONSIBILITIES | 0.993 | 0.987 | 1.000 | ✅ |
| SALARY_WAGE_EXPECTATION | 0.694 | 0.535 | 0.986 | ❌ |
| SEXUAL_ORIENTATION | 0.959 | 0.931 | 0.990 | ✅ |
| LATITUDE | 0.515 | 0.607 | 0.447 | ❌ |
| LONGITUDE | 0.308 | 0.857 | 0.188 | ❌ |
| GENDER | 0.361 | 0.591 | 0.260 | ❌ |

**Legend**: ✅ F1≥0.85 | ⚠️ F1 0.70–0.85 | ❌ F1<0.70

---

## 2. Qwen Span Classification Head

**Model**: `runs/qwen_spancls_heads/last_linear/head.pt` | **Backbone**: Qwen3.5-9B-Base (frozen) | **Temperature**: 1.035854

| Metric | Test Set (13,576 spans) |
|--------|--------------------------|
| Top-1 Accuracy | 98.53% |
| NLL | 0.085 |
| ECE | 0.055 |
| Brier Score | 0.008 |
| NON_PII Accuracy | 57.14% |

| Low-Accuracy Labels (< 80%) | Accuracy |
|------------------------------|----------|
| *4 labels below threshold* | *varies* |

---

## 3. Hybrid Pipeline — Smoke Test Results

**10 hand-crafted examples** | **Default policy**: redact≥0.60, review≥0.25

| # | Example | Expected PII | Detected | Qwen Type | Decision | Risk Score |
|---|---------|-------------|----------|-----------|----------|------------|
| 1 | DOB: 04/05/1998 | DATE_OF_BIRTH | ✅ | DATE_OF_BIRTH | review | 0.48 |
| 2 | Deadline 15/06/2025 | DATE_OF_BIRTH | ❌ | — | — | — |
| 3 | BSB 062-000 acct | BANK_ACCOUNT | ❌ | — | — | — |
| 4 | alex@example.com | EMAIL_ADDRESS | ✅ | EMAIL_ADDRESS | review | 0.42 |
| 5 | 0412 345 678 | MOBILE | ✅ | MOBILE | review | 0.26 |
| 5 | 02 9876 5432 | HOME_PHONE | ✅ | MOBILE | review | 0.28 |
| 6 | 42 Wallaby Way, Sydney | ADDRESS | ✅ | ADDRESS | review | 0.44 |
| 7 | Order ORD-987654 | *(negative)* | ✅ | — | — | — |
| 8 | identifies as male | GENDER | ❌ | — | — | — |
| 8 | he/him pronouns | PRONOUN | ✅ | PRONOUN | ignore | 0.00 |
| 9 | salary $85,000 | SALARY | ✅ | SALARY | review | 0.46 |
| 10 | Medicare 2123 45678 | MEDICARE | ❌ | — | — | — |

**Summary**: 7/10 detected · 6 review · 1 ignore · 0 redact · 3 missed (BSB, ambiguous date, Medicare)

---

## 4. Smoke Miss Investigation

| Case | Label | In OPF Space? | Training Data | Root Cause |
|------|-------|---------------|---------------|------------|
| ambiguous_date | DATE_OF_BIRTH | ✅ | Sufficient | OPF model failure — ambiguous dates without DOB context not recognized |
| bsb_account | BANK_ACCOUNT_NUMBER | ✅ | Sufficient | OPF model failure — BSB/account number pattern not learned |
| medicare | MEDICARE_NUMBER | ✅ | Sufficient | OPF model failure — Medicare number pattern not learned |
| gender_male | GENDER | ✅ | Sufficient | OPF model failure — low recall on GENDER in general |

---

## 5. Known Limitations

| # | Issue | Severity | Mitigation |
|---|-------|----------|------------|
| 1 | Default policy thresholds produce 0 redactions | Medium | Lower redact threshold to 0.40–0.50 |
| 2 | OPF misses BSB, Medicare, ambiguous dates | Medium | Augment training data for these patterns |
| 3 | GENDER F1=0.36 (near-zero recall) | High | Add more GENDER training examples |
| 4 | LONGITUDE F1=0.31, LATITUDE F1=0.52 | Medium | Add coordinate training examples |
| 5 | Hybrid eval incomplete (subprocess too slow) | Medium | Implement batched OPF inference |
| 6 | Qwen NON_PII accuracy 57% | Medium | Consider LoRA fine-tuning for better NON_PII discrimination |
| 7 | SALARY_WAGE_EXPECTATION low precision (54%) | Low | Investigate confusion with SALARY |
