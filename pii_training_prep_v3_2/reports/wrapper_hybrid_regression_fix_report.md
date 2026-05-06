# Hybrid Backend Regression Fix — Report

**Date**: 2026-05-01
**Severity**: Critical — all API responses returned empty spans

---

## Root Cause

Two functions in `redaction/core/policy.py` used hardcoded old decision values (`AUTO_REDACT`, `REVIEW`) that didn't match the new decision values (`redact`, `review`, `ignore`, `analysis`) introduced by the policy consistency fix.

| Function | Line | Old Behavior | Impact |
|----------|------|-------------|--------|
| `build_response()` | 161 | `filter spans to {AUTO_REDACT, REVIEW} only` | **All spans dropped** — 8 detected → 0 returned |
| `redact_text()` | 136-137 | `check dec == "AUTO_REDACT" or "REVIEW"` | **Zero redactions applied** — redacted_text identical to input |

The backend correctly detected, classified, and scored 8 spans. The policy layer correctly computed risk-based decisions. But `build_response` filtered them all out because `redact/review/ignore/analysis` ≠ `AUTO_REDACT/REVIEW`.

## Fix

### `policy.py` — Added helper functions + updated call sites

```python
def _is_redact_decision(decision):
    return decision in ("AUTO_REDACT", "redact")

def _is_review_decision(decision):
    return decision in ("REVIEW", "review")

def _is_visible_decision(decision):
    return decision in ("AUTO_REDACT", "REVIEW", "redact", "review", "ignore", "analysis", "PASS")
```

- `redact_text`: Uses `_is_redact_decision()` and `_is_review_decision()` — now correctly redacts `redact` spans
- `build_response`: Uses `_is_visible_decision()` — now shows all decision types in API response

### Before/After API Response

**Input**: Multi-PII text with PERSON, DRIVERS_LICENCE, PASSPORT, CENTRELINK, IHI, DOB, ADDRESS

**Before (broken)**:
```json
{"spans": [], "redacted_text": "<unchanged>", "spans": []}
```

**After (fixed)**:
```json
{
  "spans": [
    {"type": "PERSON",       "decision": "review",  "risk_score": 0.440},
    {"type": "DRIVERS_LICENCE","decision": "ignore",  "risk_score": 0.175},
    {"type": "PASSPORT_NUMBER","decision": "redact",  "risk_score": 0.877},
    {"type": "CENTRELINK_REFERENCE_NUMBER","decision": "redact", "risk_score": 0.898},
    {"type": "IHI",          "decision": "review",  "risk_score": 0.462},
    {"type": "CENTRELINK_REFERENCE_NUMBER","decision": "redact", "risk_score": 0.930},
    {"type": "DATE_OF_BIRTH", "decision": "review",  "risk_score": 0.478},
    {"type": "ADDRESS",      "decision": "review",  "risk_score": 0.471}
  ],
  "redacted_text": "...passport: [PASSPORT_NUMBER]\nnational ID card: [CENTRELINK_REFERENCE_NUMBER]..."
}
```

## Files Changed

| File | Change |
|------|--------|
| `redaction/core/policy.py` | Added 3 helper functions; fixed `redact_text` and `build_response` |

## Verification

### Regression Smoke Test Results

| Input | OPF Spans | Backend Spans | After Policy | API Spans | Redacted? |
|-------|-----------|---------------|-------------|-----------|-----------|
| Multi-PII ID text | 8 | 8 | 8 | 8 | ✅ 3 redacted |
| `student ID: 5102 88411` | 0 | 1 (fallback) | 1 | 1 | ✅ (analysis) |
| `5102 88411` | 0 | 1 (fallback) | 1 | 1 | ✅ (analysis) |
| `DOB 04/05/1998` | 1 | 1 | 1 | 1 | ✅ (review) |

### Decision Consistency

| Span | Risk | Thresholds | Expected | Actual | OK? |
|------|------|-----------|----------|--------|-----|
| PASSPORT_NUMBER | 0.877 | redact=0.70 | redact | redact | ✅ |
| CENTRELINK | 0.898 | redact=0.70 | redact | redact | ✅ |
| PERSON | 0.440 | review=0.25 | review | review | ✅ |
| IHI | 0.462 | review=0.25 | review | review | ✅ |
| DATE_OF_BIRTH | 0.478 | review=0.25 | review | review | ✅ |
| ADDRESS | 0.471 | review=0.25 | review | review | ✅ |
| DRIVERS_LICENCE | 0.175 | redact=0.70, review=0.25 | ignore | ignore | ✅ |

## Acceptance Criteria

- [x] Multi-PII input returns spans > 0 (8 spans)
- [x] `student ID: 5102 88411` still returns probability top-k
- [x] `5102 88411` still returns fallback analysis
- [x] Policy decisions consistent with risk thresholds
- [x] Redacted text correctly replaces redact spans
- [x] No training started
- [x] No checkpoints modified
