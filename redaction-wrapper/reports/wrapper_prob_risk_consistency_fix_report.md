# Prob / Risk Consistency Fix Report

**Date**: 2026-05-02  
**Scope**: `/home/admin/ZYX/redaction-wrapper`  
**Branch**: working tree (uncommitted)

## Summary

Fixed a policy/UI inconsistency where the displayed probability (`Prob` column) did not match the true Qwen classifier output from `type_distribution_topk[0][1]`, and ignored spans incorrectly showed high `Risk` scores.

## Changes

### 1. Probability Source Fix (`redaction/backends/hybrid_opf_qwen.py`)

**Before**: `top_prob` was set to `sr.get("top_probability", 0.5)` — the Qwen classifier's raw top probability, which could differ from the sorted distribution's first probability.

**After**: `top_prob = top1_prob` where `top1_prob = sorted_dist[0][1]` (the actual top-1 probability from the sorted type distribution). This ensures:
- `display_prob` (frontend Prob column) = `type_distribution_topk[0][1]`
- API field `top1_prob` = `type_distribution_topk[0][1]`
- API field `top_probability` = `top1_prob`

### 2. Top Type Consistency (`redaction/backends/hybrid_opf_qwen.py`)

**Before**: `top_type = sr.get("top_type", "NON_PII")` — Qwen's predicted top type, which could differ from the top-k distribution's first item.

**After**: `top_type = topk[0][0] if topk else qwen_top_type` — always prefers the distribution's first item when top-k exists.

### 3. Evidence Fields (`redaction/core/span.py`)

Added two new fields to the `Span` dataclass:
- `pii_evidence_passed: bool = False`
- `evidence_reason: str = ""`

Both fields are included in `to_schema()` API output.

### 4. Evidence Gate & Decision Order (`redaction/backends/hybrid_opf_qwen.py`)

The decision order was already correct. Added post-decision evidence gate zeroing:

```
if non_pii_prob >= 0.50:
    decision = "ignore"
    decision_reason = "non_pii_high"
    risk_score = 0.0
    evidence_reason = "non_pii_high"

elif top1_prob < 0.20 and top3_sum < 0.40:
    decision = "ignore"
    decision_reason = "low_pii_evidence"
    risk_score = 0.0
    evidence_reason = "low_top1_and_top3"

elif negative_context_detected and top1_prob < 0.70:
    decision = "ignore"
    decision_reason = "negative_context"
    risk_score = 0.0

else:
    risk_score = (1 - top3_sum) * data_classification_weight
    if risk_score >= review_threshold (0.25):
        decision = "review"
        decision_reason = "high_top3_uncertainty_risk"
    else:
        decision = "redact"
        decision_reason = "low_top3_uncertainty_risk"
```

For ALL ignore decisions:
- `risk_score` is zeroed (was previously always `uncertainty * dc_weight`, showing as 85% in UI)
- `pii_evidence_passed = False`
- `evidence_reason` = the decision reason

### 5. Frontend Fixes (`static/redaction_demo.html`)

**Risk Bar**: For ignore decisions, shows "—" instead of a percentage bar. Previously showed grey percentage bars even for ignored spans.

**Probability Column**: Uses `span.top1_prob` first, falling back to `span.top_probability` for backward compatibility.

## Regression Test Cases

### Case 1: Weak Bank Account (Low Evidence)
**Input**: `"Southern Mutual??"`  
**Expected**:
- Top-k first item: `BANK_ACCOUNT_NUMBER ~7.4%`
- `Prob` = `7.4%` (NOT 82.1%)
- `Decision` = `IGNORE`
- `Risk` = `—` (was 85%)
- `decision_reason` = `low_pii_evidence`

### Case 2: Strong Bank Account (High Evidence)  
**Input**: `"BSB 062-001, account 123456789"`  
**Expected**:
- High top1_prob and top3_sum
- `Decision` = `REDACT` or `REVIEW`
- `Risk` = non-zero, meaningful
- NOT ignored

### Case 3: Non-PII High
**Input**: Any text where NON_PII probability >= 50%  
**Expected**:
- `Decision` = `IGNORE`
- `Risk` = `—`
- `decision_reason` = `non_pii_high`

## Acceptance Criteria

- [x] PROB always matches top-k first probability
- [x] Low-evidence IGNORE rows do not show high Risk
- [x] Strong bank account rows still redact/review normally
- [x] No training started
- [x] No checkpoint modified
- [x] No OPF or Qwen inference weights changed

## Files Modified

| File | Changes |
|---|---|
| `redaction/core/span.py` | Added `pii_evidence_passed`, `evidence_reason` fields |
| `redaction/backends/hybrid_opf_qwen.py` | Fixed probability source, top_type, risk zeroing, evidence gate |
| `static/redaction_demo.html` | Fixed risk bar display for ignored spans, Prob column source |
