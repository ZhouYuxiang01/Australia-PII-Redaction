# Policy Consistency Fix — Smoke Report

**Thresholds**: redact=0.7, review=0.25

## Results

| Test | Spans | Decision | Reason | Risk | Consistency |
|------|-------|----------|--------|------|-------------|
| student_id | STUDENT_ID | **review** | risk_based | 0.394 | OK |
| bank_details | BANK_ACCOUNT_NUMBER | **redact** | risk_based | 0.945 | OK |
| bank_details | BANK_ACCOUNT_NUMBER | **redact** | risk_based | 0.993 | OK |
| order_neg | PAYMENT_CARD_NUMBER | **analysis** | fallback_full_input | 0.035 | OK |
| bare_5102 | DRIVERS_LICENCE | **analysis** | fallback_full_input | 0.535 | OK |
| dob | DATE_OF_BIRTH | **review** | risk_based | 0.473 | OK |

## Root Cause

`apply_policy()` in `redaction/core/policy.py` was overriding backend risk-based decisions with static `type_actions` from the policy JSON config.

## Fix

1. `policy.py`: `apply_policy()` now checks `risk_score` first. If present, derives decision from thresholds rather than `type_actions`.
2. `span.py`: Added `decision_reason`, `redact_threshold`, `review_threshold`, `policy_version` fields.
3. `hybrid_opf_qwen.py`: Disagreement safety rule — if OPF type != Qwen type and top_prob < 0.85, caps decision at "review" unless risk >= 0.85.
4. `hybrid-opf-qwen.json`: Default thresholds set to redact=0.70, review=0.25.
5. `hybrid-80class-v1.json`: Added `redact_threshold` and `review_threshold` fields for policy layer.

## Decision Precedence

1. `fallback_full_input` source → analysis
2. `risk_score >= redact_threshold` → redact
3. `risk_score >= review_threshold` → review
4. Otherwise → ignore
5. Disagreement safety: OPF≠Qwen and top_prob<0.85 → cap at review

## Acceptance Criteria

- [x] No span with risk < review_threshold gets redact or review
- [x] No span with risk < redact_threshold gets redact
- [x] fallback_full_input spans use analysis
- [x] decision_reason present on all spans
- [x] No training started, no checkpoints modified
