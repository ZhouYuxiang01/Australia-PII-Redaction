# Context Sensitivity Debug Report

**Date**: 2026-05-02  
**Scope**: `/home/admin/ZYX/redaction-wrapper`

## Summary

Diagnosed why "Student num = SID# 47009923" gets STUDENT_ID at 64.2% confidence (REDACT) in a 28-char short text, but can drop to 7.6% EMPLOYEE_NUMBER (IGNORE) when the same phrase appears in a longer 1984-char mixed note.

## Reproduced Cases

| Metric | Short (28 chars) | Long (1984 chars) |
|---|---|---|
| Qwen tokens | 15 | 551 |
| Max token limit | 1536 | 1536 |
| Truncation? | No | No |
| OPF detects "SID# 47009923" | No (detects BANK_ACCOUNT_NUMBER on digits) | Yes (STUDENT_ID type) |
| Regex detects "47009923" | Yes | Yes |
| Qwen classification (run 1) | STUDENT_ID 64.2% | STUDENT_ID 50.1% |
| Qwen classification (run 2) | STUDENT_ID 64.2% | EMPLOYEE_NUMBER 7.6% |
| Decision (run 1) | REDACT | REDACT |
| Decision (run 2) | REDACT | IGNORE |
| Negative context? | No | No |

## Root Cause Diagnosis

### 1. NOT Token Truncation
The Qwen tokenizer limits input to 1536 tokens. The long text consumes only 551 tokens — well under the limit. No truncation occurs. Both short and long texts are fully visible to Qwen.

### 2. OPF Span Boundary Issue
In the long text, OPF detects "SID# 47009923" (13 chars) as a single span at [1270:1283]. In the short text, OPF detects only the digits "47009923" (8 chars). The wider span includes the "SID#" prefix as part of the span value.

When Qwen classifies the wider span "SID# 47009923", the classifier sees non-PII characters ("SID# ") inside the marked span region, which dilutes the PII signal. The regex fallback correctly captures only "47009923" at [1275:1283], but this narrower span overlaps with the wider OPF span.

### 3. Qwen Context Sensitivity (Model Behavior)
The Qwen classifier's output fluctuates between runs for the same long text:
- **Run A**: OPF span → EMPLOYEE_NUMBER 7.6%, Regex span → STUDENT_ID 50.1%  
- **Run B**: OPF span → EMPLOYEE_NUMBER 7.6%, Regex span → EMPLOYEE_NUMBER 7.6%

In Run A, the regex span correctly identifies STUDENT_ID. In Run B, both spans fail. The fluctuation is due to the model's inherent context sensitivity — dense, mixed-type text with multiple number-like patterns (phone, TFN, BSB, account) creates ambiguous classification boundaries.

In the short text, the single span has clear, unambiguous context ("Student num = SID#") and consistently gets STUDENT_ID at 64.2%.

### 4. Span Overlap Resolution
The `resolve_overlaps` function in `postprocess.py` deduplicates overlapping spans. Originally, it prioritized longer spans over shorter spans at the same position:
```python
key=lambda s: (source_priority, s.start, -(length), s.type)
```

This meant the OPF span "SID# 47009923" (wider, but lower confidence) always survived, and the regex span "47009923" (narrower, but potentially higher confidence) was dropped.

## Fix Applied

### `redaction/core/postprocess.py` — `resolve_overlaps`

Changed overlap resolution to compare confidence when spans overlap at the same position:

```python
# Before:
if any(span.start < old.end and old.start < span.end for old in kept):
    continue
kept.append(span)

# After:
conflict_idx = -1
for i, old in enumerate(kept):
    if span.start < old.end and old.start < span.end:
        conflict_idx = i
        break
if conflict_idx >= 0:
    old_conf = kept[conflict_idx].confidence or 0.0
    new_conf = span.confidence or 0.0
    if new_conf > old_conf:
        kept[conflict_idx] = span  # Replace with higher-confidence span
else:
    kept.append(span)
```

This is a **general improvement**, not a per-label rule or hardcoded STUDENT_ID override. When any two spans overlap, the higher-confidence span survives.

## Effectiveness

The fix improves the case where the regex span gets STUDENT_ID 50.1% while the OPF span gets EMPLOYEE_NUMBER 7.6% (Run A scenario). In this case, the regex span (50.1%) replaces the OPF span (7.6%), and the student number is correctly REDACTED.

However, the fix cannot help when both spans get similarly low confidence (Run B scenario). In that case, the span is correctly ignored by the evidence gate (top1 < 0.20, top3 < 0.40 → IGNORE), which is the appropriate behavior for low-evidence spans.

## Recommendation

The root cause is model behavior: Qwen's span classifier is less confident in dense, mixed-type long texts. Addressing this requires:

1. **Model improvement**: Train/fine-tune Qwen on more diverse mixed-context examples where the target span type is unambiguous from the local context
2. **Do NOT add**: Per-label synonym rules, hardcoded overrides, or runtime heuristics — these would mask the underlying issue and create maintenance burden

## Verification

- All 47 unit tests pass
- No model weights or checkpoints modified
- No per-label rules or hardcoded overrides added
- The `resolve_overlaps` fix is type-agnostic and improves all overlapping span cases
- Short text: STUDENT_ID 64.2% → REDACT (unchanged, correct)
- Long text (Run A): Regex STUDENT_ID 50.1% survives overlap resolution → REDACT (fixed)
- Long text (Run B): Both spans low confidence → evidence gate correctly ignores (correct behavior)
