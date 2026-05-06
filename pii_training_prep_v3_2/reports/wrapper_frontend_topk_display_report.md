# Frontend Top-k Distribution Display — Update Report

**Date**: 2026-05-01
**Status**: Complete

---

## Changes Made

### File Modified
`static/redaction_demo.html` (923 → 986 lines)

### Table Column Changes

| Before | After | Purpose |
|--------|-------|---------|
| Type (28%) | Type (11%) | Primary label |
| Pos (14%) | Pos (7%) | Character offsets |
| Confidence (18%) | OPF (9%) | OPF's original label |
| Decision (14%) | Qwen (9%) | Qwen's classified label |
| Replacement (26%) | Prob (7%) | Top probability |
| — | Risk (8%) | Composite risk score |
| — | Decision (9%) | redact/review/ignore/analysis |
| — | Top-k (40%) | Top-5 labels as chips |

### New JavaScript Functions

**`renderTopkChips(topk)`**: Renders `type_distribution_topk` as compact chips
- First chip highlighted (green accent)
- Each chip: `LABEL XX.X%`
- Returns "—" if topk empty/missing

**`riskBar(score)`**: Renders risk_score as a colored progress bar
- Green bar (default, risk < 0.25)
- Orange bar (medium, risk 0.25-0.60)
- Red bar (high, risk ≥ 0.60)
- Returns "—" if score missing

### Decision Pill Updates

| Old Value | New Value | Color |
|-----------|-----------|-------|
| AUTO_REDACT | redact | Red |
| REVIEW | review | Amber |
| — | ignore | Gray |
| — | analysis | Green |

### Backward Compatibility

All new fields gracefully degrade to "—" if missing:
- `span.opf_top_type` → "—" if absent
- `span.top_type` → "—" if absent
- `span.top_probability` → "—" if absent
- `span.risk_score` → "—" if absent
- `span.type_distribution_topk` → "—" if absent/empty

### CSS Additions

- `.topk-chips` / `.topk-chip`: Flex-wrap chip layout
- `.risk-bar` / `.risk-bar i`: Progress bar with color variants
- `.pill.type-pill`: Blue pill for OPF/Qwen type labels
- `.pill.analysis`: Green pill
- `.pill.ignore`: Gray pill
- `.col-topk`: Padding for top-k column cells

### Span Count Display

Now shows all decision types:
- `N redact` (redact count)
- `N review` (review count)
- `N analysis` (fallback analysis count)

---

## Example Display

For input `sid: 5102 88411`:

| Type | Pos | OPF | Qwen | Prob | Risk | Decision | Top-k Distribution |
|------|-----|-----|------|------|------|----------|-------------------|
| STUDENT_ID | 5-15 | — | STUDENT_ID | 30.9% | ████ 0.310 | analysis | STUDENT_ID 30.9% · MOBILE 5.2% · PAYMENT_CARD_NUMBER 2.8% · AU_TFN 2.6% · DRIVERS_LICENCE 2.3% |

For input `DOB 04/05/1998`:

| Type | Pos | OPF | Qwen | Prob | Risk | Decision | Top-k Distribution |
|------|-----|-----|------|------|------|----------|-------------------|
| DATE_OF_BIRTH | 4-14 | DATE_OF_BIRTH | DATE_OF_BIRTH | 94.2% | █████████ 0.473 | redact | DATE_OF_BIRTH 94.2% · PASSPORT_EXPIRY 0.1% · MEDICARE_EXPIRY 0.1% |

---

## Acceptance Criteria

- [x] UI displays top-k distribution for "sid: 5102 88411"
- [x] UI displays OPF Type and Qwen Type if present
- [x] UI displays risk_score with color bar if present
- [x] Existing redaction output still works (backward-compatible)
- [x] No training started
- [x] No checkpoint modified
