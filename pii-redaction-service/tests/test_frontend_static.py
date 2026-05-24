from pathlib import Path


HTML = Path(__file__).resolve().parents[1] / "static" / "redaction_demo.html"


def main() -> None:
    html = HTML.read_text(encoding="utf-8")

    assert "[hidden]" in html and "display: none !important" in html, (
        "hidden output sections must stay hidden even when component classes set display"
    )
    assert "id=\"ocr-band\" hidden" in html, "file extraction panel should be hidden by default"
    assert 'decision === "PASS"' not in html, "PASS should not be a user-visible frontend state"
    assert ".pill.pass" not in html, "PASS should not be styled as a user-visible decision"
    assert "<th" in html and ">Value</th>" in html, "detected table should show source value first"
    assert ">Pos</th>" not in html, "detected table should not show offsets in the user-facing layout"
    assert ">Confidentiality</th>" in html, "detected table should show data sensitivity classification"
    assert ">OPF</th>" not in html, "OPF internals should not be shown as a detected-table column"
    assert ">Qwen</th>" not in html, "Qwen internals should not be shown as a detected-table column"
    assert "renderSpans(spans, sourceText)" in html, "span rows should derive source values from input text"
    assert "hasModelScores(span)" in html, "frontend should distinguish model-scored spans from rule-only spans"
    assert "isPolicyOnlySpan(span)" in html, "only policy-only spans should get deterministic display values"
    assert "renderTopkChips(span)" in html, "top-k rendering should handle policy-only spans"
    assert 'span.source === "rule" && span.decision_reason === "type_actions" && !hasModelScores(span)' in html, (
        "100% fallback must be limited to policy-only rule spans without model scores"
    )
    assert "riskLevelBadge(riskScore)" in html, "risk should render as a low/medium/high badge"
    assert "riskBar(" not in html, "risk should not render the old numeric progress bar"
    assert "Math.round(riskScore * 100)" not in html, "risk should not show a numeric percentage"
    assert "<colgroup>" in html, "detected table should use fixed columns for alignment"
    for cls in ("w-value", "w-type", "w-confidentiality", "w-prob", "w-risk", "w-decision", "w-topk"):
        assert f'class="{cls}"' in html, f"missing detected table column sizing class {cls}"
    assert "table-layout: fixed" in html, "detected table needs fixed layout to avoid scattered columns"
    assert "class=\"num-head\"" in html, "numeric headers should align with numeric cells"
    assert "class=\"decision-head\"" in html, "decision header should align with decision pills"
    assert "class=\"col-decision\"" in html, "decision cells should be separately alignable"


def test_static_frontend_contract() -> None:
    main()


if __name__ == "__main__":
    main()
