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
    assert "<th" in html and ">Value</th>" in html, "detected table should show source value after Pos"
    assert ">OPF</th>" not in html, "OPF internals should not be shown as a detected-table column"
    assert ">Qwen</th>" not in html, "Qwen internals should not be shown as a detected-table column"
    assert "renderSpans(spans, sourceText)" in html, "span rows should derive source values from input text"
    assert "hasModelScores(span)" in html, "frontend should distinguish model-scored spans from rule-only spans"
    assert "isPolicyOnlySpan(span)" in html, "only policy-only spans should get deterministic display values"
    assert "renderTopkChips(span)" in html, "top-k rendering should handle policy-only spans"
    assert 'span.source === "rule" && span.decision_reason === "type_actions" && !hasModelScores(span)' in html, (
        "100% fallback must be limited to policy-only rule spans without model scores"
    )
    assert "var riskScore = isPolicyOnlySpan(span) ? 0 : span.risk_score" in html, (
        "risk should only be forced to 0 for policy-only spans without model scores"
    )
    assert ".risk-bar.zero" in html and 'cls += " zero"' in html, (
        "0% risk should still render a progress track"
    )
    assert "<colgroup>" in html, "detected table should use fixed columns for alignment"
    for cls in ("w-type", "w-pos", "w-value", "w-prob", "w-risk", "w-decision", "w-topk"):
        assert f'class="{cls}"' in html, f"missing detected table column sizing class {cls}"
    assert "table-layout: fixed" in html, "detected table needs fixed layout to avoid scattered columns"
    assert "class=\"num-head\"" in html, "numeric headers should align with numeric cells"
    assert "class=\"decision-head\"" in html, "decision header should align with decision pills"
    assert "class=\"col-decision\"" in html, "decision cells should be separately alignable"


def test_static_frontend_contract() -> None:
    main()


if __name__ == "__main__":
    main()
