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


if __name__ == "__main__":
    main()
