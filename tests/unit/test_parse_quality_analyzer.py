"""Tests for PDF parse quality heuristics."""

from src.libs.loader.pdf_parsers import PdfParseQualityAnalyzer


def test_good_text_scores_high() -> None:
    analyzer = PdfParseQualityAnalyzer()
    text = "\n\n".join(
        f"This is a well formed paragraph with enough content for page {i}."
        for i in range(40)
    )

    report = analyzer.analyze(text=text, page_count=2)

    assert report.score >= 0.65
    assert report.should_use_external_api is False
    assert report.reasons == []


def test_empty_text_triggers_external_api() -> None:
    analyzer = PdfParseQualityAnalyzer()

    report = analyzer.analyze(text="", page_count=5)

    assert report.should_use_external_api is True
    assert "empty_text" in report.reasons
    assert "low_text_density" in report.reasons


def test_scanned_pdf_likely_triggers_external_api() -> None:
    analyzer = PdfParseQualityAnalyzer()

    report = analyzer.analyze(text="short text", page_count=10, image_count=10)

    assert report.should_use_external_api is True
    assert "scanned_pdf_likely" in report.reasons


def test_mojibake_triggers_external_api() -> None:
    analyzer = PdfParseQualityAnalyzer()
    text = "鍙彃鎷旀灦鏋 " * 100

    report = analyzer.analyze(text=text, page_count=1)

    assert report.should_use_external_api is True
    assert "mojibake_detected" in report.reasons


def test_table_structure_lost_reduces_score() -> None:
    analyzer = PdfParseQualityAnalyzer()
    text = "Product Price Quantity\nA 10 3\nB 20 4\n" * 30

    report = analyzer.analyze(text=text, page_count=2, table_count=1)

    assert "table_structure_lost" in report.reasons
    assert report.score < 1.0


def test_markdown_table_not_flagged_as_lost() -> None:
    analyzer = PdfParseQualityAnalyzer()
    text = "\n".join(
        [
            "| Product | Price |",
            "| --- | --- |",
            "| A | 10 |",
            "This table is followed by enough ordinary explanatory content." * 20,
        ]
    )

    report = analyzer.analyze(text=text, page_count=1, table_count=1)

    assert "table_structure_lost" not in report.reasons
