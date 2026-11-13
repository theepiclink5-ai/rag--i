"""Tests for PDF parse report persistence."""

from __future__ import annotations

import json
from pathlib import Path

from src.libs.loader.parse_report_writer import PdfParseReportWriter


def test_write_parse_report(tmp_path: Path) -> None:
    writer = PdfParseReportWriter(tmp_path)

    report_path = writer.write(
        source_path="docs/sample.pdf",
        doc_id="doc_123",
        text="parsed text",
        metadata={
            "doc_hash": "abc",
            "title": "Sample",
            "parser_provider": "mineru",
            "parser_route": "composite",
            "primary_parser_provider": "markitdown",
            "fallback_provider": "mineru",
            "fallback_used": True,
            "parse_quality_score": 0.9,
            "parse_quality_reasons": [],
            "parse_quality_metrics": {"char_count": 11},
            "should_use_external_pdf_api": False,
            "primary_parse_quality_score": 0.2,
            "primary_parse_quality_reasons": ["empty_text"],
            "primary_should_use_external_pdf_api": True,
            "mineru_batch_id": "batch-1",
            "mineru_file_name": "sample.pdf",
            "mineru_state": "done",
        },
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.name == "doc_123.parse_report.json"
    assert report["source_path"] == "docs/sample.pdf"
    assert report["doc_id"] == "doc_123"
    assert report["parser_provider"] == "mineru"
    assert report["fallback_used"] is True
    assert report["primary_parse_quality_reasons"] == ["empty_text"]
    assert report["selected_text_char_count"] == len("parsed text")
