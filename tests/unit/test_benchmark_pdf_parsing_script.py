"""Tests for PDF parsing benchmark script helpers."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_pdf_parsing import (
    aggregate,
    bad_cases,
    discover_pdfs,
    write_records_csv,
    write_summary,
)


def test_discover_pdfs_from_directory(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"pdf")
    (tmp_path / "b.PDF").write_bytes(b"pdf")
    (tmp_path / "note.txt").write_text("nope", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.pdf").write_bytes(b"pdf")

    files = discover_pdfs(tmp_path)

    assert [file.name for file in files] == ["a.pdf", "b.PDF", "c.pdf"]


def test_discover_pdfs_respects_limit(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"{index}.pdf").write_bytes(b"pdf")

    files = discover_pdfs(tmp_path, limit=2)

    assert len(files) == 2


def test_aggregate_counts_fallbacks() -> None:
    records = [
        {
            "success": True,
            "should_use_external_pdf_api": True,
            "fallback_used": False,
            "parse_quality_score": 0.4,
            "elapsed_ms": 10,
        },
        {
            "success": True,
            "should_use_external_pdf_api": False,
            "fallback_used": True,
            "parse_quality_score": 0.9,
            "elapsed_ms": 20,
        },
        {"success": False, "elapsed_ms": 5},
    ]

    result = aggregate(records, execute_fallback=True)

    assert result["total_files"] == 3
    assert result["successful"] == 2
    assert result["failed"] == 1
    assert result["fallback_needed"] == 1
    assert result["fallback_used"] == 1
    assert result["average_quality_score"] == 0.65


def test_write_summary(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "summary.json"

    write_summary(output, {"ok": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}


def test_bad_cases_extracts_failures_and_low_quality_records() -> None:
    records = [
        {
            "source_path": "ok.pdf",
            "success": True,
            "parse_quality_reasons": [],
            "should_use_external_pdf_api": False,
        },
        {
            "source_path": "scan.pdf",
            "success": True,
            "parse_quality_score": 0.2,
            "parse_quality_reasons": ["scanned_pdf_likely"],
            "should_use_external_pdf_api": True,
            "fallback_used": False,
            "parser_provider": "markitdown",
            "report_path": "scan_report.json",
        },
        {
            "source_path": "bad.pdf",
            "success": False,
            "error": "parse failed",
        },
    ]

    cases = bad_cases(records)

    assert len(cases) == 2
    assert cases[0]["source_path"] == "scan.pdf"
    assert cases[0]["suggested_action"] == "run_mineru_or_ocr"
    assert cases[1]["issue_type"] == "parse_failed"


def test_write_records_csv(tmp_path: Path) -> None:
    output = tmp_path / "summary.csv"

    write_records_csv(
        output,
        [
            {
                "source_path": "scan.pdf",
                "success": True,
                "parser_provider": "markitdown",
                "parse_quality_reasons": ["empty_text", "low_text_density"],
                "parse_quality_score": 0.1,
            }
        ],
    )

    content = output.read_text(encoding="utf-8-sig")

    assert "source_path,success" in content
    assert "empty_text|low_text_density" in content
