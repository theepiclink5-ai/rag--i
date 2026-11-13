"""Tests for composite PDF parser fallback routing."""

from pathlib import Path

import pytest

from src.libs.loader.pdf_parsers import CompositePdfParser, PdfParseResult


class StubParser:
    def __init__(self, provider_name: str, text: str, fail: bool = False) -> None:
        self.provider_name = provider_name
        self.text = text
        self.fail = fail
        self.calls = 0

    def parse(self, file_path: str | Path) -> PdfParseResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.provider_name} failed")
        return PdfParseResult(text=self.text, parser_provider=self.provider_name)


def test_high_quality_primary_does_not_use_fallback(tmp_path: Path) -> None:
    primary = StubParser(
        provider_name="primary",
        text="\n\n".join(
            f"This primary parser produced enough useful content paragraph {i}."
            for i in range(30)
        ),
    )
    fallback = StubParser(provider_name="fallback", text="fallback text")
    parser = CompositePdfParser(primary_parser=primary, fallback_parser=fallback)

    result = parser.parse(tmp_path / "doc.pdf")

    assert result.parser_provider == "primary"
    assert result.metadata["fallback_used"] is False
    assert fallback.calls == 0


def test_low_quality_primary_uses_fallback(tmp_path: Path) -> None:
    primary = StubParser(provider_name="primary", text="")
    fallback = StubParser(
        provider_name="fallback",
        text="\n\n".join(
            f"Fallback parser produced useful content paragraph {i}."
            for i in range(30)
        ),
    )
    parser = CompositePdfParser(primary_parser=primary, fallback_parser=fallback)

    result = parser.parse(tmp_path / "doc.pdf")

    assert result.parser_provider == "fallback"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["fallback_provider"] == "fallback"
    assert result.metadata["primary_parser_provider"] == "primary"
    assert "empty_text" in result.metadata["primary_parse_quality_reasons"]
    assert fallback.calls == 1


def test_primary_failure_uses_fallback(tmp_path: Path) -> None:
    primary = StubParser(provider_name="primary", text="", fail=True)
    fallback = StubParser(provider_name="fallback", text="Enough fallback text. " * 20)
    parser = CompositePdfParser(primary_parser=primary, fallback_parser=fallback)

    result = parser.parse(tmp_path / "doc.pdf")

    assert result.parser_provider == "fallback"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["primary_parse_error"] == "primary failed"


def test_primary_failure_without_fallback_raises(tmp_path: Path) -> None:
    primary = StubParser(provider_name="primary", text="", fail=True)
    parser = CompositePdfParser(primary_parser=primary, fallback_enabled=False)

    with pytest.raises(RuntimeError, match="primary failed"):
        parser.parse(tmp_path / "doc.pdf")


def test_fallback_failure_keeps_primary_result(tmp_path: Path) -> None:
    primary = StubParser(provider_name="primary", text="")
    fallback = StubParser(provider_name="fallback", text="", fail=True)
    parser = CompositePdfParser(primary_parser=primary, fallback_parser=fallback)

    result = parser.parse(tmp_path / "doc.pdf")

    assert result.parser_provider == "primary"
    assert result.metadata["fallback_used"] is False
    assert result.metadata["fallback_error"] == "fallback failed"
    assert result.metadata["should_use_external_pdf_api"] is True
