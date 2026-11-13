"""Validation tests for human-facing sample documents used in manual ingestion checks."""

from pathlib import Path


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_documents"


def test_chinese_chunk_validation_fixture_exists():
    fixture = FIXTURES_DIR / "chinese_chunk_validation.md"
    assert fixture.exists(), "Chunk validation fixture should exist for dashboard testing"


def test_chinese_chunk_validation_fixture_is_long_enough():
    fixture = FIXTURES_DIR / "chinese_chunk_validation.md"
    content = fixture.read_text(encoding="utf-8")

    assert len(content) >= 2500, "Fixture should be long enough to produce multiple chunks"
    assert content.count("## ") >= 5, "Fixture should contain multiple sections for boundary checks"
    assert "Embedding" in content
    assert "Chunk" in content
    assert "Vector Store" in content
