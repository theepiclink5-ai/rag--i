"""Tests for configured PDF parser factory."""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from src.core.settings import load_settings
from src.libs.loader.pdf_parsers import (
    CompositePdfParser,
    MarkItDownPdfParser,
    MinerUParseArtifact,
    MinerUParseOptions,
    MinerUPdfParser,
    PdfParserFactory,
)


class FakeMinerUClient:
    def parse_pdf(
        self, file_path: str | Path, options: MinerUParseOptions
    ) -> MinerUParseArtifact:
        return MinerUParseArtifact(result_dir=Path(file_path).parent)


def _write_settings(path: Path, pdf_parser_block: str) -> None:
    base_config = """
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
      max_tokens: 1024
    embedding:
      provider: openai
      model: text-embedding-3-small
      dimensions: 1536
    vector_store:
      provider: chroma
      persist_directory: ./data/db/chroma
      collection_name: knowledge_hub
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
    rerank:
      enabled: false
      provider: none
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 5
    evaluation:
      enabled: false
      provider: custom
      metrics:
        - hit_rate
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    """
    config = textwrap.dedent(base_config).strip()
    if pdf_parser_block.strip():
        config = f"{config}\n{textwrap.dedent(pdf_parser_block).strip()}"
    path.write_text(config + "\n", encoding="utf-8")


def test_factory_without_pdf_parser_config_returns_markitdown(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    _write_settings(settings_path, "")
    settings = load_settings(settings_path)

    parser = PdfParserFactory.create(settings)

    assert isinstance(parser, MarkItDownPdfParser)


def test_factory_with_disabled_fallback_returns_primary(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    _write_settings(
        settings_path,
        """
        pdf_parser:
          primary: markitdown
          fallback_enabled: false
          fallback_provider: mineru
          fallback_threshold: 0.65
        """,
    )
    settings = load_settings(settings_path)

    parser = PdfParserFactory.create(settings)

    assert isinstance(parser, MarkItDownPdfParser)


def test_factory_creates_composite_with_mineru_fallback(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    _write_settings(
        settings_path,
        """
        pdf_parser:
          primary: markitdown
          fallback_enabled: true
          fallback_provider: mineru
          fallback_threshold: 0.5
          mineru:
            api_key: test-key
            model_version: vlm
            language: ch
            is_ocr: true
            enable_table: true
            enable_formula: true
            timeout_seconds: 60
            poll_interval_seconds: 1
            save_raw_result: true
        """,
    )
    settings = load_settings(settings_path)

    parser = PdfParserFactory.create(
        settings,
        mineru_client=FakeMinerUClient(),
        output_dir=tmp_path / "mineru",
    )

    assert isinstance(parser, CompositePdfParser)
    assert isinstance(parser.primary_parser, MarkItDownPdfParser)
    assert isinstance(parser.fallback_parser, MinerUPdfParser)
    assert parser.quality_analyzer.fallback_threshold == 0.5


def test_factory_rejects_unknown_primary_provider(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    _write_settings(
        settings_path,
        """
        pdf_parser:
          primary: unknown
          fallback_enabled: false
          fallback_provider: mineru
          fallback_threshold: 0.65
        """,
    )
    settings = load_settings(settings_path)

    with pytest.raises(ValueError, match="Unsupported primary"):
        PdfParserFactory.create(settings)
