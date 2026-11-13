"""Factory for constructing configured PDF parser pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.core.settings import Settings
from src.libs.loader.pdf_parsers.base_pdf_parser import BasePdfParser
from src.libs.loader.pdf_parsers.composite_pdf_parser import CompositePdfParser
from src.libs.loader.pdf_parsers.markitdown_pdf_parser import MarkItDownPdfParser
from src.libs.loader.pdf_parsers.mineru_http_client import MinerUHttpClient
from src.libs.loader.pdf_parsers.mineru_pdf_parser import (
    MinerUClientProtocol,
    MinerUParseError,
    MinerUParseOptions,
    MinerUPdfParser,
)
from src.libs.loader.pdf_parsers.parse_quality_analyzer import PdfParseQualityAnalyzer


class PdfParserFactory:
    """Build parser instances from application settings."""

    @classmethod
    def create(
        cls,
        settings: Optional[Settings] = None,
        mineru_client: Optional[MinerUClientProtocol] = None,
        output_dir: str | Path = "data/parsed/mineru",
    ) -> BasePdfParser:
        """Create a PDF parser according to settings.

        Defaults to MarkItDown when no pdf_parser config is present.  When
        fallback is enabled, returns a CompositePdfParser with the configured
        fallback backend.
        """
        parser_settings = getattr(settings, "pdf_parser", None) if settings else None
        primary_name = (
            parser_settings.primary.lower() if parser_settings else "markitdown"
        )
        primary_parser = cls._create_primary(primary_name)

        if not parser_settings or not parser_settings.fallback_enabled:
            return primary_parser

        quality_analyzer = PdfParseQualityAnalyzer(
            fallback_threshold=parser_settings.fallback_threshold
        )
        fallback_parser = cls._create_fallback(
            settings=settings,
            provider=parser_settings.fallback_provider.lower(),
            mineru_client=mineru_client,
            output_dir=output_dir,
        )
        return CompositePdfParser(
            primary_parser=primary_parser,
            fallback_parser=fallback_parser,
            quality_analyzer=quality_analyzer,
            fallback_enabled=True,
        )

    @classmethod
    def _create_primary(cls, provider: str) -> BasePdfParser:
        if provider == "markitdown":
            return MarkItDownPdfParser()
        raise ValueError(f"Unsupported primary PDF parser provider: {provider}")

    @classmethod
    def _create_fallback(
        cls,
        settings: Settings,
        provider: str,
        mineru_client: Optional[MinerUClientProtocol],
        output_dir: str | Path,
    ) -> BasePdfParser:
        if provider == "mineru":
            return cls._create_mineru(settings, mineru_client, output_dir)
        raise ValueError(f"Unsupported fallback PDF parser provider: {provider}")

    @classmethod
    def _create_mineru(
        cls,
        settings: Settings,
        mineru_client: Optional[MinerUClientProtocol],
        output_dir: str | Path,
    ) -> MinerUPdfParser:
        parser_settings = getattr(settings, "pdf_parser", None)
        mineru_settings = getattr(parser_settings, "mineru", None)
        if mineru_settings is None:
            raise MinerUParseError("Missing pdf_parser.mineru configuration")

        options = MinerUParseOptions(
            model_version=mineru_settings.model_version,
            language=mineru_settings.language,
            is_ocr=mineru_settings.is_ocr,
            enable_table=mineru_settings.enable_table,
            enable_formula=mineru_settings.enable_formula,
            timeout_seconds=mineru_settings.timeout_seconds,
            poll_interval_seconds=mineru_settings.poll_interval_seconds,
        )
        client = mineru_client or MinerUHttpClient(api_key=mineru_settings.api_key or "")
        return MinerUPdfParser(
            client=client,
            options=options,
            output_dir=output_dir,
            save_raw_result=mineru_settings.save_raw_result,
        )
