"""Composite PDF parser that routes low-quality parses to a fallback backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.libs.loader.pdf_parsers.base_pdf_parser import BasePdfParser, PdfParseResult
from src.libs.loader.pdf_parsers.parse_quality_analyzer import (
    PdfParseQualityAnalyzer,
    PdfParseQualityReport,
)

logger = logging.getLogger(__name__)


class CompositePdfParser:
    """Run a primary parser first, then use fallback when quality is poor."""

    provider_name = "composite"

    def __init__(
        self,
        primary_parser: BasePdfParser,
        fallback_parser: Optional[BasePdfParser] = None,
        quality_analyzer: Optional[PdfParseQualityAnalyzer] = None,
        fallback_enabled: bool = True,
    ) -> None:
        self.primary_parser = primary_parser
        self.fallback_parser = fallback_parser
        self.quality_analyzer = quality_analyzer or PdfParseQualityAnalyzer()
        self.fallback_enabled = fallback_enabled

    def parse(self, file_path: str | Path) -> PdfParseResult:
        path = Path(file_path)
        try:
            primary_result = self.primary_parser.parse(path)
        except Exception as exc:
            if not self._can_fallback:
                raise
            logger.warning("Primary PDF parser failed for %s: %s", path, exc)
            primary_report = self.quality_analyzer.analyze(
                text="", file_path=path, parse_error=str(exc)
            )
            return self._parse_fallback(
                path=path,
                primary_report=primary_report,
                primary_error=str(exc),
            )

        primary_report = self._analyze(primary_result, path)
        if not primary_report.should_use_external_api or not self._can_fallback:
            return self._with_route_metadata(
                result=primary_result,
                selected_report=primary_report,
                fallback_used=False,
            )

        try:
            return self._parse_fallback(path=path, primary_report=primary_report)
        except Exception as exc:
            logger.warning("Fallback PDF parser failed for %s: %s", path, exc)
            return self._with_route_metadata(
                result=primary_result,
                selected_report=primary_report,
                fallback_used=False,
                fallback_error=str(exc),
            )

    @property
    def _can_fallback(self) -> bool:
        return self.fallback_enabled and self.fallback_parser is not None

    def _parse_fallback(
        self,
        path: Path,
        primary_report: PdfParseQualityReport,
        primary_error: Optional[str] = None,
    ) -> PdfParseResult:
        if self.fallback_parser is None:
            raise RuntimeError("Fallback parser is not configured")

        fallback_result = self.fallback_parser.parse(path)
        fallback_report = self._analyze(fallback_result, path)
        return self._with_route_metadata(
            result=fallback_result,
            selected_report=fallback_report,
            fallback_used=True,
            primary_report=primary_report,
            primary_error=primary_error,
        )

    def _analyze(self, result: PdfParseResult, path: Path) -> PdfParseQualityReport:
        return self.quality_analyzer.analyze(
            text=result.text,
            file_path=path,
            page_count=result.metadata.get("page_count"),
            image_count=len(result.images),
            table_count=len(result.tables),
        )

    def _with_route_metadata(
        self,
        result: PdfParseResult,
        selected_report: PdfParseQualityReport,
        fallback_used: bool,
        primary_report: Optional[PdfParseQualityReport] = None,
        primary_error: Optional[str] = None,
        fallback_error: Optional[str] = None,
    ) -> PdfParseResult:
        metadata: Dict[str, Any] = {
            **result.metadata,
            "parser_route": self.provider_name,
            "parse_quality_score": selected_report.score,
            "parse_quality_reasons": selected_report.reasons,
            "parse_quality_metrics": selected_report.metrics,
            "should_use_external_pdf_api": selected_report.should_use_external_api,
            "fallback_used": fallback_used,
            "fallback_provider": self._fallback_provider_name(),
            "primary_parser_provider": self.primary_parser.provider_name,
        }
        if primary_report is not None:
            metadata.update(
                {
                    "primary_parse_quality_score": primary_report.score,
                    "primary_parse_quality_reasons": primary_report.reasons,
                    "primary_should_use_external_pdf_api": (
                        primary_report.should_use_external_api
                    ),
                }
            )
        if primary_error:
            metadata["primary_parse_error"] = primary_error
        if fallback_error:
            metadata["fallback_error"] = fallback_error

        return PdfParseResult(
            text=result.text,
            parser_provider=result.parser_provider,
            metadata=metadata,
            pages=list(result.pages),
            tables=list(result.tables),
            images=list(result.images),
            quality_score=selected_report.score,
            warnings=list(result.warnings),
            raw_output_dir=result.raw_output_dir,
        )

    def _fallback_provider_name(self) -> Optional[str]:
        if self.fallback_parser is None:
            return None
        return self.fallback_parser.provider_name
