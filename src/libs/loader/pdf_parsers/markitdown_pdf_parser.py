"""MarkItDown-backed PDF parser."""

from __future__ import annotations

import logging
from pathlib import Path

from src.libs.loader.pdf_parsers.base_pdf_parser import PdfParseResult

try:
    from markitdown import MarkItDown

    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False


logger = logging.getLogger(__name__)


class MarkItDownPdfParser:
    """Parse PDFs into Markdown text using MarkItDown."""

    provider_name = "markitdown"

    def __init__(self, markitdown: MarkItDown | None = None) -> None:
        if not MARKITDOWN_AVAILABLE:
            raise ImportError(
                "MarkItDown is required for PdfLoader. "
                "Install with: pip install markitdown"
            )
        self._markitdown = markitdown or MarkItDown()

    def parse(self, file_path: str | Path) -> PdfParseResult:
        path = Path(file_path)
        try:
            result = self._markitdown.convert(str(path))
            text = result.text_content if hasattr(result, "text_content") else str(result)
        except Exception as exc:
            logger.error("Failed to parse PDF %s with MarkItDown: %s", path, exc)
            raise RuntimeError(f"PDF parsing failed: {exc}") from exc

        return PdfParseResult(
            text=text,
            parser_provider=self.provider_name,
        )
