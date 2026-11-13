"""Common PDF parser interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class PdfParseResult:
    """Normalized output from a PDF parser backend."""

    text: str
    parser_provider: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    raw_output_dir: Optional[str] = None


class BasePdfParser(Protocol):
    """Protocol implemented by concrete PDF parser backends."""

    provider_name: str

    def parse(self, file_path: str | Path) -> PdfParseResult:
        """Parse a PDF and return normalized text plus parser metadata."""
