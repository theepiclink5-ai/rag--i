"""Generic document loader for ingestion-supported file formats."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.pdf_parsers import BasePdfParser


class DocumentLoader(BaseLoader):
    """Load PDF, text, Markdown, and DOCX files into a unified Document."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
        pdf_parser: Optional[BasePdfParser] = None,
        parse_report_dir: str | Path | None = None,
    ) -> None:
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        self._pdf_loader = PdfLoader(
            extract_images=extract_images,
            image_storage_dir=image_storage_dir,
            parser=pdf_parser,
            parse_report_dir=parse_report_dir,
        )
        self._markitdown = MarkItDown() if MARKITDOWN_AVAILABLE else None

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported document format '{suffix}'. Supported: {supported}")

        if suffix == ".pdf":
            return self._pdf_loader.load(path)

        doc_hash = self._compute_file_hash(path)
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": suffix.lstrip("."),
            "doc_hash": doc_hash,
        }

        if suffix in {".txt", ".md"}:
            text_content = self._read_text_file(path)
        else:
            text_content = self._convert_with_markitdown(path)

        title = self._extract_title(text_content)
        if title:
            metadata["title"] = title

        return Document(
            id=f"doc_{doc_hash[:16]}",
            text=text_content,
            metadata=metadata,
        )

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        for line in text.splitlines()[:20]:
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
            if stripped:
                return stripped
        return None

    @staticmethod
    def _read_text_file(path: Path) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="replace")

    def _convert_with_markitdown(self, path: Path) -> str:
        if self._markitdown is None:
            raise ImportError(
                "MarkItDown is required for DOCX loading. Install with: pip install markitdown"
            )

        try:
            result = self._markitdown.convert(str(path))
            return result.text_content if hasattr(result, "text_content") else str(result)
        except Exception as exc:
            raise RuntimeError(f"Document parsing failed: {exc}") from exc
