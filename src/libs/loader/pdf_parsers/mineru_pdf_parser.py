"""MinerU-backed PDF parser.

This module keeps MinerU result handling separate from the HTTP client.  The
parser accepts a client object that returns a MinerU artifact, then normalizes
the extracted package into PdfParseResult.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from src.libs.loader.pdf_parsers.base_pdf_parser import PdfParseResult


class MinerUParseError(RuntimeError):
    """Raised when MinerU parsing or result normalization fails."""


@dataclass(frozen=True)
class MinerUParseOptions:
    """Options sent to MinerU when requesting a parse."""

    model_version: str = "vlm"
    language: str = "ch"
    is_ocr: bool = True
    enable_table: bool = True
    enable_formula: bool = True
    timeout_seconds: int = 600
    poll_interval_seconds: int = 5


@dataclass(frozen=True)
class MinerUParseArtifact:
    """Raw artifact returned by a MinerU client."""

    full_zip_bytes: Optional[bytes] = None
    full_zip_path: Optional[str | Path] = None
    result_dir: Optional[str | Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MinerUClientProtocol(Protocol):
    """Client protocol used by MinerUPdfParser."""

    def parse_pdf(
        self, file_path: str | Path, options: MinerUParseOptions
    ) -> MinerUParseArtifact:
        """Submit a PDF to MinerU and return its parsed artifact."""


class MissingMinerUClient:
    """Default client that makes missing wiring explicit."""

    def parse_pdf(
        self, file_path: str | Path, options: MinerUParseOptions
    ) -> MinerUParseArtifact:
        raise MinerUParseError(
            "MinerU client is not configured. Inject a client before using "
            "MinerUPdfParser for real API calls."
        )


class MinerUPdfParser:
    """Normalize MinerU parse artifacts into the project PDF parser contract."""

    provider_name = "mineru"

    def __init__(
        self,
        client: Optional[MinerUClientProtocol] = None,
        options: Optional[MinerUParseOptions] = None,
        output_dir: str | Path = "data/parsed/mineru",
        save_raw_result: bool = True,
    ) -> None:
        self.client = client or MissingMinerUClient()
        self.options = options or MinerUParseOptions()
        self.output_dir = Path(output_dir)
        self.save_raw_result = save_raw_result

    def parse(self, file_path: str | Path) -> PdfParseResult:
        path = Path(file_path)
        artifact = self.client.parse_pdf(path, self.options)
        result_dir = self._materialize_artifact(artifact, path)

        full_md = self._find_required_file(result_dir, "full.md")
        text = full_md.read_text(encoding="utf-8")
        content_list = self._read_optional_json(result_dir, "content_list")
        layout = self._read_optional_json(result_dir, "layout")

        metadata = {
            **artifact.metadata,
            "mineru_result_dir": str(result_dir),
            "mineru_has_content_list": content_list is not None,
            "mineru_has_layout": layout is not None,
        }

        return PdfParseResult(
            text=text,
            parser_provider=self.provider_name,
            metadata=metadata,
            pages=self._extract_pages(layout),
            tables=self._extract_blocks(content_list, block_type="table"),
            images=self._extract_blocks(content_list, block_type="image"),
            raw_output_dir=str(result_dir),
        )

    def _materialize_artifact(
        self, artifact: MinerUParseArtifact, source_path: Path
    ) -> Path:
        if artifact.result_dir:
            result_dir = Path(artifact.result_dir)
            if not result_dir.exists():
                raise MinerUParseError(f"MinerU result directory not found: {result_dir}")
            return result_dir

        result_dir = self._result_dir_for(source_path)
        result_dir.mkdir(parents=True, exist_ok=True)

        if artifact.full_zip_path:
            self._extract_zip(Path(artifact.full_zip_path), result_dir)
            return result_dir

        if artifact.full_zip_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
                temp_file.write(artifact.full_zip_bytes)
                temp_zip_path = Path(temp_file.name)
            try:
                self._extract_zip(temp_zip_path, result_dir)
            finally:
                temp_zip_path.unlink(missing_ok=True)
            return result_dir

        raise MinerUParseError(
            "MinerU artifact must include result_dir, full_zip_path, or full_zip_bytes."
        )

    def _result_dir_for(self, source_path: Path) -> Path:
        stem = source_path.stem or "document"
        return self.output_dir / stem

    def _extract_zip(self, zip_path: Path, result_dir: Path) -> None:
        if not zip_path.exists():
            raise MinerUParseError(f"MinerU zip artifact not found: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(result_dir)
        except zipfile.BadZipFile as exc:
            raise MinerUParseError(f"Invalid MinerU zip artifact: {zip_path}") from exc

    def _find_required_file(self, result_dir: Path, filename: str) -> Path:
        matches = list(result_dir.rglob(filename))
        if not matches:
            raise MinerUParseError(f"MinerU result is missing required {filename}")
        return matches[0]

    def _read_optional_json(self, result_dir: Path, name_fragment: str) -> Any:
        matches = sorted(result_dir.rglob(f"*{name_fragment}*.json"))
        if not matches:
            return None
        try:
            return json.loads(matches[0].read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MinerUParseError(f"Invalid MinerU JSON artifact: {matches[0]}") from exc

    def _extract_pages(self, layout: Any) -> list[dict[str, Any]]:
        if not isinstance(layout, list):
            return []
        pages = []
        for item in layout:
            if isinstance(item, dict):
                pages.append(item)
        return pages

    def _extract_blocks(self, content_list: Any, block_type: str) -> list[dict[str, Any]]:
        if not isinstance(content_list, list):
            return []
        blocks = []
        for item in content_list:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or item.get("block_type") or "").lower()
            if block_type in item_type:
                blocks.append(item)
        return blocks
