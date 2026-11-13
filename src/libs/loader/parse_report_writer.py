"""Write parse reports for PDF ingestion audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class PdfParseReportWriter:
    """Persist compact JSON reports for parsed PDFs."""

    def __init__(self, report_dir: str | Path) -> None:
        self.report_dir = Path(report_dir)

    def write(
        self,
        *,
        source_path: str | Path,
        doc_id: str,
        metadata: Dict[str, Any],
        text: str,
    ) -> Path:
        """Write a parse report and return its path."""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.report_dir / f"{doc_id}.parse_report.json"
        report = self._build_report(
            source_path=source_path,
            doc_id=doc_id,
            metadata=metadata,
            text=text,
        )
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return report_path

    def _build_report(
        self,
        *,
        source_path: str | Path,
        doc_id: str,
        metadata: Dict[str, Any],
        text: str,
    ) -> Dict[str, Any]:
        metrics = metadata.get("parse_quality_metrics", {})
        return {
            "source_path": str(source_path),
            "doc_id": doc_id,
            "doc_hash": metadata.get("doc_hash"),
            "title": metadata.get("title"),
            "parser_provider": metadata.get("parser_provider"),
            "parser_route": metadata.get("parser_route"),
            "primary_parser_provider": metadata.get("primary_parser_provider"),
            "fallback_provider": metadata.get("fallback_provider"),
            "fallback_used": bool(metadata.get("fallback_used", False)),
            "fallback_error": metadata.get("fallback_error"),
            "parse_quality_score": metadata.get("parse_quality_score"),
            "parse_quality_reasons": metadata.get("parse_quality_reasons", []),
            "parse_quality_metrics": metrics,
            "should_use_external_pdf_api": bool(
                metadata.get("should_use_external_pdf_api", False)
            ),
            "primary_parse_quality_score": metadata.get("primary_parse_quality_score"),
            "primary_parse_quality_reasons": metadata.get(
                "primary_parse_quality_reasons", []
            ),
            "primary_should_use_external_pdf_api": metadata.get(
                "primary_should_use_external_pdf_api"
            ),
            "selected_text_char_count": len(text or ""),
            "image_count": len(metadata.get("images", [])),
            "mineru_result_dir": metadata.get("mineru_result_dir"),
            "mineru_batch_id": metadata.get("mineru_batch_id"),
            "mineru_file_name": metadata.get("mineru_file_name"),
            "mineru_state": metadata.get("mineru_state"),
        }
