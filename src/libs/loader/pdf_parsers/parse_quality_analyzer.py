"""Heuristic quality checks for locally parsed PDF text."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


MOJIBAKE_TOKENS = (
    "涓",
    "锛",
    "銆",
    "绋",
    "鐨",
    "杩",
    "妫",
    "鍙",
    "鎻",
    "鎷",
    "彃",
    "€",
    "�",
    "□",
)


@dataclass(frozen=True)
class PdfParseQualityReport:
    """Quality report for one parsed PDF."""

    score: float
    reasons: List[str] = field(default_factory=list)
    should_use_external_api: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)


class PdfParseQualityAnalyzer:
    """Score parsed PDF output to decide whether a fallback parser is needed."""

    def __init__(self, fallback_threshold: float = 0.65) -> None:
        self.fallback_threshold = fallback_threshold

    def analyze(
        self,
        text: str,
        file_path: str | Path | None = None,
        page_count: Optional[int] = None,
        image_count: int = 0,
        table_count: int = 0,
        markdown_table_count: Optional[int] = None,
        parse_error: Optional[str] = None,
    ) -> PdfParseQualityReport:
        """Analyze parse output and return a deterministic quality report."""
        if parse_error:
            return PdfParseQualityReport(
                score=0.0,
                reasons=["parse_error"],
                should_use_external_api=True,
                metrics={"parse_error": parse_error},
            )

        inferred = self._infer_pdf_stats(file_path)
        if page_count is None:
            page_count = inferred.get("page_count")
        if image_count == 0:
            image_count = int(inferred.get("image_count", 0))

        normalized_text = text or ""
        lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
        char_count = len(normalized_text)
        effective_pages = max(page_count or 1, 1)
        avg_chars_per_page = char_count / effective_pages
        markdown_table_count = (
            self._count_markdown_tables(lines)
            if markdown_table_count is None
            else markdown_table_count
        )

        metrics: Dict[str, Any] = {
            "page_count": page_count,
            "char_count": char_count,
            "avg_chars_per_page": round(avg_chars_per_page, 2),
            "image_count": image_count,
            "table_count": table_count,
            "markdown_table_count": markdown_table_count,
            "mojibake_ratio": round(self._mojibake_ratio(normalized_text), 4),
            "duplicate_line_ratio": round(self._duplicate_line_ratio(lines), 4),
            "short_line_ratio": round(self._short_line_ratio(lines), 4),
        }

        score = 1.0
        reasons: List[str] = []

        if not normalized_text.strip():
            score -= 0.7
            reasons.append("empty_text")

        if avg_chars_per_page < 80:
            score -= 0.4
            reasons.append("low_text_density")

        if image_count >= effective_pages and avg_chars_per_page < 120:
            score -= 0.35
            reasons.append("scanned_pdf_likely")

        if metrics["mojibake_ratio"] > 0.01:
            score -= 0.35
            reasons.append("mojibake_detected")

        if metrics["duplicate_line_ratio"] > 0.25:
            score -= 0.2
            reasons.append("repeated_text_noise")

        if table_count > 0 and markdown_table_count == 0:
            score -= 0.25
            reasons.append("table_structure_lost")

        if image_count >= 3 and "[IMAGE:" not in normalized_text:
            score -= 0.15
            reasons.append("image_content_missing")

        if metrics["short_line_ratio"] > 0.55 and len(lines) >= 20:
            score -= 0.15
            reasons.append("fragmented_lines")

        final_score = max(round(score, 4), 0.0)
        should_fallback = final_score < self.fallback_threshold or any(
            reason in reasons
            for reason in ("empty_text", "scanned_pdf_likely", "mojibake_detected")
        )

        return PdfParseQualityReport(
            score=final_score,
            reasons=reasons,
            should_use_external_api=should_fallback,
            metrics=metrics,
        )

    def _infer_pdf_stats(self, file_path: str | Path | None) -> Dict[str, int]:
        if not file_path or not PYMUPDF_AVAILABLE:
            return {}

        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            image_count = sum(len(page.get_images(full=True)) for page in doc)
            doc.close()
            return {"page_count": page_count, "image_count": image_count}
        except Exception:
            return {}

    def _mojibake_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        hits = sum(text.count(token) for token in MOJIBAKE_TOKENS)
        return hits / max(len(text), 1)

    def _duplicate_line_ratio(self, lines: List[str]) -> float:
        if not lines:
            return 0.0
        counts = Counter(lines)
        duplicated = sum(count - 1 for count in counts.values() if count > 1)
        return duplicated / len(lines)

    def _short_line_ratio(self, lines: List[str]) -> float:
        if not lines:
            return 0.0
        short_count = sum(1 for line in lines if len(line) < 20)
        return short_count / len(lines)

    def _count_markdown_tables(self, lines: List[str]) -> int:
        table_lines = [line for line in lines if line.count("|") >= 2]
        return 1 if len(table_lines) >= 2 else 0
