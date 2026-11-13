"""Pre-generation evidence checks for citable retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.core.types import RetrievalResult


class EvidenceStatus(str, Enum):
    """Confidence that the retrieved evidence can support an answer."""

    SUPPORTED = "supported"
    UNCERTAIN = "uncertain"
    NO_EVIDENCE = "no_evidence"


@dataclass(frozen=True)
class EvidenceGuardConfig:
    enabled: bool = True
    min_text_length: int = 30
    strong_score_threshold: float = 0.60
    weak_score_threshold: float = 0.20
    min_strong_evidence: int = 1


@dataclass
class EvidenceDecision:
    status: EvidenceStatus
    citable_results: List[RetrievalResult] = field(default_factory=list)
    rejected_count: int = 0
    top_rerank_score: Optional[float] = None
    reason: str = ""

    @property
    def answerable(self) -> bool:
        return self.status is not EvidenceStatus.NO_EVIDENCE


class EvidenceGuard:
    """Filter untraceable chunks and gate clearly unsupported queries."""

    def __init__(self, config: Optional[EvidenceGuardConfig] = None) -> None:
        self.config = config or EvidenceGuardConfig()

    @classmethod
    def from_settings(cls, settings: object) -> "EvidenceGuard":
        guard = getattr(settings, "citation_guard", None)
        if guard is None:
            return cls()
        return cls(
            EvidenceGuardConfig(
                enabled=guard.enabled,
                min_text_length=guard.min_text_length,
                strong_score_threshold=guard.strong_score_threshold,
                weak_score_threshold=guard.weak_score_threshold,
                min_strong_evidence=guard.min_strong_evidence,
            )
        )

    def evaluate(self, results: List[RetrievalResult]) -> EvidenceDecision:
        if not self.config.enabled:
            return EvidenceDecision(
                status=EvidenceStatus.UNCERTAIN,
                citable_results=list(results),
                reason="evidence guard disabled",
            )

        structurally_valid = self._filter_citable(results)
        rejected_count = len(results) - len(structurally_valid)
        if not structurally_valid:
            return EvidenceDecision(
                status=EvidenceStatus.NO_EVIDENCE,
                rejected_count=rejected_count,
                reason="no traceable text chunks were retrieved",
            )

        scored = [
            (result, score)
            for result in structurally_valid
            if (score := self._rerank_score(result)) is not None
        ]
        if not scored:
            return EvidenceDecision(
                status=EvidenceStatus.UNCERTAIN,
                citable_results=structurally_valid,
                rejected_count=rejected_count,
                reason="no Cross-Encoder scores available",
            )

        scored.sort(key=lambda item: item[1], reverse=True)
        top_score = scored[0][1]
        relevant_results = [
            result
            for result, score in scored
            if score >= self.config.weak_score_threshold
        ]
        strong_count = sum(
            score >= self.config.strong_score_threshold for _, score in scored
        )

        if strong_count >= self.config.min_strong_evidence:
            return EvidenceDecision(
                status=EvidenceStatus.SUPPORTED,
                citable_results=relevant_results,
                rejected_count=rejected_count + len(scored) - len(relevant_results),
                top_rerank_score=top_score,
                reason="strong reranked evidence found",
            )
        if relevant_results:
            return EvidenceDecision(
                status=EvidenceStatus.UNCERTAIN,
                citable_results=relevant_results,
                rejected_count=rejected_count + len(scored) - len(relevant_results),
                top_rerank_score=top_score,
                reason="only weak reranked evidence found",
            )
        return EvidenceDecision(
            status=EvidenceStatus.NO_EVIDENCE,
            rejected_count=rejected_count + len(scored),
            top_rerank_score=top_score,
            reason="all reranked evidence fell below the weak threshold",
        )

    def _filter_citable(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        accepted: List[RetrievalResult] = []
        seen_chunk_ids = set()
        for result in results:
            metadata = result.metadata or {}
            source = (
                metadata.get("source_path")
                or metadata.get("source")
                or metadata.get("source_ref")
            )
            text = (result.text or "").strip()
            if (
                not result.chunk_id
                or result.chunk_id in seen_chunk_ids
                or not source
                or len(text) < self.config.min_text_length
            ):
                continue
            seen_chunk_ids.add(result.chunk_id)
            accepted.append(result)
        return accepted

    @staticmethod
    def _rerank_score(result: RetrievalResult) -> Optional[float]:
        metadata = result.metadata or {}
        if not metadata.get("reranked"):
            return None
        score = metadata.get("rerank_score", result.score)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        return float(score)
