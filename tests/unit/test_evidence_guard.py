"""Tests for pre-generation evidence filtering and gating."""

from src.core.response.evidence_guard import (
    EvidenceGuard,
    EvidenceGuardConfig,
    EvidenceStatus,
)
from src.core.types import RetrievalResult


def _result(
    chunk_id: str,
    score: float,
    *,
    text: str = "A sufficiently long reference passage for evidence checks.",
    source: str = "docs/reference.pdf",
    reranked: bool = True,
) -> RetrievalResult:
    metadata = {"source_path": source}
    if reranked:
        metadata.update({"reranked": True, "rerank_score": score})
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata=metadata,
    )


def test_strong_evidence_is_supported_and_low_scores_are_removed() -> None:
    decision = EvidenceGuard().evaluate([
        _result("strong", 0.91),
        _result("irrelevant", 0.01),
    ])

    assert decision.status is EvidenceStatus.SUPPORTED
    assert [result.chunk_id for result in decision.citable_results] == ["strong"]
    assert decision.rejected_count == 1
    assert decision.top_rerank_score == 0.91


def test_weak_evidence_is_uncertain() -> None:
    decision = EvidenceGuard().evaluate([_result("weak", 0.35)])

    assert decision.status is EvidenceStatus.UNCERTAIN
    assert decision.answerable is True
    assert len(decision.citable_results) == 1


def test_all_low_scored_evidence_is_rejected() -> None:
    decision = EvidenceGuard().evaluate([_result("low", 0.05)])

    assert decision.status is EvidenceStatus.NO_EVIDENCE
    assert decision.answerable is False
    assert decision.citable_results == []


def test_unscored_retrieval_results_are_not_compared_to_rerank_thresholds() -> None:
    decision = EvidenceGuard().evaluate([_result("rrf", 0.02, reranked=False)])

    assert decision.status is EvidenceStatus.UNCERTAIN
    assert len(decision.citable_results) == 1
    assert decision.top_rerank_score is None


def test_untraceable_short_and_duplicate_chunks_are_filtered() -> None:
    valid = _result("valid", 0.8)
    decision = EvidenceGuard().evaluate([
        valid,
        _result("valid", 0.9),
        _result("short", 0.9, text="too short"),
        _result("no-source", 0.9, source=""),
    ])

    assert decision.status is EvidenceStatus.SUPPORTED
    assert [result.chunk_id for result in decision.citable_results] == ["valid"]
    assert decision.rejected_count == 3


def test_guard_can_be_disabled() -> None:
    guard = EvidenceGuard(EvidenceGuardConfig(enabled=False))
    result = _result("raw", 0.01, source="", text="short")

    decision = guard.evaluate([result])

    assert decision.status is EvidenceStatus.UNCERTAIN
    assert decision.citable_results == [result]
