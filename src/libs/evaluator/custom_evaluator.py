"""Custom evaluator implementation for lightweight metrics.

This evaluator computes simple, deterministic metrics such as hit rate and MRR.
It is designed for fast regression checks and sanity validation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Custom evaluator for lightweight metrics (hit_rate, mrr).

    The evaluator expects retrieved chunks to contain an identifier field.
    Supported id fields: id, chunk_id, document_id, doc_id.
    Ground truth can also provide source filenames via ``sources``; those are
    matched against retrieved chunk metadata source/source_path basenames.
    """

    SUPPORTED_METRICS = {"hit_rate", "mrr"}
    _ID_FIELDS = ("id", "chunk_id", "document_id", "doc_id")

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalized = [str(metric).strip().lower() for metric in (metrics or [])]
        if not normalized:
            normalized = ["hit_rate", "mrr"]

        unsupported = [metric for metric in normalized if metric not in self.SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                "Unsupported custom metrics: "
                f"{', '.join(unsupported)}. Supported: {', '.join(sorted(self.SUPPORTED_METRICS))}"
            )

        self.metrics = normalized

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute requested metrics for the given retrieval results.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks or records.
            generated_answer: Optional generated answer (unused).
            ground_truth: Ground truth ids or structure.
            trace: Optional TraceContext (unused).
            **kwargs: Additional parameters (unused).

        Returns:
            Dictionary of metric name to float value.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        retrieved_ids = self._extract_ids(retrieved_chunks, label="retrieved_chunks")
        retrieved_sources = self._extract_sources(retrieved_chunks)
        ground_truth_ids = self._extract_ground_truth_ids(ground_truth)
        if self._uses_source_only_ground_truth(ground_truth):
            retrieved_keys = retrieved_sources
        else:
            retrieved_keys = retrieved_ids

        results: Dict[str, float] = {}

        if "hit_rate" in self.metrics:
            results["hit_rate"] = self._compute_hit_rate(retrieved_keys, ground_truth_ids)
        if "mrr" in self.metrics:
            results["mrr"] = self._compute_mrr(retrieved_keys, ground_truth_ids)

        return results

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        metrics = getattr(getattr(settings, "evaluation", None), "metrics", None)
        if metrics is None:
            return []
        return [str(metric) for metric in metrics]

    def _extract_ground_truth_ids(self, ground_truth: Optional[Any]) -> List[str]:
        """Extract ground truth ids/source keys from various input shapes."""
        if ground_truth is None:
            return []
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, dict):
            keys: List[str] = []
            if "ids" in ground_truth and isinstance(ground_truth["ids"], list):
                keys.extend(self._extract_ids(ground_truth["ids"], label="ground_truth.ids"))
            if "sources" in ground_truth and isinstance(ground_truth["sources"], list):
                keys.extend(self._normalize_source(item) for item in ground_truth["sources"])
            if keys:
                return keys
            return self._extract_ids([ground_truth], label="ground_truth")
        if isinstance(ground_truth, list):
            return self._extract_ids(ground_truth, label="ground_truth")

        raise ValueError(
            f"Unsupported ground_truth type: {type(ground_truth).__name__}. "
            "Expected str, dict, list, or None."
        )

    def _uses_source_only_ground_truth(self, ground_truth: Optional[Any]) -> bool:
        """Return True when the test case should be ranked by source names."""
        if not isinstance(ground_truth, dict):
            return False
        ids = ground_truth.get("ids")
        sources = ground_truth.get("sources")
        return isinstance(sources, list) and bool(sources) and not ids

    def _extract_ids(self, items: Iterable[Any], label: str) -> List[str]:
        """Extract ids from a list of items."""
        ids: List[str] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                ids.append(item)
                continue
            if isinstance(item, dict):
                for field in self._ID_FIELDS:
                    if field in item:
                        ids.append(str(item[field]))
                        break
                else:
                    raise ValueError(
                        f"Missing id field in {label}[{index}]. "
                        f"Expected one of {', '.join(self._ID_FIELDS)}"
                    )
                continue
            if hasattr(item, "id"):
                ids.append(str(getattr(item, "id")))
                continue
            if hasattr(item, "chunk_id"):
                ids.append(str(getattr(item, "chunk_id")))
                continue

            raise ValueError(
                f"Unable to extract id from {label}[{index}] of type "
                f"{type(item).__name__}"
            )

        return ids

    def _extract_sources(self, items: Iterable[Any]) -> List[str]:
        """Extract normalized source basenames from retrieved chunk metadata."""
        sources: List[str] = []
        for item in items:
            metadata = self._get_metadata(item)
            if not metadata:
                continue
            source = metadata.get("source_path") or metadata.get("source")
            if source:
                sources.append(self._normalize_source(source))
        return sources

    def _get_metadata(self, item: Any) -> Dict[str, Any]:
        """Return metadata for dict or object retrieval results."""
        if isinstance(item, dict):
            metadata = item.get("metadata")
            return metadata if isinstance(metadata, dict) else {}
        metadata = getattr(item, "metadata", None)
        return metadata if isinstance(metadata, dict) else {}

    def _normalize_source(self, source: Any) -> str:
        """Normalize a source path to a case-insensitive filename key."""
        value = str(source).replace("\\", "/").rstrip("/")
        return value.rsplit("/", 1)[-1].lower()

    def _compute_hit_rate(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """Compute hit rate (binary)."""
        if not ground_truth_ids:
            return 0.0
        return 1.0 if any(item in ground_truth_ids for item in retrieved_ids) else 0.0

    def _compute_mrr(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """Compute Mean Reciprocal Rank (MRR)."""
        if not ground_truth_ids:
            return 0.0
        for rank, item in enumerate(retrieved_ids, start=1):
            if item in ground_truth_ids:
                return 1.0 / rank
        return 0.0
