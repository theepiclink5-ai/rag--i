"""TraceService – read and parse traces from logs/traces.jsonl.

Provides a typed, filterable interface over the raw JSONL trace log.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path

logger = logging.getLogger(__name__)

# Default path to the traces file (absolute, CWD-independent)
DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceService:
    """Read-only service for querying recorded traces.

    Args:
        traces_path: Path to the JSONL file.  Defaults to
            ``logs/traces.jsonl``.
    """

    def __init__(self, traces_path: Optional[str | Path] = None) -> None:
        self.traces_path = Path(traces_path) if traces_path else DEFAULT_TRACES_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_traces(
        self,
        trace_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return traces in reverse-chronological order.

        Args:
            trace_type: Filter by ``trace_type`` field (e.g.
                ``"ingestion"`` or ``"query"``).  ``None`` = all.
            limit: Maximum number of traces to return.

        Returns:
            List of trace dicts (newest first).
        """
        traces = self._load_all()

        if trace_type:
            traces = [t for t in traces if t.get("trace_type") == trace_type]

        # Newest first
        traces.sort(key=lambda t: t.get("started_at", ""), reverse=True)

        return traces[:limit]

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single trace by its ``trace_id``.

        Returns:
            Trace dict, or ``None`` if not found.
        """
        for t in self._load_all():
            if t.get("trace_id") == trace_id:
                return t
        return None

    def get_stage_timings(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract stage timings from a trace.

        Returns:
            List of dicts with keys: stage_name, elapsed_ms, data.
            Ordered by appearance.
        """
        stages = trace.get("stages", [])
        timings: List[Dict[str, Any]] = []
        for s in stages:
            # The raw stage dict has: stage, timestamp, data (dict), elapsed_ms
            # Extract the inner 'data' dict directly rather than flattening
            stage_data = s.get("data", {})
            if not isinstance(stage_data, dict):
                stage_data = {}
            # Older traces stored stage-specific fields at the top level.
            if not stage_data:
                stage_data = {
                    key: value
                    for key, value in s.items()
                    if key not in {"stage", "timestamp", "elapsed_ms", "data"}
                }
            timings.append(
                {
                    "stage_name": s.get("stage"),
                    "elapsed_ms": s.get("elapsed_ms", 0),
                    "data": stage_data,
                }
            )
        return timings

    def get_latency_breakdown(self, trace: Dict[str, Any]) -> Dict[str, float]:
        """Return wall-clock latency buckets without double-counting retrieval.

        Dense and sparse retrieval run concurrently, so their contribution to
        the critical path is the slower branch rather than their sum. New
        traces record answer-ready latency directly; timestamps provide a
        backwards-compatible estimate for older traces.
        """
        timings = self.get_stage_timings(trace)
        by_name = {item["stage_name"]: item for item in timings}
        total_ms = float(
            trace.get("total_elapsed_ms", trace.get("elapsed_ms", 0)) or 0
        )
        evaluation_ms = float(
            by_name.get("ragas_evaluation", {}).get("elapsed_ms", 0) or 0
        )

        metadata = trace.get("metadata") or {}
        answer_ready = metadata.get("answer_ready_elapsed_ms")
        if not isinstance(answer_ready, (int, float)):
            answer_ready = self._elapsed_to_answer_stage(trace)
        if answer_ready is None:
            answer_ready = max(total_ms - evaluation_ms, 0.0)
        answer_ready_ms = min(max(float(answer_ready), 0.0), total_ms or float(answer_ready))

        dense_ms = float(by_name.get("dense_retrieval", {}).get("elapsed_ms", 0) or 0)
        sparse_ms = float(by_name.get("sparse_retrieval", {}).get("elapsed_ms", 0) or 0)
        retrieval_wall_ms = max(dense_ms, sparse_ms)
        known_answer_ms = sum(
            float(by_name.get(name, {}).get("elapsed_ms", 0) or 0)
            for name in (
                "reranker_startup", "initialization", "query_processing",
                "fusion", "rerank", "evidence_guard", "llm_generation",
            )
        ) + retrieval_wall_ms

        return {
            "answer_ready_ms": answer_ready_ms,
            "evaluation_ms": evaluation_ms,
            "total_ms": total_ms,
            "other_answer_ms": max(answer_ready_ms - known_answer_ms, 0.0),
            "retrieval_wall_ms": retrieval_wall_ms,
        }

    @staticmethod
    def _elapsed_to_answer_stage(trace: Dict[str, Any]) -> Optional[float]:
        """Estimate answer-ready latency for traces created before that field."""
        stages = trace.get("stages") or []
        preferred = ("llm_generation", "evidence_guard")
        target = next(
            (
                stage
                for name in preferred
                for stage in reversed(stages)
                if stage.get("stage") == name
            ),
            None,
        )
        if not target or not trace.get("started_at") or not target.get("timestamp"):
            return None
        try:
            started = datetime.fromisoformat(trace["started_at"])
            finished = datetime.fromisoformat(target["timestamp"])
            return max((finished - started).total_seconds() * 1000.0, 0.0)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> List[Dict[str, Any]]:
        """Parse every line in the JSONL file.

        Silently skips malformed lines.
        """
        if not self.traces_path.exists():
            return []

        traces: List[Dict[str, Any]] = []
        with self.traces_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed trace line: %s", line[:80])
        return traces
