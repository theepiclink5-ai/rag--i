"""Persistence for per-request evaluation results."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.core.settings import resolve_path

logger = logging.getLogger(__name__)


class EvaluationCollector:
    """Append detailed evaluation results to a JSON Lines file."""

    def __init__(
        self,
        evaluations_path: str | Path = resolve_path("logs/evaluations.jsonl"),
    ) -> None:
        self._path = Path(evaluations_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def collect(
        self,
        *,
        trace_id: str,
        query: str,
        collection: str | None,
        metrics: Dict[str, float],
        status: str = "completed",
        error: str | None = None,
    ) -> None:
        record: Dict[str, Any] = {
            "trace_id": trace_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "collection": collection,
            "status": status,
            "metrics": metrics,
        }
        if error:
            record["error"] = error

        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Failed to persist evaluation for trace %s", trace_id)

    @property
    def path(self) -> Path:
        return self._path
