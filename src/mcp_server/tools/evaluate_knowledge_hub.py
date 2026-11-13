"""MCP tool for an end-to-end RAG answer with detailed Ragas evaluation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from mcp import types

from src.core.response import MCPToolResponse
from src.core.settings import Settings, load_settings
from src.core.trace import TraceCollector, TraceContext
from src.mcp_server.tools.ask_knowledge_hub import AskKnowledgeHubTool
from src.observability.evaluation.evaluation_collector import EvaluationCollector
from src.observability.evaluation.ragas_evaluator import (
    ANSWER_RELEVANCY,
    CONTEXT_PRECISION,
    CONTEXT_RECALL,
    FAITHFULNESS,
    RagasEvaluator,
    SUPPORTED_METRICS,
)
from src.libs.evaluator.correctness_evaluator import (
    CorrectnessEvaluator,
    SUPPORTED_METRICS as CORRECTNESS_METRICS,
)

logger = logging.getLogger(__name__)

TOOL_NAME = "evaluate_knowledge_hub"
TOOL_DESCRIPTION = """Answer a knowledge-base question and evaluate that exact run.

Unlike ask_knowledge_hub, this route runs Ragas after generation. Evaluation uses
the final contexts and answer from the same request, never a second retrieval.
Use it for quality analysis because it adds LLM latency and cost.
"""

ALL_SUPPORTED_METRICS = SUPPORTED_METRICS | CORRECTNESS_METRICS
DEFAULT_METRICS = [FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION, CONTEXT_RECALL]

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The question to answer and evaluate.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of reranked evidence chunks.",
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
        "collection": {
            "type": "string",
            "description": "Optional collection name to search.",
        },
        "metrics": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(ALL_SUPPORTED_METRICS)},
            "description": "Metrics to calculate. Includes Ragas metrics and answer_correctness.",
            "default": DEFAULT_METRICS,
            "minItems": 1,
            "uniqueItems": True,
        },
        "reference_answer": {
            "type": "string",
            "description": "Optional reference answer for context_recall and answer_correctness evaluation.",
        },
    },
    "required": ["question"],
}


class EvaluateKnowledgeHubTool:
    """Run the shared answer pipeline and evaluate its frozen output."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        answer_tool: Optional[AskKnowledgeHubTool] = None,
        trace_collector: Optional[TraceCollector] = None,
        evaluation_collector: Optional[EvaluationCollector] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.answer_tool = answer_tool or AskKnowledgeHubTool(settings=self.settings)
        self.trace_collector = trace_collector or TraceCollector()
        self.evaluation_collector = evaluation_collector or EvaluationCollector()

    async def execute(
        self,
        question: str,
        top_k: int = 5,
        collection: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
    ) -> MCPToolResponse:
        selected_metrics = self._validate_metrics(metrics or DEFAULT_METRICS)
        trace = TraceContext(trace_type="query")
        trace.metadata.update({
            "mode": "evaluation",
            "source": "mcp",
            "evaluation_status": "running",
            "evaluation_metrics_requested": selected_metrics,
        })

        try:
            response = await self.answer_tool.execute(
                question=question,
                top_k=top_k,
                collection=collection,
                trace=trace,
                collect_trace=False,
                mode="evaluation",
                source="mcp",
            )

            if response.metadata.get("status") != "answered":
                self._record_outcome(
                    trace=trace,
                    response=response,
                    question=question,
                    collection=collection,
                    status="skipped",
                    detail=response.metadata.get("status", "answer_not_generated"),
                )
                return response

            try:
                chunks = list(trace.metadata.get("final_results") or [])
                if not chunks:
                    raise ValueError("The completed answer trace has no final contexts")

                # Split metrics into ragas vs correctness
                ragas_metrics = [m for m in selected_metrics if m in SUPPORTED_METRICS]
                correctness_metrics = [m for m in selected_metrics if m in CORRECTNESS_METRICS]

                scores: Dict[str, float] = {}

                # Run Ragas evaluation
                if ragas_metrics:
                    evaluator = RagasEvaluator(settings=self.settings, metrics=ragas_metrics)
                    ground_truth = {"reference_answer": reference_answer} if reference_answer else None
                    started = time.monotonic()
                    ragas_scores = await asyncio.to_thread(
                        evaluator.evaluate,
                        query=question,
                        retrieved_chunks=chunks,
                        generated_answer=response.content,
                        ground_truth=ground_truth,
                        trace=trace,
                    )
                    scores.update(ragas_scores)
                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    trace.record_stage(
                        "ragas_evaluation",
                        {
                            "provider": "ragas",
                            "metrics": ragas_scores,
                            "context_count": len(chunks),
                        },
                        elapsed_ms=elapsed_ms,
                    )

                # Run Correctness evaluation
                if correctness_metrics and reference_answer:
                    corr_evaluator = CorrectnessEvaluator(
                        settings=self.settings, metrics=correctness_metrics,
                    )
                    corr_gt = {"reference_answer": reference_answer}
                    started = time.monotonic()
                    corr_scores = await asyncio.to_thread(
                        corr_evaluator.evaluate,
                        query=question,
                        retrieved_chunks=chunks,
                        generated_answer=response.content,
                        ground_truth=corr_gt,
                        trace=trace,
                    )
                    scores.update(corr_scores)
                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    trace.record_stage(
                        "correctness_evaluation",
                        {
                            "provider": "correctness",
                            "metrics": corr_scores,
                        },
                        elapsed_ms=elapsed_ms,
                    )

                self._record_outcome(
                    trace=trace,
                    response=response,
                    question=question,
                    collection=collection,
                    status="completed",
                    metrics=scores,
                )
            except Exception as exc:
                # The answer is still useful when the optional judge fails.
                logger.exception("Ragas evaluation failed for trace %s", trace.trace_id)
                self._record_outcome(
                    trace=trace,
                    response=response,
                    question=question,
                    collection=collection,
                    status="error",
                    detail=str(exc),
                )
            return response
        except Exception as exc:
            logger.exception("evaluate_knowledge_hub failed: %s", exc)
            trace.metadata.update({
                "evaluation_status": "error",
                "evaluation_error": str(exc),
            })
            self.evaluation_collector.collect(
                trace_id=trace.trace_id,
                query=question,
                collection=collection,
                metrics={},
                status="error",
                error=str(exc),
            )
            return MCPToolResponse(
                content="The evaluated knowledge-base answer could not be completed.",
                metadata={
                    "mode": "evaluation",
                    "status": "error",
                    "trace_id": trace.trace_id,
                    "evaluation_status": "error",
                    "error": str(exc),
                },
                is_empty=True,
            )
        finally:
            self.trace_collector.collect(trace)

    def _record_outcome(
        self,
        *,
        trace: TraceContext,
        response: MCPToolResponse,
        question: str,
        collection: Optional[str],
        status: str,
        metrics: Optional[Dict[str, float]] = None,
        detail: Optional[str] = None,
    ) -> None:
        """Write one evaluation outcome consistently to all output surfaces."""
        scores = metrics or {}
        trace_update: Dict[str, Any] = {
            "evaluation_status": status,
            "evaluation_metrics": scores,
        }
        response_update = dict(trace_update)
        if detail:
            key = "evaluation_reason" if status == "skipped" else "evaluation_error"
            trace_update[key] = detail
            response_update[key] = detail

        trace.metadata.update(trace_update)
        response.metadata.update(response_update)
        self.evaluation_collector.collect(
            trace_id=trace.trace_id,
            query=question,
            collection=response.metadata.get("collection", collection),
            metrics=scores,
            status=status,
            error=detail if status == "error" else None,
        )

    @staticmethod
    def _validate_metrics(metrics: List[str]) -> List[str]:
        normalized = list(dict.fromkeys(metric.strip().lower() for metric in metrics))
        unsupported = [metric for metric in normalized if metric not in ALL_SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(ALL_SUPPORTED_METRICS))}"
            )
        if not normalized:
            raise ValueError("At least one evaluation metric is required")
        return normalized


_tool_instance: Optional[EvaluateKnowledgeHubTool] = None


def get_tool_instance(settings: Optional[Settings] = None) -> EvaluateKnowledgeHubTool:
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = EvaluateKnowledgeHubTool(settings=settings)
    return _tool_instance


async def evaluate_knowledge_hub_handler(
    question: str,
    top_k: int = 5,
    collection: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    reference_answer: Optional[str] = None,
) -> types.CallToolResult:
    try:
        response = await get_tool_instance().execute(
            question=question,
            top_k=top_k,
            collection=collection,
            metrics=metrics,
            reference_answer=reference_answer,
        )
        return types.CallToolResult(
            content=response.to_mcp_content(),
            isError=response.metadata.get("status") == "error",
        )
    except ValueError as exc:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Invalid parameters: {exc}")],
            isError=True,
        )


def register_tool(protocol_handler: Any) -> None:
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=evaluate_knowledge_hub_handler,
    )
    logger.info("Registered MCP tool: %s", TOOL_NAME)
