"""Evaluate a small golden-set sample through the full RAG and Ragas pipeline."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import statistics
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.core.response.prompt_config import RagAnswerPrompt
from src.core.response.citation_generator import CitationGenerator
from src.core.response.response_builder import ResponseBuilder
from src.core.settings import load_settings
from src.libs.llm import LLMFactory, Message
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool
from src.observability.evaluation.ragas_evaluator import RagasEvaluator


GOLDEN_PATH = _REPO_ROOT / "data/public_sets/pdf_mixed_100/golden_test_set.json"
OUTPUT_PATH = _REPO_ROOT / "logs/ragas_sample_10.json"
COLLECTION = "pdf_mixed_100"
METRICS = ["faithfulness", "answer_relevancy", "context_precision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed/rejected cases and retry errors or missing cases.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Reuse completed answers but recompute retrieval and Ragas metrics.",
    )
    return parser.parse_args()


def select_cases(limit: int) -> list[dict[str, Any]]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    selected = []
    seen_sources: set[str] = set()
    for case in data["test_cases"]:
        if case.get("metadata", {}).get("answer_type") != "rag_benchmark_answer":
            continue
        sources = case.get("expected_sources") or []
        source = sources[0] if sources else ""
        if not source or source in seen_sources:
            continue
        selected.append(case)
        seen_sources.add(source)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise ValueError(f"Only found {len(selected)} distinct-source golden cases")
    return selected


def save_report(report: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item.get("status") == "complete"]
    averages = {}
    for metric in METRICS:
        values = [item["ragas_metrics"][metric] for item in completed]
        averages[metric] = statistics.fmean(values) if values else None
    return {
        "total": len(results),
        "completed": len(completed),
        "rejected_before_llm": sum(
            item.get("status") == "rejected_before_llm" for item in results
        ),
        "errors": sum(item.get("status") == "error" for item in results),
        "expected_source_hit_rate": (
            sum(bool(item.get("expected_source_hit")) for item in results) / len(results)
            if results else 0.0
        ),
        "average_metrics": averages,
    }


async def main() -> int:
    args = parse_args()
    cases = select_cases(args.limit)
    settings = load_settings()
    answer_prompt = RagAnswerPrompt.load()
    response_builder = ResponseBuilder(
        citation_generator=CitationGenerator(snippet_max_length=1200)
    )
    tool = QueryKnowledgeHubTool(
        settings=settings,
        response_builder=response_builder,
    )
    llm = LLMFactory.create(settings)
    evaluator = RagasEvaluator(settings=settings, metrics=METRICS)
    previous_by_query: dict[str, dict[str, Any]] = {}
    if args.resume and OUTPUT_PATH.exists():
        previous = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        previous_by_query = {
            item["query"]: item for item in previous.get("results", [])
        }

    report: dict[str, Any] = {
        "status": "running",
        "collection": COLLECTION,
        "selection": "first golden rag_benchmark_answer per distinct source",
        "limit": args.limit,
        "prompt": answer_prompt.metadata(),
        "results": [],
    }
    save_report(report)

    for index, case in enumerate(cases, start=1):
        query = case["query"]
        previous_item = previous_by_query.get(query)
        should_keep_previous = (
            previous_item
            and previous_item.get("status") == "rejected_before_llm"
        ) or (
            previous_item
            and previous_item.get("status") == "complete"
            and not args.rescore
        )
        if should_keep_previous:
            report["results"].append(previous_item)
            report["aggregate"] = aggregate(report["results"])
            save_report(report)
            print(f"[{index}/{args.limit}] resumed: {query[:72]}")
            continue
        expected_sources = case.get("expected_sources", [])
        item: dict[str, Any] = {
            "index": index,
            "query": query,
            "reference_answer": case.get("reference_answer"),
            "expected_sources": expected_sources,
        }
        try:
            retrieval = await tool.execute(
                query=query,
                top_k=5,
                collection=COLLECTION,
            )
            retrieved_sources = [citation.source for citation in retrieval.citations]
            item.update({
                "retrieval_metadata": retrieval.metadata,
                "retrieved_sources": retrieved_sources,
                "expected_source_hit": any(
                    Path(actual).name == Path(expected).name
                    for actual in retrieved_sources
                    for expected in expected_sources
                ),
            })

            if not retrieval.metadata.get("answerable", False):
                item.update({
                    "status": "rejected_before_llm",
                    "answer": retrieval.content,
                })
            else:
                if previous_item and previous_item.get("status") == "complete":
                    answer = previous_item["answer"]
                    answer_model = previous_item.get("answer_model")
                else:
                    completion = await asyncio.to_thread(
                        llm.chat,
                        [
                            Message(role="system", content=answer_prompt.system_prompt),
                            Message(
                                role="user",
                                content=answer_prompt.build_user_prompt(
                                    query=query,
                                    evidence=retrieval.content,
                                ),
                            ),
                        ],
                    )
                    answer = completion.content
                    answer_model = completion.model
                contexts = [
                    citation.text_snippet
                    for citation in retrieval.citations
                    if citation.text_snippet.strip()
                ]
                ragas_metrics = await asyncio.to_thread(
                    evaluator.evaluate,
                    query=query,
                    retrieved_chunks=contexts,
                    generated_answer=answer,
                    ground_truth=case.get("reference_answer"),
                )
                item.update({
                    "status": "complete",
                    "answer": answer,
                    "answer_model": answer_model,
                    "ragas_context_count": len(contexts),
                    "ragas_metrics": ragas_metrics,
                })
        except Exception as exc:
            item.update({
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

        report["results"].append(item)
        report["aggregate"] = aggregate(report["results"])
        save_report(report)
        print(
            f"[{index}/{args.limit}] {item['status']}: {query[:72]} "
            f"metrics={item.get('ragas_metrics')}"
        )

    report["status"] = "complete"
    report["aggregate"] = aggregate(report["results"])
    save_report(report)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    return 0 if report["aggregate"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
