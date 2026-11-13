"""Run one real question through retrieval, reranking, evidence guard, and LLM."""

from __future__ import annotations

import asyncio
import argparse
import io
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.core.settings import load_settings
from src.core.response.prompt_config import RagAnswerPrompt
from src.libs.llm import LLMFactory, Message
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool
from src.observability.evaluation.ragas_evaluator import RagasEvaluator


QUERY = "What is the trading symbol for NetApp, Inc. common stock?"
COLLECTION = "pdf_mixed_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Score the generated answer with the configured Ragas judge.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = load_settings()
    answer_prompt = RagAnswerPrompt.load()
    tool = QueryKnowledgeHubTool(settings=settings)
    retrieval_response = await tool.execute(
        query=QUERY,
        top_k=5,
        collection=COLLECTION,
    )

    report = {
        "query": QUERY,
        "collection": COLLECTION,
        "retrieval_metadata": retrieval_response.metadata,
        "evidence": retrieval_response.content,
        **answer_prompt.metadata(),
    }

    if not retrieval_response.metadata.get("answerable", False):
        report["status"] = "rejected_before_llm"
        report["answer"] = retrieval_response.content
    else:
        llm = LLMFactory.create(settings)
        completion = await asyncio.to_thread(
            llm.chat,
            [
                Message(
                    role="system",
                    content=answer_prompt.system_prompt,
                ),
                Message(
                    role="user",
                    content=answer_prompt.build_user_prompt(
                        query=QUERY,
                        evidence=retrieval_response.content,
                    ),
                ),
            ],
        )
        report.update({
            "status": "complete",
            "answer": completion.content,
            "model": completion.model,
            "usage": completion.usage,
        })
        if args.with_ragas:
            evaluator = RagasEvaluator(
                settings=settings,
                metrics=[
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                ],
            )
            report["ragas_metrics"] = await asyncio.to_thread(
                evaluator.evaluate,
                query=QUERY,
                retrieved_chunks=[retrieval_response.content],
                generated_answer=completion.content,
            )

    output_path = _REPO_ROOT / "logs" / "rag_answer_verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
