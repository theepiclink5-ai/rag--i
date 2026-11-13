"""Run one real end-to-end answer plus Ragas evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from src.mcp_server.tools.evaluate_knowledge_hub import EvaluateKnowledgeHubTool


async def _run(question: str, collection: str, top_k: int) -> None:
    response = await EvaluateKnowledgeHubTool().execute(
        question=question,
        collection=collection,
        top_k=top_k,
    )
    print(json.dumps({
        "answer": response.content,
        "metadata": response.metadata,
        "citations": [citation.to_dict() for citation in response.citations],
    }, ensure_ascii=False, indent=2))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        default="What is the trading symbol for NetApp, Inc. common stock?",
    )
    parser.add_argument("--collection", default="pdf_mixed_100")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(_run(args.question, args.collection, args.top_k))


if __name__ == "__main__":
    main()
