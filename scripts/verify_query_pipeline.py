"""Verify retrieval, reranking, and evidence gating with two real queries."""

from __future__ import annotations

import asyncio
import argparse
import io
import json
import sys
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool
from src.core.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disable-rerank", action="store_true")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = load_settings()
    if args.disable_rerank:
        settings = replace(
            settings,
            rerank=replace(settings.rerank, enabled=False, provider="none"),
        )
    tool = QueryKnowledgeHubTool(settings=settings)
    report = []
    output_path = _REPO_ROOT / "logs" / "query_pipeline_verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"status": "started", "results": report}, indent=2),
        encoding="utf-8",
    )
    cases = [
        (
            "IN_DOMAIN",
            "How many broadcast zones does BC-AMP-500Z support?",
        ),
        (
            "OUT_OF_DOMAIN",
            "What ingredients are needed to bake a chocolate cake on Mars?",
        ),
    ]
    for label, query in cases:
        try:
            response = await tool.execute(
                query=query,
                top_k=5,
                collection="pdf_mixed_100",
            )
        except BaseException as exc:
            output_path.write_text(
                json.dumps(
                    {
                        "status": f"failed_{label.lower()}",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "results": report,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise
        report.append({
            "label": label,
            "query": query,
            "metadata": response.metadata,
            "content": response.content,
        })
        output_path.write_text(
            json.dumps(
                {"status": f"completed_{label.lower()}", "results": report},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n==={label}===")
        print(json.dumps(response.metadata, ensure_ascii=False, indent=2))
        print(response.content[:1200])
    output_path.write_text(
        json.dumps({"status": "complete", "results": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
