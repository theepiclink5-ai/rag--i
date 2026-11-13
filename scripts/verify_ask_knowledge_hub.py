"""Run one real end-to-end request through the ask_knowledge_hub tool."""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

from src.mcp_server.tools.ask_knowledge_hub import AskKnowledgeHubTool


async def main() -> int:
    response = await AskKnowledgeHubTool().execute(
        question="What is the trading symbol for NetApp, Inc. common stock?",
        top_k=5,
        collection="pdf_mixed_100",
    )
    print(json.dumps({
        "answer": response.content,
        "citations": [citation.to_dict() for citation in response.citations],
        "metadata": response.metadata,
    }, ensure_ascii=False, indent=2))
    return 0 if response.metadata.get("status") == "answered" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
