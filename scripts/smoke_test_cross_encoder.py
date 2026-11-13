"""Run a small, deterministic smoke test against the configured reranker model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.core.settings import load_settings
from src.libs.reranker.reranker_factory import RerankerFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="什么是混合检索？",
        help="Query used to rank the built-in sample passages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = [
        {
            "id": "exact",
            "text": "混合检索把 BM25 关键词检索与向量语义检索结合，并使用 RRF 融合结果。",
        },
        {
            "id": "semantic",
            "text": "系统同时考虑词面匹配和语义相似度，再合并两路召回的排名。",
        },
        {
            "id": "irrelevant",
            "text": "今天上海天气晴朗，最高气温二十五度。",
        },
        {
            "id": "english",
            "text": "Hybrid retrieval combines sparse keyword search with dense semantic search.",
        },
    ]

    settings = load_settings()
    started = time.perf_counter()
    reranker = RerankerFactory.create(settings)
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    results = reranker.rerank(args.query, candidates, top_k=len(candidates))
    rerank_seconds = time.perf_counter() - started

    print(f"model={settings.rerank.model}")
    print(f"load_seconds={load_seconds:.2f}")
    print(f"rerank_seconds={rerank_seconds:.2f}")
    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['id']} score={result['rerank_score']:.6f}")
    close = getattr(reranker, "close", None)
    if callable(close):
        close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
