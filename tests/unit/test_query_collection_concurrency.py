import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from src.core.trace import TraceContext
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool


def test_collection_initialization_and_search_are_atomic() -> None:
    tool = QueryKnowledgeHubTool(settings=SimpleNamespace())
    active = {"collection": None}

    def initialize(collection: str) -> None:
        active["collection"] = collection
        time.sleep(0.02)

    def search(query: str, top_k: int, trace: TraceContext):
        return [(query, active["collection"])]

    tool._ensure_initialized = initialize  # type: ignore[method-assign]
    tool._perform_search = search  # type: ignore[method-assign]

    def run(collection: str):
        return tool._search_collection(
            collection=collection,
            query=collection,
            top_k=1,
            trace=TraceContext(),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, "A")
        second = pool.submit(run, "B")

    assert first.result() == [("A", "A")]
    assert second.result() == [("B", "B")]
