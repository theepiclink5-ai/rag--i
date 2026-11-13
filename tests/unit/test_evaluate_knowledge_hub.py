from types import SimpleNamespace

import pytest

from src.core.response import MCPToolResponse
from src.mcp_server.protocol_handler import ProtocolHandler, _register_default_tools
from src.mcp_server.tools import evaluate_knowledge_hub as evaluate_module
from src.mcp_server.tools.evaluate_knowledge_hub import EvaluateKnowledgeHubTool


class FakeAnswerTool:
    def __init__(self) -> None:
        self.calls = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        trace = kwargs["trace"]
        trace.metadata.update({
            "final_results": [
                {
                    "chunk_id": "chunk-1",
                    "text": "NetApp common stock trades under NTAP.",
                    "score": 0.96,
                }
            ],
            "answer": "NetApp's trading symbol is NTAP. [1]",
        })
        return MCPToolResponse(
            content="NetApp's trading symbol is NTAP. [1]",
            metadata={"status": "answered", "collection": "docs"},
        )


class MissingContextAnswerTool(FakeAnswerTool):
    async def execute(self, **kwargs):
        response = await super().execute(**kwargs)
        kwargs["trace"].metadata["final_results"] = []
        return response


class FakeRagasEvaluator:
    def __init__(self, settings, metrics):
        self.metrics = metrics

    def evaluate(self, **kwargs):
        assert kwargs["retrieved_chunks"][0]["chunk_id"] == "chunk-1"
        assert kwargs["generated_answer"].endswith("[1]")
        return {metric: 0.9 for metric in self.metrics}


class FailingRagasEvaluator(FakeRagasEvaluator):
    def evaluate(self, **kwargs):
        raise RuntimeError("judge unavailable")


class MemoryCollector:
    def __init__(self) -> None:
        self.items = []

    def collect(self, trace=None, **kwargs):
        if trace is not None:
            trace.finish()
            self.items.append(trace)
        else:
            self.items.append(kwargs)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_evaluation_uses_same_answer_trace_without_second_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(evaluate_module, "RagasEvaluator", FakeRagasEvaluator)
    answer_tool = FakeAnswerTool()
    traces = MemoryCollector()
    evaluations = MemoryCollector()
    tool = EvaluateKnowledgeHubTool(
        settings=SimpleNamespace(),
        answer_tool=answer_tool,
        trace_collector=traces,
        evaluation_collector=evaluations,
    )

    response = await tool.execute("What is the symbol?", collection="docs")

    assert len(answer_tool.calls) == 1
    assert answer_tool.calls[0]["collect_trace"] is False
    assert answer_tool.calls[0]["mode"] == "evaluation"
    assert response.metadata["evaluation_status"] == "completed"
    assert response.metadata["evaluation_metrics"]["faithfulness"] == 0.9
    assert traces.items[0].metadata["evaluation_status"] == "completed"
    assert evaluations.items[0]["trace_id"] == traces.items[0].trace_id


@pytest.mark.anyio
async def test_evaluation_failure_preserves_generated_answer(monkeypatch) -> None:
    monkeypatch.setattr(evaluate_module, "RagasEvaluator", FailingRagasEvaluator)
    tool = EvaluateKnowledgeHubTool(
        settings=SimpleNamespace(),
        answer_tool=FakeAnswerTool(),
        trace_collector=MemoryCollector(),
        evaluation_collector=MemoryCollector(),
    )

    response = await tool.execute("What is the symbol?", collection="docs")

    assert response.content == "NetApp's trading symbol is NTAP. [1]"
    assert response.metadata["status"] == "answered"
    assert response.metadata["evaluation_status"] == "error"


@pytest.mark.anyio
async def test_missing_context_is_an_evaluation_error_not_an_answer_error() -> None:
    tool = EvaluateKnowledgeHubTool(
        settings=SimpleNamespace(),
        answer_tool=MissingContextAnswerTool(),
        trace_collector=MemoryCollector(),
        evaluation_collector=MemoryCollector(),
    )

    response = await tool.execute("What is the symbol?", collection="docs")

    assert response.content == "NetApp's trading symbol is NTAP. [1]"
    assert response.metadata["status"] == "answered"
    assert response.metadata["evaluation_status"] == "error"
    assert "no final contexts" in response.metadata["evaluation_error"]


def test_default_tools_include_explicit_evaluation_route() -> None:
    handler = ProtocolHandler("test", "1.0")

    _register_default_tools(handler)

    names = {tool.name for tool in handler.get_tool_schemas()}
    assert "evaluate_knowledge_hub" in names
