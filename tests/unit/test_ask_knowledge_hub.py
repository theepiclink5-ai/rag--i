from types import SimpleNamespace

import pytest

from src.core.response import Citation, MCPToolResponse, RagAnswerPrompt
from src.libs.llm import ChatResponse
from src.mcp_server.protocol_handler import ProtocolHandler, _register_default_tools
from src.mcp_server.tools.ask_knowledge_hub import AskKnowledgeHubTool


class FakeRetrievalTool:
    def __init__(self, response: MCPToolResponse) -> None:
        self.response = response
        self.calls = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["trace"].metadata["final_results"] = [
            {
                "chunk_id": "chunk-1",
                "score": 0.91,
                "text": "NetApp common stock trades under NTAP.",
                "source": "netapp.pdf",
            }
        ]
        return self.response


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    def chat(self, messages, trace=None):
        self.calls.append((messages, trace))
        return ChatResponse(
            content="NetApp's trading symbol is NTAP. [1]",
            model="fake-model",
            usage={"total_tokens": 42},
        )


class MemoryTraceCollector:
    def __init__(self) -> None:
        self.traces = []

    def collect(self, trace) -> None:
        trace.finish()
        self.traces.append(trace)


def make_prompt() -> RagAnswerPrompt:
    return RagAnswerPrompt(
        name="rag_answer",
        version="test",
        system_prompt="Use only supplied evidence.",
        user_template="Question: {query}\nEvidence: {evidence}",
        evidence_preamble="Evidence follows.",
        insufficient_evidence_template="Insufficient evidence for {query}",
    )


def make_settings():
    return SimpleNamespace(llm=SimpleNamespace(provider="fake"))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_complete_answer_uses_retrieval_llm_and_one_shared_trace() -> None:
    citation = Citation(
        index=1,
        chunk_id="chunk-1",
        source="netapp.pdf",
        score=0.91,
        text_snippet="NetApp common stock trades under NTAP.",
    )
    retrieval = FakeRetrievalTool(
        MCPToolResponse(
            content="[1] NetApp common stock trades under NTAP.",
            citations=[citation],
            metadata={"answerable": True, "evidence_status": "supported"},
        )
    )
    llm = FakeLLM()
    collector = MemoryTraceCollector()
    tool = AskKnowledgeHubTool(
        settings=make_settings(),
        retrieval_tool=retrieval,
        llm=llm,
        answer_prompt=make_prompt(),
        trace_collector=collector,
    )

    response = await tool.execute("What is the symbol?", collection="docs")

    assert response.content == "NetApp's trading symbol is NTAP. [1]"
    assert response.metadata["mode"] == "answer"
    assert response.metadata["status"] == "answered"
    assert response.metadata["trace_id"] == collector.traces[0].trace_id
    assert retrieval.calls[0]["collect_trace"] is False
    assert llm.calls[0][0][1].content.endswith(
        "Evidence: [1] NetApp common stock trades under NTAP."
    )
    assert collector.traces[0].metadata["answer"] == response.content
    assert collector.traces[0].get_stage_data("llm_generation")["model"] == "fake-model"


@pytest.mark.anyio
async def test_insufficient_evidence_returns_without_calling_llm() -> None:
    retrieval = FakeRetrievalTool(
        MCPToolResponse(
            content="Insufficient evidence.",
            metadata={"answerable": False, "evidence_status": "no_evidence"},
            is_empty=True,
        )
    )
    llm = FakeLLM()
    collector = MemoryTraceCollector()
    tool = AskKnowledgeHubTool(
        settings=make_settings(),
        retrieval_tool=retrieval,
        llm=llm,
        answer_prompt=make_prompt(),
        trace_collector=collector,
    )

    response = await tool.execute("Unknown question")

    assert response.metadata["status"] == "insufficient_evidence"
    assert response.metadata["mode"] == "answer"
    assert llm.calls == []
    assert collector.traces[0].metadata["status"] == "insufficient_evidence"


@pytest.mark.anyio
async def test_retrieval_error_is_not_reported_as_insufficient_evidence() -> None:
    retrieval = FakeRetrievalTool(
        MCPToolResponse(
            content="Retrieval failed.",
            metadata={"error": "database unavailable"},
            is_empty=True,
        )
    )
    llm = FakeLLM()
    collector = MemoryTraceCollector()
    tool = AskKnowledgeHubTool(
        settings=make_settings(),
        retrieval_tool=retrieval,
        llm=llm,
        answer_prompt=make_prompt(),
        trace_collector=collector,
    )

    response = await tool.execute("A valid question")

    assert response.metadata["status"] == "error"
    assert llm.calls == []
    assert collector.traces[0].metadata["status"] == "error"


def test_default_tools_expose_retrieval_and_answer_modes() -> None:
    handler = ProtocolHandler("test", "1.0")

    _register_default_tools(handler)

    names = {tool.name for tool in handler.get_tool_schemas()}
    assert "query_knowledge_hub" in names
    assert "ask_knowledge_hub" in names
