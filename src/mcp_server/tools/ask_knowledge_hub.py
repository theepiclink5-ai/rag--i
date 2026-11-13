"""MCP tool that runs the complete grounded RAG answer pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from mcp import types

from src.core.response import MCPToolResponse, RagAnswerPrompt
from src.core.settings import Settings, load_settings
from src.core.trace import TraceCollector, TraceContext
from src.libs.llm import BaseLLM, LLMFactory, Message
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool

logger = logging.getLogger(__name__)

TOOL_NAME = "ask_knowledge_hub"
TOOL_DESCRIPTION = """Answer a question using the knowledge base end to end.

This tool performs hybrid retrieval, Cross-Encoder reranking, evidence checks,
and grounded LLM generation inside the RAG service. It returns a final answer
with citations. Use query_knowledge_hub instead when you only need source text
for the calling agent to reason over itself.
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The question to answer from the knowledge base.",
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
    },
    "required": ["question"],
}


class AskKnowledgeHubTool:
    """Coordinate retrieval and grounded answer generation as one trace."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        retrieval_tool: Optional[QueryKnowledgeHubTool] = None,
        llm: Optional[BaseLLM] = None,
        answer_prompt: Optional[RagAnswerPrompt] = None,
        trace_collector: Optional[TraceCollector] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.retrieval_tool = retrieval_tool or QueryKnowledgeHubTool(
            settings=self.settings
        )
        self._llm = llm
        self.answer_prompt = answer_prompt or RagAnswerPrompt.load()
        self.trace_collector = trace_collector or TraceCollector()

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = LLMFactory.create(self.settings)
        return self._llm

    async def execute(
        self,
        question: str,
        top_k: int = 5,
        collection: Optional[str] = None,
        *,
        trace: Optional[TraceContext] = None,
        collect_trace: bool = True,
        mode: str = "answer",
        source: str = "mcp",
    ) -> MCPToolResponse:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        trace = trace or TraceContext(trace_type="query")
        trace.metadata.update({
            "query": question[:200],
            "mode": mode,
            "source": source,
        })

        try:
            retrieval = await self.retrieval_tool.execute(
                query=question,
                top_k=top_k,
                collection=collection,
                trace=trace,
                collect_trace=False,
            )
            retrieval.metadata.update({"mode": mode, "trace_id": trace.trace_id})

            if "error" in retrieval.metadata:
                answer_ready_ms = trace.elapsed_ms()
                trace.metadata.update({
                    "status": "error",
                    "error": retrieval.metadata["error"],
                    "answer_ready_elapsed_ms": round(answer_ready_ms, 2),
                })
                retrieval.metadata.update({
                    "status": "error",
                    "answer_ready_elapsed_ms": round(answer_ready_ms, 2),
                })
                return retrieval

            if not retrieval.metadata.get("answerable", False):
                answer_ready_ms = trace.elapsed_ms()
                retrieval.metadata["status"] = "insufficient_evidence"
                retrieval.metadata["answer_ready_elapsed_ms"] = round(
                    answer_ready_ms, 2
                )
                trace.metadata.update({
                    "status": "insufficient_evidence",
                    "answer": retrieval.content,
                    "citations": [],
                    "answer_ready_elapsed_ms": round(answer_ready_ms, 2),
                    **self.answer_prompt.metadata(),
                })
                return retrieval

            messages = [
                Message(role="system", content=self.answer_prompt.system_prompt),
                Message(
                    role="user",
                    content=self.answer_prompt.build_user_prompt(
                        query=question,
                        evidence=retrieval.content,
                    ),
                ),
            ]
            started = time.monotonic()
            completion = await asyncio.to_thread(self.llm.chat, messages, trace)
            llm_elapsed_ms = (time.monotonic() - started) * 1000.0
            trace.record_stage(
                "llm_generation",
                {
                    "provider": self.settings.llm.provider,
                    "model": completion.model,
                    "prompt_name": self.answer_prompt.name,
                    "prompt_version": self.answer_prompt.version,
                    "usage": completion.usage or {},
                },
                elapsed_ms=llm_elapsed_ms,
            )
            answer_ready_ms = trace.elapsed_ms()

            metadata = dict(retrieval.metadata)
            metadata.update({
                "mode": mode,
                "status": "answered",
                "trace_id": trace.trace_id,
                "model": completion.model,
                "usage": completion.usage or {},
                "answer_ready_elapsed_ms": round(answer_ready_ms, 2),
                **self.answer_prompt.metadata(),
            })
            trace.metadata.update({
                "status": "answered",
                "answer": completion.content,
                "citations": [citation.to_dict() for citation in retrieval.citations],
                "model": completion.model,
                "usage": completion.usage or {},
                "answer_ready_elapsed_ms": round(answer_ready_ms, 2),
                **self.answer_prompt.metadata(),
            })
            return MCPToolResponse(
                content=completion.content,
                citations=retrieval.citations,
                metadata=metadata,
                is_empty=False,
                image_contents=retrieval.image_contents,
            )
        except Exception as exc:
            answer_ready_ms = trace.elapsed_ms()
            trace.metadata.update({
                "status": "error",
                "error": str(exc),
                "answer_ready_elapsed_ms": round(answer_ready_ms, 2),
            })
            logger.exception("ask_knowledge_hub failed: %s", exc)
            return MCPToolResponse(
                content="The knowledge-base answer could not be generated.",
                metadata={
                    "mode": mode,
                    "status": "error",
                    "trace_id": trace.trace_id,
                    "error": str(exc),
                    "answer_ready_elapsed_ms": round(answer_ready_ms, 2),
                },
                is_empty=True,
            )
        finally:
            if collect_trace:
                self.trace_collector.collect(trace)


_tool_instance: Optional[AskKnowledgeHubTool] = None


def get_tool_instance(settings: Optional[Settings] = None) -> AskKnowledgeHubTool:
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = AskKnowledgeHubTool(settings=settings)
    return _tool_instance


async def ask_knowledge_hub_handler(
    question: str,
    top_k: int = 5,
    collection: Optional[str] = None,
) -> types.CallToolResult:
    try:
        response = await get_tool_instance().execute(question, top_k, collection)
        return types.CallToolResult(
            content=response.to_mcp_content(),
            isError=response.metadata.get("status") == "error",
        )
    except ValueError as exc:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Invalid parameters: {exc}")],
            isError=True,
        )
    except Exception:
        logger.exception("ask_knowledge_hub handler error")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="Internal answer error")],
            isError=True,
        )


def register_tool(protocol_handler: Any) -> None:
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=ask_knowledge_hub_handler,
    )
    logger.info("Registered MCP tool: %s", TOOL_NAME)
