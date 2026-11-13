"""
MCP Server Tools.

This package contains the MCP tool definitions exposed to clients.
"""

from src.mcp_server.tools.query_knowledge_hub import (
    TOOL_NAME as QUERY_KNOWLEDGE_HUB_NAME,
    TOOL_DESCRIPTION as QUERY_KNOWLEDGE_HUB_DESCRIPTION,
    TOOL_INPUT_SCHEMA as QUERY_KNOWLEDGE_HUB_SCHEMA,
    QueryKnowledgeHubTool,
    query_knowledge_hub_handler,
    register_tool as register_query_knowledge_hub,
)
from src.mcp_server.tools.ask_knowledge_hub import (
    TOOL_NAME as ASK_KNOWLEDGE_HUB_NAME,
    TOOL_DESCRIPTION as ASK_KNOWLEDGE_HUB_DESCRIPTION,
    TOOL_INPUT_SCHEMA as ASK_KNOWLEDGE_HUB_SCHEMA,
    AskKnowledgeHubTool,
    ask_knowledge_hub_handler,
    register_tool as register_ask_knowledge_hub,
)
from src.mcp_server.tools.evaluate_knowledge_hub import (
    TOOL_NAME as EVALUATE_KNOWLEDGE_HUB_NAME,
    TOOL_DESCRIPTION as EVALUATE_KNOWLEDGE_HUB_DESCRIPTION,
    TOOL_INPUT_SCHEMA as EVALUATE_KNOWLEDGE_HUB_SCHEMA,
    EvaluateKnowledgeHubTool,
    evaluate_knowledge_hub_handler,
    register_tool as register_evaluate_knowledge_hub,
)

__all__ = [
    "QUERY_KNOWLEDGE_HUB_NAME",
    "QUERY_KNOWLEDGE_HUB_DESCRIPTION",
    "QUERY_KNOWLEDGE_HUB_SCHEMA",
    "QueryKnowledgeHubTool",
    "query_knowledge_hub_handler",
    "register_query_knowledge_hub",
    "ASK_KNOWLEDGE_HUB_NAME",
    "ASK_KNOWLEDGE_HUB_DESCRIPTION",
    "ASK_KNOWLEDGE_HUB_SCHEMA",
    "AskKnowledgeHubTool",
    "ask_knowledge_hub_handler",
    "register_ask_knowledge_hub",
    "EVALUATE_KNOWLEDGE_HUB_NAME",
    "EVALUATE_KNOWLEDGE_HUB_DESCRIPTION",
    "EVALUATE_KNOWLEDGE_HUB_SCHEMA",
    "EvaluateKnowledgeHubTool",
    "evaluate_knowledge_hub_handler",
    "register_evaluate_knowledge_hub",
]
