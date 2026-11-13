__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
"""
Modular RAG MCP Server - Main Entry Point
This is the entry point for the MCP Server. It delegates to the actual
server implementation in src.mcp_server.server.
"""
from src.mcp_server.server import main

if __name__ == "__main__":
    sys.exit(main())
