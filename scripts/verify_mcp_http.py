"""Verify a running Streamable HTTP MCP endpoint."""

from __future__ import annotations

import argparse
import asyncio
import os

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def verify(url: str, token: str | None) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        async with streamable_http_client(url, http_client=client) as streams:
            read_stream, write_stream, _ = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = [tool.name for tool in tools.tools]
                print(f"Connected: {url}")
                print(f"Tools ({len(names)}): {', '.join(names)}")
                result = await session.call_tool("list_collections", {})
                print(f"list_collections: {'ok' if not result.isError else 'error'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an MCP HTTP endpoint")
    parser.add_argument("--url", default="http://127.0.0.1:8000/mcp")
    args = parser.parse_args()
    asyncio.run(verify(args.url, os.getenv("MCP_HTTP_BEARER_TOKEN")))


if __name__ == "__main__":
    main()
