"""Streamable HTTP transport for the Modular RAG MCP server."""

from __future__ import annotations
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


import argparse
import hmac
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Sequence

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.mcp_server.protocol_handler import create_mcp_server
from src.mcp_server.server import (
    SERVER_NAME,
    SERVER_VERSION,
    _preload_heavy_imports,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MCP_PATH = "/mcp"
TOKEN_ENV_NAME = "MCP_HTTP_BEARER_TOKEN"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class StreamableHTTPApp:
    """Expose the SDK session manager as an ASGI application."""

    def __init__(
        self,
        manager: StreamableHTTPSessionManager,
        bearer_token: Optional[str] = None,
    ) -> None:
        self.manager = manager
        self.bearer_token = bearer_token

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if self.bearer_token and not self._is_authorized(scope):
            response = JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.manager.handle_request(scope, receive, send)

    def _is_authorized(self, scope: Any) -> bool:
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        supplied = headers.get(b"authorization", b"").decode("latin-1")
        expected = f"Bearer {self.bearer_token}"
        return hmac.compare_digest(supplied, expected)


async def health(_: Request) -> JSONResponse:
    """Return a small readiness response for local and cloud probes."""
    return JSONResponse(
        {
            "status": "ok",
            "service": SERVER_NAME,
            "version": SERVER_VERSION,
            "transport": "streamable-http",
        }
    )


def create_http_app(
    *,
    host: str = DEFAULT_HOST,
    mcp_path: str = DEFAULT_MCP_PATH,
    bearer_token: Optional[str] = None,
    stateless: bool = False,
    allowed_hosts: Optional[Sequence[str]] = None,
    allowed_origins: Optional[Sequence[str]] = None,
) -> Starlette:
    """Build a Starlette app around the existing low-level MCP server."""
    path = "/" + mcp_path.strip("/")
    hosts = list(allowed_hosts or _default_allowed_hosts(host))
    origins = list(allowed_origins or _default_allowed_origins(host))
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )
    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)
    manager = StreamableHTTPSessionManager(
        app=server,
        stateless=stateless,
        security_settings=security,
    )
    mcp_app = StreamableHTTPApp(manager, bearer_token=bearer_token)

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route(path, mcp_app, methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lifespan,
    )


def _default_allowed_hosts(host: str) -> list[str]:
    if host in LOOPBACK_HOSTS:
        return ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    return [f"{host}:*"]


def _default_allowed_origins(host: str) -> list[str]:
    if host in LOOPBACK_HOSTS:
        return [
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://[::1]:*",
        ]
    return [f"http://{host}:*", f"https://{host}:*"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the Streamable HTTP MCP server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--path", default=DEFAULT_MCP_PATH)
    parser.add_argument("--stateless", action="store_true")
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Allowed HTTP Host value. Repeat for multiple hosts.",
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        help="Allowed Origin value. Repeat for multiple origins.",
    )
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="Allow a non-loopback listener without a bearer token.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    token = os.getenv(TOKEN_ENV_NAME)
    if args.host not in LOOPBACK_HOSTS and not token and not args.allow_unauthenticated:
        raise SystemExit(
            f"Refusing unauthenticated non-loopback listener. Set {TOKEN_ENV_NAME} "
            "or explicitly pass --allow-unauthenticated for a trusted test network."
        )

    _preload_heavy_imports()
    app = create_http_app(
        host=args.host,
        mcp_path=args.path,
        bearer_token=token,
        stateless=args.stateless,
        allowed_hosts=args.allowed_host or None,
        allowed_origins=args.allowed_origin or None,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
