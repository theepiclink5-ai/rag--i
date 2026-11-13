from starlette.testclient import TestClient

from src.mcp_server.http_server import create_http_app


def test_health_endpoint() -> None:
    with TestClient(create_http_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["transport"] == "streamable-http"


def test_mcp_endpoint_requires_configured_bearer_token() -> None:
    with TestClient(create_http_app(bearer_token="secret")) as client:
        response = client.post("/mcp", json={})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_non_loopback_listener_requires_explicit_security(monkeypatch) -> None:
    from src.mcp_server import http_server

    monkeypatch.delenv(http_server.TOKEN_ENV_NAME, raising=False)

    try:
        http_server.main(["--host", "0.0.0.0"])
    except SystemExit as exc:
        assert "Refusing unauthenticated" in str(exc)
    else:
        raise AssertionError("non-loopback listener started without authentication")
