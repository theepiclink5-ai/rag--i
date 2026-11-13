import json
from types import SimpleNamespace

import httpx
import pytest

from src.libs.reranker.jina_reranker import JinaRerankError, JinaReranker


def make_settings(**overrides):
    values = {
        "model": "jina-reranker-v2-base-multilingual",
        "top_k": 2,
        "timeout": 5.0,
        "api_key": "test-key",
        "base_url": "https://api.jina.ai/v1",
        "max_retries": 0,
        "retry_base_delay": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(rerank=SimpleNamespace(**values))


def test_rerank_maps_jina_indexes_to_original_candidates():
    captured = {}

    def handler(request):
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reranker = JinaReranker(make_settings(), http_client=client)
    candidates = [
        {"id": "a", "text": "first", "score": 0.03},
        {"id": "b", "text": "second", "score": 0.02},
        {"id": "c", "text": "third", "score": 0.01},
    ]

    result = reranker.rerank("question", candidates, top_k=2)

    assert [item["id"] for item in result] == ["b", "a"]
    assert [item["rerank_score"] for item in result] == [0.91, 0.42]
    assert all(item["rerank_provider"] == "jina" for item in result)
    assert captured["authorization"] == "Bearer test-key"
    assert json.loads(captured["payload"])["top_n"] == 2
    assert "rerank_score" not in candidates[0]


def test_empty_candidates_do_not_call_api():
    def handler(request):
        raise AssertionError("API should not be called")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reranker = JinaReranker(make_settings(), http_client=client)

    assert reranker.rerank("question", []) == []


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Jina API key"):
        JinaReranker(make_settings(api_key=None))


def test_malformed_response_raises_provider_error():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"results": [{"index": 9}]})
        )
    )
    reranker = JinaReranker(make_settings(), http_client=client)
