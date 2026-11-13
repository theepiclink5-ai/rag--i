"""Tests for the lightweight subprocess Cross-Encoder proxy."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.libs.reranker.subprocess_reranker import (
    SubprocessCrossEncoderReranker,
    SubprocessRerankError,
)


def _settings():
    return SimpleNamespace(rerank=SimpleNamespace(
        model="local-model",
        timeout=2.0,
        startup_timeout=3.0,
        batch_size=4,
        max_length=128,
        cache_dir="model-cache",
        local_files_only=True,
    ))


def _proxy(message):
    connection = MagicMock()
    connection.poll.return_value = True
    connection.recv.side_effect = [{"type": "ready"}, message]
    process = MagicMock()
    process.is_alive.return_value = True
    context = MagicMock()
    context.Pipe.return_value = (connection, MagicMock())
    context.Process.return_value = process
    with patch("src.libs.reranker.subprocess_reranker.multiprocessing.get_context", return_value=context):
        proxy = SubprocessCrossEncoderReranker(_settings())
    return proxy, connection


def test_rerank_round_trip_returns_worker_results():
    expected = [{"id": "b", "text": "best", "rerank_score": 0.9}]
    proxy, connection = _proxy({"type": "result", "results": expected})

    result = proxy.rerank(
        "query",
        [{"id": "a", "text": "weak"}, {"id": "b", "text": "best"}],
        top_k=1,
    )

    assert result == expected
    assert connection.send.call_args.args[0]["type"] == "rerank"
    assert connection.send.call_args.args[0]["top_k"] == 1


def test_worker_error_is_exposed_to_core_fallback():
    proxy, _ = _proxy({"type": "error", "error": "model failure"})

    with pytest.raises(SubprocessRerankError, match="model failure"):
        proxy.rerank("query", [{"id": "a", "text": "passage"}])
