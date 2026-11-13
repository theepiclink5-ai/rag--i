"""Unit tests for Jina Embedding provider implementation."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.jina_embedding import JinaEmbedding, JinaEmbeddingError


@pytest.fixture
def mock_settings_jina() -> Any:
    settings = Mock()
    settings.embedding = Mock()
    settings.embedding.provider = "jina"
    settings.embedding.model = "jina-embeddings-v5-text-small"
    settings.embedding.dimensions = 1024
    settings.embedding.api_key = "test-key"
    settings.embedding.base_url = "https://api.jina.ai/v1"
    return settings


class TestJinaEmbedding:
    def test_initialization_default(self, mock_settings_jina: Any) -> None:
        embedding = JinaEmbedding(mock_settings_jina)

        assert embedding.model == "jina-embeddings-v5-text-small"
        assert embedding.dimensions == 1024
        assert embedding.base_url == "https://api.jina.ai/v1"

    @patch("httpx.Client")
    def test_embed_multiple_texts(
        self,
        mock_client_class: Mock,
        mock_settings_jina: Any,
    ) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_client_class.return_value.__enter__.return_value.post.return_value = mock_response

        embedding = JinaEmbedding(mock_settings_jina)
        result = embedding.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch("httpx.Client")
    def test_embed_http_status_error(
        self,
        mock_client_class: Mock,
        mock_settings_jina: Any,
    ) -> None:
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = __import__("httpx").HTTPStatusError(
            "Unauthorized", request=Mock(), response=mock_response
        )
        mock_client_class.return_value.__enter__.return_value.post.return_value = mock_response

        embedding = JinaEmbedding(mock_settings_jina)

        with pytest.raises(JinaEmbeddingError, match="status 401"):
            embedding.embed(["test"])


class TestJinaEmbeddingFactoryIntegration:
    def test_factory_creates_jina_embedding(self, mock_settings_jina: Any) -> None:
        EmbeddingFactory.register_provider("jina", JinaEmbedding)

        embedding = EmbeddingFactory.create(mock_settings_jina)

        assert isinstance(embedding, JinaEmbedding)
