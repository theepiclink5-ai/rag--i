"""Jina Embedding implementation.

This module provides an Embedding provider implementation for Jina AI's
cloud embeddings API.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class JinaEmbeddingError(RuntimeError):
    """Raised when a Jina Embeddings API call fails."""


class JinaEmbedding(BaseEmbedding):
    """Jina Embedding provider implementation."""

    DEFAULT_BASE_URL = "https://api.jina.ai/v1"
    DEFAULT_TIMEOUT = 120.0

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.model = settings.embedding.model
        self.dimensions = getattr(settings.embedding, "dimensions", None)
        self.api_key = (
            api_key
            or getattr(settings.embedding, "api_key", None)
            or os.environ.get("JINA_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Jina API key not provided. Set embedding.api_key, "
                "JINA_API_KEY, or pass api_key explicitly."
            )

        configured_base_url = getattr(settings.embedding, "base_url", None)
        self.base_url = base_url or configured_base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.max_retries = int(kwargs.get("max_retries", 3))
        self.retry_base_delay = float(kwargs.get("retry_base_delay", 2.0))
        self._extra_config = kwargs

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        self.validate_texts(texts)

        try:
            import httpx
        except ImportError as e:
            raise JinaEmbeddingError(
                "httpx library is required for Jina Embedding. "
                "Install with: pip install httpx"
            ) from e

        url = f"{self.base_url.rstrip('/')}/embeddings"
        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                result = self._post_with_retries(client, url, payload)
        except httpx.ConnectError as e:
            raise JinaEmbeddingError(
                f"Failed to connect to Jina Embeddings API at {self.base_url}."
            ) from e
        except httpx.TimeoutException as e:
            raise JinaEmbeddingError("Jina Embeddings API request timed out.") from e
        except httpx.RequestError as e:
            raise JinaEmbeddingError(
                f"Jina Embeddings API request failed: {str(e)}"
            ) from e
        except (KeyError, ValueError, TypeError) as e:
            raise JinaEmbeddingError(
                f"Failed to parse Jina Embeddings API response: {str(e)}"
            ) from e

        try:
            data = result["data"]
            embeddings = [item["embedding"] for item in data]
        except (KeyError, TypeError) as e:
            raise JinaEmbeddingError(
                "Unexpected response format from Jina Embeddings API."
            ) from e

        if len(embeddings) != len(texts):
            raise JinaEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )

        return embeddings

    def get_dimension(self) -> Optional[int]:
        return self.dimensions

    def _post_with_retries(
        self,
        client: Any,
        url: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: httpx.HTTPStatusError | None = None
        for attempt in range(self.max_retries + 1):
            response = client.post(url, json=payload, headers=headers)
            try:
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                retryable = status == 429 or 500 <= status < 600
                if not retryable or attempt >= self.max_retries:
                    raise JinaEmbeddingError(
                        "Jina Embeddings API request failed with "
                        f"status {status}."
                    ) from exc
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = self.retry_base_delay * (2 ** attempt)
                else:
                    delay = self.retry_base_delay * (2 ** attempt)
                time.sleep(delay)

        raise JinaEmbeddingError("Jina Embeddings API request failed.") from last_error
