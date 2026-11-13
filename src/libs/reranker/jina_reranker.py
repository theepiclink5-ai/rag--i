"""Jina AI reranker provider."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

from src.libs.reranker.base_reranker import BaseReranker


class JinaRerankError(RuntimeError):
    """Raised when Jina cannot produce a valid reranking result."""


class JinaReranker(BaseReranker):
    """Rerank candidates through Jina's hosted rerank API."""

    DEFAULT_BASE_URL = "https://api.jina.ai/v1"

    def __init__(self, settings: Any, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, http_client: Any = None,
                 **kwargs: Any) -> None:
        config = settings.rerank
        self.model = kwargs.get("model") or config.model
        self.base_url = (base_url or getattr(config, "base_url", None)
                         or os.environ.get("JINA_API_BASE")
                         or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = (api_key or getattr(config, "api_key", None)
                        or os.environ.get("JINA_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "Jina API key not provided. Set rerank.api_key, "
                "JINA_API_KEY, or pass api_key explicitly."
            )
        self.timeout = float(kwargs.get("timeout", config.timeout))
        self.max_retries = int(kwargs.get(
            "max_retries", getattr(config, "max_retries", 2)
        ))
        self.retry_base_delay = float(kwargs.get(
            "retry_base_delay", getattr(config, "retry_base_delay", 0.5)
        ))
        self.default_top_k = int(config.top_k)
        self._client = http_client

    def rerank(self, query: str, candidates: List[Dict[str, Any]],
               trace: Any = None, **kwargs: Any) -> List[Dict[str, Any]]:
        del trace
        if not candidates:
            return []
        top_k = max(1, min(
            int(kwargs.get("top_k", self.default_top_k)), len(candidates)
        ))
        payload = {
            "model": self.model,
            "query": query,
            "documents": [str(item.get("text", "")) for item in candidates],
            "top_n": top_k,
            "return_documents": False,
        }
        if self._client is not None:
            data = self._post_with_retries(self._client, payload)
        else:
            with httpx.Client(timeout=self.timeout) as client:
                data = self._post_with_retries(client, payload)
        results = data.get("results")
        if not isinstance(results, list):
            raise JinaRerankError("Jina response does not contain a results list")

        reranked: List[Dict[str, Any]] = []
        seen_indexes = set()
        for item in results:
            if not isinstance(item, dict):
                raise JinaRerankError("Jina returned a malformed result")
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                raise JinaRerankError(f"Jina returned an invalid index: {index!r}")
            if index in seen_indexes or not isinstance(score, (int, float)):
                raise JinaRerankError(
                    "Jina returned duplicate indexes or an invalid score"
                )
            seen_indexes.add(index)
            candidate = dict(candidates[index])
            candidate["rerank_score"] = float(score)
            candidate["rerank_provider"] = "jina"
            reranked.append(candidate)
        if not reranked:
            raise JinaRerankError("Jina returned no reranking results")
        return reranked[:top_k]

    def _post_with_retries(self, client: Any,
                           payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.post(
                    f"{self.base_url}/rerank", headers=headers, json=payload
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise JinaRerankError("Jina returned a non-object response")
                return data
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                if not (status == 429 or 500 <= status < 600):
                    break
                if attempt >= self.max_retries:
                    break
                retry_after = exc.response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else self.retry_base_delay * (2 ** attempt)
                except ValueError:
                    delay = self.retry_base_delay * (2 ** attempt)
                time.sleep(delay)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_base_delay * (2 ** attempt))
            except (ValueError, JinaRerankError) as exc:
                last_error = exc
                break
        raise JinaRerankError(
            f"Jina rerank request failed: {last_error}"
        ) from last_error
