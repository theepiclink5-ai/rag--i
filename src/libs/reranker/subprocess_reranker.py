"""Minimal subprocess wrapper for local Cross-Encoder reranking."""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class SubprocessRerankError(RuntimeError):
    """Raised when the reranker worker cannot serve a request."""


def _worker_main(connection: Any, config: Dict[str, Any]) -> None:
    """Load the model in the child process and serve requests over a Pipe."""
    try:
        from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker

        settings = SimpleNamespace(rerank=SimpleNamespace(**config))
        reranker = CrossEncoderReranker(settings=settings)
        connection.send({"type": "ready"})

        while True:
            request = connection.recv()
            if request.get("type") == "stop":
                break
            if request.get("type") != "rerank":
                continue

            try:
                results = reranker.rerank(
                    query=request["query"],
                    candidates=request["candidates"],
                    top_k=request["top_k"],
                )
                connection.send({"type": "result", "results": results})
            except Exception as exc:
                connection.send({
                    "type": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    except BaseException as exc:
        try:
            connection.send({
                "type": "startup_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
        except BaseException:
            pass
    finally:
        connection.close()


class SubprocessCrossEncoderReranker(BaseReranker):
    """Cross-Encoder proxy whose model lives in one long-running child process."""

    def __init__(self, settings: Any, **kwargs: Any) -> None:
        self.settings = settings
        rerank = settings.rerank
        self.timeout = float(getattr(rerank, "timeout", 15.0))
        self.startup_timeout = float(getattr(rerank, "startup_timeout", 60.0))
        self._lock = threading.Lock()
        self._process: Optional[multiprocessing.Process] = None
        self._connection: Any = None
        self._config = {
            "model": rerank.model,
            "timeout": self.timeout,
            "batch_size": int(getattr(rerank, "batch_size", 16)),
            "max_length": int(getattr(rerank, "max_length", 512)),
            "cache_dir": str(getattr(rerank, "cache_dir", "./data/models/huggingface/hub")),
            "local_files_only": bool(getattr(rerank, "local_files_only", False)),
        }
        self._start_worker()
        atexit.register(self.close)

    def _start_worker(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_worker_main,
            args=(child, self._config),
            name="cross-encoder-worker",
            daemon=True,
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent

        if not parent.poll(self.startup_timeout):
            self._terminate_worker()
            raise SubprocessRerankError(
                f"Cross-Encoder worker startup timed out after {self.startup_timeout:.1f}s"
            )
        try:
            message = parent.recv()
        except (EOFError, OSError) as exc:
            self._terminate_worker()
            raise SubprocessRerankError("Cross-Encoder worker exited during startup") from exc
        if message.get("type") != "ready":
            self._terminate_worker()
            raise SubprocessRerankError(
                f"Cross-Encoder worker failed to start: {message.get('error', message)}"
            )
        logger.info("Cross-Encoder subprocess is ready (pid=%s)", process.pid)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)
        top_k = kwargs.get("top_k", len(candidates))
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        with self._lock:
            if self._process is None or not self._process.is_alive():
                raise SubprocessRerankError("Cross-Encoder worker is not running")
            try:
                self._connection.send({
                    "type": "rerank",
                    "query": query,
                    "candidates": candidates,
                    "top_k": top_k,
                })
                if not self._connection.poll(self.timeout):
                    self._terminate_worker()
                    raise SubprocessRerankError(
                        f"Cross-Encoder worker timed out after {self.timeout:.1f}s"
                    )
                message = self._connection.recv()
            except (EOFError, OSError, BrokenPipeError) as exc:
                self._terminate_worker()
                raise SubprocessRerankError("Cross-Encoder worker disconnected") from exc

        if message.get("type") == "error":
            raise SubprocessRerankError(message.get("error", "Unknown worker error"))
        if message.get("type") != "result":
            raise SubprocessRerankError(f"Unexpected worker response: {message}")
        return message["results"]

    def _terminate_worker(self) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

    def close(self) -> None:
        with self._lock:
            if self._connection is not None and self._process is not None:
                try:
                    if self._process.is_alive():
                        self._connection.send({"type": "stop"})
                        self._process.join(timeout=2.0)
                except (OSError, BrokenPipeError):
                    pass
            self._terminate_worker()
