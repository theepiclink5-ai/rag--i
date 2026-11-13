"""HTTP client for MinerU precise parsing API."""

from __future__ import annotations

import json
import http.client
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from src.libs.loader.pdf_parsers.mineru_pdf_parser import (
    MinerUParseArtifact,
    MinerUParseError,
    MinerUParseOptions,
)


@dataclass(frozen=True)
class HttpResponse:
    """Minimal HTTP response used by MinerUHttpClient."""

    status_code: int
    body: bytes
    headers: Dict[str, str] | None = None

    def json(self) -> Dict[str, Any]:
        try:
            return json.loads(self.body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise MinerUParseError("MinerU returned invalid JSON") from exc


class HttpTransport(Protocol):
    """Small transport interface for mockable HTTP calls."""

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        """Execute one HTTP request."""


class UrllibHttpTransport:
    """Default HTTP transport implemented with Python standard-library HTTP."""

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        request_headers = dict(headers or {})
        request_data = data
        if json_body is not None:
            request_data = json.dumps(json_body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        if request_data is not None:
            request_headers.setdefault("Content-Length", str(len(request_data)))

        return self._request_once(
            method=method,
            url=url,
            headers=request_headers,
            data=request_data,
            timeout=timeout,
            redirects_left=3,
        )

    def _request_once(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        data: Optional[bytes],
        timeout: Optional[float],
        redirects_left: int,
    ) -> HttpResponse:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            raise MinerUParseError(f"Unsupported MinerU URL scheme: {parsed.scheme}")

        connection_class = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection = connection_class(parsed.netloc, timeout=timeout)
        try:
            connection.request(
                method.upper(),
                path,
                body=data,
                headers=headers,
            )
            response = connection.getresponse()
            status_code = response.status
            body = response.read()
            response_headers = dict(response.getheaders())
            if status_code in {301, 302, 303, 307, 308} and redirects_left > 0:
                location = response_headers.get("Location") or response_headers.get("location")
                if location:
                    redirect_url = urllib.parse.urljoin(url, location)
                    redirect_method = "GET" if status_code == 303 else method
                    redirect_data = None if redirect_method.upper() == "GET" else data
                    redirect_headers = dict(headers)
                    if redirect_data is None:
                        redirect_headers.pop("Content-Length", None)
                    return self._request_once(
                        method=redirect_method,
                        url=redirect_url,
                        headers=redirect_headers,
                        data=redirect_data,
                        timeout=timeout,
                        redirects_left=redirects_left - 1,
                    )

            return HttpResponse(
                status_code=response.status,
                body=body,
                headers=response_headers,
            )
        except OSError as exc:
            raise MinerUParseError(f"MinerU request failed: {exc}") from exc
        finally:
            connection.close()


class MinerUHttpClient:
    """Client for MinerU precise local-file parsing flow."""

    BASE_URL = "https://mineru.net"
    RUNNING_STATES = {"waiting-file", "pending", "running", "converting"}

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        transport: Optional[HttpTransport] = None,
        sleep_func: Any = time.sleep,
    ) -> None:
        if not api_key:
            raise MinerUParseError("MinerU API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.transport = transport or UrllibHttpTransport()
        self.sleep_func = sleep_func

    def parse_pdf(
        self, file_path: str | Path, options: MinerUParseOptions
    ) -> MinerUParseArtifact:
        path = Path(file_path)
        upload_info = self._request_upload_url(path, options)
        upload_url = upload_info["file_urls"][0]
        self._upload_file(upload_url, path, timeout=options.timeout_seconds)
        result = self._poll_batch_result(
            batch_id=upload_info["batch_id"],
            file_name=path.name,
            timeout_seconds=options.timeout_seconds,
            poll_interval_seconds=options.poll_interval_seconds,
        )
        full_zip_url = result.get("full_zip_url")
        if not full_zip_url:
            raise MinerUParseError("MinerU result did not include full_zip_url")
        zip_bytes = self._download_zip(full_zip_url, timeout=options.timeout_seconds)
        return MinerUParseArtifact(
            full_zip_bytes=zip_bytes,
            metadata={
                "mineru_batch_id": upload_info["batch_id"],
                "mineru_file_name": result.get("file_name", path.name),
                "mineru_state": result.get("state"),
                "mineru_full_zip_url": full_zip_url,
            },
        )

    def _request_upload_url(
        self, file_path: Path, options: MinerUParseOptions
    ) -> Dict[str, Any]:
        payload = {
            "files": [
                {
                    "name": file_path.name,
                    "data_id": file_path.stem[:128],
                    "is_ocr": options.is_ocr,
                }
            ],
            "model_version": options.model_version,
            "language": options.language,
            "enable_table": options.enable_table,
            "enable_formula": options.enable_formula,
        }
        response = self.transport.request(
            method="POST",
            url=f"{self.base_url}/api/v4/file-urls/batch",
            headers=self._json_headers(),
            json_body=payload,
            timeout=options.timeout_seconds,
        )
        data = self._api_json(response, "request MinerU upload URL")
        batch_id = data.get("data", {}).get("batch_id")
        file_urls = data.get("data", {}).get("file_urls")
        if not batch_id or not isinstance(file_urls, list) or not file_urls:
            raise MinerUParseError("MinerU upload URL response is missing batch_id/file_urls")
        return {"batch_id": batch_id, "file_urls": file_urls}

    def _upload_file(self, upload_url: str, file_path: Path, timeout: float) -> None:
        response = self.transport.request(
            method="PUT",
            url=upload_url,
            headers={},
            data=file_path.read_bytes(),
            timeout=timeout,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise MinerUParseError(f"MinerU file upload failed: HTTP {response.status_code}")

    def _poll_batch_result(
        self,
        batch_id: str,
        file_name: str,
        timeout_seconds: int,
        poll_interval_seconds: int,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                response = self.transport.request(
                    method="GET",
                    url=f"{self.base_url}/api/v4/extract-results/batch/{batch_id}",
                    headers=self._json_headers(),
                    timeout=timeout_seconds,
                )
            except MinerUParseError:
                if time.monotonic() >= deadline:
                    raise
                self.sleep_func(poll_interval_seconds)
                continue
            data = self._api_json(response, "poll MinerU batch result")
            result = self._select_result(data.get("data", {}).get("extract_result"), file_name)
            state = str(result.get("state", "")).lower()
            if state == "done":
                return result
            if state == "failed":
                raise MinerUParseError(
                    f"MinerU parsing failed: {result.get('err_msg') or 'unknown error'}"
                )
            if state not in self.RUNNING_STATES:
                raise MinerUParseError(f"Unknown MinerU task state: {state}")
            if time.monotonic() >= deadline:
                raise MinerUParseError("MinerU parsing timed out")
            self.sleep_func(poll_interval_seconds)

    def _download_zip(self, full_zip_url: str, timeout: float) -> bytes:
        last_error: Optional[MinerUParseError] = None
        for attempt in range(3):
            try:
                response = self.transport.request(
                    method="GET",
                    url=full_zip_url,
                    headers={},
                    timeout=timeout,
                )
                break
            except MinerUParseError as exc:
                last_error = exc
                if attempt == 2:
                    raise
                self.sleep_func(1)
        else:
            raise last_error or MinerUParseError("MinerU zip download failed")

        if response.status_code < 200 or response.status_code >= 300:
            raise MinerUParseError(
                f"MinerU zip download failed: HTTP {response.status_code}"
            )
        return response.body

    def _api_json(self, response: HttpResponse, action: str) -> Dict[str, Any]:
        if response.status_code < 200 or response.status_code >= 300:
            raise MinerUParseError(f"Failed to {action}: HTTP {response.status_code}")
        data = response.json()
        if data.get("code") != 0:
            raise MinerUParseError(f"Failed to {action}: {data.get('msg') or data.get('code')}")
        return data

    def _select_result(self, results: Any, file_name: str) -> Dict[str, Any]:
        if not isinstance(results, list) or not results:
            raise MinerUParseError("MinerU batch result is missing extract_result")
        for result in results:
            if isinstance(result, dict) and result.get("file_name") == file_name:
                return result
        first = results[0]
        if not isinstance(first, dict):
            raise MinerUParseError("MinerU batch result contains invalid item")
        return first

    def _json_headers(self) -> Dict[str, str]:
        return {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
