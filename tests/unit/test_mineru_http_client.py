"""Tests for MinerU HTTP client using a fake transport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from src.libs.loader.pdf_parsers import (
    HttpResponse,
    MinerUHttpClient,
    MinerUParseError,
    MinerUParseOptions,
)


class FakeTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers or {},
                "json_body": json_body,
                "data": data,
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise AssertionError("No fake response configured")
        return self.responses.pop(0)


def _json_response(payload: dict[str, Any], status_code: int = 200) -> HttpResponse:
    return HttpResponse(
        status_code=status_code,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def test_parse_pdf_runs_mineru_batch_flow(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    transport = FakeTransport(
        [
            _json_response(
                {
                    "code": 0,
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": ["https://upload.example/doc.pdf"],
                    },
                }
            ),
            HttpResponse(status_code=200, body=b""),
            _json_response(
                {
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "file_name": "doc.pdf",
                                "state": "done",
                                "full_zip_url": "https://download.example/full.zip",
                            }
                        ]
                    },
                }
            ),
            HttpResponse(status_code=200, body=b"zip-bytes"),
        ]
    )
    client = MinerUHttpClient(
        api_key="test-key",
        transport=transport,
        sleep_func=lambda seconds: None,
    )

    artifact = client.parse_pdf(pdf, MinerUParseOptions(timeout_seconds=30))

    assert artifact.full_zip_bytes == b"zip-bytes"
    assert artifact.metadata["mineru_batch_id"] == "batch-1"
    assert artifact.metadata["mineru_file_name"] == "doc.pdf"
    assert [call["method"] for call in transport.calls] == ["POST", "PUT", "GET", "GET"]
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert transport.calls[0]["json_body"]["files"][0]["name"] == "doc.pdf"
    assert transport.calls[1]["data"] == b"%PDF-1.4 test"


def test_parse_pdf_polls_running_state(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"pdf")
    sleeps: list[int] = []
    transport = FakeTransport(
        [
            _json_response(
                {
                    "code": 0,
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": ["https://upload.example/doc.pdf"],
                    },
                }
            ),
            HttpResponse(status_code=200, body=b""),
            _json_response(
                {
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {"file_name": "doc.pdf", "state": "running"}
                        ]
                    },
                }
            ),
            _json_response(
                {
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "file_name": "doc.pdf",
                                "state": "done",
                                "full_zip_url": "https://download.example/full.zip",
                            }
                        ]
                    },
                }
            ),
            HttpResponse(status_code=200, body=b"zip"),
        ]
    )
    client = MinerUHttpClient(
        api_key="test-key",
        transport=transport,
        sleep_func=lambda seconds: sleeps.append(seconds),
    )

    artifact = client.parse_pdf(
        pdf,
        MinerUParseOptions(timeout_seconds=30, poll_interval_seconds=3),
    )

    assert artifact.full_zip_bytes == b"zip"
    assert sleeps == [3]


def test_upload_url_api_error_raises(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"pdf")
    transport = FakeTransport([_json_response({"code": 1001, "msg": "bad key"})])
    client = MinerUHttpClient(api_key="test-key", transport=transport)

    with pytest.raises(MinerUParseError, match="bad key"):
        client.parse_pdf(pdf, MinerUParseOptions())


def test_failed_task_raises(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"pdf")
    transport = FakeTransport(
        [
            _json_response(
                {
                    "code": 0,
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": ["https://upload.example/doc.pdf"],
                    },
                }
            ),
            HttpResponse(status_code=200, body=b""),
            _json_response(
                {
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "file_name": "doc.pdf",
                                "state": "failed",
                                "err_msg": "parse failed",
                            }
                        ]
                    },
                }
            ),
        ]
    )
    client = MinerUHttpClient(api_key="test-key", transport=transport)

    with pytest.raises(MinerUParseError, match="parse failed"):
        client.parse_pdf(pdf, MinerUParseOptions())


def test_missing_api_key_raises() -> None:
    with pytest.raises(MinerUParseError, match="API key is required"):
        MinerUHttpClient(api_key="")
