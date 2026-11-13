"""Tests for MinerU parser artifact normalization."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from src.libs.loader.pdf_parsers import (
    MinerUParseArtifact,
    MinerUParseError,
    MinerUParseOptions,
    MinerUPdfParser,
)


class FakeMinerUClient:
    def __init__(self, artifact: MinerUParseArtifact) -> None:
        self.artifact = artifact
        self.calls: list[tuple[Path, MinerUParseOptions]] = []

    def parse_pdf(
        self, file_path: str | Path, options: MinerUParseOptions
    ) -> MinerUParseArtifact:
        self.calls.append((Path(file_path), options))
        return self.artifact


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_parse_from_zip_bytes(tmp_path: Path) -> None:
    artifact = MinerUParseArtifact(
        full_zip_bytes=_zip_bytes(
            {
                "result/full.md": "# Parsed by MinerU\n\nUseful text.",
                "result/doc_content_list.json": json.dumps(
                    [
                        {"type": "table", "text": "| A | B |"},
                        {"type": "image", "path": "images/1.png"},
                    ]
                ),
                "result/doc_layout.json": json.dumps([{"page_id": 1}]),
            }
        ),
        metadata={"task_id": "task-123"},
    )
    client = FakeMinerUClient(artifact)
    parser = MinerUPdfParser(client=client, output_dir=tmp_path / "mineru")

    result = parser.parse(tmp_path / "doc.pdf")

    assert result.parser_provider == "mineru"
    assert result.text.startswith("# Parsed by MinerU")
    assert result.metadata["task_id"] == "task-123"
    assert result.metadata["mineru_has_content_list"] is True
    assert result.metadata["mineru_has_layout"] is True
    assert len(result.tables) == 1
    assert len(result.images) == 1
    assert result.pages == [{"page_id": 1}]
    assert client.calls[0][0] == tmp_path / "doc.pdf"


def test_parse_from_existing_result_dir(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "full.md").write_text("Existing MinerU markdown.", encoding="utf-8")

    parser = MinerUPdfParser(
        client=FakeMinerUClient(MinerUParseArtifact(result_dir=result_dir))
    )

    result = parser.parse(tmp_path / "doc.pdf")

    assert result.text == "Existing MinerU markdown."
    assert result.raw_output_dir == str(result_dir)


def test_missing_full_md_raises(tmp_path: Path) -> None:
    artifact = MinerUParseArtifact(
        full_zip_bytes=_zip_bytes({"result/other.md": "not the required file"})
    )
    parser = MinerUPdfParser(
        client=FakeMinerUClient(artifact), output_dir=tmp_path / "mineru"
    )

    with pytest.raises(MinerUParseError, match="missing required full.md"):
        parser.parse(tmp_path / "doc.pdf")


def test_bad_zip_raises_parse_error(tmp_path: Path) -> None:
    artifact = MinerUParseArtifact(full_zip_bytes=b"not-a-zip")
    parser = MinerUPdfParser(
        client=FakeMinerUClient(artifact), output_dir=tmp_path / "mineru"
    )

    with pytest.raises(MinerUParseError, match="Invalid MinerU zip artifact"):
        parser.parse(tmp_path / "doc.pdf")


def test_missing_client_raises_clear_error(tmp_path: Path) -> None:
    parser = MinerUPdfParser()

    with pytest.raises(MinerUParseError, match="client is not configured"):
        parser.parse(tmp_path / "doc.pdf")
