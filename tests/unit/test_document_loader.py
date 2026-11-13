"""Unit tests for the generic DocumentLoader."""

from __future__ import annotations

import pytest

from src.libs.loader.document_loader import DocumentLoader
from src.libs.loader.pdf_parsers import PdfParseResult


def test_load_txt_file(tmp_path):
    txt_file = tmp_path / "note.txt"
    txt_file.write_text("Hello RAG\nThis is a text file.", encoding="utf-8")

    loader = DocumentLoader(extract_images=False)
    document = loader.load(txt_file)

    assert document.text == "Hello RAG\nThis is a text file."
    assert document.metadata["doc_type"] == "txt"
    assert document.metadata["source_path"].endswith("note.txt")
    assert document.metadata["title"] == "Hello RAG"


def test_load_md_file_extracts_heading_title(tmp_path):
    md_file = tmp_path / "note.md"
    md_file.write_text("# Project Notes\n\nContent", encoding="utf-8")

    loader = DocumentLoader(extract_images=False)
    document = loader.load(md_file)

    assert document.metadata["doc_type"] == "md"
    assert document.metadata["title"] == "Project Notes"


def test_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b\n1,2", encoding="utf-8")

    loader = DocumentLoader(extract_images=False)

    with pytest.raises(ValueError, match="Unsupported document format"):
        loader.load(file_path)


def test_pdf_loader_uses_injected_parser(tmp_path):
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n")

    class StubPdfParser:
        provider_name = "stub"

        def parse(self, file_path):
            return PdfParseResult(
                text="# Stub PDF\n\nParsed through injected parser.",
                parser_provider=self.provider_name,
            )

    loader = DocumentLoader(extract_images=False, pdf_parser=StubPdfParser())
    document = loader.load(pdf_file)

    assert document.text.startswith("# Stub PDF")
    assert document.metadata["parser_provider"] == "stub"
