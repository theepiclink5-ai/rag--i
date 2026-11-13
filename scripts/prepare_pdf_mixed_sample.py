"""Prepare a mixed 100-PDF benchmark set.

The output combines public PDFs and document-image pages converted to scanned
PDFs. It creates a manifest and a project-compatible golden test set with two
QA cases per document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


PARSEBENCH_DATASET = "huggingworld/ParseBench"
PARSEBENCH_BASE = f"https://huggingface.co/datasets/{PARSEBENCH_DATASET}/resolve/main"
PARSEBENCH_API = f"https://huggingface.co/api/datasets/{PARSEBENCH_DATASET}/tree/main"

DOCPARSING_DATASET = "SoMarkAI/DocParsingBench"
DOCPARSING_BASE = f"https://huggingface.co/datasets/{DOCPARSING_DATASET}/resolve/main"
DOCPARSING_API = f"https://huggingface.co/api/datasets/{DOCPARSING_DATASET}/tree/main"

IRS_FORMS = [
    ("fw9.pdf", "Form W-9", "Request for Taxpayer Identification Number and Certification"),
    ("f1040.pdf", "Form 1040", "U.S. Individual Income Tax Return"),
    ("fw4.pdf", "Form W-4", "Employee's Withholding Certificate"),
    ("f941.pdf", "Form 941", "Employer's Quarterly Federal Tax Return"),
    ("f4506t.pdf", "Form 4506-T", "Request for Transcript of Tax Return"),
    ("f1099msc.pdf", "Form 1099-MISC", "Miscellaneous Information"),
    ("f1099nec.pdf", "Form 1099-NEC", "Nonemployee Compensation"),
    ("f1098.pdf", "Form 1098", "Mortgage Interest Statement"),
    ("f8863.pdf", "Form 8863", "Education Credits"),
    ("f1040s1.pdf", "Schedule 1", "Additional Income and Adjustments to Income"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare mixed 100-PDF benchmark data.")
    parser.add_argument(
        "--output-dir",
        default="data/public_sets/pdf_mixed_100",
        help="Directory for the mixed benchmark set.",
    )
    parser.add_argument(
        "--open-rag-dir",
        default="data/public_sets/open_ragbench/sample_50",
        help="Existing Open RAG Benchmark sample directory used for academic PDFs.",
    )
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--allow-insecure-downloads",
        action="store_true",
        help="Disable TLS verification for public PDF/image downloads.",
    )
    return parser.parse_args()


def configure_console_encoding() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    decoded = urllib.parse.unquote(Path(value).name)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", decoded).strip("_")
    return cleaned or f"{stable_key(value)[:12]}.pdf"


def request_url(url: str, timeout: int, insecure: bool = False) -> bytes:
    context = ssl._create_unverified_context() if insecure else None
    request = urllib.request.Request(url, headers={"User-Agent": "modular-rag-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read()


def download_file(url: str, dest: Path, timeout: int, insecure: bool = False) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.write_bytes(request_url(url, timeout=timeout, insecure=insecure))


def load_json_url(url: str, timeout: int) -> Any:
    return json.loads(request_url(url, timeout=timeout).decode("utf-8"))


def load_jsonl_url(url: str, timeout: int) -> list[dict[str, Any]]:
    lines = request_url(url, timeout=timeout).decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def list_hf_paths(api_base: str, subdir: str, suffix: str, timeout: int) -> list[str]:
    data = load_json_url(f"{api_base}/{subdir}?recursive=true", timeout)
    paths = [item["path"] for item in data if item.get("path", "").lower().endswith(suffix)]
    return sorted(paths, key=stable_key)


def group_annotations(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pdf = row.get("pdf")
        if pdf:
            grouped[str(pdf)].append(row)
    return grouped


def rule_json(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("rule") or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def compact_text(value: str, limit: int = 280) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    return text[:limit].rstrip()


def parsebench_qa(category: str, rows: list[dict[str, Any]], filename: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows[:4]:
        rule = rule_json(row)
        if category == "chart":
            labels = rule.get("labels") or []
            value = str(rule.get("value", "")).strip()
            if labels and value:
                label_text = " / ".join(str(label) for label in labels)
                cases.append({
                    "query": f"In the chart PDF, what value is associated with {label_text}?",
                    "reference_answer": value,
                    "answer_type": "chart_data_point",
                    "evidence": label_text,
                })
        elif category == "table":
            markdown = row.get("expected_markdown") or ""
            if markdown:
                cases.append({
                    "query": "Extract the main table content from this PDF page.",
                    "reference_answer": compact_text(markdown, 600),
                    "answer_type": "table_extraction",
                    "evidence": "expected_markdown",
                })
        elif category == "layout":
            content = rule.get("content") or {}
            text = content.get("text")
            klass = rule.get("canonical_class")
            if text:
                cases.append({
                    "query": f"What {klass or 'layout'} text element appears on this page?",
                    "reference_answer": str(text),
                    "answer_type": "layout_element",
                    "evidence": str(text),
                })
        elif category == "text":
            if "bag_of_sentence" in rule:
                sentences = list(rule["bag_of_sentence"].keys())
                if sentences:
                    cases.append({
                        "query": "Name one factual sentence that should appear in this dense text PDF.",
                        "reference_answer": str(sentences[0]),
                        "answer_type": "text_content",
                        "evidence": str(sentences[0]),
                    })
            elif "text" in rule:
                cases.append({
                    "query": "Which heading or emphasized text should be preserved from this PDF?",
                    "reference_answer": str(rule["text"]),
                    "answer_type": "text_formatting",
                    "evidence": str(rule["text"]),
                })
    if not cases:
        cases.append({
            "query": "What type of document parsing case is this PDF used for?",
            "reference_answer": category,
            "answer_type": "document_type",
            "evidence": category,
        })
    while len(cases) < 2:
        cases.append({
            "query": "Which source file should support this answer?",
            "reference_answer": filename,
            "answer_type": "source_identification",
            "evidence": filename,
        })
    return cases[:2]


def add_case(
    golden_cases: list[dict[str, Any]],
    doc: dict[str, Any],
    qa: dict[str, Any],
) -> None:
    golden_cases.append({
        "query": qa["query"],
        "expected_chunk_ids": [],
        "expected_sources": [doc["pdf_filename"]],
        "reference_answer": qa["reference_answer"],
        "metadata": {
            "dataset": doc["dataset"],
            "doc_id": doc["doc_id"],
            "document_type": doc["document_type"],
            "source_url": doc.get("source_url"),
            "answer_type": qa.get("answer_type"),
            "evidence": qa.get("evidence"),
            "qa_quality": doc.get("qa_quality", "derived"),
        },
    })


def add_parsebench_docs(
    manifest: list[dict[str, Any]],
    golden_cases: list[dict[str, Any]],
    pdf_dir: Path,
    timeout: int,
    insecure: bool,
) -> None:
    category_specs = [
        ("chart", "chart.jsonl", 15),
        ("table", "table.jsonl", 15),
        ("layout", "layout.jsonl", 15),
        ("text", "text_content.jsonl", 8),
        ("text", "text_formatting.jsonl", 7),
    ]
    used: set[str] = set()
    for category, jsonl_name, count in category_specs:
        rows = load_jsonl_url(f"{PARSEBENCH_BASE}/{jsonl_name}", timeout)
        grouped = group_annotations(rows)
        pdf_paths = [
            path for path in sorted(grouped, key=lambda p: (-len(grouped[p]), stable_key(p)))
            if path.startswith(f"docs/{category}/") and path not in used
        ][:count]
        for pdf_path in pdf_paths:
            used.add(pdf_path)
            filename = f"parsebench_{category}_{safe_name(pdf_path)}"
            dest = pdf_dir / filename
            url = f"{PARSEBENCH_BASE}/{urllib.parse.quote(pdf_path)}"
            download_file(url, dest, timeout, insecure)
            doc = {
                "doc_id": f"parsebench:{pdf_path}",
                "pdf_filename": filename,
                "document_type": f"parsebench_{category}",
                "dataset": PARSEBENCH_DATASET,
                "source_url": url,
                "source_path": pdf_path,
                "page_count_hint": 1,
                "qa_quality": "dataset_rule_derived",
            }
            manifest.append(doc)
            for qa in parsebench_qa(category, grouped[pdf_path], filename):
                add_case(golden_cases, doc, qa)


def add_openrag_docs(
    manifest: list[dict[str, Any]],
    golden_cases: list[dict[str, Any]],
    pdf_dir: Path,
    open_rag_dir: Path,
) -> None:
    docs = json.loads((open_rag_dir / "selected_pdfs.json").read_text(encoding="utf-8"))
    golden = json.loads((open_rag_dir / "golden_test_set.json").read_text(encoding="utf-8"))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in golden["test_cases"]:
        for source in case.get("expected_sources", []):
            by_source[source].append(case)

    for src_doc in docs[:15]:
        original = open_rag_dir / "pdfs" / src_doc["pdf_filename"]
        filename = f"openrag_academic_{src_doc['pdf_filename']}"
        shutil.copy2(original, pdf_dir / filename)
        doc = {
            "doc_id": f"openrag:{src_doc['doc_id']}",
            "pdf_filename": filename,
            "document_type": "academic_double_column",
            "dataset": "deepmatics/open_ragbench",
            "source_url": src_doc["pdf_url"],
            "source_path": str(original),
            "qa_quality": "dataset_answer",
        }
        manifest.append(doc)
        source_cases = by_source[src_doc["pdf_filename"]][:2]
        for case in source_cases:
            add_case(golden_cases, doc, {
                "query": case["query"],
                "reference_answer": case["reference_answer"],
                "answer_type": "rag_benchmark_answer",
                "evidence": src_doc["pdf_filename"],
            })


def add_irs_forms(
    manifest: list[dict[str, Any]],
    golden_cases: list[dict[str, Any]],
    pdf_dir: Path,
    timeout: int,
    insecure: bool,
) -> None:
    for pdf_name, form_name, title in IRS_FORMS:
        url = f"https://www.irs.gov/pub/irs-pdf/{pdf_name}"
        filename = f"irs_form_{pdf_name}"
        download_file(url, pdf_dir / filename, timeout, insecure)
        doc = {
            "doc_id": f"irs:{pdf_name}",
            "pdf_filename": filename,
            "document_type": "government_form",
            "dataset": "irs.gov public forms",
            "source_url": url,
            "qa_quality": "metadata_derived",
        }
        manifest.append(doc)
        for qa in [
            {
                "query": "Which IRS form is this document?",
                "reference_answer": form_name,
                "answer_type": "form_identity",
                "evidence": form_name,
            },
            {
                "query": "What is the stated purpose or title of this IRS form?",
                "reference_answer": title,
                "answer_type": "form_title",
                "evidence": title,
            },
        ]:
            add_case(golden_cases, doc, qa)


def markdown_excerpt(markdown: str) -> str:
    lines = [
        re.sub(r"[#*_`>|-]+", " ", line).strip()
        for line in markdown.splitlines()
        if line.strip()
    ]
    text = compact_text(" ".join(lines), 500)
    return text or "The page contains scanned document content."


def add_scanned_docs(
    manifest: list[dict[str, Any]],
    golden_cases: list[dict[str, Any]],
    pdf_dir: Path,
    output_dir: Path,
    timeout: int,
    insecure: bool,
) -> None:
    image_paths = list_hf_paths(DOCPARSING_API, "images", ".png", timeout)
    selected: list[str] = []
    prefixes: set[str] = set()
    for path in image_paths:
        prefix = re.sub(r"_\d+\.png$", "", Path(path).name)
        if prefix in prefixes:
            continue
        prefixes.add(prefix)
        selected.append(path)
        if len(selected) == 15:
            break
    if len(selected) < 15:
        selected_set = set(selected)
        for path in image_paths:
            if path in selected_set:
                continue
            selected.append(path)
            selected_set.add(path)
            if len(selected) == 15:
                break

    image_dir = output_dir / "source_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for image_path in selected:
        stem = Path(image_path).stem
        image_file = image_dir / f"{stem}.png"
        md_file = image_dir / f"{stem}.md"
        image_url = f"{DOCPARSING_BASE}/{urllib.parse.quote(image_path)}"
        markdown_path = f"markdowns/{stem}.md"
        markdown_url = f"{DOCPARSING_BASE}/{urllib.parse.quote(markdown_path)}"
        download_file(image_url, image_file, timeout, insecure)
        download_file(markdown_url, md_file, timeout, insecure)

        filename = f"scanned_docparsing_{stem}.pdf"
        with Image.open(image_file) as img:
            rgb = img.convert("RGB")
            rgb.save(pdf_dir / filename, "PDF", resolution=150.0)

        excerpt = markdown_excerpt(md_file.read_text(encoding="utf-8", errors="replace"))
        doc = {
            "doc_id": f"docparsing:{stem}",
            "pdf_filename": filename,
            "document_type": "scanned_image_pdf",
            "dataset": DOCPARSING_DATASET,
            "source_url": image_url,
            "source_path": image_path,
            "qa_quality": "markdown_derived",
        }
        manifest.append(doc)
        for qa in [
            {
                "query": "What visible text or content appears in this scanned PDF page?",
                "reference_answer": excerpt,
                "answer_type": "ocr_content",
                "evidence": excerpt[:120],
            },
            {
                "query": "What kind of PDF input is this document testing?",
                "reference_answer": "A scanned image-based PDF that requires OCR or visual document parsing.",
                "answer_type": "document_type",
                "evidence": "scanned image-based PDF",
            },
        ]:
            add_case(golden_cases, doc, qa)


def write_outputs(output_dir: Path, manifest: list[dict[str, Any]], golden_cases: list[dict[str, Any]]) -> None:
    manifest.sort(key=lambda item: item["pdf_filename"])
    (output_dir / "manifest.json").write_text(
        json.dumps({
            "description": "Mixed 100-PDF benchmark set for PDF parsing and RAG evaluation.",
            "total_documents": len(manifest),
            "documents": manifest,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "golden_test_set.json").write_text(
        json.dumps({
            "description": (
                "Mixed 100-PDF golden set. Each case includes expected_sources and "
                "reference_answer; chunk IDs should be filled after ingestion if needed."
            ),
            "version": "1.0",
            "test_cases": golden_cases,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    counts: dict[str, int] = defaultdict(int)
    for doc in manifest:
        counts[doc["document_type"]] += 1
    (output_dir / "source_notes.md").write_text(
        "\n".join([
            "# Mixed 100 PDF Benchmark",
            "",
            "This set intentionally mixes born-digital PDFs, academic papers, government forms,",
            "chart/table/layout benchmark pages, dense text pages, and scanned image-based PDFs.",
            "",
            "## Distribution",
            "",
            *[f"- {key}: {value}" for key, value in sorted(counts.items())],
            "",
            "QA cases are derived from public dataset rules, public metadata, or dataset answers.",
            "For interview-grade reporting, manually review and upgrade selected cases with page",
            "numbers and chunk IDs after ingestion.",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    output_dir = Path(args.output_dir)
    pdf_dir = output_dir / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    golden_cases: list[dict[str, Any]] = []

    print("[1/4] ParseBench PDFs")
    add_parsebench_docs(manifest, golden_cases, pdf_dir, args.timeout, args.allow_insecure_downloads)
    print("[2/4] Open RAG Benchmark academic PDFs")
    add_openrag_docs(manifest, golden_cases, pdf_dir, Path(args.open_rag_dir))
    print("[3/4] IRS public forms")
    add_irs_forms(manifest, golden_cases, pdf_dir, args.timeout, args.allow_insecure_downloads)
    print("[4/4] Scanned image PDFs")
    add_scanned_docs(manifest, golden_cases, pdf_dir, output_dir, args.timeout, args.allow_insecure_downloads)

    write_outputs(output_dir, manifest, golden_cases)
    print(f"Documents: {len(manifest)}")
    print(f"QA cases: {len(golden_cases)}")
    print(f"Output: {output_dir}")
    time.sleep(0.1)
    return 0 if len(manifest) == 100 and len(golden_cases) == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
