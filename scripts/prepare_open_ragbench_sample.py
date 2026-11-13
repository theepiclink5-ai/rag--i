"""Prepare a reproducible Open RAG Benchmark PDF sample.

The script downloads the small metadata files, selects documents with complete
query/qrel/answer coverage, and emits a project-compatible golden test set.
PDF downloads are opt-in because they can be large.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import ssl
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BASE_URLS = (
    "https://huggingface.co/datasets/deepmatics/open_ragbench/resolve/main/official/pdf/arxiv",
    "https://huggingface.co/datasets/deepmatics/open_ragbench/resolve/main/pdf/arxiv",
)
META_FILES = ("pdf_urls.json", "queries.json", "qrels.json", "answers.json")


@dataclass(frozen=True)
class PreparedPaths:
    meta_dir: Path
    output_dir: Path
    pdf_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a 50-PDF Open RAG Benchmark sample for local RAG testing."
    )
    parser.add_argument("--count", type=int, default=50, help="Number of PDFs to select.")
    parser.add_argument(
        "--max-qa-per-doc",
        type=int,
        default=4,
        help="Maximum number of QA cases to keep per selected PDF.",
    )
    parser.add_argument(
        "--seed",
        default="modular-rag-open-ragbench-v1",
        help="Stable seed used for deterministic selection.",
    )
    parser.add_argument(
        "--meta-dir",
        default="data/public_sets/open_ragbench/meta",
        help="Directory for downloaded Open RAG Benchmark metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/public_sets/open_ragbench/sample_50",
        help="Directory for selected metadata and golden test set.",
    )
    parser.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Download selected PDF files into the output pdfs directory.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Network timeout in seconds.",
    )
    parser.add_argument(
        "--allow-insecure-downloads",
        action="store_true",
        help="Disable TLS certificate verification for PDF downloads only.",
    )
    return parser.parse_args()


def configure_console_encoding() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def prepare_paths(args: argparse.Namespace) -> PreparedPaths:
    meta_dir = Path(args.meta_dir)
    output_dir = Path(args.output_dir)
    pdf_dir = output_dir / "pdfs"
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.download_pdfs:
        pdf_dir.mkdir(parents=True, exist_ok=True)
    return PreparedPaths(meta_dir=meta_dir, output_dir=output_dir, pdf_dir=pdf_dir)


def download_url(
    url: str,
    dest: Path,
    timeout: int,
    *,
    insecure: bool = False,
) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "modular-rag-benchmark/1.0"})
    context = ssl._create_unverified_context() if insecure else None
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        dest.write_bytes(response.read())


def ensure_metadata(meta_dir: Path, timeout: int) -> None:
    for name in META_FILES:
        dest = meta_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        last_error: Exception | None = None
        for base_url in BASE_URLS:
            try:
                download_url(f"{base_url}/{name}", dest, timeout)
                last_error = None
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                last_error = exc
        if last_error is not None:
            raise last_error


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        result: dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict):
                key = item.get("id") or item.get("query_id") or item.get("qid") or item.get("doc_id")
                if key is not None:
                    result[str(key)] = item
        return result
    raise ValueError(f"Unsupported metadata shape: {type(value).__name__}")


def normalize_pdf_urls(raw: Any) -> dict[str, str]:
    data = as_mapping(raw)
    urls: dict[str, str] = {}
    for doc_id, value in data.items():
        if isinstance(value, str):
            urls[str(doc_id)] = value
        elif isinstance(value, dict):
            url = value.get("url") or value.get("pdf_url") or value.get("download_url")
            if url:
                urls[str(doc_id)] = str(url)
    return urls


def normalize_queries(raw: Any) -> dict[str, str]:
    data = as_mapping(raw)
    queries: dict[str, str] = {}
    for qid, value in data.items():
        if isinstance(value, str):
            queries[str(qid)] = value
        elif isinstance(value, dict):
            text = value.get("query") or value.get("question") or value.get("text")
            if text:
                queries[str(qid)] = str(text)
    return queries


def normalize_answers(raw: Any) -> dict[str, str]:
    data = as_mapping(raw)
    answers: dict[str, str] = {}
    for qid, value in data.items():
        if isinstance(value, str):
            answers[str(qid)] = value
        elif isinstance(value, list):
            answers[str(qid)] = "; ".join(str(v) for v in value if str(v).strip())
        elif isinstance(value, dict):
            answer = value.get("answer") or value.get("answers") or value.get("reference_answer")
            if isinstance(answer, list):
                answers[str(qid)] = "; ".join(str(v) for v in answer if str(v).strip())
            elif answer:
                answers[str(qid)] = str(answer)
    return answers


def iter_qrels(raw: Any) -> Iterable[tuple[str, str, float]]:
    if isinstance(raw, dict):
        for qid, value in raw.items():
            if isinstance(value, dict):
                if "doc_id" in value:
                    yield str(qid), str(value["doc_id"]), 1.0
                elif "document_id" in value:
                    yield str(qid), str(value["document_id"]), 1.0
                elif "corpus_id" in value:
                    yield str(qid), str(value["corpus_id"]), 1.0
                else:
                    for doc_id, score in value.items():
                        yield str(qid), str(doc_id), float(score)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        doc_id = item.get("doc_id") or item.get("document_id") or item.get("corpus_id")
                        score = item.get("score") or item.get("relevance") or 1
                        if doc_id is not None:
                            yield str(qid), str(doc_id), float(score)
                    else:
                        yield str(qid), str(item), 1.0
        return
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            qid = item.get("query_id") or item.get("qid") or item.get("id")
            doc_id = item.get("doc_id") or item.get("document_id") or item.get("corpus_id")
            score = item.get("score") or item.get("relevance") or 1
            if qid is not None and doc_id is not None:
                yield str(qid), str(doc_id), float(score)
        return
    raise ValueError(f"Unsupported qrels shape: {type(raw).__name__}")


def stable_rank_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def pdf_filename(doc_id: str, url: str) -> str:
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    if not name.lower().endswith(".pdf"):
        name = f"{doc_id}.pdf"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return safe or f"{doc_id}.pdf"


def build_sample(
    pdf_urls: dict[str, str],
    queries: dict[str, str],
    qrels_raw: Any,
    answers: dict[str, str],
    count: int,
    max_qa_per_doc: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_doc: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for qid, doc_id, score in iter_qrels(qrels_raw):
        if score <= 0:
            continue
        if doc_id in pdf_urls and qid in queries and qid in answers:
            by_doc[doc_id].append((qid, score))

    candidates = [
        doc_id for doc_id, items in by_doc.items()
        if items and pdf_urls.get(doc_id)
    ]
    candidates.sort(key=lambda doc_id: (-len(by_doc[doc_id]), stable_rank_key(seed, doc_id)))
    selected_doc_ids = candidates[:count]
    if len(selected_doc_ids) < count:
        raise RuntimeError(f"Only found {len(selected_doc_ids)} complete PDFs, need {count}.")

    selected_docs: list[dict[str, Any]] = []
    golden_cases: list[dict[str, Any]] = []
    for doc_id in selected_doc_ids:
        url = pdf_urls[doc_id]
        filename = pdf_filename(doc_id, url)
        selected_docs.append({
            "doc_id": doc_id,
            "pdf_url": url,
            "pdf_filename": filename,
            "available_qa_count": len(by_doc[doc_id]),
        })
        qids = [qid for qid, _ in sorted(
            by_doc[doc_id],
            key=lambda pair: (-pair[1], stable_rank_key(seed, pair[0])),
        )[:max_qa_per_doc]]
        for qid in qids:
            golden_cases.append({
                "query": queries[qid],
                "expected_chunk_ids": [],
                "expected_sources": [filename],
                "reference_answer": answers[qid],
                "metadata": {
                    "dataset": "deepmatics/open_ragbench",
                    "subset": "official/pdf/arxiv",
                    "query_id": qid,
                    "doc_id": doc_id,
                    "pdf_url": url,
                },
            })
    return selected_docs, golden_cases


def write_outputs(output_dir: Path, docs: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    (output_dir / "selected_pdfs.json").write_text(
        json.dumps(docs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "golden_test_set.json").write_text(
        json.dumps({
            "description": (
                "Open RAG Benchmark 50-PDF sample generated for Modular RAG MCP Server. "
                "Ground truth uses expected_sources at PDF-file level."
            ),
            "version": "1.0",
            "dataset": "deepmatics/open_ragbench",
            "test_cases": cases,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join([
            "# Open RAG Benchmark Sample",
            "",
            f"- PDFs selected: {len(docs)}",
            f"- QA cases selected: {len(cases)}",
            "- Golden set: `golden_test_set.json`",
            "- PDF list: `selected_pdfs.json`",
            "",
            "The golden set uses `expected_sources` because chunk IDs are created only after ingestion.",
            "",
        ]),
        encoding="utf-8",
    )


def download_pdfs(
    docs: list[dict[str, Any]],
    pdf_dir: Path,
    timeout: int,
    *,
    insecure: bool = False,
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for index, doc in enumerate(docs, start=1):
        dest = pdf_dir / doc["pdf_filename"]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[{index}/{len(docs)}] skip {dest.name}")
            continue
        try:
            print(f"[{index}/{len(docs)}] download {dest.name}")
            download_url(doc["pdf_url"], dest, timeout, insecure=insecure)
            time.sleep(0.2)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failures.append({
                "doc_id": doc["doc_id"],
                "pdf_url": doc["pdf_url"],
                "error": str(exc),
            })
    return failures


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    paths = prepare_paths(args)
    ensure_metadata(paths.meta_dir, args.timeout)

    pdf_urls = normalize_pdf_urls(load_json(paths.meta_dir / "pdf_urls.json"))
    queries = normalize_queries(load_json(paths.meta_dir / "queries.json"))
    answers = normalize_answers(load_json(paths.meta_dir / "answers.json"))
    qrels = load_json(paths.meta_dir / "qrels.json")

    docs, cases = build_sample(
        pdf_urls=pdf_urls,
        queries=queries,
        qrels_raw=qrels,
        answers=answers,
        count=args.count,
        max_qa_per_doc=args.max_qa_per_doc,
        seed=args.seed,
    )
    write_outputs(paths.output_dir, docs, cases)

    failures: list[dict[str, str]] = []
    if args.download_pdfs:
        failures = download_pdfs(
            docs,
            paths.pdf_dir,
            args.timeout,
            insecure=args.allow_insecure_downloads,
        )
        (paths.output_dir / "pdf_download_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Selected PDFs: {len(docs)}")
    print(f"Selected QA cases: {len(cases)}")
    print(f"Output directory: {paths.output_dir}")
    if args.download_pdfs:
        print(f"PDF directory: {paths.pdf_dir}")
        print(f"Download failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
