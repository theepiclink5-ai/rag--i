#!/usr/bin/env python
"""Benchmark PDF parsing quality over a file or directory.

By default this script is safe and local-only: it uses MarkItDown plus the
parse-quality analyzer and does not call MinerU.  Add --execute-fallback to use
the configured CompositePdfParser, which may call MinerU for low-quality PDFs.

Usage:
    python scripts/benchmark_pdf_parsing.py --path data/real_pdfs
    python scripts/benchmark_pdf_parsing.py --path data/real_pdfs --execute-fallback
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.settings import load_settings, resolve_path
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.pdf_parsers import MarkItDownPdfParser, PdfParserFactory


def configure_console_encoding() -> None:
    if sys.platform != "win32":
        return

    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PDF parse quality benchmark over one file or directory."
    )
    parser.add_argument("--path", "-p", required=True, help="PDF file or directory.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "settings.yaml"),
        help="Settings YAML path. Defaults to config/settings.yaml.",
    )
    parser.add_argument(
        "--collection",
        default="pdf_benchmark",
        help="Collection/run name used for output directories.",
    )
    parser.add_argument(
        "--execute-fallback",
        action="store_true",
        help="Allow configured fallback parser calls, including MinerU API calls.",
    )
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Extract local PDF images while benchmarking.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of PDFs to process.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for per-document parse reports.",
    )
    parser.add_argument(
        "--mineru-output-dir",
        default=None,
        help="Directory for extracted MinerU result packages.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Path for benchmark summary JSON.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Path for per-document CSV summary.",
    )
    parser.add_argument(
        "--bad-cases-json",
        default=None,
        help="Path for low-quality/failure bad-case JSON output.",
    )
    return parser.parse_args()


def discover_pdfs(path: str | Path, limit: int | None = None) -> List[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    if root.is_file():
        if root.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file, got: {root}")
        files = [root]
    else:
        files = sorted(
            item for item in root.rglob("*") if item.is_file() and item.suffix.lower() == ".pdf"
        )

    return files[:limit] if limit is not None else files


def build_loader(args: argparse.Namespace) -> PdfLoader:
    settings = load_settings(args.config)
    collection = args.collection
    report_dir = (
        resolve_path(args.report_dir)
        if args.report_dir
        else resolve_path(f"data/parsed/reports/{collection}")
    )

    if args.execute_fallback:
        mineru_output_dir = (
            resolve_path(args.mineru_output_dir)
            if args.mineru_output_dir
            else resolve_path(f"data/parsed/mineru/{collection}")
        )
        parser = PdfParserFactory.create(
            settings=settings,
            output_dir=mineru_output_dir,
        )
    else:
        parser = MarkItDownPdfParser()

    return PdfLoader(
        extract_images=args.extract_images,
        parser=parser,
        image_storage_dir=resolve_path(f"data/images/{collection}"),
        parse_report_dir=report_dir,
    )


def record_success(file_path: Path, elapsed_ms: float, document: Any) -> Dict[str, Any]:
    metadata = document.metadata
    metrics = metadata.get("parse_quality_metrics", {})
    return {
        "source_path": str(file_path),
        "success": True,
        "doc_id": document.id,
        "parser_provider": metadata.get("parser_provider"),
        "parser_route": metadata.get("parser_route"),
        "fallback_used": bool(metadata.get("fallback_used", False)),
        "fallback_provider": metadata.get("fallback_provider"),
        "should_use_external_pdf_api": bool(
            metadata.get("should_use_external_pdf_api", False)
        ),
        "parse_quality_score": metadata.get("parse_quality_score"),
        "parse_quality_reasons": metadata.get("parse_quality_reasons", []),
        "char_count": metrics.get("char_count", len(document.text or "")),
        "page_count": metrics.get("page_count"),
        "image_count": len(metadata.get("images", [])) or metrics.get("image_count", 0),
        "report_path": metadata.get("parse_report_path"),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def record_failure(file_path: Path, elapsed_ms: float, error: Exception) -> Dict[str, Any]:
    return {
        "source_path": str(file_path),
        "success": False,
        "error": str(error),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def aggregate(records: List[Dict[str, Any]], execute_fallback: bool) -> Dict[str, Any]:
    successful = [record for record in records if record.get("success")]
    failed = [record for record in records if not record.get("success")]
    fallback_needed = [
        record for record in successful if record.get("should_use_external_pdf_api")
    ]
    fallback_used = [record for record in successful if record.get("fallback_used")]
    quality_scores = [
        float(record["parse_quality_score"])
        for record in successful
        if record.get("parse_quality_score") is not None
    ]
    elapsed = [float(record["elapsed_ms"]) for record in records]

    return {
        "execute_fallback": execute_fallback,
        "total_files": len(records),
        "successful": len(successful),
        "failed": len(failed),
        "fallback_needed": len(fallback_needed),
        "fallback_used": len(fallback_used),
        "average_quality_score": (
            round(sum(quality_scores) / len(quality_scores), 4)
            if quality_scores
            else None
        ),
        "total_elapsed_ms": round(sum(elapsed), 1),
        "average_elapsed_ms": round(sum(elapsed) / len(elapsed), 1) if elapsed else 0,
    }


def write_summary(path: str | Path, summary: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def bad_cases(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cases = []
    for record in records:
        if not record.get("success"):
            cases.append(
                {
                    "source_path": record.get("source_path"),
                    "issue_type": "parse_failed",
                    "error": record.get("error"),
                    "report_path": record.get("report_path"),
                }
            )
            continue

        reasons = record.get("parse_quality_reasons") or []
        if record.get("should_use_external_pdf_api") or reasons:
            cases.append(
                {
                    "source_path": record.get("source_path"),
                    "issue_type": "low_parse_quality",
                    "parse_quality_score": record.get("parse_quality_score"),
                    "reasons": reasons,
                    "fallback_needed": record.get("should_use_external_pdf_api"),
                    "fallback_used": record.get("fallback_used"),
                    "parser_provider": record.get("parser_provider"),
                    "report_path": record.get("report_path"),
                    "suggested_action": _suggest_action(record),
                }
            )
    return cases


def _suggest_action(record: Dict[str, Any]) -> str:
    reasons = set(record.get("parse_quality_reasons") or [])
    if not record.get("success"):
        return "inspect_parse_error"
    if "scanned_pdf_likely" in reasons or "empty_text" in reasons:
        return "run_mineru_or_ocr"
    if "mojibake_detected" in reasons:
        return "run_mineru_or_check_encoding"
    if "table_structure_lost" in reasons:
        return "run_mineru_and_check_tables"
    if "image_content_missing" in reasons:
        return "run_mineru_or_enable_image_captioning"
    if "repeated_text_noise" in reasons:
        return "inspect_header_footer_noise"
    if "fragmented_lines" in reasons:
        return "inspect_layout_or_chunking"
    if record.get("should_use_external_pdf_api"):
        return "run_mineru_fallback"
    return "manual_review"


def write_records_csv(path: str | Path, records: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "source_path",
        "success",
        "parser_provider",
        "parser_route",
        "fallback_used",
        "fallback_provider",
        "should_use_external_pdf_api",
        "parse_quality_score",
        "parse_quality_reasons",
        "char_count",
        "page_count",
        "image_count",
        "elapsed_ms",
        "report_path",
        "error",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {column: record.get(column, "") for column in columns}
            if isinstance(row["parse_quality_reasons"], list):
                row["parse_quality_reasons"] = "|".join(row["parse_quality_reasons"])
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    try:
        files = discover_pdfs(args.path, limit=args.limit)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    if not files:
        print("[WARN] No PDFs found.")
        return 0

    try:
        loader = build_loader(args)
    except Exception as exc:
        print(f"[FAIL] Failed to initialize parser: {exc}", file=sys.stderr)
        return 2

    if args.execute_fallback:
        print("[WARN] Fallback execution is enabled; low-quality PDFs may call MinerU.")
    else:
        print("[INFO] Local-only mode; MinerU fallback will not be called.")

    records: List[Dict[str, Any]] = []
    for index, file_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {file_path}")
        started = time.monotonic()
        try:
            document = loader.load(file_path)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            record = record_success(file_path, elapsed_ms, document)
            records.append(record)
            reasons = ",".join(record.get("parse_quality_reasons", [])) or "-"
            print(
                "  [OK] "
                f"provider={record.get('parser_provider')} "
                f"quality={record.get('parse_quality_score')} "
                f"fallback_needed={record.get('should_use_external_pdf_api')} "
                f"fallback_used={record.get('fallback_used')} "
                f"reasons={reasons}"
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            records.append(record_failure(file_path, elapsed_ms, exc))
            print(f"  [FAIL] {exc}")

    summary = {
        "aggregate": aggregate(records, execute_fallback=args.execute_fallback),
        "records": records,
    }
    summary_path = (
        resolve_path(args.summary_json)
        if args.summary_json
        else resolve_path(f"data/parsed/reports/{args.collection}/benchmark_summary.json")
    )
    write_summary(summary_path, summary)
    csv_path = (
        resolve_path(args.summary_csv)
        if args.summary_csv
        else resolve_path(f"data/parsed/reports/{args.collection}/benchmark_summary.csv")
    )
    write_records_csv(csv_path, records)
    bad_cases_path = (
        resolve_path(args.bad_cases_json)
        if args.bad_cases_json
        else resolve_path(f"data/parsed/reports/{args.collection}/bad_cases.json")
    )
    write_summary(bad_cases_path, {"bad_cases": bad_cases(records)})

    agg = summary["aggregate"]
    print("\n" + "=" * 60)
    print("PDF PARSE BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total files:       {agg['total_files']}")
    print(f"Successful:        {agg['successful']}")
    print(f"Failed:            {agg['failed']}")
    print(f"Fallback needed:   {agg['fallback_needed']}")
    print(f"Fallback used:     {agg['fallback_used']}")
    print(f"Avg quality score: {agg['average_quality_score']}")
    print(f"Summary JSON:      {summary_path}")
    print(f"Summary CSV:       {csv_path}")
    print(f"Bad cases JSON:    {bad_cases_path}")
    print("=" * 60)
    return 0 if not agg["failed"] else 1


if __name__ == "__main__":
    configure_console_encoding()
    sys.exit(main())
