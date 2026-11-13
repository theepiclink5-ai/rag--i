#!/usr/bin/env python
"""Validate MinerU parsing against a single local PDF.

This script intentionally processes one PDF at a time.  It is meant for
checking API credentials, upload/poll/download behavior, and parse report
generation before running a larger benchmark.

Usage:
    python scripts/test_mineru_parse.py --path documents/sample.pdf
    python scripts/test_mineru_parse.py --path documents/sample.pdf --output-dir data/parsed/mineru/smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.settings import load_settings, resolve_path
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.pdf_parsers import (
    MinerUHttpClient,
    MinerUParseOptions,
    MinerUPdfParser,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-file MinerU parsing smoke test."
    )
    parser.add_argument(
        "--path",
        "-p",
        required=True,
        help="Local PDF file to submit to MinerU.",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "settings.yaml"),
        help="Settings YAML path. Defaults to config/settings.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/parsed/mineru/smoke",
        help="Directory for extracted MinerU result packages.",
    )
    parser.add_argument(
        "--report-dir",
        default="data/parsed/reports/smoke",
        help="Directory for parse_report JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.path)
    if not pdf_path.exists():
        print(f"[FAIL] PDF not found: {pdf_path}", file=sys.stderr)
        return 2
    if pdf_path.suffix.lower() != ".pdf":
        print(f"[FAIL] Expected a PDF file: {pdf_path}", file=sys.stderr)
        return 2

    try:
        settings = load_settings(args.config)
        mineru_settings = getattr(getattr(settings, "pdf_parser", None), "mineru", None)
        if mineru_settings is None:
            print("[FAIL] Missing pdf_parser.mineru configuration.", file=sys.stderr)
            return 2
        if not mineru_settings.api_key:
            print("[FAIL] MINERU_API_KEY is not configured.", file=sys.stderr)
            return 2
    except Exception as exc:
        print(f"[FAIL] Failed to load configuration: {exc}", file=sys.stderr)
        return 2

    options = MinerUParseOptions(
        model_version=mineru_settings.model_version,
        language=mineru_settings.language,
        is_ocr=mineru_settings.is_ocr,
        enable_table=mineru_settings.enable_table,
        enable_formula=mineru_settings.enable_formula,
        timeout_seconds=mineru_settings.timeout_seconds,
        poll_interval_seconds=mineru_settings.poll_interval_seconds,
    )
    parser = MinerUPdfParser(
        client=MinerUHttpClient(api_key=mineru_settings.api_key),
        options=options,
        output_dir=resolve_path(args.output_dir),
        save_raw_result=mineru_settings.save_raw_result,
    )
    loader = PdfLoader(
        extract_images=False,
        parser=parser,
        parse_report_dir=resolve_path(args.report_dir),
    )

    print("[INFO] Submitting one PDF to MinerU...")
    print(f"[INFO] File: {pdf_path}")
    try:
        document = loader.load(pdf_path)
    except Exception as exc:
        print(f"[FAIL] MinerU parse failed: {exc}", file=sys.stderr)
        return 1

    metadata = document.metadata
    print("[OK] MinerU parse completed")
    print(f"  Document ID: {document.id}")
    print(f"  Text chars:  {len(document.text)}")
    print(f"  Quality:     {metadata.get('parse_quality_score')}")
    print(f"  Reasons:     {metadata.get('parse_quality_reasons', [])}")
    print(f"  Result dir:  {metadata.get('mineru_result_dir')}")
    print(f"  Report:      {metadata.get('parse_report_path')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
