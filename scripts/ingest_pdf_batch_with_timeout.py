"""Ingest PDFs one by one with a per-file timeout.

This is a thin orchestration wrapper around ``scripts/ingest.py``. It keeps the
production ingestion path unchanged while preventing one problematic PDF from
blocking an entire benchmark run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ingest PDFs with per-file timeout.")
    parser.add_argument("--path", required=True, help="PDF file or directory.")
    parser.add_argument("--collection", required=True, help="Target collection.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Maximum seconds allowed for one PDF.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to write the batch ingest summary JSON.",
    )
    parser.add_argument("--force", action="store_true", help="Forward --force to ingest.py.")
    parser.add_argument("--verbose", action="store_true", help="Forward --verbose to ingest.py.")
    parser.add_argument(
        "--start-after",
        default=None,
        help="Skip files through this filename, then process the remaining files.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only process a specific filename. Can be repeated.",
    )
    return parser.parse_args()


def discover_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.pdf"))


def run_one(pdf: Path, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/ingest.py",
        "--path",
        str(pdf),
        "--collection",
        args.collection,
    ]
    if args.force:
        cmd.append("--force")
    if args.verbose:
        cmd.append("--verbose")

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=args.timeout_seconds,
        )
        status = "success" if proc.returncode == 0 else "failed"
        return {
            "file": str(pdf),
            "status": status,
            "returncode": proc.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "file": str(pdf),
            "status": "timeout",
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }


def main() -> int:
    args = parse_args()
    pdfs = discover_pdfs(Path(args.path))
    if args.include:
        include = set(args.include)
        pdfs = [pdf for pdf in pdfs if pdf.name in include]
    if args.start_after:
        names = [pdf.name for pdf in pdfs]
        if args.start_after in names:
            pdfs = pdfs[names.index(args.start_after) + 1:]

    output_json = Path(
        args.output_json
        or f"data/parsed/reports/{args.collection}/batch_ingest_summary.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for index, pdf in enumerate(pdfs, start=1):
        print(f"[{index}/{len(pdfs)}] {pdf}")
        result = run_one(pdf, args)
        results.append(result)
        print(f"  {result['status']} in {result['elapsed_seconds']}s")
        output_json.write_text(
            json.dumps({
                "collection": args.collection,
                "timeout_seconds": args.timeout_seconds,
                "total": len(pdfs),
                "results": results,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    failures = [item for item in results if item["status"] != "success"]
    print(f"Completed: {len(results) - len(failures)}/{len(results)}")
    print(f"Failures/timeouts: {len(failures)}")
    print(f"Summary: {output_json}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
