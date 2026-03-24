#!/usr/bin/env python3
"""
scripts/ingest_bulk.py — Bulk-ingest all resumes from an S3 bucket prefix.

Usage:
    python scripts/ingest_bulk.py --prefix resumes/ --workers 4
    python scripts/ingest_bulk.py --bucket my-resume-bucket --prefix 2024/
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging, get_logger, settings
from src.mcp_server.tools.resume_tool import IngestResumeInput, ingest_resume
from src.rag.embeddings import embedder
from src.rag.vector_store import vector_store
from src.s3.client import s3_client

configure_logging()
logger = get_logger("bulk_ingest")


async def process_single(key: str, bucket: str, semaphore: asyncio.Semaphore, force: bool) -> dict:
    async with semaphore:
        try:
            result = await ingest_resume(
                IngestResumeInput(s3_key=key, s3_bucket=bucket, force_reindex=force)
            )
            return {"key": key, "success": True, "candidate": result.candidate_name, "chunks": result.chunks_indexed}
        except Exception as exc:
            logger.error("bulk_ingest.error", key=key, error=str(exc))
            return {"key": key, "success": False, "error": str(exc)}


async def bulk_ingest(bucket: str, prefix: str, workers: int, force: bool) -> None:
    logger.info("bulk_ingest.start", bucket=bucket, prefix=prefix, workers=workers)

    # Pre-load model
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, embedder.load)
    await loop.run_in_executor(None, vector_store.connect)

    # List all objects
    objects = await s3_client.list_objects(bucket=bucket, prefix=prefix)
    supported = {fmt.lower() for fmt in settings.supported_formats}
    to_process = [
        o for o in objects
        if Path(o["key"]).suffix.lower().lstrip(".") in supported
    ]

    logger.info("bulk_ingest.files_found", total=len(to_process))
    if not to_process:
        print("No supported files found.")
        return

    semaphore = asyncio.Semaphore(workers)
    tasks = [process_single(o["key"], bucket, semaphore, force) for o in to_process]

    start = time.monotonic()
    results = await asyncio.gather(*tasks, return_exceptions=False)
    elapsed = time.monotonic() - start

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]

    print(f"\n{'─'*60}")
    print(f"  Bulk Ingest Complete")
    print(f"{'─'*60}")
    print(f"  Total files : {len(to_process)}")
    print(f"  Succeeded   : {len(successes)}")
    print(f"  Failed      : {len(failures)}")
    print(f"  Time elapsed: {elapsed:.1f}s  ({len(to_process)/elapsed:.1f} files/sec)")
    print(f"{'─'*60}")

    if failures:
        print("\nFailed files:")
        for f in failures:
            print(f"  ✗ {f['key']}: {f.get('error', 'unknown')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk ingest resumes from S3")
    parser.add_argument("--bucket", default=settings.aws.s3_resume_bucket)
    parser.add_argument(
        "--prefix",
        default=settings.aws.resume_key_prefix,
        help="S3 key prefix to scan (defaults to AWS_RESUME_KEY_PREFIX)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers")
    parser.add_argument("--force", action="store_true", help="Force reindex existing")
    args = parser.parse_args()

    asyncio.run(bulk_ingest(
        bucket=args.bucket,
        prefix=args.prefix,
        workers=args.workers,
        force=args.force,
    ))
