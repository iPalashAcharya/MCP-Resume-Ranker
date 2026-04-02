#!/usr/bin/env python3
"""
scripts/test_rank.py — Quick CLI test for the ranking pipeline.
Useful for smoke-testing after deployment or config changes.

Usage:
    python scripts/test_rank.py --jd-text "Python engineer, 3+ years, AWS required"
    python scripts/test_rank.py --jd-key "development/jd/senior-engineer.pdf" --top-k 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging, settings
from src.mcp_server.tools.ranking_tool import RankCandidatesInput, rank_candidates_for_job
from src.rag.embeddings import embedder
from src.rag.vector_store import vector_store

configure_logging()


async def run(args):
    loop = asyncio.get_event_loop()
    print("⟳ Loading embedding model...")
    await loop.run_in_executor(None, embedder.load)
    print("⟳ Connecting to Milvus...")
    await loop.run_in_executor(None, vector_store.connect)

    ref_keys = args.reference_key or None
    resume_bucket = args.resume_bucket or None

    params = RankCandidatesInput(
        jd_text=args.jd_text or None,
        jd_s3_key=args.jd_key or None,
        jd_s3_bucket=args.bucket or None,
        top_k=args.top_k,
        use_cache=not args.no_cache,
        reference_selected_resume_s3_keys=ref_keys,
        resume_s3_bucket=resume_bucket,
    )

    print(f"⟳ Ranking candidates (top_k={args.top_k})...")
    result = await rank_candidates_for_job(params)

    print(f"\n{'═'*70}")
    print(f"  Job: {result.jd_title}  |  Company: {result.jd_company or 'N/A'}")
    print(f"  Retrieved: {result.total_retrieved}  |  Returned: {result.total_returned}")
    print(f"  Method: {result.ranking_method}  |  Time: {result.processing_time_ms}ms")
    print(f"{'═'*70}\n")

    for c in result.candidates:
        bar = "█" * int(c.weighted_score / 10) + "░" * (10 - int(c.weighted_score / 10))
        flags = f"  ⚠ {', '.join(c.red_flags)}" if c.red_flags else ""
        print(f"  #{c.rank:2d}  {c.candidate_name:<30} {bar} {c.weighted_score:5.1f}/100")
        print(f"       Skills: {', '.join(c.skills[:5])}")
        if c.project_skills:
            print(f"       Project skills: {', '.join(c.project_skills[:8])}")
        print(f"       {c.summary}{flags}")
        print()

    if args.json:
        print("\nFull JSON output:")
        print(json.dumps(result.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd-text", help="Inline job description text")
    parser.add_argument("--jd-key", help="S3 key of the JD file")
    parser.add_argument(
        "--bucket",
        default=settings.aws.s3_jd_bucket,
        help="S3 bucket for JD (defaults to AWS_S3_JD_BUCKET)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--reference-key",
        action="append",
        dest="reference_key",
        default=None,
        help="S3 key of an already-selected resume (LLM calibration); repeat for multiple",
    )
    parser.add_argument(
        "--resume-bucket",
        default=None,
        help="S3 bucket for --reference-key (defaults to AWS_S3_RESUME_BUCKET)",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    args = parser.parse_args()

    if not args.jd_text and not args.jd_key:
        parser.error("Provide --jd-text or --jd-key")

    asyncio.run(run(args))
