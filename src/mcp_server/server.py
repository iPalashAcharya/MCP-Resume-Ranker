"""
mcp_server/server.py — Main FastMCP server entry point.

Registers all MCP tools and exposes health + status endpoints.
Supports both SSE (HTTP streaming) and stdio transports.

Run with:
    python -m src.mcp_server.server          # stdio (for Claude Desktop)
    uvicorn src.mcp_server.server:app        # SSE (for Node.js backend)
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from src.config import configure_logging, get_logger, settings
from src.mcp_server.tools.ranking_tool import (
    RankCandidatesInput,
    RankCandidatesOutput,
    rank_candidates_for_job,
)
from src.mcp_server.tools.resume_tool import (
    IngestResumeInput,
    IngestResumeOutput,
    ingest_resume,
)
from src.rag.cache import cache
from src.rag.embeddings import embedder
from src.rag.vector_store import vector_store

# ─── Logging ─────────────────────────────────────────────────────────────────

configure_logging()
logger = get_logger(__name__)

# ─── Startup / Shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None, None]:
    """Initialise heavy resources once at startup."""
    logger.info("server.startup", name=settings.mcp.name, version="1.0.0")

    # Pre-load embedding model in background
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, embedder.load)
    logger.info("server.embeddings_ready", model=settings.embedding.model_name)

    # Connect to Milvus
    await loop.run_in_executor(None, vector_store.connect)
    logger.info("server.milvus_ready")

    # Log cache status
    ch = cache.health_check()
    logger.info("server.cache_status", **ch)

    logger.info("server.ready")
    yield

    logger.info("server.shutdown")


# ─── MCP App ─────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name=settings.mcp.name,
    instructions="""
Resume Ranker MCP Server.

Tools available:
  1. ingest_resume         — Index a new resume from AWS S3 into the vector store
  2. rank_candidates       — Retrieve and LLM-rank candidates for a given job profile
  3. delete_resume         — Remove a resume from the vector store
  4. health_check          — System health status
  5. list_indexed_resumes  — Count and metadata of all indexed resumes
""",
    lifespan=lifespan,
)


# ─── Tool: ingest_resume ──────────────────────────────────────────────────────

@mcp.tool(
    name="ingest_resume",
    description=(
        "Ingest a resume from AWS S3 into the RAG vector store. "
        "Called automatically when a new resume is uploaded to S3. "
        "Parses PDF/DOCX/TXT, chunks the text, generates Qwen embeddings, "
        "and upserts into Milvus."
    ),
)
async def tool_ingest_resume(
    s3_key: str,
    s3_bucket: str | None = None,
    resume_id: str | None = None,
    force_reindex: bool = False,
) -> dict:
    """
    Args:
        s3_key: S3 object key for the resume file (e.g. "resumes/john_doe.pdf")
        s3_bucket: S3 bucket name (default: AWS_S3_RESUME_BUCKET env var)
        resume_id: Optional stable ID (auto-derived from ETag if not given)
        force_reindex: Re-process even if already indexed
    """
    try:
        result = await ingest_resume(
            IngestResumeInput(
                s3_key=s3_key,
                s3_bucket=s3_bucket,
                resume_id=resume_id,
                force_reindex=force_reindex,
            )
        )
        return result.model_dump()
    except FileNotFoundError as exc:
        raise ToolError(f"Resume not found in S3: {exc}") from exc
    except ValueError as exc:
        raise ToolError(f"Invalid resume: {exc}") from exc
    except Exception as exc:
        logger.exception("tool.ingest_resume.error", error=str(exc))
        raise ToolError(f"Ingestion failed: {exc}") from exc


# ─── Tool: rank_candidates ────────────────────────────────────────────────────

@mcp.tool(
    name="rank_candidates",
    description=(
        "Rank candidates from the vector store against a job profile. "
        "Accepts either a raw JD text string or an S3 key pointing to a JD file. "
        "Optional reference_selected_resume_s3_keys: resumes already selected for this JD "
        "are sent to the LLM as calibration context only (not ranked). "
        "Returns a ranked list of candidates with LLM-generated scores, summaries, "
        "and red flags."
    ),
)
async def tool_rank_candidates(
    jd_text: str | None = None,
    jd_s3_key: str | None = None,
    jd_s3_bucket: str | None = None,
    jd_id: str | None = None,
    top_k: int = 10,
    min_experience_years: float | None = None,
    required_skills_filter: list[str] | None = None,
    include_presigned_urls: bool = False,
    use_cache: bool = True,
    reference_selected_resume_s3_keys: list[str] | None = None,
    resume_s3_bucket: str | None = None,
) -> dict:
    """
    Args:
        jd_text: Raw job description text (use this OR jd_s3_key)
        jd_s3_key: S3 key of the JD file (use this OR jd_text)
        jd_s3_bucket: S3 bucket for the JD (default: AWS_S3_JD_BUCKET)
        jd_id: Optional stable ID for caching (auto-generated if not given)
        top_k: Number of top candidates to return (1–50, default 10)
        min_experience_years: Hard filter — exclude candidates below this threshold
        required_skills_filter: Hard filter — candidates must have ALL listed skills
        include_presigned_urls: Return 1-hour pre-signed S3 URLs for each resume
        use_cache: If false, bypass cached ranking and re-run from scratch
        reference_selected_resume_s3_keys: Optional S3 keys of already-selected resumes (LLM calibration only)
        resume_s3_bucket: Bucket for those reference resumes (default: resume bucket)
    """
    try:
        result = await rank_candidates_for_job(
            RankCandidatesInput(
                jd_text=jd_text,
                jd_s3_key=jd_s3_key,
                jd_s3_bucket=jd_s3_bucket,
                jd_id=jd_id,
                top_k=top_k,
                min_experience_years=min_experience_years,
                required_skills_filter=required_skills_filter,
                include_presigned_urls=include_presigned_urls,
                use_cache=use_cache,
                reference_selected_resume_s3_keys=reference_selected_resume_s3_keys,
                resume_s3_bucket=resume_s3_bucket,
            )
        )
        return result.model_dump()
    except ValueError as exc:
        raise ToolError(f"Invalid input: {exc}") from exc
    except Exception as exc:
        logger.exception("tool.rank_candidates.error", error=str(exc))
        raise ToolError(f"Ranking failed: {exc}") from exc


# ─── Tool: delete_resume ──────────────────────────────────────────────────────

@mcp.tool(
    name="delete_resume",
    description="Remove a resume and all its chunks from the vector store.",
)
async def tool_delete_resume(resume_id: str) -> dict:
    """
    Args:
        resume_id: The resume_id returned when the resume was ingested
    """
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, vector_store.delete_resume, resume_id)
        cache.invalidate_resume(resume_id)
        return {"success": True, "resume_id": resume_id, "message": "Resume deleted"}
    except Exception as exc:
        raise ToolError(f"Deletion failed: {exc}") from exc


# ─── Tool: health_check ───────────────────────────────────────────────────────

@mcp.tool(
    name="health_check",
    description="Returns health status of all system components (Milvus, Redis, Embeddings).",
)
async def tool_health_check() -> dict:
    loop = asyncio.get_event_loop()
    milvus_health = await loop.run_in_executor(None, vector_store.health_check)
    cache_health = cache.health_check()
    emb_ready = embedder._loaded

    return {
        "status": "healthy" if milvus_health["status"] == "healthy" else "degraded",
        "timestamp": time.time(),
        "components": {
            "milvus": milvus_health,
            "cache": cache_health,
            "embeddings": {
                "status": "loaded" if emb_ready else "not_loaded",
                "model": settings.embedding.model_name,
            },
        },
        "config": {
            "embedding_model": settings.embedding.model_name,
            "llm_model": settings.llm.model_name,
            "llm_provider": settings.llm.provider.value,
            "rag_top_k": settings.rag.top_k,
            "final_rank_k": settings.rag.final_rank_k,
        },
    }


# ─── Tool: list_indexed_resumes ───────────────────────────────────────────────

@mcp.tool(
    name="list_indexed_resumes",
    description="Returns the count of indexed resumes in the vector store.",
)
async def tool_list_indexed() -> dict:
    loop = asyncio.get_event_loop()
    count = await loop.run_in_executor(None, vector_store.count_resumes)
    return {
        "total_indexed": count,
        "collection": settings.milvus.collection_name,
    }


# ─── ASGI app (for SSE/HTTP transport) ───────────────────────────────────────

# When running under uvicorn with SSE transport:
#   uvicorn src.mcp_server.server:app --host 0.0.0.0 --port 8000
app = mcp.http_app(path="/mcp")


# ─── stdio entrypoint ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    transport = settings.mcp.transport.value

    if transport == "stdio":
        logger.info("server.running_stdio")
        mcp.run(transport="stdio")
    else:
        import uvicorn
        logger.info("server.running_sse", host=settings.mcp.host, port=settings.mcp.port)
        uvicorn.run(
            "src.mcp_server.server:app",
            host=settings.mcp.host,
            port=settings.mcp.port,
            reload=False,
            workers=settings.max_workers,
            log_config=None,  # structlog handles logging
        )