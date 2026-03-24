"""
mcp_server/tools/ranking_tool.py — MCP Tool: rank_candidates_for_job

Accepts a job profile (text or S3 key) and returns the best-matched,
LLM-ranked candidates from the vector store.

Pipeline:
  1. Parse / resolve the job description
  2. Embed the JD (with cache)
  3. Vector search in Milvus (top-K candidates)
  4. LLM re-ranking (multi-criteria scoring)
  5. (Optional) Enrich with pre-signed S3 download URLs
  6. Return structured ranking response
"""
from __future__ import annotations

import hashlib
import time
import uuid
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

from src.config import get_logger, settings
from src.llm.ranker import candidate_ranker
from src.parsers.jd_parser import JobDescription, jd_parser
from src.rag.cache import cache
from src.rag.embeddings import embedder
from src.rag.vector_store import vector_store
from src.s3.client import s3_client

logger = get_logger(__name__)


# ─── Input / Output Schemas ───────────────────────────────────────────────────

class RankCandidatesInput(BaseModel):
    """
    Exactly one of `jd_text` or `jd_s3_key` must be provided.
    """
    jd_text: Optional[str] = Field(
        None,
        description="Raw job description text (if not stored in S3)",
    )
    jd_s3_key: Optional[str] = Field(
        None,
        description="S3 key of the JD file (PDF/DOCX/TXT)",
    )
    jd_s3_bucket: Optional[str] = Field(
        None,
        description="S3 bucket for the JD. Defaults to AWS_S3_JD_BUCKET env var",
    )
    jd_id: Optional[str] = Field(
        None,
        description="Stable identifier for this JD (used for caching). Auto-generated if not given",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top candidates to return",
    )
    min_experience_years: Optional[float] = Field(
        None,
        description="Optional hard filter: exclude candidates below this experience threshold",
    )
    required_skills_filter: Optional[List[str]] = Field(
        None,
        description="Optional hard filter: candidates must possess ALL of these skills",
    )
    include_presigned_urls: bool = Field(
        False,
        description="If true, include 1-hour pre-signed S3 download URLs for each resume",
    )
    use_cache: bool = Field(
        True,
        description="If false, bypass cached ranking results and re-rank from scratch",
    )

    @model_validator(mode="after")
    def require_one_jd_source(self) -> "RankCandidatesInput":
        if not self.jd_text and not self.jd_s3_key:
            raise ValueError("Provide either jd_text or jd_s3_key")
        return self


class CandidateResult(BaseModel):
    rank: int
    candidate_name: str
    s3_key: str
    s3_bucket: Optional[str]
    email: Optional[str]
    skills: List[str]
    experience_years: Optional[float]
    education: List[str]
    weighted_score: float
    skills_match: Optional[float]
    experience_fit: Optional[float]
    education_fit: Optional[float]
    overall_relevance: Optional[float]
    vector_score: Optional[float]
    summary: str
    red_flags: List[str]
    resume_download_url: Optional[str] = None


class RankCandidatesOutput(BaseModel):
    success: bool
    jd_id: str
    jd_title: str
    jd_company: Optional[str]
    total_retrieved: int
    total_returned: int
    candidates: List[CandidateResult]
    processing_time_ms: int
    ranking_method: str
    message: str


# ─── Tool Implementation ──────────────────────────────────────────────────────

async def rank_candidates_for_job(params: RankCandidatesInput) -> RankCandidatesOutput:
    """
    Retrieve and LLM-rank the best candidates for a given job profile.
    """
    start = time.monotonic()
    logger.info("tool.rank_candidates.start", jd_s3_key=params.jd_s3_key, top_k=params.top_k)

    # ── Step 1: Resolve / parse the JD ───────────────────────────────────────
    jd = await _resolve_jd(params)

    # ── Step 2: Check ranking cache ───────────────────────────────────────────
    if params.use_cache:
        cached_result = cache.get_ranking(jd.jd_id)
        if cached_result is not None:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info("tool.rank_candidates.cache_hit", jd_id=jd.jd_id)
            cached_result["processing_time_ms"] = elapsed_ms
            cached_result["message"] += " (cached)"
            return RankCandidatesOutput(**cached_result)

    # ── Step 3: Embed the JD ─────────────────────────────────────────────────
    jd_emb_key = hashlib.sha256(jd.embedding_text.encode()).hexdigest()
    jd_embedding = cache.get_embedding(jd_emb_key)
    if jd_embedding is None:
        jd_embedding = await embedder.aembed_single(jd.embedding_text)
        cache.set_embedding(jd_emb_key, jd_embedding)

    # Optionally store JD in its own collection for analytics
    try:
        await vector_store.aupsert_jd(
            jd_id=jd.jd_id,
            metadata=jd.to_metadata(),
            embedding=jd_embedding,
        )
    except Exception as exc:
        logger.warning("tool.rank_candidates.jd_upsert_warning", error=str(exc))

    # ── Step 4: Milvus vector search ──────────────────────────────────────────
    filters = _build_milvus_filters(params)
    retrieved = await vector_store.asearch_resumes(
        query_embedding=jd_embedding,
        top_k=params.top_k * 3,  # over-fetch to give LLM more candidates
    )

    # Apply hard skill / experience filters post-retrieval
    retrieved = _apply_hard_filters(retrieved, params)

    logger.info("tool.rank_candidates.retrieved", count=len(retrieved))

    if not retrieved:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return RankCandidatesOutput(
            success=True,
            jd_id=jd.jd_id,
            jd_title=jd.title,
            jd_company=jd.company,
            total_retrieved=0,
            total_returned=0,
            candidates=[],
            processing_time_ms=elapsed_ms,
            ranking_method="vector",
            message="No candidates found in the vector store for this job profile.",
        )

    # ── Step 5: LLM re-ranking ────────────────────────────────────────────────
    ranked_raw = await candidate_ranker.rank_candidates(
        jd=jd,
        retrieved_candidates=retrieved,
        final_k=params.top_k,
    )

    # ── Step 6: Build output (optionally enrich with presigned URLs) ──────────
    candidates: List[CandidateResult] = []
    for r in ranked_raw:
        presigned = None
        if params.include_presigned_urls and r.get("s3_key") and r.get("s3_bucket"):
            try:
                presigned = await s3_client.presigned_url(r["s3_bucket"], r["s3_key"])
            except Exception as exc:
                logger.warning("tool.rank_candidates.presign_error", error=str(exc))

        candidates.append(
            CandidateResult(
                rank=r.get("rank", 0),
                candidate_name=r.get("candidate_name", "Unknown"),
                s3_key=r.get("s3_key", ""),
                s3_bucket=r.get("s3_bucket"),
                email=r.get("email"),
                skills=r.get("skills", []) if isinstance(r.get("skills"), list)
                        else str(r.get("skills", "")).split(","),
                experience_years=r.get("experience_years"),
                education=r.get("education", []) if isinstance(r.get("education"), list)
                           else str(r.get("education", "")).split(" | "),
                weighted_score=r.get("weighted_score", 0.0),
                skills_match=r.get("skills_match"),
                experience_fit=r.get("experience_fit"),
                education_fit=r.get("education_fit"),
                overall_relevance=r.get("overall_relevance"),
                vector_score=r.get("vector_score"),
                summary=r.get("summary", ""),
                red_flags=r.get("red_flags", []),
                resume_download_url=presigned,
            )
        )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    ranking_method = "llm+vector" if ranked_raw and "skills_match" in ranked_raw[0] else "vector_fallback"

    output = RankCandidatesOutput(
        success=True,
        jd_id=jd.jd_id,
        jd_title=jd.title,
        jd_company=jd.company,
        total_retrieved=len(retrieved),
        total_returned=len(candidates),
        candidates=candidates,
        processing_time_ms=elapsed_ms,
        ranking_method=ranking_method,
        message=f"Ranked {len(candidates)} candidates for '{jd.title}' in {elapsed_ms}ms",
    )

    # Cache the result
    cache.set_ranking(jd.jd_id, output.model_dump())

    logger.info(
        "tool.rank_candidates.done",
        jd_id=jd.jd_id,
        returned=len(candidates),
        elapsed_ms=elapsed_ms,
        method=ranking_method,
    )
    return output


# ─── Helpers ─────────────────────────────────────────────────────────────────

async def _resolve_jd(params: RankCandidatesInput) -> JobDescription:
    """Parse JD from text or download from S3."""
    if params.jd_text:
        jd_id = params.jd_id or _jd_id_from_text(params.jd_text)
        return jd_parser.parse_from_text(params.jd_text, jd_id=jd_id)

    # S3 path
    bucket = params.jd_s3_bucket or settings.aws.s3_jd_bucket
    key = _normalize_and_guard_jd_key(bucket, params.jd_s3_key or "")
    meta = await s3_client.head_object(bucket, key)
    jd_id = params.jd_id or _jd_id_from_etag(meta["etag"], key)
    ext = s3_client.infer_file_extension(key)
    raw_bytes = await s3_client.download_bytes(bucket, key)
    return jd_parser.parse(
        data=raw_bytes,
        extension=ext,
        jd_id=jd_id,
        s3_key=key,
        s3_bucket=bucket,
    )


def _normalize_and_guard_jd_key(bucket: str, raw_key: str) -> str:
    """Normalise accepted key formats and enforce configured JD prefix guard."""
    key = raw_key.strip()

    # Accept s3://bucket/path and convert to plain key.
    if key.startswith("s3://"):
        parsed = urlparse(key)
        if parsed.netloc and parsed.netloc != bucket:
            raise ValueError(f"S3 URI bucket '{parsed.netloc}' does not match configured bucket '{bucket}'")
        key = parsed.path.lstrip("/")

    # Accept bucket/key and convert to plain key.
    bucket_prefix = f"{bucket}/"
    if key.startswith(bucket_prefix):
        key = key[len(bucket_prefix):]

    allowed_prefix = settings.aws.jd_key_prefix.strip().lstrip("/")
    if allowed_prefix and not key.startswith(allowed_prefix):
        raise ValueError(
            f"JD key must start with '{allowed_prefix}'. Received: '{key}'"
        )

    return key


def _jd_id_from_text(text: str) -> str:
    return "jd_" + hashlib.sha256(text.encode()).hexdigest()[:20]


def _jd_id_from_etag(etag: str, key: str) -> str:
    return "jd_" + hashlib.sha256(f"{etag}:{key}".encode()).hexdigest()[:20]


def _build_milvus_filters(params: RankCandidatesInput) -> Optional[str]:
    """Build a Milvus boolean expression for hard filters."""
    exprs = []
    if params.min_experience_years is not None:
        exprs.append(f"experience_years >= {params.min_experience_years}")
    return " and ".join(exprs) if exprs else None


def _apply_hard_filters(candidates: List[dict], params: RankCandidatesInput) -> List[dict]:
    """Post-retrieval hard filters (skill-based)."""
    if not params.required_skills_filter:
        return candidates

    required = {s.lower() for s in params.required_skills_filter}
    filtered = []
    for c in candidates:
        candidate_skills = {s.lower() for s in c.get("skills", [])}
        if required.issubset(candidate_skills):
            filtered.append(c)

    logger.info(
        "tool.rank_candidates.hard_filter",
        before=len(candidates),
        after=len(filtered),
        required_skills=params.required_skills_filter,
    )
    return filtered
