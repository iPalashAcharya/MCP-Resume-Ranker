"""
mcp_server/tools/resume_tool.py — MCP Tool: ingest_resume

Called by the Node.js backend whenever a new resume is uploaded to S3.
Pipeline:
  1. Validate the S3 object (format, size)
  2. Download bytes from S3
  3. Parse text + structure (PDF/DOCX/TXT)
  4. Chunk text
  5. Embed chunks with Qwen
  6. Upsert into Milvus
  7. Cache the parsed document
  8. Return ingestion summary
"""
from __future__ import annotations

import hashlib
import time
from typing import Optional

from urllib.parse import urlparse

from pydantic import BaseModel, Field

from src.config import get_logger, settings
from src.parsers.resume_parser import resume_parser
from src.rag.cache import cache
from src.rag.chunker import chunk_resume_document
from src.rag.embeddings import embedder
from src.rag.vector_store import vector_store
from src.s3.client import s3_client

logger = get_logger(__name__)


# ─── Input / Output Schemas ───────────────────────────────────────────────────

class IngestResumeInput(BaseModel):
    s3_key: str = Field(..., description="S3 object key for the resume file")
    s3_bucket: Optional[str] = Field(
        None,
        description="S3 bucket name. Defaults to AWS_S3_RESUME_BUCKET env var",
    )
    resume_id: Optional[str] = Field(
        None,
        description="Unique ID for this resume. Auto-derived from S3 ETag if not provided",
    )
    force_reindex: bool = Field(
        False,
        description="If true, re-process even if this resume_id already exists",
    )


class IngestResumeOutput(BaseModel):
    success: bool
    resume_id: str
    candidate_name: str
    s3_key: str
    s3_bucket: str
    chunks_indexed: int
    skills_detected: list
    experience_years: float
    education: list
    email: Optional[str]
    processing_time_ms: int
    message: str


# ─── Tool Implementation ──────────────────────────────────────────────────────

async def ingest_resume(params: IngestResumeInput) -> IngestResumeOutput:
    """
    Ingest a resume from S3 into the vector store.

    Triggers: S3 event → Node.js backend → MCP tool call.
    """
    start = time.monotonic()
    bucket = params.s3_bucket or settings.aws.s3_resume_bucket
    key = _normalize_and_guard_resume_key(bucket, params.s3_key)

    logger.info("tool.ingest_resume.start", s3_key=key, bucket=bucket)

    try:
        # 1. Validate file
        meta = await s3_client.validate_file(bucket, key)
        ext = s3_client.infer_file_extension(key)

        # 2. Derive or use provided resume_id
        resume_id = params.resume_id or _derive_resume_id(meta["etag"], key)

        # 3. Check cache — skip if already indexed (unless forced)
        if not params.force_reindex:
            cached = cache.get_resume_doc(resume_id)
            if cached is not None:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                logger.info("tool.ingest_resume.cache_hit", resume_id=resume_id)
                return IngestResumeOutput(
                    success=True,
                    resume_id=resume_id,
                    candidate_name=cached.candidate_name,
                    s3_key=key,
                    s3_bucket=bucket,
                    chunks_indexed=0,
                    skills_detected=cached.skills,
                    experience_years=cached.experience_years,
                    education=cached.education,
                    email=cached.email,
                    processing_time_ms=elapsed_ms,
                    message="Resume already indexed (cache hit). Use force_reindex=true to re-process.",
                )

        # 4. Download from S3
        logger.info("tool.ingest_resume.downloading", s3_key=key)
        raw_bytes = await s3_client.download_bytes(bucket, key)

        # 5. Parse
        logger.info("tool.ingest_resume.parsing", resume_id=resume_id, ext=ext)
        doc = resume_parser.parse(
            data=raw_bytes,
            extension=ext,
            resume_id=resume_id,
            s3_key=key,
            s3_bucket=bucket,
        )

        # 6. Chunk
        structured_chunks = chunk_resume_document(
            doc,
            chunk_size=settings.rag.chunk_size,
            overlap=settings.rag.chunk_overlap,
        )
        chunks = [c.text for c in structured_chunks]
        logger.info("tool.ingest_resume.chunking", resume_id=resume_id, n_chunks=len(chunks))

        # 7. Embed (with per-chunk cache)
        embeddings = []
        texts_to_embed = []
        cache_hits_idx = []

        for i, chunk in enumerate(chunks):
            cached_emb = cache.get_embedding(chunk)
            if cached_emb:
                embeddings.append((i, cached_emb))
                cache_hits_idx.append(i)
            else:
                texts_to_embed.append((i, chunk))

        if texts_to_embed:
            idxs, raw_texts = zip(*texts_to_embed)
            new_embs = await embedder.aembed_texts(list(raw_texts))
            for idx, emb, text in zip(idxs, new_embs, raw_texts):
                cache.set_embedding(text, emb)
                embeddings.append((idx, emb))

        # Restore original order
        embeddings.sort(key=lambda x: x[0])
        ordered_embeddings = [e for _, e in embeddings]

        # 8. Upsert into Milvus
        await vector_store.aupsert_resume(
            resume_id=resume_id,
            metadata=doc.to_metadata(),
            embeddings=ordered_embeddings,
            chunks=chunks,
        )

        # 9. Cache the parsed doc (so re-calls skip re-parsing)
        cache.set_resume_doc(resume_id, doc)

        # 10. Invalidate stale ranking caches (candidate pool changed)
        cache.invalidate_all_rankings()

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "tool.ingest_resume.done",
            resume_id=resume_id,
            candidate=doc.candidate_name,
            chunks=len(chunks),
            elapsed_ms=elapsed_ms,
        )

        return IngestResumeOutput(
            success=True,
            resume_id=resume_id,
            candidate_name=doc.candidate_name,
            s3_key=key,
            s3_bucket=bucket,
            chunks_indexed=len(chunks),
            skills_detected=doc.skills,
            experience_years=doc.experience_years,
            education=doc.education,
            email=doc.email,
            processing_time_ms=elapsed_ms,
            message=f"Successfully indexed {len(chunks)} chunks for {doc.candidate_name}",
        )

    except FileNotFoundError as exc:
        logger.error("tool.ingest_resume.not_found", error=str(exc))
        raise
    except ValueError as exc:
        logger.error("tool.ingest_resume.validation_error", error=str(exc))
        raise
    except Exception as exc:
        logger.exception("tool.ingest_resume.unexpected_error", error=str(exc))
        raise



def _normalize_and_guard_resume_key(bucket: str, raw_key: str) -> str:
    """Normalise accepted key formats and enforce the configured dev-prefix guard."""
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

    allowed_prefix = settings.aws.resume_key_prefix.strip().lstrip("/")
    if allowed_prefix and not key.startswith(allowed_prefix):
        raise ValueError(
            f"Resume key must start with '{allowed_prefix}'. Received: '{key}'"
        )

    return key

def _derive_resume_id(etag: str, s3_key: str) -> str:
    """Derive a stable, unique resume_id from S3 ETag + key."""
    raw = f"{etag}:{s3_key}"
    return "r_" + hashlib.sha256(raw.encode()).hexdigest()[:20]