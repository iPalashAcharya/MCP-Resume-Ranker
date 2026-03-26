from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from src.config import settings
from src.parsers.resume_parser import ResumeDocument, ResumeSection

@dataclass
class ResumeChunk:
    text: str
    section_title: str
    importance: float
    resume_id: str
    chunk_index: int
    metadata: dict[str, Any]  # carries resume-level metadata for vector store

def chunk_resume_document(
    doc: ResumeDocument,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[ResumeChunk]:
    """
    Chunk a fully parsed ResumeDocument section-by-section.
    Each chunk stays within its section boundary — no cross-section bleed.
    """
    chunk_size, overlap = _normalize_chunk_params(chunk_size, overlap)
    chunks: List[ResumeChunk] = []
    base_metadata = doc.to_metadata()  # reuse your existing to_metadata()
    chunk_index = 0

    for section in doc.sections:
        section_chunks = _chunk_section(section, chunk_size, overlap)

        for chunk_text in section_chunks:
            chunks.append(ResumeChunk(
                text=_prefix(doc.candidate_name, section.title, chunk_text),
                section_title=section.title,
                importance=section.importance,
                resume_id=doc.resume_id,
                chunk_index=chunk_index,
                metadata={
                    **base_metadata,
                    "section": section.title,
                    "importance": section.importance,
                },
            ))
            chunk_index += 1

    if chunks:
        return chunks

    # Defensive fallback: parser should usually provide sections, but if not,
    # keep ingestion resilient by chunking the raw resume text.
    for idx, chunk_text in enumerate(chunk_text_from_raw(doc.full_text, chunk_size, overlap)):
        chunks.append(
            ResumeChunk(
                text=chunk_text,
                section_title="General",
                importance=1.0,
                resume_id=doc.resume_id,
                chunk_index=idx,
                metadata={
                    **base_metadata,
                    "section": "General",
                    "importance": 1.0,
                },
            )
        )
    return chunks


def chunk_text_from_raw(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[str]:
    """
    Split free text into overlapping word chunks.
    Useful for fallback/compatibility paths outside structured section chunking.
    """
    chunk_size, overlap = _normalize_chunk_params(chunk_size, overlap)
    clean_text = text.strip()
    if not clean_text:
        return []

    words = clean_text.split()
    if len(words) <= chunk_size:
        return [clean_text]

    chunks: List[str] = []
    step = chunk_size - overlap
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        i += step

    return chunks or [clean_text]


def _chunk_section(section: ResumeSection, chunk_size: int, overlap: int) -> List[str]:
    """
    Word-level chunking scoped to a single section.
    Short sections (e.g. Skills) are returned as-is.
    """
    clean_content = section.content.strip()
    if not clean_content:
        return []

    words = clean_content.split()

    # Short sections don't need splitting — return whole
    if len(words) <= chunk_size:
        return [clean_content]

    chunks: List[str] = []
    step = chunk_size - overlap
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        i += step

    return chunks


def _prefix(candidate_name: str, section_title: str, chunk_text: str) -> str:
    """
    Prepend candidate + section context to every chunk.
    This greatly improves embedding relevance for reranking.
    """
    return f"Candidate: {candidate_name}\nSection: {section_title}\n\n{chunk_text}"


def _normalize_chunk_params(chunk_size: int | None, overlap: int | None) -> tuple[int, int]:
    """
    Resolve chunk parameters from explicit values or settings, then validate.
    Prevents invalid configs (e.g., overlap >= chunk_size) from causing hangs.
    """
    resolved_chunk_size = chunk_size if chunk_size is not None else settings.rag.chunk_size
    resolved_overlap = overlap if overlap is not None else settings.rag.chunk_overlap

    if resolved_chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {resolved_chunk_size}")
    if resolved_overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {resolved_overlap}")
    if resolved_overlap >= resolved_chunk_size:
        raise ValueError(
            f"overlap must be smaller than chunk_size (overlap={resolved_overlap}, chunk_size={resolved_chunk_size})"
        )

    return resolved_chunk_size, resolved_overlap