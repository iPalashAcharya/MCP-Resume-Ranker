"""
llm/ranker.py — LLM-powered re-ranking of candidates against a job description.

Stage 1 (Vector Search): Milvus retrieves top-K candidates by semantic similarity.
Stage 2 (LLM Re-rank):   The LLM scores each candidate on multiple criteria and
                          returns a structured JSON ranking.

Recommended LLMs (fastest → most capable):
  • groq/llama-3.3-70b-versatile   — sub-second latency, great for production
  • qwen/qwen-2.5-72b-instruct     — consistent with embedding model family
  • anthropic/claude-3-haiku       — excellent instruction following, fast
  • openai/gpt-4o-mini             — reliable JSON output, cost-effective
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI  # works for OpenAI, Groq, Ollama (OpenAI-compatible)

from src.config import get_logger, settings
from src.parsers.jd_parser import JobDescription

logger = get_logger(__name__)

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert technical recruiter and AI system specialised in
candidate ranking. You will be given a job description and a list of candidate profiles
retrieved from a vector search.

Your task:
1. Score each candidate from 0–100 on EACH of these dimensions:
   - skills_match      : overlap between candidate skills and JD requirements
   - experience_fit    : years of experience vs. minimum required
   - education_fit     : education meets or exceeds requirement
   - overall_relevance : holistic judgment of how well this candidate fits the role

2. Compute a weighted_score: (skills_match * 0.40) + (experience_fit * 0.30) +
   (education_fit * 0.15) + (overall_relevance * 0.15)

3. Write a 1-2 sentence `summary` for each candidate explaining the match.

4. Flag any `red_flags` (e.g., career gaps, skill mismatch) as a list of strings.

Return ONLY a valid JSON array, no markdown fences, no extra text:
[
  {
    "s3_key": "...",
    "candidate_name": "...",
    "skills_match": 85,
    "experience_fit": 70,
    "education_fit": 90,
    "overall_relevance": 80,
    "weighted_score": 80.5,
    "summary": "...",
    "red_flags": [],
    "rank": 1
  }
]
"""

USER_PROMPT_TEMPLATE = """
## Job Description
Title: {title}
Company: {company}
Required Skills: {required_skills}
Preferred Skills: {preferred_skills}
Minimum Experience: {min_experience} years
Education Requirements: {education_reqs}

## Responsibilities
{responsibilities}

## Candidates to Rank ({n_candidates} candidates)
{candidates_json}

Rank all {n_candidates} candidates from most to least suitable. Return full JSON array.
"""


# ─── LLM Client factory ───────────────────────────────────────────────────────

def _get_llm_client() -> AsyncOpenAI:
    """
    Returns an AsyncOpenAI client configured for the chosen provider.
    All providers expose an OpenAI-compatible API:
      - OpenAI: https://api.openai.com/v1
      - Groq: https://api.groq.com/openai/v1
      - Ollama: http://localhost:11434/v1
    """
    base_url = settings.llm.base_url
    if settings.llm.provider.value == "groq" and not base_url:
        base_url = "https://api.groq.com/openai/v1"
    elif settings.llm.provider.value == "ollama" and not base_url:
        base_url = "http://localhost:11434/v1"

    return AsyncOpenAI(
        api_key=settings.llm.api_key or "ollama",  # ollama ignores key
        base_url=base_url,
    )


# ─── Ranker ───────────────────────────────────────────────────────────────────

class CandidateRanker:
    """Two-stage ranker: vector retrieval → LLM re-ranking."""

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    def _client_(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = _get_llm_client()
        return self._client

    def _build_user_prompt(self, jd: JobDescription, candidates: List[dict]) -> str:
        candidates_summary = []
        for i, c in enumerate(candidates, 1):
            candidates_summary.append({
                "index": i,
                "s3_key": c.get("s3_key", ""),
                "candidate_name": c.get("candidate_name", "Unknown"),
                "skills": c.get("skills", []),
                "experience_years": c.get("experience_years", 0),
                "education": c.get("education", []),
                "vector_similarity_score": round(c.get("score", 0), 4),
                "relevant_excerpt": c.get("best_chunk", "")[:300],
            })

        return USER_PROMPT_TEMPLATE.format(
            title=jd.title,
            company=jd.company or "Not specified",
            required_skills=", ".join(jd.required_skills[:15]),
            preferred_skills=", ".join(jd.preferred_skills[:10]),
            min_experience=jd.min_experience_years,
            education_reqs=", ".join(jd.education_requirements[:4]) or "Not specified",
            responsibilities="\n".join(f"• {r}" for r in jd.responsibilities[:8]),
            n_candidates=len(candidates),
            candidates_json=json.dumps(candidates_summary, indent=2),
        )

    async def rerank_with_llm(
        self,
        jd: JobDescription,
        candidates: List[dict],
    ) -> List[dict]:
        """Ask the LLM to score and re-rank candidates."""
        if not candidates:
            return []

        user_prompt = self._build_user_prompt(jd, candidates)
        logger.info("ranker.llm_rerank", model=settings.llm.model_name, n=len(candidates))

        try:
            response = await self._client_().chat.completions.create(
                model=settings.llm.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=settings.llm.max_tokens,
                temperature=settings.llm.temperature,
                response_format={"type": "json_object"} if _supports_json_mode() else None,
            )

            content = response.choices[0].message.content or "[]"
            ranked = self._parse_llm_response(content, candidates)

            logger.info("ranker.llm_rerank_done", ranked_count=len(ranked))
            return ranked

        except Exception as exc:
            logger.error("ranker.llm_error", error=str(exc), fallback="vector_score")
            # Graceful fallback: return candidates sorted by vector score
            return self._fallback_rank(candidates)

    def _parse_llm_response(self, content: str, fallback_candidates: List[dict]) -> List[dict]:
        """Parse LLM JSON response; fall back to vector scores on failure."""
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", content.strip())

        # Handle {"candidates": [...]} wrapper
        if clean.startswith("{"):
            try:
                wrapper = json.loads(clean)
                for key in ("candidates", "results", "rankings", "data"):
                    if key in wrapper and isinstance(wrapper[key], list):
                        clean = json.dumps(wrapper[key])
                        break
            except json.JSONDecodeError:
                pass

        try:
            ranked = json.loads(clean)
            if not isinstance(ranked, list):
                raise ValueError("Expected JSON array")

            # Re-sort by weighted_score descending and assign ranks
            ranked.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
            for i, r in enumerate(ranked, 1):
                r["rank"] = i

            return ranked
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("ranker.parse_error", error=str(exc))
            return self._fallback_rank(fallback_candidates)

    @staticmethod
    def _fallback_rank(candidates: List[dict]) -> List[dict]:
        sorted_c = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        result = []
        for i, c in enumerate(sorted_c, 1):
            result.append({
                **c,
                "rank": i,
                "weighted_score": round(c.get("score", 0) * 100, 2),
                "skills_match": None,
                "experience_fit": None,
                "education_fit": None,
                "overall_relevance": None,
                "summary": f"Vector similarity: {c.get('score', 0):.4f}",
                "red_flags": [],
                "ranking_method": "vector_fallback",
            })
        return result

    async def rank_candidates(
        self,
        jd: JobDescription,
        retrieved_candidates: List[dict],
        final_k: int = None,
    ) -> List[dict]:
        """Full two-stage ranking pipeline."""
        final_k = final_k or settings.rag.final_rank_k

        # Stage 2: LLM re-ranking
        ranked = await self.rerank_with_llm(jd, retrieved_candidates)

        # Enrich results with original metadata fields if missing
        ranked_keys = {r.get("s3_key") for r in ranked}
        retrieval_map = {c["s3_key"]: c for c in retrieved_candidates}
        for r in ranked:
            original = retrieval_map.get(r.get("s3_key"), {})
            r.setdefault("email", original.get("email"))
            r.setdefault("vector_score", original.get("score"))
            r.setdefault("skills", original.get("skills", []))
            r.setdefault("experience_years", original.get("experience_years"))
            r.setdefault("education", original.get("education", []))

        return ranked[:final_k]


def _supports_json_mode() -> bool:
    """Return True for providers known to support json_object response format."""
    model = settings.llm.model_name.lower()
    return any(p in model for p in ("gpt-4", "gpt-3.5", "qwen"))


# Module-level singleton
candidate_ranker = CandidateRanker()