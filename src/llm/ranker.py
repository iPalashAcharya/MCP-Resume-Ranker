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

# Final score = formula minus red-flag penalty (see `_apply_deterministic_scoring`).
_WEIGHT_SKILLS = 0.40
_WEIGHT_EXPERIENCE = 0.30
_WEIGHT_EDUCATION = 0.15
_WEIGHT_OVERALL = 0.15
_RED_FLAG_PENALTY_PER_ITEM = 5.0
_RED_FLAG_PENALTY_CAP = 15.0

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a strict technical recruiter scoring candidates against ONE job description.

You receive a job title, required/preferred skills, minimum experience, education expectations,
responsibilities, and for each candidate: structured skills, experience_years, education, vector similarity,
and resume text (merged chunks when available — may still be truncated).

When experience_years is greater than zero or education is non-empty, treat those structured fields as
ground truth from the applicant tracking system. Do NOT add red_flags like "lack of experience" or
"insufficient education details" solely because the resume text snippet omits those sections — score
experience_fit and education_fit using structured fields plus any supporting text.

Scoring rules (use the FULL 0–100 range; avoid clustering everyone in the 80s–90s):
- Typical strong-but-not-perfect matches: about 65–85 on relevant dimensions.
- Reserve 90–100 for rare, near-perfect alignment on title/domain, skills, and experience together.
- skills_match: overlap and depth vs JD **required** skills (not just keyword overlap).
- experience_fit: compares candidate years to JD minimum; penalize if leadership/seniority in a **different domain** does not compensate for wrong track.
- education_fit: meets or exceeds stated education requirements; use 70 if education is unknown from structured fields and resume_text.
- overall_relevance: holistic fit to THIS role — **job title and domain matter**. Examples:
  • If the JD is for a Web Application Technical Lead and the candidate’s primary leadership is
    Mobile (or another clearly different product/domain), overall_relevance must be **low** (about 25–55)
    unless the excerpt shows direct web-app / full-stack technical leadership for similar systems.
  • Generic “strong engineer” without domain/title fit must not exceed about 75 on overall_relevance.

red_flags:
- List concrete issues (e.g. title/domain mismatch, missing critical required skills, thin evidence in resume_text).
- If red_flags is non-empty, **overall_relevance must reflect that** (typically cap overall_relevance at about 60).

Return ONLY JSON (no markdown). Use EITHER:
- a JSON array of objects, one per candidate, OR
- a single object: {"candidates": [ ... ] }

Each object MUST include:
  "s3_key", "candidate_name",
  "skills_match", "experience_fit", "education_fit", "overall_relevance" (integers 0–100),
  "summary" (1–2 sentences),
  "red_flags" (array of strings; [] if none)

You MAY include "weighted_score" but it WILL BE IGNORED — the server recomputes it from the four scores.
Do not assign rank; the server will sort.
"""

REFERENCE_CALIBRATION_SUFFIX = """
When a "Reference profiles" section is present in the user message:
Those profiles are real candidates already selected for this job. Use them ONLY to calibrate
seniority, skills depth, domain, and presentation style—what "good" looks like for this hire.
Do NOT score reference profiles. Do NOT include them in your JSON output.
Score ONLY the people listed under "Candidates to Rank".
"""

USER_PROMPT_JD_BODY = """## Job Description
Title: {title}
Company: {company}
Required Skills: {required_skills}
Preferred Skills: {preferred_skills}
Minimum Experience: {min_experience} years
Education Requirements: {education_reqs}

## Responsibilities
{responsibilities}
"""

USER_PROMPT_REFERENCE_BLOCK = """
## Reference profiles (already selected for this role)
Use only to calibrate your scoring bar. Do not score these people; they must not appear in your JSON output.
{reference_json}
"""

USER_PROMPT_CANDIDATES_BLOCK = """
## Candidates to Rank ({n_candidates} candidates)
{candidates_json}

Score all {n_candidates} candidates with the strict rules above. Return JSON array or {{"candidates": [...]}}.
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


def _clamp_0_100(value: Any, default: float = 50.0) -> float:
    if value is None:
        return default
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return default


def _apply_deterministic_scoring(row: dict) -> dict:
    """Clamp subscores and set weighted_score from the agreed formula + red-flag penalty."""
    row["skills_match"] = _clamp_0_100(row.get("skills_match"), 50.0)
    row["experience_fit"] = _clamp_0_100(row.get("experience_fit"), 50.0)
    row["education_fit"] = _clamp_0_100(row.get("education_fit"), 50.0)
    row["overall_relevance"] = _clamp_0_100(row.get("overall_relevance"), 50.0)

    base = (
        _WEIGHT_SKILLS * row["skills_match"]
        + _WEIGHT_EXPERIENCE * row["experience_fit"]
        + _WEIGHT_EDUCATION * row["education_fit"]
        + _WEIGHT_OVERALL * row["overall_relevance"]
    )

    flags = row.get("red_flags")
    penalty = 0.0
    if isinstance(flags, list) and flags:
        penalty = min(_RED_FLAG_PENALTY_CAP, _RED_FLAG_PENALTY_PER_ITEM * len(flags))

    row["weighted_score"] = round(max(0.0, min(100.0, base - penalty)), 2)
    if penalty > 0:
        row["weighted_score_before_red_flags"] = round(max(0.0, min(100.0, base)), 2)
    return row


# ─── Ranker ───────────────────────────────────────────────────────────────────

class CandidateRanker:
    """Two-stage ranker: vector retrieval → LLM re-ranking."""

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    def _client_(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = _get_llm_client()
        return self._client

    @staticmethod
    def _system_content(has_references: bool) -> str:
        if has_references:
            return SYSTEM_PROMPT + "\n\n" + REFERENCE_CALIBRATION_SUFFIX
        return SYSTEM_PROMPT

    @staticmethod
    def _budget_reference_profiles(
        refs: List[dict],
        *,
        per_max: Optional[int] = None,
        section_max: Optional[int] = None,
    ) -> List[dict]:
        """Apply per-reference and section-level character caps (mutates copies)."""
        if not refs:
            return []
        per = (
            settings.rag.llm_per_reference_resume_max_chars
            if per_max is None
            else per_max
        )
        section = (
            settings.rag.llm_reference_section_max_chars
            if section_max is None
            else section_max
        )
        out = []
        for r in refs:
            text = (r.get("profile_text") or "").strip()
            if not text:
                continue
            text = text[: max(per, 0)]
            out.append({
                "s3_key": r.get("s3_key", ""),
                "candidate_name": r.get("candidate_name", "Unknown"),
                "profile_text": text,
            })
        if section > 0 and out:
            total = sum(len(x["profile_text"]) for x in out)
            if total > section:
                per_alloc = max(section // len(out), 150)
                for x in out:
                    x["profile_text"] = x["profile_text"][:per_alloc]
        return out

    def _build_user_prompt(
        self,
        jd: JobDescription,
        candidates: List[dict],
        reference_profiles: Optional[List[dict]] = None,
        max_user_override: Optional[int] = None,
    ) -> str:
        max_user = (
            max_user_override
            if max_user_override is not None
            else settings.rag.llm_user_prompt_max_chars
        )
        if max_user <= 0:
            max_user = 26000

        jd_raw_full = ""
        raw_max = settings.rag.llm_jd_raw_max_chars
        if raw_max > 0 and (jd.raw_text or "").strip():
            snippet = jd.raw_text[:raw_max]
            jd_raw_full = f"\n## Verbatim JD excerpt (truncated)\n{snippet}\n"

        jd_body = USER_PROMPT_JD_BODY.format(
            title=jd.title,
            company=jd.company or "Not specified",
            required_skills=", ".join(jd.required_skills[:15]),
            preferred_skills=", ".join(jd.preferred_skills[:10]),
            min_experience=jd.min_experience_years,
            education_reqs=", ".join(jd.education_requirements[:4]) or "Not specified",
            responsibilities="\n".join(f"• {r}" for r in jd.responsibilities[:8]),
        )

        ref_in = reference_profiles or []
        n_cand = len(candidates)

        per_body = settings.rag.llm_candidate_context_max_chars
        if per_body <= 0:
            per_body = 2200
        per_body = max(per_body, 400)

        per_ref = max(settings.rag.llm_per_reference_resume_max_chars, 300)
        sec_ref = settings.rag.llm_reference_section_max_chars

        include_jd_raw = bool(jd_raw_full)
        user_prompt = ""

        def pack(jd_raw_use: str, p_body: int, p_ref: int, s_ref: int) -> str:
            refs = self._budget_reference_profiles(
                ref_in, per_max=p_ref, section_max=s_ref,
            )
            summaries = []
            for i, c in enumerate(candidates, 1):
                body = (c.get("merged_chunk_text") or c.get("best_chunk") or "").strip()
                body = body[:p_body]
                summaries.append({
                    "index": i,
                    "s3_key": c.get("s3_key", ""),
                    "candidate_name": c.get("candidate_name", "Unknown"),
                    "skills": c.get("skills", []),
                    "experience_years": c.get("experience_years", 0),
                    "education": c.get("education", []),
                    "vector_similarity_score": round(c.get("score", 0), 4),
                    "resume_text": body,
                })
            cj = json.dumps(summaries, indent=2)
            if refs:
                return (
                    jd_body
                    + jd_raw_use
                    + USER_PROMPT_REFERENCE_BLOCK.format(
                        reference_json=json.dumps(refs, indent=2),
                    )
                    + USER_PROMPT_CANDIDATES_BLOCK.format(
                        n_candidates=n_cand,
                        candidates_json=cj,
                    )
                )
            return (
                jd_body
                + jd_raw_use
                + USER_PROMPT_CANDIDATES_BLOCK.format(
                    n_candidates=n_cand,
                    candidates_json=cj,
                )
            )

        for iteration in range(28):
            jd_raw_use = jd_raw_full if include_jd_raw else ""
            user_prompt = pack(jd_raw_use, per_body, per_ref, sec_ref)
            if len(user_prompt) <= max_user:
                break
            if per_body > 500:
                per_body = int(per_body * 0.84)
            elif per_ref > 350:
                per_ref = int(per_ref * 0.84)
            elif sec_ref > 800:
                sec_ref = int(sec_ref * 0.84)
            elif include_jd_raw:
                include_jd_raw = False
            else:
                per_body = max(400, int(per_body * 0.9))
                per_ref = max(300, int(per_ref * 0.9))

        if len(user_prompt) > max_user:
            user_prompt = pack("", 400, 300, min(sec_ref, 4000) if sec_ref > 0 else 0)

        if len(user_prompt) > max_user:
            logger.warning(
                "ranker.prompt_still_over_budget",
                chars=len(user_prompt),
                max_user=max_user,
            )

        est_tokens = max(len(user_prompt) // 4, 1)
        logger.info(
            "ranker.user_prompt_size",
            chars=len(user_prompt),
            est_tokens=est_tokens,
            max_user=max_user,
        )
        return user_prompt

    async def rerank_with_llm(
        self,
        jd: JobDescription,
        candidates: List[dict],
        reference_profiles: Optional[List[dict]] = None,
    ) -> List[dict]:
        """Ask the LLM to score and re-rank candidates."""
        if not candidates:
            return []

        user_prompt = self._build_user_prompt(jd, candidates, reference_profiles)
        has_refs = "## Reference profiles" in user_prompt
        logger.info(
            "ranker.llm_rerank",
            model=settings.llm.model_name,
            n=len(candidates),
            references=has_refs,
        )

        async def _call_llm(umax: Optional[int] = None) -> Any:
            up = (
                user_prompt
                if umax is None
                else self._build_user_prompt(
                    jd, candidates, reference_profiles, max_user_override=umax,
                )
            )
            hr = "## Reference profiles" in up
            return await self._client_().chat.completions.create(
                model=settings.llm.model_name,
                messages=[
                    {"role": "system", "content": self._system_content(hr)},
                    {"role": "user", "content": up},
                ],
                max_tokens=settings.llm.max_tokens,
                temperature=settings.llm.temperature,
                response_format={"type": "json_object"} if _supports_json_mode() else None,
            )

        try:
            response = await _call_llm()
        except Exception as exc:
            err_s = str(exc).lower()
            if "413" in str(exc) or "request too large" in err_s or "tokens per minute" in err_s:
                logger.warning("ranker.llm_oversized_retry", error=str(exc)[:200])
                try:
                    response = await _call_llm(umax=20000)
                except Exception as exc2:
                    logger.error("ranker.llm_error", error=str(exc2), fallback="vector_score")
                    return self._fallback_rank(candidates)
            else:
                logger.error("ranker.llm_error", error=str(exc), fallback="vector_score")
                return self._fallback_rank(candidates)

        try:
            content = response.choices[0].message.content or "[]"
            ranked = self._parse_llm_response(content, candidates)

            logger.info("ranker.llm_rerank_done", ranked_count=len(ranked))
            return ranked

        except Exception as exc:
            logger.error("ranker.llm_parse_or_rank_error", error=str(exc), fallback="vector_score")
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

            for r in ranked:
                if isinstance(r, dict):
                    _apply_deterministic_scoring(r)

            # Re-sort by deterministic weighted_score descending and assign ranks
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
        reference_profiles: Optional[List[dict]] = None,
    ) -> List[dict]:
        """Full two-stage ranking pipeline."""
        final_k = final_k or settings.rag.final_rank_k

        # Stage 2: LLM re-ranking
        ranked = await self.rerank_with_llm(
            jd,
            retrieved_candidates,
            reference_profiles=reference_profiles,
        )

        # Enrich results with original metadata fields if missing
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
    """Return True for providers/models known to support OpenAI-style json_object responses."""
    model = settings.llm.model_name.lower()
    prov = settings.llm.provider.value
    if any(p in model for p in ("gpt-4", "gpt-3.5", "qwen")):
        return True
    if prov == "groq" and any(p in model for p in ("llama", "mixtral", "gemma")):
        return True
    return False


# Module-level singleton
candidate_ranker = CandidateRanker()
