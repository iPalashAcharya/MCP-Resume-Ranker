"""
Non-LLM ranking boosts layered on top of vector similarity.

1) Skills-list vs project narrative: tokens from the Skills section are checked against
   the text of the Projects section (same resume). Skills that also appear in project
   descriptions get credit.

2) Job description vs stored signals: required/preferred JD phrases are matched against
   heuristic `skill_signals` saved on Projects chunks at ingest.

Both signals blend into one multiplicative boost capped by RAG_RANKING_SKILL_BOOST_MAX_MULTIPLIER.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.config import settings
from src.parsers.jd_parser import JobDescription
from src.rag.skill_signals import strip_chunk_prefix, tokenize_skills_section_text


def _norm_section(sec: str) -> str:
    return (sec or "").strip().lower()


def skills_list_vs_project_narrative(
    chunk_rows: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], float]:
    """
    Parse Skills-section chunk bodies into tokens; see which appear (case-insensitive)
    in concatenated Projects-section text.

    Returns (skills_in_projects, skills_not_in_projects, overlap_ratio) where overlap_ratio is
    len(in_projects) / max(1, total_skill_tokens), or 0 if there are no skill tokens.
    """
    skills_blobs: List[str] = []
    projects_blobs: List[str] = []

    for r in chunk_rows:
        sec = _norm_section(str(r.get("section") or ""))
        raw = str(r.get("chunk_text") or "")
        body = strip_chunk_prefix(raw)
        if not body:
            continue
        if sec == "skills":
            skills_blobs.append(body)
        elif sec == "projects":
            projects_blobs.append(body)

    skills_text = "\n".join(skills_blobs)
    projects_haystack = "\n".join(projects_blobs).lower()

    tokens = tokenize_skills_section_text(skills_text)
    if not tokens:
        return [], [], 0.0

    in_projects: List[str] = []
    not_in: List[str] = []
    for t in tokens:
        tl = t.lower()
        if tl and tl in projects_haystack:
            in_projects.append(t)
        else:
            not_in.append(t)

    ratio = len(in_projects) / max(1, len(tokens))
    return in_projects, not_in, ratio


def _project_chunk_skill_signals(chunk_rows: List[Dict[str, Any]]) -> List[str]:
    """Union of CSV skill_signals from rows whose section is Projects."""
    out: List[str] = []
    seen: set[str] = set()
    for r in chunk_rows:
        if _norm_section(str(r.get("section") or "")) != "projects":
            continue
        csv = str(r.get("skill_signals") or "")
        for part in csv.split(","):
            s = part.strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
    return out


def _jd_skill_phrases(jd: JobDescription) -> List[str]:
    phrases: List[str] = []
    for s in list(jd.required_skills) + list(jd.preferred_skills):
        t = str(s).strip()
        if t:
            phrases.append(t)
    return phrases


def jd_requirements_vs_project_signals(
    jd: JobDescription,
    chunk_rows: List[Dict[str, Any]],
) -> Tuple[int, int, float]:
    """
    Count how many JD required/preferred skill phrases match project-chunk skill_signals
    (case-insensitive substring either way).

    Returns (hit_count, phrase_count, match_ratio) with match_ratio = hits / max(1, phrase_count).
    """
    phrases = _jd_skill_phrases(jd)
    if not phrases:
        return 0, 0, 0.0

    signals = _project_chunk_skill_signals(chunk_rows)
    if not signals:
        return 0, len(phrases), 0.0

    sl = [s.lower() for s in signals]
    hits = 0
    for phrase in phrases:
        pl = phrase.lower()
        matched = False
        for sig in sl:
            if pl in sig or sig in pl:
                matched = True
                break
        if matched:
            hits += 1

    ratio = hits / max(1, len(phrases))
    return hits, len(phrases), ratio


def compute_skill_boost_multiplier(
    skills_projects_overlap_ratio: float,
    jd_signals_match_ratio: float,
    *,
    jd_phrase_count: int = 0,
) -> float:
    """
    Map the two overlap ratios into [1.0, max_mult] using configured weights.
    If the JD lists no required/preferred skills, the JD-signals term is omitted (list-vs-projects only).
    """
    if not settings.rag.ranking_skill_boost_enabled:
        return 1.0

    w_list = max(0.0, settings.rag.ranking_skill_boost_skills_list_vs_projects_weight)
    w_jd = max(0.0, settings.rag.ranking_skill_boost_jd_vs_signals_weight)
    if jd_phrase_count <= 0:
        w_jd = 0.0
    wsum = w_list + w_jd
    if wsum <= 0:
        return 1.0
    w_list, w_jd = w_list / wsum, w_jd / wsum

    blend = (
        w_list * max(0.0, min(1.0, skills_projects_overlap_ratio))
        + w_jd * max(0.0, min(1.0, jd_signals_match_ratio))
    )
    max_mult = max(1.0, settings.rag.ranking_skill_boost_max_multiplier)
    return 1.0 + (max_mult - 1.0) * blend


def compute_candidate_skill_boost(
    jd: JobDescription,
    chunk_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combine skills-list vs projects and JD vs project_signals; return diagnostics + multiplier."""
    in_p, not_in, r_list = skills_list_vs_project_narrative(chunk_rows)
    _hits, nph, r_jd = jd_requirements_vs_project_signals(jd, chunk_rows)
    mult = compute_skill_boost_multiplier(r_list, r_jd, jd_phrase_count=nph)
    return {
        "skills_in_projects": in_p,
        "skills_not_in_projects": not_in,
        "skills_projects_overlap_ratio": r_list,
        "jd_project_signals_match_ratio": r_jd,
        "skill_boost_multiplier": mult,
    }
