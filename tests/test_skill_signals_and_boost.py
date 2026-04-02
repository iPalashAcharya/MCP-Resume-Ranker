"""Tests for heuristic skill_signals, Skills tokenization, and ranking boost helpers."""
from __future__ import annotations

import pytest

from src.parsers.jd_parser import JobDescription
from src.rag.ranking_skill_boost import (
    compute_candidate_skill_boost,
    jd_requirements_vs_project_signals,
    skills_list_vs_project_narrative,
)
from src.rag.skill_signals import (
    extract_skill_signals_from_text,
    strip_chunk_prefix,
    tokenize_skills_section_text,
)


def test_tokenize_skills_delimiters() -> None:
    s = "Python, Java | Go\n• Rust"
    out = tokenize_skills_section_text(s)
    assert {x.lower() for x in out} >= {"python", "java", "go", "rust"}


def test_extract_skill_signals_shapes() -> None:
    text = "Built TypeScript frontend; used AWS and PostgreSQL with GPT-4 style eval."
    sig = extract_skill_signals_from_text(text)
    joined = " ".join(sig).lower()
    assert "typescript" in joined or any("TypeScript" == x for x in sig)
    assert "aws" in joined


def test_strip_chunk_prefix() -> None:
    raw = "Candidate: Jane\nSection: skills\n\nPython, SQL"
    assert strip_chunk_prefix(raw) == "Python, SQL"


def test_skills_list_vs_project_narrative_overlap() -> None:
    rows = [
        {
            "chunk_index": 0,
            "section": "skills",
            "skill_signals": "",
            "chunk_text": "Candidate: X\nSection: skills\n\nPython, Docker",
        },
        {
            "chunk_index": 1,
            "section": "projects",
            "skill_signals": "",
            "chunk_text": "Candidate: X\nSection: projects\n\nUsed Python and Kubernetes.",
        },
    ]
    inn, outt, ratio = skills_list_vs_project_narrative(rows)
    assert "Python" in inn or "python" in [x.lower() for x in inn]
    assert ratio > 0


def test_jd_requirements_vs_project_signals() -> None:
    jd = JobDescription(
        jd_id="j1",
        title="Eng",
        company=None,
        raw_text="",
        required_skills=["Django"],
        preferred_skills=[],
        min_experience_years=0.0,
        education_requirements=[],
        responsibilities=[],
        s3_key=None,
        s3_bucket=None,
    )
    rows = [
        {
            "chunk_index": 0,
            "section": "projects",
            "skill_signals": "Django,PostgreSQL",
            "chunk_text": "",
        },
    ]
    hits, n, r = jd_requirements_vs_project_signals(jd, rows)
    assert hits >= 1
    assert r > 0


def test_compute_candidate_skill_boost_multiplier(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import settings

    monkeypatch.setattr(settings.rag, "ranking_skill_boost_enabled", True)
    monkeypatch.setattr(settings.rag, "ranking_skill_boost_max_multiplier", 1.3)
    jd = JobDescription(
        jd_id="j2",
        title="Eng",
        company=None,
        raw_text="",
        required_skills=["AWS"],
        preferred_skills=[],
        min_experience_years=0.0,
        education_requirements=[],
        responsibilities=[],
        s3_key=None,
        s3_bucket=None,
    )
    rows = [
        {
            "chunk_index": 0,
            "section": "projects",
            "skill_signals": "AWS",
            "chunk_text": "",
        },
    ]
    out = compute_candidate_skill_boost(jd, rows)
    assert 1.0 <= out["skill_boost_multiplier"] <= 1.3
