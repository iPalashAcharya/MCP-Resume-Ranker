"""Tests for project section detection and parser output."""
from __future__ import annotations

import pytest

from src.parsers.resume_parser import (
    ResumeParser,
    _extract_sections,
    _is_project_section_header,
    _normalize_extracted_skills,
)


@pytest.mark.parametrize(
    "line",
    [
        "Projects",
        "PROJECTS",
        "Portfolio",
        "Selected Projects",
        "Technical Projects",
        "Personal Projects",
        "Academic Projects",
        "Key Projects",
        "Related Projects",
        "Project Work",
        "Project Experience",
        "Open Source Projects",
        "• Selected Projects",
        "1. Projects",
        "- Portfolio",
    ],
)
def test_project_section_headers_recognized(line: str) -> None:
    assert _is_project_section_header(line), f"expected project header: {line!r}"


@pytest.mark.parametrize(
    "line",
    [
        "Built several projects using Python and AWS",
        "This project used React for the frontend",
        "Work Experience",
        "Professional Experience",
        "Research",
        "A" * 70,
    ],
)
def test_non_project_headers_rejected(line: str) -> None:
    assert not _is_project_section_header(line)


def test_sections_skills_and_projects_split() -> None:
    text = """Jane Doe
Skills
Python, AWS | Docker

Selected Projects
Built API with Python and FastAPI.
"""
    sections = _extract_sections(text)
    by_title = {s.title: s.content for s in sections}
    assert "skills" in by_title and "Python" in by_title["skills"]
    assert "projects" in by_title and "FastAPI" in by_title["projects"]


def test_normalize_skills_canonical_casing() -> None:
    found = {"python", "PYTHON", "Ci/Cd", "node.js"}
    out = _normalize_extracted_skills(found)
    assert "Python" in out
    assert "CI/CD" in out
    assert "Node.js" in out


def test_parse_project_skills_empty_until_ingest_aggregation() -> None:
    p = ResumeParser()
    data = b"""Alex Dev
Projects
Shipped pipeline: PyTorch, Kafka.

Skills
Leadership
"""
    doc = p.parse(data, "txt", "rid", "k.pdf", "b")
    assert doc.project_skills == []
