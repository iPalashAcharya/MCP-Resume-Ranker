"""
parsers/jd_parser.py — Extract structured requirements from Job Descriptions.
JD files may be PDF, DOCX, or plain text stored in AWS S3.
Returns a JobDescription dataclass for RAG retrieval and LLM ranking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import get_logger
from src.parsers.resume_parser import (
    _extract_text_from_bytes,
    _SKILL_KEYWORDS,
)

logger = get_logger(__name__)


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class JobDescription:
    """Parsed job description ready for embedding and candidate matching."""
    jd_id: str
    title: str
    company: Optional[str]
    raw_text: str
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    min_experience_years: float = 0.0
    education_requirements: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    location: Optional[str] = None
    employment_type: Optional[str] = None
    s3_key: Optional[str] = None
    s3_bucket: Optional[str] = None

    @property
    def embedding_text(self) -> str:
        """Structured text optimised for embedding."""
        parts = [
            f"Job Title: {self.title}",
        ]
        if self.company:
            parts.append(f"Company: {self.company}")
        if self.required_skills:
            parts.append("Required Skills: " + ", ".join(self.required_skills))
        if self.preferred_skills:
            parts.append("Preferred Skills: " + ", ".join(self.preferred_skills))
        if self.min_experience_years > 0:
            parts.append(f"Minimum Experience: {self.min_experience_years} years")
        if self.education_requirements:
            parts.append("Education: " + " | ".join(self.education_requirements))
        if self.responsibilities:
            parts.append("Responsibilities:\n" + "\n".join(f"- {r}" for r in self.responsibilities[:10]))
        parts.append("\nFull Description:\n" + self.raw_text[:1500])
        return "\n".join(parts)

    def to_metadata(self) -> dict:
        return {
            "jd_id": self.jd_id,
            "title": self.title,
            "company": self.company or "",
            "required_skills": ",".join(self.required_skills),
            "preferred_skills": ",".join(self.preferred_skills),
            "min_experience_years": self.min_experience_years,
            "location": self.location or "",
            "employment_type": self.employment_type or "",
            "s3_key": self.s3_key or "",
        }


# ─── Parsing Helpers ──────────────────────────────────────────────────────────

_REQUIRED_MARKERS = re.compile(
    r"(?i)(required|must have|mandatory|essential|minimum qualifications?)",
)
_PREFERRED_MARKERS = re.compile(
    r"(?i)(preferred|nice to have|plus|bonus|desired|advantage)",
)
_EXP_PATTERN = re.compile(r"(\d+)\+?\s*(?:to\s*\d+)?\s*years?\s+(?:of\s+)?(?:experience|exp\.?)", re.IGNORECASE)
_TITLE_PATTERN = re.compile(
    r"(?i)(job title|position|role)[:\s]+(.{3,80})",
)
_COMPANY_PATTERN = re.compile(
    r"(?i)(company|organization|employer|about us)[:\s]+(.{2,80})",
)
_LOCATION_PATTERN = re.compile(
    r"(?i)(location|work location|office|city|country)[:\s]+(.{2,80})",
)
_EMPLOYMENT_PATTERN = re.compile(
    r"(?i)(full[\s-]?time|part[\s-]?time|contract|remote|hybrid|on[\s-]?site)",
)
_EDUCATION_RE = re.compile(
    r"(?i)(bachelor|master|ph\.?d|mba|b\.?tech|m\.?tech|degree|graduate)[^.\n]{0,60}",
)
_BULLET_RE = re.compile(r"^[\•\-\*\u2022\u25cf]\s+", re.MULTILINE)


def _extract_bullets(block: str) -> List[str]:
    lines = _BULLET_RE.sub("", block).splitlines()
    return [l.strip() for l in lines if l.strip() and len(l.strip()) > 5]


def _detect_section_blocks(text: str) -> dict[str, str]:
    """Split the JD into named blocks using common section headers."""
    section_patterns = [
        ("responsibilities", r"(?i)(responsibilities|duties|what you.?ll do|role overview)"),
        ("requirements", r"(?i)(requirements?|qualifications?|what we.?re looking for|must have)"),
        ("preferred", r"(?i)(preferred|nice to have|bonus|desired)"),
        ("benefits", r"(?i)(benefits?|perks?|what we offer|compensation)"),
        ("about", r"(?i)(about (us|the company|the role)|overview|description)"),
    ]

    blocks: dict[str, list[str]] = {name: [] for name, _ in section_patterns}
    blocks["general"] = []
    current = "general"

    for line in text.splitlines():
        matched = False
        for name, pattern in section_patterns:
            if re.match(pattern, line.strip()) and len(line.strip()) < 80:
                current = name
                matched = True
                break
        if not matched:
            blocks[current].append(line)

    return {k: "\n".join(v) for k, v in blocks.items()}


class JDParser:
    """Stateless JD parser."""

    def parse(
        self,
        data: bytes,
        extension: str,
        jd_id: str,
        s3_key: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        raw_text_override: Optional[str] = None,
    ) -> JobDescription:
        logger.info("jd_parser.parsing", jd_id=jd_id)

        raw_text = raw_text_override or _extract_text_from_bytes(data, extension)
        if not raw_text.strip():
            raise ValueError(f"No text extractable from JD {jd_id}")

        blocks = _detect_section_blocks(raw_text)

        # Title
        title_match = _TITLE_PATTERN.search(raw_text)
        title = title_match.group(2).strip() if title_match else "Unknown Position"

        # Company
        company_match = _COMPANY_PATTERN.search(raw_text)
        company = company_match.group(2).strip() if company_match else None

        # Location
        location_match = _LOCATION_PATTERN.search(raw_text)
        location = location_match.group(2).strip() if location_match else None

        # Employment type
        emp_match = _EMPLOYMENT_PATTERN.search(raw_text)
        employment_type = emp_match.group(0).strip() if emp_match else None

        # Experience years
        exp_years = 0.0
        for m in _EXP_PATTERN.finditer(raw_text):
            exp_years = max(exp_years, float(m.group(1)))

        # Education
        education = list({m.group().strip() for m in _EDUCATION_RE.finditer(raw_text)})[:4]

        # Skills
        all_skills = list(set(_SKILL_KEYWORDS.findall(raw_text)))

        req_block = blocks.get("requirements", "")
        pref_block = blocks.get("preferred", "")
        required_skills = list(set(_SKILL_KEYWORDS.findall(req_block))) if req_block else all_skills
        preferred_skills = [s for s in list(set(_SKILL_KEYWORDS.findall(pref_block))) if s not in required_skills]

        # Responsibilities
        resp_block = blocks.get("responsibilities", "")
        responsibilities = _extract_bullets(resp_block)[:15] if resp_block else []

        # Benefits
        ben_block = blocks.get("benefits", "")
        benefits = _extract_bullets(ben_block)[:10] if ben_block else []

        jd = JobDescription(
            jd_id=jd_id,
            title=title,
            company=company,
            raw_text=raw_text,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            min_experience_years=exp_years,
            education_requirements=education,
            responsibilities=responsibilities,
            benefits=benefits,
            location=location,
            employment_type=employment_type,
            s3_key=s3_key,
            s3_bucket=s3_bucket,
        )

        logger.info(
            "jd_parser.done",
            jd_id=jd_id,
            title=title,
            required_skills=len(required_skills),
            exp_years=exp_years,
        )
        return jd

    def parse_from_text(self, text: str, jd_id: str, title: Optional[str] = None) -> JobDescription:
        """Parse directly from a text string (e.g., from API body)."""
        return self.parse(
            data=b"",
            extension="txt",
            jd_id=jd_id,
            raw_text_override=text,
        )


jd_parser = JDParser()