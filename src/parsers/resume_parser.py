"""
parsers/resume_parser.py — Extract structured information from resumes.
Supports PDF, DOCX, DOC, and plain-text formats.
Returns a ResumeDocument dataclass for downstream embedding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.config import get_logger

logger = get_logger(__name__)


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class ResumeSection:
    title: str
    content: str
    importance: float = 1.0   # weight hint for chunking strategy


@dataclass
class ResumeDocument:
    """Parsed, structured resume ready for embedding."""
    resume_id: str
    s3_key: str
    s3_bucket: str
    candidate_name: str
    raw_text: str
    sections: List[ResumeSection] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    experience_years: float = 0.0
    education: List[str] = field(default_factory=list)
    email: Optional[str] = None
    phone: Optional[str] = None
    word_count: int = 0
    char_count: int = 0

    @property
    def full_text(self) -> str:
        """Returns enriched text suitable for embedding (sections joined)."""
        parts = [f"Candidate: {self.candidate_name}"]
        for sec in self.sections:
            parts.append(f"{sec.title}:\n{sec.content}")
        return "\n\n".join(parts) if parts else self.raw_text

    def to_metadata(self) -> dict:
        return {
            "resume_id": self.resume_id,
            "s3_key": self.s3_key,
            "s3_bucket": self.s3_bucket,
            "candidate_name": self.candidate_name,
            "skills": ",".join(self.skills),
            "experience_years": self.experience_years,
            "education": " | ".join(self.education),
            "email": self.email or "",
            "word_count": self.word_count,
        }


# ─── Section Patterns ─────────────────────────────────────────────────────────

_SECTION_PATTERNS = [
    (r"(?i)(summary|objective|profile|about me)", "summary", 1.5),
    (r"(?i)(experience|work history|employment|career)", "experience", 2.0),
    (r"(?i)(education|academic|qualification)", "education", 1.5),
    (r"(?i)(skills?|technical skills?|core competencies|technologies)", "skills", 2.0),
    (r"(?i)(projects?|portfolio)", "projects", 1.3),
    (r"(?i)(certifications?|licenses?|accreditations?)", "certifications", 1.2),
    (r"(?i)(awards?|achievements?|honors?)", "achievements", 1.0),
    (r"(?i)(publications?|research|papers?)", "publications", 1.0),
    (r"(?i)(languages?|spoken languages?)", "languages", 0.8),
    (r"(?i)(volunteer|community|social)", "volunteer", 0.7),
]

_SKILL_KEYWORDS = re.compile(
    r"\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Swift|Kotlin|"
    r"React|Angular|Vue|Node\.js|FastAPI|Django|Flask|Spring|Rails|"
    r"AWS|GCP|Azure|Docker|Kubernetes|Terraform|CI/CD|"
    r"SQL|PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Milvus|"
    r"Machine Learning|Deep Learning|NLP|LLM|RAG|"
    r"TensorFlow|PyTorch|HuggingFace|LangChain|OpenAI|"
    r"REST|GraphQL|gRPC|Microservices|Kafka|RabbitMQ)\b",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"[\+\(]?[0-9][0-9\s\-\(\)\.]{7,}[0-9]")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


# ─── Text Extractors ─────────────────────────────────────────────────────────

def _extract_text_from_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber (primary) with pypdf fallback."""
    try:
        import pdfplumber
        import io

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except Exception as pdf_err:
        logger.warning("pdfplumber_failed", error=str(pdf_err), fallback="pypdf")

    try:
        import pypdf
        import io

        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as pypdf_err:
        raise ValueError(f"PDF extraction failed: {pypdf_err}") from pypdf_err


def _extract_text_from_docx(data: bytes) -> str:
    import docx2txt
    import io
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        text = docx2txt.process(tmp_path)
    finally:
        os.unlink(tmp_path)

    return text or ""


def _extract_text_from_bytes(data: bytes, extension: str) -> str:
    ext = extension.lower().lstrip(".")
    if ext == "pdf":
        return _extract_text_from_pdf(data)
    elif ext in ("docx", "doc"):
        return _extract_text_from_docx(data)
    elif ext == "txt":
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported extension: {ext}")


# ─── Analysis helpers ─────────────────────────────────────────────────────────

def _extract_sections(text: str) -> List[ResumeSection]:
    lines = text.splitlines()
    sections: List[ResumeSection] = []
    current_title = "General"
    current_content: List[str] = []
    current_importance = 1.0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_content.append("")
            continue

        matched_section = None
        for pattern, name, importance in _SECTION_PATTERNS:
            if re.match(pattern, stripped) and len(stripped) < 60:
                matched_section = (name, importance)
                break

        if matched_section:
            if current_content:
                sections.append(
                    ResumeSection(
                        title=current_title,
                        content="\n".join(current_content).strip(),
                        importance=current_importance,
                    )
                )
            current_title, current_importance = matched_section
            current_content = []
        else:
            current_content.append(stripped)

    # Flush last section
    if current_content:
        sections.append(
            ResumeSection(
                title=current_title,
                content="\n".join(current_content).strip(),
                importance=current_importance,
            )
        )
    return [s for s in sections if s.content]


def _extract_skills(text: str) -> List[str]:
    found = set(_SKILL_KEYWORDS.findall(text))
    return sorted(found)


def _estimate_experience_years(text: str) -> float:
    """Rough heuristic: count year ranges mentioned in the text."""
    years = sorted(set(int(y) for y in _YEAR_RE.findall(text)))
    if len(years) >= 2:
        span = years[-1] - years[0]
        return min(float(span), 40.0)
    return 0.0


def _extract_name_heuristic(text: str) -> str:
    """Naive: assume the first non-empty line ≤ 5 words is the candidate name."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped.split()) <= 5 and not re.search(r"[@:/]", stripped):
            return stripped
    return "Unknown Candidate"


def _extract_education(text: str) -> List[str]:
    pattern = re.compile(
        r"(B\.?S\.?|B\.?E\.?|B\.?Tech|M\.?S\.?|M\.?Tech|MBA|PhD|Bachelor|Master|Doctor)[\w\s,\.]{0,80}",
        re.IGNORECASE,
    )
    return list({m.group().strip() for m in pattern.finditer(text)})[:5]


# ─── Main Parser Class ────────────────────────────────────────────────────────

class ResumeParser:
    """Stateless parser: call `parse()` with raw bytes and metadata."""

    def parse(
        self,
        data: bytes,
        extension: str,
        resume_id: str,
        s3_key: str,
        s3_bucket: str,
    ) -> ResumeDocument:
        logger.info("resume_parser.parsing", resume_id=resume_id, ext=extension)

        raw_text = _extract_text_from_bytes(data, extension)
        if not raw_text.strip():
            raise ValueError(f"No text could be extracted from resume {s3_key}")

        sections = _extract_sections(raw_text)
        skills = _extract_skills(raw_text)
        experience_years = _estimate_experience_years(raw_text)
        education = _extract_education(raw_text)
        candidate_name = _extract_name_heuristic(raw_text)
        email_match = _EMAIL_RE.search(raw_text)
        phone_match = _PHONE_RE.search(raw_text)

        doc = ResumeDocument(
            resume_id=resume_id,
            s3_key=s3_key,
            s3_bucket=s3_bucket,
            candidate_name=candidate_name,
            raw_text=raw_text,
            sections=sections,
            skills=skills,
            experience_years=experience_years,
            education=education,
            email=email_match.group() if email_match else None,
            phone=phone_match.group() if phone_match else None,
            word_count=len(raw_text.split()),
            char_count=len(raw_text),
        )

        logger.info(
            "resume_parser.done",
            resume_id=resume_id,
            candidate=candidate_name,
            skills_count=len(skills),
            experience_years=experience_years,
            sections=len(sections),
        )
        return doc


# Module-level singleton
resume_parser = ResumeParser()