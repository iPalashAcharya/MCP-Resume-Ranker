"""
Lightweight, non-LLM heuristics for skill-like tokens in resume text.

- `extract_skill_signals_from_text`: used on Projects section bodies at chunk/ingest time.
  Uses shape-based patterns (camelCase, PascalCase, ALL_CAPS acronyms, version suffixes).
  Filters common English stopwords and resume boilerplate verbs — no catalog of technology names.

- `tokenize_skills_section_text`: splits Skills-section prose into candidate tokens for
  cross-checking against Projects text at ranking time.
"""
from __future__ import annotations

import re
from typing import List, Set

# Common English function words / glue — not a "skill list"; avoids obvious noise.
# Keep compact for speed; extend if needed.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if in on at to for of as by with from into through during
    before after above below between under again further then once here there when
    where why how all each both few more most other some such no nor not only own
    same so than too very can will just don should now been being have has had
    having do does did doing was were being is am are isn aren wasn weren
    it its we you he she they them our your my their this that these those
    what which who whom whose about against between into through while
    also back even ever still just own same such than very just
    project projects portfolio experience work role team company client
    built using used use via including include includes focused focus
    led lead leading developed develop development designed design implemented
    implementation created create creating managed manage management
    responsible responsibilities responsibility various multiple several
    strong excellent good great key main core various highly
    """.split()
)

# Resume-y verbs / section fluff (lowercase); not exhaustive.
_VERB_LIKE: frozenset[str] = frozenset(
    """
    built using used utilize utilized leveraging leveraged helped help helped
    worked working collaborate collaborated collaboration delivered delivery
    improved improve optimized optimize maintained maintain supported support
    integrated integration automated automation deployed deploy deployment
    """.split()
)

# Shape: StudlyCaps / camelCase inside a word boundary (e.g. TypeScript, McKinseyStyle).
_PASCAL_MULTI_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")

# Short acronyms (SQL, AWS, NLP) — length 3–5 all caps.
_ACRONYM_RE = re.compile(r"\b[A-Z]{3,5}\b")

# Token with digit(s): Python3, GPT4, v2
_ALNUM_VERSION_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*\d+[A-Za-z0-9]*\b")

# Hyphen + number: GPT-4, ci-cd style fragments (we keep the whole token)
_HYPHEN_NUM_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-\d+(?:\.\d+)?)+\b", re.IGNORECASE)

# Split Skills bullets / CSV-ish lines (ranking: list vs project narrative)
_SKILLS_DELIM_RE = re.compile(r"[,|;\n\r•·\u2022\u25aa\u25cf\-]+")


def _keep_token(t: str) -> bool:
    tl = t.strip().lower()
    if len(tl) < 2:
        return False
    if tl in _STOPWORDS or tl in _VERB_LIKE:
        return False
    # Drop pure years / numbers
    if tl.isdigit():
        return False
    return True


def extract_skill_signals_from_text(text: str, *, max_signals: int = 48) -> List[str]:
    """
    Extract heuristic skill-like tokens from raw Projects-section text.
    No fixed technology whitelist — only orthographic / numeric patterns + stopword filter.
    """
    if not (text or "").strip():
        return []

    candidates: List[str] = []

    for pattern in (
        _PASCAL_MULTI_RE,
        _ACRONYM_RE,
        _ALNUM_VERSION_RE,
        _HYPHEN_NUM_RE,
    ):
        for m in pattern.finditer(text):
            candidates.append(m.group(0).strip())

    seen_lower: Set[str] = set()
    out: List[str] = []
    for raw in candidates:
        if not _keep_token(raw):
            continue
        key = raw.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        out.append(raw)
        if len(out) >= max_signals:
            break
    return out


def tokenize_skills_section_text(text: str) -> List[str]:
    """
    Split Skills-section content into individual skill tokens using common delimiters.
    Used at ranking time to compare against Projects narrative (case-insensitive).
    """
    if not (text or "").strip():
        return []
    # Normalize bullets to comma
    normalized = _SKILLS_DELIM_RE.sub(",", text)
    parts = [p.strip() for p in normalized.split(",")]
    out: List[str] = []
    seen: Set[str] = set()
    for p in parts:
        if len(p) < 2:
            continue
        # Drop parenthetical-only noise occasionally
        p = p.strip(" \t()[]")
        if len(p) < 2:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def strip_chunk_prefix(chunk_stored_text: str) -> str:
    """
    Remove `Candidate: ...\\nSection: ...\\n\\n` prefix from stored Milvus chunk_text if present.
    Returns body suitable for delimiter tokenization.
    """
    if not chunk_stored_text:
        return ""
    # Match our chunker prefix format
    m = re.search(
        r"(?is)^Candidate:\s*.+?\nSection:\s*.+?\n\n(.*)$",
        chunk_stored_text,
    )
    if m:
        return m.group(1).strip()
    return chunk_stored_text.strip()
