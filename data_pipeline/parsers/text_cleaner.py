"""Text clean-up, section splitting and chunking for scientific full-text."""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

SECTION_NAMES = [
    "abstract", "introduction", "background", "methods", "materials and methods", "methodology",
    "results", "results and discussion", "discussion", "conclusion", "conclusions", "references",
    "acknowledgements", "acknowledgments", "funding", "conflict of interest", "supplementary",
]
# Sections most likely to contain *stated findings* come first when we must cap chunk count.
SECTION_PRIORITY = {
    "abstract": 0, "results": 1, "results and discussion": 1, "discussion": 2, "conclusion": 2,
    "conclusions": 2, "introduction": 3, "background": 3, "body": 4, "methods": 6,
    "materials and methods": 6, "methodology": 6,
}
_SECTION_RE = re.compile(
    r"^\s*(?:#{1,4}\s*)?(?:\d{1,2}\.?\s+)?(" + "|".join(re.escape(s) for s in SECTION_NAMES) + r")\s*:?\s*$",
    re.I,
)
_STOP_SECTIONS = {"references", "acknowledgements", "acknowledgments", "funding", "conflict of interest"}

_CITATION_BRACKETS = re.compile(r"\[\s*\d+(?:\s*[–\-,]\s*\d+)*\s*\]")
_CITATION_PAREN = re.compile(r"\((?:[A-Z][A-Za-z\-]+(?: et al\.?)?(?:,? (?:and|&) [A-Z][A-Za-z\-]+)?,? \d{4}[a-z]?(?:; )?)+\)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _is_formula_or_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if re.fullmatch(r"\d{1,4}", s):  # bare page numbers
        return True
    letters = sum(c.isalpha() for c in s)
    if len(s) >= 8 and letters / len(s) < 0.4:      # mostly digits/symbols -> formula / table residue
        return True
    if len(s) < 4 and not s.startswith("#"):
        return True
    return False


def clean_text(raw: str, strip_references: bool = True) -> str:
    """Normalise unicode, de-hyphenate line wraps, drop formulas/citation markers/reference list."""
    text = unicodedata.normalize("NFKC", raw or "")
    text = _CONTROL.sub(" ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # inhibi-\ntion -> inhibition
    lines = [ln.rstrip() for ln in text.splitlines()]

    if strip_references:
        ref_idx = None
        for i, ln in enumerate(lines):
            m = _SECTION_RE.match(ln)
            if m and m.group(1).lower() == "references" and i > len(lines) * 0.4:
                ref_idx = i
                break
        if ref_idx is not None:
            lines = lines[:ref_idx]

    lines = [ln for ln in lines if not _is_formula_or_noise(ln)]

    # rebuild paragraphs: blank line separates; single newlines inside are soft wraps
    paras, buf = [], []
    for ln in lines:
        if not ln.strip():
            if buf:
                paras.append(" ".join(buf)); buf = []
        elif ln.lstrip().startswith("#"):
            if buf:
                paras.append(" ".join(buf)); buf = []
            paras.append(ln.strip())
        else:
            buf.append(ln.strip())
    if buf:
        paras.append(" ".join(buf))

    cleaned = []
    for p in paras:
        if p.startswith("#"):
            cleaned.append(p)
            continue
        p = _CITATION_BRACKETS.sub("", p)
        p = _CITATION_PAREN.sub("", p)
        p = _URL.sub("", p)
        p = _EMAIL.sub("", p)
        p = re.sub(r"\s+([,.;:])", r"\1", p)
        p = re.sub(r"\s{2,}", " ", p).strip()
        if len(p) > 1:
            cleaned.append(p)
    return "\n\n".join(cleaned).strip()


def split_sections(text: str) -> List[Dict[str, str]]:
    """Split into [{"section": name, "text": ...}] using heading lines; unknown leading text -> 'body'."""
    sections: List[Dict[str, str]] = []
    current, buf = "body", []
    for para in text.split("\n\n"):
        first = para.strip().splitlines()[0] if para.strip() else ""
        m = _SECTION_RE.match(first) if len(first) < 60 else None
        if m:
            if buf:
                sections.append({"section": current, "text": "\n\n".join(buf)})
                buf = []
            current = m.group(1).lower()
            rest = para.strip()[len(first):].strip()
            if rest:
                buf.append(rest)
            if current in _STOP_SECTIONS:
                break
        else:
            buf.append(para)
    if buf and current not in _STOP_SECTIONS:
        sections.append({"section": current, "text": "\n\n".join(buf)})
    return [s for s in sections if s["text"].strip()]


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text) if s.strip()]


def chunk_text(text: str, max_chars: int = 2500, overlap: int = 250) -> List[str]:
    """Sentence-aware chunking with character overlap. Never splits mid-sentence unless a single
    sentence exceeds `max_chars`."""
    text = text.strip()
    if not text:
        return []
    sentences = []
    for s in _split_sentences(text.replace("\n\n", " ")):
        while len(s) > max_chars:                   # pathological unbroken text
            sentences.append(s[:max_chars]); s = s[max_chars:]
        sentences.append(s)

    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        if cur and cur_len + len(s) + 1 > max_chars:
            chunks.append(" ".join(cur))
            # overlap: carry trailing sentences totalling <= overlap chars
            carry, carry_len = [], 0
            for prev in reversed(cur):
                if carry_len + len(prev) > overlap:
                    break
                carry.insert(0, prev); carry_len += len(prev) + 1
            cur, cur_len = carry, carry_len
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        tail = " ".join(cur)
        if not chunks or tail != chunks[-1]:
            chunks.append(tail)
    return chunks


def build_chunks(markdown_or_text: str, max_chars: int = 2500, overlap: int = 250,
                 max_chunks: int | None = None) -> List[Dict]:
    """Full text -> [{"index", "section", "text"}] ordered by extraction priority, capped."""
    cleaned = clean_text(markdown_or_text)
    chunks: List[Dict] = []
    for sec in split_sections(cleaned):
        body = re.sub(r"^#+\s*", "", sec["text"], flags=re.M)
        for piece in chunk_text(body, max_chars, overlap):
            if len(piece) >= 120:
                chunks.append({"section": sec["section"], "text": piece})
    for i, c in enumerate(chunks):
        c["index"] = i
    if max_chunks and len(chunks) > max_chunks:
        chunks.sort(key=lambda c: (SECTION_PRIORITY.get(c["section"], 5), c["index"]))
        chunks = sorted(chunks[:max_chunks], key=lambda c: c["index"])
    return chunks
