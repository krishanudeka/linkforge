"""PDF -> Markdown-ish text using PyMuPDF with multi-column reading-order recovery.

Strategy per page:
  * take text blocks, drop images/tables-of-glyphs
  * detect full-width blocks (titles, abstracts, figure captions) vs left/right column blocks
  * read: full-width blocks split the page into bands; inside a band read left column then right
  * repeated headers/footers (same text on many pages) and bare page numbers are removed
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import fitz  # PyMuPDF

from backend.config import settings
from data_pipeline.parsers.text_cleaner import SECTION_NAMES

logger = logging.getLogger("PDFParser")

_HEADING_RE = re.compile(
    r"^(?:\d{1,2}\.?\s+)?(" + "|".join(re.escape(s) for s in SECTION_NAMES) + r")\s*:?$", re.I
)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+[^\s\"<>.,;)])", re.I)


@dataclass
class ParsedDocument:
    markdown: str
    n_pages: int
    title: Optional[str] = None
    doi: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class PDFParser:
    def __init__(self, header_footer_margin: float = 0.07, repeat_ratio: float = 0.4):
        self.margin = header_footer_margin
        self.repeat_ratio = repeat_ratio

    # ------------------------------------------------------------------ page level
    def _page_blocks(self, page) -> List[Dict]:
        blocks = []
        for x0, y0, x1, y1, text, _no, btype in page.get_text("blocks"):
            if btype != 0:
                continue
            text = text.strip()
            if text:
                blocks.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text})
        return blocks

    def _order_blocks(self, blocks: List[Dict], page_width: float) -> List[Dict]:
        if not blocks:
            return []
        mid = page_width / 2.0
        full, left, right = [], [], []
        for b in blocks:
            width = b["x1"] - b["x0"]
            if width > 0.62 * page_width or (b["x0"] < mid - 0.08 * page_width and b["x1"] > mid + 0.08 * page_width):
                full.append(b)
            elif (b["x0"] + b["x1"]) / 2 < mid:
                left.append(b)
            else:
                right.append(b)

        # Single-column page: nothing to reorder, just top-to-bottom.
        if not left or not right:
            return sorted(blocks, key=lambda b: (round(b["y0"]), b["x0"]))

        ordered: List[Dict] = []
        full_sorted = sorted(full, key=lambda b: b["y0"])
        boundaries = [b["y0"] for b in full_sorted]

        def in_band(b, lo, hi):
            return lo <= b["y0"] < hi

        lo = -1.0
        for fb, hi in zip(full_sorted, boundaries):
            band_l = sorted([b for b in left if in_band(b, lo, hi)], key=lambda b: b["y0"])
            band_r = sorted([b for b in right if in_band(b, lo, hi)], key=lambda b: b["y0"])
            ordered += band_l + band_r + [fb]
            lo = fb["y1"] - 0.5  # blocks that start within the full-width block are below it
        tail_l = sorted([b for b in left if b["y0"] >= lo], key=lambda b: b["y0"])
        tail_r = sorted([b for b in right if b["y0"] >= lo], key=lambda b: b["y0"])
        ordered += tail_l + tail_r
        return ordered

    # ------------------------------------------------------------------ document level
    def parse(self, pdf_path: str) -> ParsedDocument:
        if settings.pdf_parser.strip().lower() == "marker":
            try:
                return self._parse_marker(pdf_path)
            except Exception as exc:  # noqa: BLE001 - marker is optional/heavy; fall back silently
                logger.warning("Marker failed (%s); falling back to PyMuPDF", exc)
        return self._parse_pymupdf(pdf_path)

    _marker_converter = None

    def _parse_marker(self, pdf_path: str) -> ParsedDocument:
        """Optional Marker backend (UNTESTED here). Needs `pip install marker-pdf` and PDF_PARSER=marker."""
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        if PDFParser._marker_converter is None:
            PDFParser._marker_converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = PDFParser._marker_converter(pdf_path)
        text, _, _ = text_from_rendered(rendered)
        with fitz.open(pdf_path) as d:
            n_pages, meta = d.page_count, d.metadata or {}
            first = d[0].get_text("text") if d.page_count else ""
        m = _DOI_RE.search(first)
        return ParsedDocument(markdown=text, n_pages=n_pages, title=(meta.get("title") or "").strip() or None,
                              doi=m.group(1).lower() if m else None, metadata=meta)

    def _parse_pymupdf(self, pdf_path: str) -> ParsedDocument:
        doc = fitz.open(pdf_path)
        try:
            pages = []
            for page in doc:
                blocks = self._page_blocks(page)
                pages.append((page.rect, blocks))
            repeated = self._repeated_margin_text(pages)

            parts: List[str] = []
            for rect, blocks in pages:
                kept = []
                for b in blocks:
                    in_margin = b["y1"] < rect.height * self.margin or b["y0"] > rect.height * (1 - self.margin)
                    norm = self._norm_margin(b["text"])
                    if in_margin and (norm in repeated or re.fullmatch(r"\d{1,4}", b["text"].strip())):
                        continue
                    kept.append(b)
                for b in self._order_blocks(kept, rect.width):
                    parts.append(self._block_to_md(b["text"]))
            markdown = "\n\n".join(p for p in parts if p)

            meta = doc.metadata or {}
            title = (meta.get("title") or "").strip() or None
            first_text = doc[0].get_text("text") if doc.page_count else ""
            m = _DOI_RE.search(first_text)
            return ParsedDocument(
                markdown=markdown,
                n_pages=doc.page_count,
                title=title,
                doi=m.group(1).lower() if m else None,
                metadata=meta,
            )
        finally:
            doc.close()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _norm_margin(text: str) -> str:
        return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip().lower()))

    def _repeated_margin_text(self, pages) -> set:
        if len(pages) < 3:
            return set()
        counts: Counter = Counter()
        for rect, blocks in pages:
            seen = set()
            for b in blocks:
                if b["y1"] < rect.height * self.margin or b["y0"] > rect.height * (1 - self.margin):
                    seen.add(self._norm_margin(b["text"]))
            counts.update(seen)
        need = max(2, int(len(pages) * self.repeat_ratio))
        return {t for t, c in counts.items() if c >= need}

    @staticmethod
    def _block_to_md(text: str) -> str:
        # join hard-wrapped lines inside a block; keep a blank line between blocks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        flat = re.sub(r"\s*\n\s*", " ", text).strip()
        if len(flat) < 80 and _HEADING_RE.match(flat):
            return f"## {_HEADING_RE.match(flat).group(1).title()}"
        return flat

    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Backwards-compatible helper returning just the text."""
        return PDFParser().parse(pdf_path).markdown
