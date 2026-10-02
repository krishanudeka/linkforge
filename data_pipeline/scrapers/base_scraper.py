"""Shared scraper plumbing: PaperRecord dataclass, HTTP session with retries, PDF download."""
from __future__ import annotations

import hashlib
import html
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.config import settings

logger = logging.getLogger("Scraper")

_TAG_RE = re.compile(r"<[^>]+>")


def strip_markup(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", text))).strip()


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip().lower()
    d = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", d)
    return d or None


def make_paper_id(doi: Optional[str] = None, pmcid: Optional[str] = None,
                  pmid: Optional[str] = None, fallback: str = "") -> str:
    """Stable identifier: DOI > PMCID > PMID > hash(fallback)."""
    if normalize_doi(doi):
        return normalize_doi(doi)
    if pmcid:
        return pmcid.upper()
    if pmid:
        return f"PMID:{pmid}"
    return "hash:" + hashlib.sha1(fallback.encode("utf-8")).hexdigest()[:16]


@dataclass
class PaperRecord:
    paper_id: str
    title: str = "Untitled"
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    authors: Optional[str] = None
    abstract: str = ""
    pdf_url: Optional[str] = None
    url: Optional[str] = None
    source: str = ""
    pdf_path: Optional[str] = None
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def parse_year(value) -> Optional[int]:
    try:
        y = int(str(value)[:4])
        return y if 1800 < y < 2200 else None
    except (TypeError, ValueError):
        return None


class BaseScraper(ABC):
    source_name = "base"

    def __init__(self, download_dir: Optional[str | Path] = None, request_delay: float = 0.4):
        self.download_dir = Path(download_dir or settings.pdf_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.session = requests.Session()
        retry = Retry(total=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET"]), respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({
            "User-Agent": f"LinkForge/1.0 (academic research; mailto:{settings.contact_email})"
        })

    # ---- interface ---------------------------------------------------------
    @abstractmethod
    def search(self, query: str, limit: int = 20) -> List[PaperRecord]:
        """Return metadata records (abstract + pdf_url when available). No downloading here."""

    # ---- shared ------------------------------------------------------------
    def _get_json(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None):
        time.sleep(self.request_delay)
        r = self.session.get(url, params=params, headers=headers, timeout=45)
        r.raise_for_status()
        return r.json()

    def pdf_path_for(self, record: PaperRecord) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", record.paper_id).strip("_")[:120]
        return self.download_dir / f"{safe}.pdf"

    def download_pdf(self, record: PaperRecord) -> Optional[Path]:
        """Download record.pdf_url -> disk. Verifies the payload really is a PDF."""
        if not record.pdf_url:
            return None
        path = self.pdf_path_for(record)
        if path.exists() and path.stat().st_size > 1024:
            record.pdf_path = str(path)
            return path
        try:
            time.sleep(self.request_delay)
            r = self.session.get(record.pdf_url, timeout=90, allow_redirects=True)
            if r.status_code != 200:
                logger.warning("PDF HTTP %s for %s", r.status_code, record.pdf_url)
                return None
            if not r.content.startswith(b"%PDF"):
                logger.warning("Not a PDF (got %s) from %s", r.headers.get("content-type"), record.pdf_url)
                return None
            path.write_bytes(r.content)
            record.pdf_path = str(path)
            return path
        except requests.RequestException as exc:
            logger.warning("PDF download failed for %s: %s", record.pdf_url, exc)
            return None

    def search_and_download(self, query: str, limit: int = 20) -> List[PaperRecord]:
        records = self.search(query, limit)
        for rec in records:
            self.download_pdf(rec)
        return records
