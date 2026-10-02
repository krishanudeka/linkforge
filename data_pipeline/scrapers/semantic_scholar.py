"""Semantic Scholar Graph API harvester (open-access PDFs only)."""
from __future__ import annotations

import logging
import time
from typing import List

from backend.config import settings
from data_pipeline.scrapers.base_scraper import (
    BaseScraper, PaperRecord, make_paper_id, normalize_doi, parse_year,
)

logger = logging.getLogger("SemanticScholarScraper")

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,venue,externalIds,openAccessPdf,authors,url"


class SemanticScholarScraper(BaseScraper):
    source_name = "semantic_scholar"

    def __init__(self, *args, **kwargs):
        # ~1 request/second for the shared unauthenticated pool
        super().__init__(*args, request_delay=0.3 if settings.semantic_scholar_api_key else 1.1, **kwargs)

    def search(self, query: str, limit: int = 20) -> List[PaperRecord]:
        headers = {"x-api-key": settings.semantic_scholar_api_key} if settings.semantic_scholar_api_key else None
        records: List[PaperRecord] = []
        offset = 0
        while len(records) < limit and offset < 1000:
            params = {"query": query, "offset": offset, "limit": min(100, max(limit, 10)),
                      "fields": FIELDS, "openAccessPdf": ""}
            try:
                data = self._get_json(SEARCH_URL, params=params, headers=headers)
            except Exception as exc:  # 429s surface here after urllib3 retries are exhausted
                logger.warning("Semantic Scholar request failed: %s", exc)
                time.sleep(5)
                break
            items = data.get("data", [])
            if not items:
                break
            for it in items:
                pdf = (it.get("openAccessPdf") or {}).get("url")
                if not pdf:
                    continue
                ext = it.get("externalIds") or {}
                doi = normalize_doi(ext.get("DOI"))
                pmcid = ext.get("PubMedCentral")
                pmcid = f"PMC{pmcid}" if pmcid and not str(pmcid).upper().startswith("PMC") else pmcid
                title = it.get("title") or "Untitled"
                records.append(PaperRecord(
                    paper_id=make_paper_id(doi, pmcid, ext.get("PubMed"), fallback=it.get("paperId") or title),
                    title=title, doi=doi, pmid=ext.get("PubMed"), pmcid=pmcid,
                    year=parse_year(it.get("year")), journal=it.get("venue") or None,
                    authors=", ".join(a.get("name", "") for a in (it.get("authors") or [])[:8]) or None,
                    abstract=it.get("abstract") or "", pdf_url=pdf, url=it.get("url"),
                    source=self.source_name,
                ))
                if len(records) >= limit:
                    break
            offset += len(items)
            if data.get("next") is None:
                break
        logger.info("Semantic Scholar: %d records for %r", len(records), query)
        return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for rec in SemanticScholarScraper().search("flavanols neuroinflammation", limit=3):
        print(rec.paper_id, rec.title[:70], rec.pdf_url)
