"""Europe PMC REST harvester (open-access papers with PDFs)."""
from __future__ import annotations

import logging
from typing import List

from data_pipeline.scrapers.base_scraper import (
    BaseScraper, PaperRecord, make_paper_id, normalize_doi, parse_year, strip_markup,
)

logger = logging.getLogger("EuropePMCScraper")

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
RENDER_URL = "https://europepmc.org/backend/ptpmcrender.fcgi"


class EuropePMCScraper(BaseScraper):
    source_name = "europe_pmc"

    def search(self, query: str, limit: int = 20) -> List[PaperRecord]:
        records: List[PaperRecord] = []
        cursor = "*"
        full_query = f"({query}) AND OPEN_ACCESS:y AND HAS_PDF:y"
        while len(records) < limit:
            params = {
                "query": full_query, "format": "json", "resultType": "core",
                "pageSize": min(100, max(limit - len(records), 1)), "cursorMark": cursor,
                "sort": "CITED desc",
            }
            data = self._get_json(SEARCH_URL, params=params)
            results = data.get("resultList", {}).get("result", [])
            if not results:
                break
            for item in results:
                rec = self._to_record(item)
                if rec:
                    records.append(rec)
                if len(records) >= limit:
                    break
            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        logger.info("Europe PMC: %d records for %r", len(records), query)
        return records

    def _to_record(self, item: dict) -> PaperRecord | None:
        title = strip_markup(item.get("title")) or "Untitled"
        doi = normalize_doi(item.get("doi"))
        pmcid, pmid = item.get("pmcid"), item.get("pmid")

        pdf_url = None
        for u in (item.get("fullTextUrlList") or {}).get("fullTextUrl", []):
            if u.get("documentStyle") == "pdf" and u.get("availability", "").lower() in ("open access", "free"):
                pdf_url = u.get("url")
                break
        if not pdf_url and pmcid:
            pdf_url = f"{RENDER_URL}?accid={pmcid}&blobtype=pdf"

        journal = (item.get("journalInfo") or {}).get("journal", {}).get("title") or item.get("journalTitle")
        abstract = strip_markup(item.get("abstractText"))
        if not abstract and not pdf_url:
            return None
        return PaperRecord(
            paper_id=make_paper_id(doi, pmcid, pmid, fallback=title),
            title=title, doi=doi, pmid=pmid, pmcid=pmcid,
            year=parse_year(item.get("pubYear")), journal=journal,
            authors=item.get("authorString"), abstract=abstract, pdf_url=pdf_url,
            url=f"https://europepmc.org/article/{item.get('source', 'MED')}/{pmid or pmcid or ''}",
            source=self.source_name,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = EuropePMCScraper()
    recs = scraper.search_and_download("neuroinflammation flavan-3-ols", limit=5)
    for r in recs:
        print(r.paper_id, "|", r.title[:70], "|", r.pdf_path)
