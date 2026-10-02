"""PubMed E-utilities harvester (metadata + abstracts; PDFs resolved via Europe PMC by PMCID)."""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import List

from backend.config import settings
from data_pipeline.scrapers.base_scraper import (
    BaseScraper, PaperRecord, make_paper_id, normalize_doi, parse_year,
)
from data_pipeline.scrapers.europe_pmc_scraper import RENDER_URL

logger = logging.getLogger("PubMedScraper")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _text(node) -> str:
    return "".join(node.itertext()).strip() if node is not None else ""


class PubMedScraper(BaseScraper):
    source_name = "pubmed"

    def __init__(self, *args, **kwargs):
        # NCBI allows 3 req/s without a key, 10 with one.
        super().__init__(*args, request_delay=0.12 if settings.ncbi_api_key else 0.4, **kwargs)

    def _common(self) -> dict:
        p = {"tool": "linkforge", "email": settings.contact_email}
        if settings.ncbi_api_key:
            p["api_key"] = settings.ncbi_api_key
        return p

    def search(self, query: str, limit: int = 20) -> List[PaperRecord]:
        term = f"({query}) AND free full text[filter] AND hasabstract"
        ids = self._get_json(
            f"{EUTILS}/esearch.fcgi",
            params={**self._common(), "db": "pubmed", "term": term, "retmax": limit,
                    "retmode": "json", "sort": "relevance"},
        )["esearchresult"].get("idlist", [])
        if not ids:
            return []

        import time
        time.sleep(self.request_delay)
        r = self.session.get(
            f"{EUTILS}/efetch.fcgi",
            params={**self._common(), "db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
            timeout=60,
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)

        records: List[PaperRecord] = []
        for art in root.findall(".//PubmedArticle"):
            pmid = _text(art.find(".//MedlineCitation/PMID"))
            title = _text(art.find(".//ArticleTitle")) or "Untitled"
            abstract = " ".join(_text(a) for a in art.findall(".//Abstract/AbstractText"))
            doi = pmcid = None
            for aid in art.findall(".//PubmedData/ArticleIdList/ArticleId"):
                kind = aid.get("IdType")
                if kind == "doi":
                    doi = normalize_doi(aid.text)
                elif kind == "pmc":
                    pmcid = (aid.text or "").strip()
            year = parse_year(
                _text(art.find(".//JournalIssue/PubDate/Year"))
                or _text(art.find(".//ArticleDate/Year"))
                or _text(art.find(".//PubDate/MedlineDate"))
            )
            authors = ", ".join(
                f"{_text(a.find('LastName'))} {_text(a.find('Initials'))}".strip()
                for a in art.findall(".//AuthorList/Author")[:8] if a.find("LastName") is not None
            )
            records.append(PaperRecord(
                paper_id=make_paper_id(doi, pmcid, pmid, fallback=title),
                title=title, doi=doi, pmid=pmid, pmcid=pmcid, year=year,
                journal=_text(art.find(".//Journal/Title")) or None, authors=authors or None,
                abstract=abstract,
                pdf_url=f"{RENDER_URL}?accid={pmcid}&blobtype=pdf" if pmcid else None,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", source=self.source_name,
            ))
        logger.info("PubMed: %d records for %r", len(records), query)
        return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for rec in PubMedScraper().search("curcumin alzheimer", limit=3):
        print(rec.paper_id, rec.title[:70], rec.pdf_url)
