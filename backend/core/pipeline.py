"""Ingestion pipeline: scrape -> parse -> chunk -> extract -> resolve -> store."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from backend.config import settings
from backend.core.services import Services, get_services
from data_pipeline.parsers.text_cleaner import build_chunks
from data_pipeline.scrapers import get_scraper
from data_pipeline.scrapers.base_scraper import make_paper_id

logger = logging.getLogger("Pipeline")

PAPER_FIELDS = ("paper_id", "doi", "pmid", "pmcid", "title", "year", "journal", "abstract",
                "source", "url", "authors")


def _meta(record: Dict[str, Any]) -> Dict[str, Any]:
    return {k: record.get(k) for k in PAPER_FIELDS}


class IngestionPipeline:
    def __init__(self, services: Optional[Services] = None):
        self.s = services or get_services()

    # ------------------------------------------------------------------ one paper
    def process_paper(self, record: Dict[str, Any], pdf_path: Optional[str] = None,
                      force: bool = False, progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Process one paper. `record` is a PaperRecord.to_dict() (or any dict with title/doi/abstract)."""
        say = progress or (lambda m: None)
        record = dict(record)
        if not record.get("paper_id"):
            record["paper_id"] = make_paper_id(record.get("doi"), record.get("pmcid"), record.get("pmid"),
                                               record.get("title") or "")
        pid = record["paper_id"]
        pdf_path = pdf_path or record.get("pdf_path")
        result = {"paper_id": pid, "title": record.get("title"), "status": "ok",
                  "chunks": 0, "triplets": 0, "used": None, "error": None}

        neo4j = self.s.neo4j
        neo4j.upsert_paper(_meta(record))
        if not force and neo4j.paper_processed(pid):
            result["status"] = "skipped"
            return result

        # ---- 1. text ------------------------------------------------------
        text, used = "", None
        if pdf_path and Path(pdf_path).exists():
            try:
                doc = self.s.parser.parse(str(pdf_path))
                text, used = doc.markdown, "pdf"
                self.s.store.save_markdown(pid, text)
                if doc.doi and not record.get("doi"):
                    record["doi"] = doc.doi
                    neo4j.upsert_paper(_meta(record))
            except Exception as exc:  # noqa: BLE001
                logger.warning("PDF parse failed for %s: %s", pid, exc)
        if len(text.strip()) < 200 and record.get("abstract"):
            text = f"{record.get('title') or ''}\n\nAbstract\n{record['abstract']}"
            used = "abstract"
        if len(text.strip()) < 100:
            result.update(status="no_text", error="no usable text (no PDF and no abstract)")
            return result
        result["used"] = used
        say(f"{pid}: parsed ({used})")

        # ---- 2. chunks + vectors -----------------------------------------
        chunks = build_chunks(text, settings.chunk_max_chars, settings.chunk_overlap_chars,
                              settings.max_chunks_per_paper)
        if not chunks:  # abstract too short for the 120-char chunk floor
            chunks = [{"index": 0, "section": "abstract", "text": text.strip()}]
        result["chunks"] = self.s.vectors.add_chunks(pid, chunks, record)

        # ---- 3. LLM extraction -------------------------------------------
        from ai_engine.extraction.llm_extractor import LLMUnavailable
        triplets: List[Dict[str, Any]] = []
        for c in chunks:
            try:
                triplets.extend(self.s.extractor.extract_triplets(c["text"], title=record.get("title")))
            except LLMUnavailable as exc:
                result.update(status="llm_unavailable", error=str(exc))
                logger.error("LLM unavailable, aborting %s: %s", pid, exc)
                return result      # not marked processed -> will be retried later
            except Exception as exc:  # noqa: BLE001
                logger.warning("Extraction failed on a chunk of %s: %s", pid, exc)
        say(f"{pid}: {len(triplets)} raw triplets")

        # ---- 4. resolve + store ------------------------------------------
        with self.s.ingest_lock:
            canon = self.s.resolver.canonicalize_triplets(triplets)
            n = neo4j.insert_triplets(canon, pid)
            self.s.vectors.add_provenance(pid, canon, record)
            neo4j.mark_paper_processed(pid, n, result["chunks"])
        self.s.graph_cache.invalidate()
        result["triplets"] = n
        say(f"{pid}: stored {n} triplets")
        return result

    # ------------------------------------------------------------------ batches
    def ingest_query(self, query: str, source: str = "europe_pmc", limit: int = 10,
                     download: bool = True, force: bool = False, progress: Optional[Callable[[str], None]] = None
                     ) -> Dict[str, Any]:
        scraper = get_scraper(source)
        records = scraper.search(query, limit)
        results = []
        for rec in records:
            if download:
                scraper.download_pdf(rec)
            try:
                results.append(self.process_paper(rec.to_dict(), force=force, progress=progress))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Paper %s failed", rec.paper_id)
                results.append({"paper_id": rec.paper_id, "status": "error", "error": str(exc)})
            if results and results[-1].get("status") == "llm_unavailable":
                break
        return self._summary(query, results)

    def ingest_pdf_file(self, pdf_path: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = dict(meta or {})
        meta.setdefault("title", Path(pdf_path).stem.replace("_", " "))
        meta.setdefault("source", "upload")
        return self.process_paper(meta, pdf_path=pdf_path)

    @staticmethod
    def _summary(query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        by: Dict[str, int] = {}
        for r in results:
            by[r["status"]] = by.get(r["status"], 0) + 1
        return {"query": query, "papers": len(results), "by_status": by,
                "triplets": sum(r.get("triplets", 0) for r in results), "results": results}
