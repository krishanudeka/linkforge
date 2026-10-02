import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.api import jobs
from backend.api.security import require_admin
from backend.core.pipeline import IngestionPipeline
from backend.core.services import get_services
from data_pipeline.scrapers import SCRAPERS
from data_pipeline.storage.pdf_store import safe_name

router = APIRouter(prefix="/ingest", tags=["ingest"], dependencies=[Depends(require_admin)])
MAX_UPLOAD_BYTES = 40 * 1024 * 1024


class QueryIngest(BaseModel):
    query: str = Field(..., min_length=2, max_length=300)
    source: str = "europe_pmc"
    limit: int = Field(10, ge=1, le=50)


def _run_query(jid: str, req: QueryIngest):
    jobs.update_job(jid, status="running")
    try:
        res = IngestionPipeline(get_services()).ingest_query(
            req.query, req.source, req.limit, progress=lambda m: jobs.log_job(jid, m))
        jobs.update_job(jid, status="done", result=res)
    except Exception as exc:  # noqa: BLE001
        jobs.update_job(jid, status="failed", error=str(exc))


def _run_pdf(jid: str, path: str, meta: dict):
    jobs.update_job(jid, status="running")
    try:
        res = IngestionPipeline(get_services()).ingest_pdf_file(path, meta)
        jobs.update_job(jid, status="done", result=res)
    except Exception as exc:  # noqa: BLE001
        jobs.update_job(jid, status="failed", error=str(exc))


@router.post("/query", status_code=202)
def ingest_query(req: QueryIngest, bg: BackgroundTasks):
    if req.source not in SCRAPERS:
        raise HTTPException(400, f"source must be one of {sorted(SCRAPERS)}")
    jid = jobs.create_job("query", req.model_dump())
    bg.add_task(_run_query, jid, req)
    return {"job_id": jid}


@router.post("/upload", status_code=202)
async def ingest_upload(bg: BackgroundTasks, file: UploadFile = File(...),
                        title: Optional[str] = Form(None), doi: Optional[str] = Form(None),
                        year: Optional[int] = Form(None)):
    content = await file.read(MAX_UPLOAD_BYTES + 1)     # never buffer more than the limit
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "PDF too large (40 MB max)")
    store = get_services().store
    name = safe_name(Path(file.filename or "upload").stem)
    try:
        path = store.save_pdf(f"upload_{name}", content)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    meta = {"title": title or name.replace("_", " "), "doi": doi, "year": year, "source": "upload"}
    if doi:
        from data_pipeline.scrapers.base_scraper import normalize_doi
        meta["paper_id"] = normalize_doi(doi)
    else:
        meta["paper_id"] = f"upload:{name}"
    jid = jobs.create_job("upload", {"file": file.filename, **meta})
    bg.add_task(_run_pdf, jid, str(path), meta)
    return {"job_id": jid}


@router.get("/jobs")
def list_jobs():
    return {"jobs": jobs.list_jobs()}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    j = jobs.get_job(job_id)
    if not j:
        raise HTTPException(404, "Unknown job")
    return j


@router.get("/papers")
def papers(limit: int = 50, skip: int = 0):
    return {"papers": get_services().neo4j.list_papers(limit, skip)}
