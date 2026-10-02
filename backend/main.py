"""FastAPI entrypoint:  uvicorn backend.main:app --reload   (run from the project root)"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.router import api_router
from backend.config import settings
from backend.core.services import get_services
from backend.tasks.scheduler import start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("linkforge")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_services().neo4j  # connect + ensure schema; fail loudly but let /health explain
    except Exception as exc:  # noqa: BLE001
        logger.error("Neo4j not reachable at startup: %s", exc)
    if not settings.admin_api_key:
        logger.warning("ADMIN_API_KEY is not set: ingest/admin endpoints are OPEN. Set it before any public deployment.")
    start_scheduler()
    yield
    stop_scheduler()
    get_services().close()


app = FastAPI(title="LinkForge API", version="1.0.0",
              description="Graph-RAG & predictive knowledge discovery over scientific literature",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router)


@app.get("/")
def root():
    return {"name": "LinkForge", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    out = {"status": "ok", "neo4j": "unknown", "llm": "unknown"}
    try:
        get_services().neo4j.stats()
        out["neo4j"] = "ok"
    except Exception as exc:  # noqa: BLE001
        logger.error("health: neo4j error: %s", exc)
        out.update(status="degraded", neo4j="unreachable (see server logs)")
    try:
        s = get_services()
        out["llm"] = "ok" if s.llm.is_available() else "not configured/unreachable"
    except Exception as exc:  # noqa: BLE001
        out["llm"] = "error (see server logs)"
    return out
