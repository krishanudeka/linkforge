"""Optional background auto-scraper (APScheduler, in-process – no Redis)."""
from __future__ import annotations

import logging
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler

from backend.config import settings
from backend.core.pipeline import IngestionPipeline
from backend.core.services import get_services

logger = logging.getLogger("Scheduler")
_scheduler: Optional[BackgroundScheduler] = None


def scrape_job() -> None:
    pipe = IngestionPipeline(get_services())
    for q in settings.scrape_queries:
        try:
            res = pipe.ingest_query(q, settings.auto_scrape_source, settings.auto_scrape_limit)
            logger.info("Auto-scrape '%s': %s", q, res["by_status"])
            if res["by_status"].get("llm_unavailable"):
                break
        except Exception:  # noqa: BLE001
            logger.exception("Auto-scrape failed for '%s'", q)


def start_scheduler() -> Optional[BackgroundScheduler]:
    global _scheduler
    if not settings.auto_scrape_enabled or not settings.scrape_queries:
        return None
    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(scrape_job, "interval", hours=settings.auto_scrape_interval_hours,
                       id="auto_scrape", max_instances=1, coalesce=True)
    _scheduler.start()
    logger.info("Auto-scraper every %.1fh for %d queries", settings.auto_scrape_interval_hours,
                len(settings.scrape_queries))
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
