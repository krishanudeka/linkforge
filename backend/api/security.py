"""Admin-key guard and a tiny per-IP rate limiter (in-memory; fine for a single free-tier instance)."""
from __future__ import annotations

import hmac
import logging
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from backend.config import settings

logger = logging.getLogger("Security")


def require_admin(x_api_key: str = Header(default="")) -> None:
    """Dependency for endpoints that write data or spend LLM quota.
    If ADMIN_API_KEY is unset the endpoints stay open (local dev only) - main.py logs a warning."""
    if not settings.admin_api_key:
        return
    if not hmac.compare_digest(x_api_key.encode(), settings.admin_api_key.encode()):
        raise HTTPException(401, "Missing or invalid X-API-Key")


_hits: dict = defaultdict(deque)


def rate_limit(request: Request) -> None:
    limit = settings.rate_limit_per_min
    if limit <= 0:
        return
    ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
          or (request.client.host if request.client else "unknown"))
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many requests - slow down")
    q.append(now)
    if len(_hits) > 5000:          # bound memory
        for k in [k for k, v in _hits.items() if not v][:1000]:
            _hits.pop(k, None)
