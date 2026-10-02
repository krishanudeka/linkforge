"""Tiny in-memory job registry for background ingestion (no Redis needed)."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
MAX_JOBS = 100


def create_job(kind: str, params: Dict[str, Any]) -> str:
    jid = uuid.uuid4().hex[:12]
    with _lock:
        if len(_jobs) >= MAX_JOBS:
            for k in sorted(_jobs, key=lambda k: _jobs[k]["created"])[: MAX_JOBS // 4]:
                _jobs.pop(k, None)
        _jobs[jid] = {"id": jid, "kind": kind, "params": params, "status": "queued",
                      "created": time.time(), "log": [], "result": None, "error": None}
    return jid


def update_job(jid: str, **fields: Any) -> None:
    with _lock:
        if jid in _jobs:
            _jobs[jid].update(fields)


def log_job(jid: str, msg: str) -> None:
    with _lock:
        if jid in _jobs:
            _jobs[jid]["log"].append(msg)
            del _jobs[jid]["log"][:-200]


def get_job(jid: str) -> Dict[str, Any] | None:
    with _lock:
        j = _jobs.get(jid)
        return dict(j) if j else None


def list_jobs() -> List[Dict[str, Any]]:
    with _lock:
        return sorted((dict(j) for j in _jobs.values()), key=lambda j: -j["created"])
