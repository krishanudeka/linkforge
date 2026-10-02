"""LLM access (Groq free tier / local Ollama) and triplet extraction.

`LLMClient`    – provider-agnostic chat with rate-limit spacing, retries and optional fallback.
`LLMExtractor` – chunk -> validated triplets (relation normalisation, snippet verification, dedup).
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from ai_engine.extraction.prompts import (
    ENTITY_TYPES, GENERIC_ENTITY_STOPLIST, RELATION_ALIASES, RELATION_TYPES, build_messages,
)
from backend.config import settings

logger = logging.getLogger("LLM")


class LLMUnavailable(RuntimeError):
    pass


# =============================================================================== client
class LLMClient:
    def __init__(self):
        self._groq = None
        self._last_call = 0.0
        self._lock = threading.Lock()

    # ---- provider plumbing ---------------------------------------------------
    @property
    def providers(self) -> List[str]:
        order = [settings.llm_provider.strip().lower()]
        fb = settings.llm_fallback_provider.strip().lower()
        if fb and fb not in order:
            order.append(fb)
        return [p for p in order if p in ("groq", "ollama")]

    def _groq_client(self):
        if self._groq is None:
            if not settings.groq_api_key:
                raise LLMUnavailable("GROQ_API_KEY is not set")
            from groq import Groq

            self._groq = Groq(api_key=settings.groq_api_key, timeout=settings.llm_timeout_sec, max_retries=0)
        return self._groq

    def is_available(self) -> bool:
        for p in self.providers:
            if p == "groq" and settings.groq_api_key:
                return True
            if p == "ollama":
                try:
                    requests.get(f"{settings.ollama_host}/api/tags", timeout=2).raise_for_status()
                    return True
                except Exception:
                    continue
        return False

    def _throttle(self):
        with self._lock:
            wait = settings.llm_min_interval_sec - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def _call_groq(self, messages, json_mode: bool, temperature: float, max_tokens: int) -> str:
        kwargs: Dict[str, Any] = dict(
            model=settings.groq_model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._groq_client().chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _call_ollama(self, messages, json_mode: bool, temperature: float, max_tokens: int) -> str:
        payload: Dict[str, Any] = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        r = requests.post(f"{settings.ollama_host}/api/chat", json=payload, timeout=settings.llm_timeout_sec)
        r.raise_for_status()
        return r.json()["message"]["content"]

    @staticmethod
    def _retry_after(exc: Exception) -> Optional[float]:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if headers:
            try:
                return float(headers.get("retry-after"))
            except (TypeError, ValueError):
                return None
        return None

    def chat(self, messages: List[Dict[str, str]], json_mode: bool = False,
             temperature: float = 0.1, max_tokens: int = 1500) -> str:
        if not self.providers:
            raise LLMUnavailable("No valid LLM_PROVIDER configured (use 'groq' or 'ollama').")
        last_exc: Optional[Exception] = None
        for provider in self.providers:
            if provider == "groq" and not settings.groq_api_key:
                last_exc = LLMUnavailable("GROQ_API_KEY is not set")
                continue
            call = self._call_groq if provider == "groq" else self._call_ollama
            for attempt in range(settings.llm_max_retries):
                self._throttle()
                try:
                    return call(messages, json_mode, temperature, max_tokens)
                except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
                    last_exc = exc
                    status = getattr(exc, "status_code", None) or getattr(
                        getattr(exc, "response", None), "status_code", None)
                    retryable = status in (408, 409, 429, 500, 502, 503, 504) or isinstance(
                        exc, (requests.ConnectionError, requests.Timeout))
                    if not retryable:
                        logger.warning("%s call failed (non-retryable): %s", provider, exc)
                        break
                    delay = self._retry_after(exc) or min(2 ** attempt * 2.0, 60.0)
                    logger.warning("%s call failed (%s), retry %d/%d in %.1fs",
                                   provider, status or type(exc).__name__,
                                   attempt + 1, settings.llm_max_retries, delay)
                    time.sleep(delay)
        raise LLMUnavailable(f"All LLM providers failed: {last_exc}")


# =============================================================================== helpers
_WS = re.compile(r"\s+")


def parse_json_payload(raw: str) -> Any:
    """Tolerant JSON parse: strips code fences, finds the outermost object/array."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def normalize_relation(rel: str) -> str:
    r = re.sub(r"[^A-Za-z0-9]+", "_", (rel or "").strip()).strip("_").upper()
    if not r:
        return ""
    if r in RELATION_TYPES:
        return r
    if r in RELATION_ALIASES:
        return RELATION_ALIASES[r]
    return "ASSOCIATED_WITH" if len(r) > 40 else r  # keep novel-but-sane relations


def normalize_type(t: Optional[str]) -> str:
    if not t:
        return "Other"
    t = t.strip().capitalize()
    return t if t in ENTITY_TYPES else "Other"


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text) if s.strip()]


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def verify_or_find_snippet(snippet: str, chunk: str, subject: str, obj: str) -> str:
    """Return a sentence that really occurs in `chunk`. Prefer the LLM's snippet if it matches,
    otherwise pick the sentence sharing the most tokens with subject+object."""
    chunk_norm = _WS.sub(" ", chunk)
    snip_norm = _WS.sub(" ", (snippet or "")).strip()
    if snip_norm and snip_norm.lower() in chunk_norm.lower():
        return snip_norm
    sentences = _split_sentences(chunk_norm)
    if not sentences:
        return ""
    want = _tokens(subject) | _tokens(obj)
    if snip_norm:
        want |= _tokens(snip_norm)
    best, best_score = "", 0.0
    for s in sentences:
        toks = _tokens(s)
        if not toks:
            continue
        score = len(toks & want) / (len(want) or 1)
        if score > best_score:
            best, best_score = s, score
    return best if best_score >= 0.3 else ""


# =============================================================================== extractor
class LLMExtractor:
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()

    def _valid_entity(self, name: str) -> bool:
        n = name.strip().lower()
        if len(n) < 2 or len(n) > 80:
            return False
        if n in GENERIC_ENTITY_STOPLIST:
            return False
        if re.fullmatch(r"[\d\W_]+", n):  # only digits/punctuation
            return False
        return True

    def extract_triplets(self, text_chunk: str, title: Optional[str] = None,
                         min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """Extract validated triplets from one chunk. Returns [] on unparseable model output."""
        min_conf = settings.min_triplet_confidence if min_confidence is None else min_confidence
        text_chunk = text_chunk.strip()
        if len(text_chunk) < 40:
            return []
        raw = self.client.chat(build_messages(text_chunk[:6000], title), json_mode=True,
                               temperature=0.1, max_tokens=1800)
        payload = parse_json_payload(raw)
        if isinstance(payload, dict):
            items = payload.get("triplets") or payload.get("relations") or []
        elif isinstance(payload, list):
            items = payload
        else:
            logger.warning("Unparseable LLM output, skipping chunk: %.120s", raw)
            return []
        return self._validate(items, text_chunk, min_conf)

    def _validate(self, items: Any, chunk: str, min_conf: float) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []
        best: Dict[tuple, Dict[str, Any]] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            s, o = str(it.get("subject", "")).strip(), str(it.get("object", "")).strip()
            rel = normalize_relation(str(it.get("relation", "")))
            if not (s and o and rel) or s.lower() == o.lower():
                continue
            if not (self._valid_entity(s) and self._valid_entity(o)):
                continue
            try:
                conf = float(it.get("confidence", 0.6))
            except (TypeError, ValueError):
                conf = 0.6
            conf = max(0.0, min(1.0, conf))
            if conf < min_conf:
                continue
            snippet = verify_or_find_snippet(str(it.get("provenance_snippet", "")), chunk, s, o)
            trip = {
                "subject": s, "relation": rel, "object": o, "confidence": round(conf, 3),
                "subject_type": normalize_type(it.get("subject_type")),
                "object_type": normalize_type(it.get("object_type")),
                "provenance_snippet": snippet,
            }
            key = (s.lower(), rel, o.lower())
            if key not in best or trip["confidence"] > best[key]["confidence"]:
                best[key] = trip
        return list(best.values())
