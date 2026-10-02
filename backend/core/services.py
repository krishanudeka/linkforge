"""Shared, lazily-built service container.

Everything heavy (Neo4j driver, embedding model, ChromaDB, LLM client, GNN) is created on first use,
so importing the API or running unit tests never loads a model or opens a connection.
Tests can build a `Services(...)` with fakes, or call `set_services()`.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from backend.config import settings

logger = logging.getLogger("Services")


class Services:
    def __init__(self, neo4j: Any = None, embedder: Any = None, vectors: Any = None, llm: Any = None,
                 extractor: Any = None, resolver: Any = None, link_predictor: Any = None,
                 store: Any = None, parser: Any = None, graph_cache: Any = None):
        self._o = dict(neo4j=neo4j, embedder=embedder, vectors=vectors, llm=llm, extractor=extractor,
                       resolver=resolver, link_predictor=link_predictor, store=store, parser=parser,
                       graph_cache=graph_cache)
        self._lock = threading.RLock()
        self._resolver_loaded = resolver is not None
        self.ingest_lock = threading.Lock()   # one ingestion at a time (LLM rate limits, resolver state)

    def _get(self, key: str, factory):
        with self._lock:
            if self._o[key] is None:
                self._o[key] = factory()
            return self._o[key]

    @property
    def neo4j(self):
        def make():
            from backend.core.neo4j_client import Neo4jClient
            return Neo4jClient().connect()
        return self._get("neo4j", make)

    @property
    def embedder(self):
        def make():
            from backend.core.embeddings import Embedder
            return Embedder()
        return self._get("embedder", make)

    @property
    def vectors(self):
        def make():
            from backend.core.vector_client import VectorClient
            return VectorClient(self.embedder)
        return self._get("vectors", make)

    @property
    def llm(self):
        def make():
            from ai_engine.extraction.llm_extractor import LLMClient
            return LLMClient()
        return self._get("llm", make)

    @property
    def extractor(self):
        def make():
            from ai_engine.extraction.llm_extractor import LLMExtractor
            return LLMExtractor(self.llm)
        return self._get("extractor", make)

    @property
    def resolver(self):
        def make():
            from ai_engine.resolution.entity_linker import EntityResolver
            return EntityResolver(self.embedder)
        r = self._get("resolver", make)
        with self._lock:
            if not self._resolver_loaded:
                try:
                    r.load_entities(self.neo4j.fetch_entities())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not preload entity registry: %s", exc)
                self._resolver_loaded = True
        return r

    @property
    def link_predictor(self):
        def make():
            from ai_engine.gnn.predict import LinkPredictionService
            return LinkPredictionService(self.embedder)
        return self._get("link_predictor", make)

    @property
    def store(self):
        def make():
            from data_pipeline.storage.pdf_store import PDFStore
            return PDFStore()
        return self._get("store", make)

    @property
    def parser(self):
        def make():
            from data_pipeline.parsers.pdf_parser import PDFParser
            return PDFParser()
        return self._get("parser", make)

    @property
    def graph_cache(self):
        def make():
            from backend.core.graph_analytics import GraphCache
            return GraphCache(self.neo4j)
        return self._get("graph_cache", make)

    @property
    def snapshot(self):
        return self.graph_cache.get()

    def close(self) -> None:
        n = self._o.get("neo4j")
        if n is not None:
            try:
                n.close()
            except Exception:  # noqa: BLE001
                pass


_services: Optional[Services] = None
_services_lock = threading.Lock()


def get_services() -> Services:
    global _services
    with _services_lock:
        if _services is None:
            _services = Services()
        return _services


def set_services(s: Optional[Services]) -> None:
    global _services
    with _services_lock:
        _services = s
