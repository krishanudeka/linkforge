"""Embedded ChromaDB store (local, zero-cost).

Two collections:
  * `paper_chunks` – text chunks of each paper, used for Graph-RAG retrieval.
  * `provenance`   – the exact sentence that supports each extracted triplet.
Embeddings are computed by our shared `Embedder` (BGE) and passed to Chroma explicitly.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.core.embeddings import Embedder

logger = logging.getLogger("VectorClient")


def _sid(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()


def _sanitize(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata may only hold str/int/float/bool – drop None, stringify the rest."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return out


class VectorClient:
    def __init__(self, embedder: Embedder, path: Optional[str] = None):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.embedder = embedder
        persist = str(path or settings.chroma_dir)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist, settings=ChromaSettings(anonymized_telemetry=False)
        )
        cosine = {"hnsw:space": "cosine"}
        self.chunks = self.client.get_or_create_collection("paper_chunks", metadata=cosine)
        self.provenance = self.client.get_or_create_collection("provenance", metadata=cosine)

    # ------------------------------------------------------------------ writes
    def add_chunks(self, paper_id: str, chunks: List[Dict[str, Any]], paper_meta: Dict[str, Any]) -> int:
        """chunks: [{"text":..., "section":..., "index": int}]"""
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        vecs = self.embedder.embed_documents(texts)
        ids = [_sid(paper_id, "chunk", str(c.get("index", i)), c["text"][:64]) for i, c in enumerate(chunks)]
        metas = [
            _sanitize({
                "paper_id": paper_id,
                "doi": paper_meta.get("doi"),
                "title": paper_meta.get("title"),
                "year": paper_meta.get("year"),
                "section": c.get("section"),
                "chunk_index": c.get("index", i),
            })
            for i, c in enumerate(chunks)
        ]
        self.chunks.upsert(ids=ids, documents=texts, embeddings=vecs.tolist(), metadatas=metas)
        return len(ids)

    def add_provenance(self, paper_id: str, triplets: List[Dict[str, Any]], paper_meta: Dict[str, Any]) -> int:
        """triplets need subject/relation/object/confidence/provenance_snippet (canonical names)."""
        rows = [t for t in triplets if (t.get("provenance_snippet") or "").strip()]
        if not rows:
            return 0
        texts = [t["provenance_snippet"].strip() for t in rows]
        vecs = self.embedder.embed_documents(texts)
        ids = [_sid(paper_id, "prov", str(i), t["subject"], t["relation"], t["object"], t["provenance_snippet"][:48])
               for i, t in enumerate(rows)]
        metas = [
            _sanitize({
                "paper_id": paper_id,
                "doi": paper_meta.get("doi"),
                "title": paper_meta.get("title"),
                "year": paper_meta.get("year"),
                "subject": t["subject"],
                "relation": t["relation"],
                "object": t["object"],
                "confidence": float(t.get("confidence", 0.5)),
            })
            for t in rows
        ]
        self.provenance.upsert(ids=ids, documents=texts, embeddings=vecs.tolist(), metadatas=metas)
        return len(ids)

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _flatten(res: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not res or not res.get("ids") or not res["ids"][0]:
            return []
        out = []
        for i, _id in enumerate(res["ids"][0]):
            dist = res["distances"][0][i] if res.get("distances") else None
            out.append({
                "id": _id,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i] or {},
                "score": None if dist is None else round(1.0 - float(dist), 4),  # cosine similarity
            })
        return out

    def search_chunks(self, query: str, limit: int = 4, paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if self.chunks.count() == 0:
            return []
        where = {"paper_id": {"$in": paper_ids}} if paper_ids else None
        res = self.chunks.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(limit, self.chunks.count()),
            where=where,
        )
        return self._flatten(res)

    def search_provenance(self, query: str, limit: int = 4) -> List[Dict[str, Any]]:
        if self.provenance.count() == 0:
            return []
        res = self.provenance.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(limit, self.provenance.count()),
        )
        return self._flatten(res)

    def get_edge_provenance(self, a: str, b: str, limit: int = 5,
                            relation: Optional[str] = None,
                            paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Supporting sentences for the relation between entities a and b (either direction).

        Optionally narrowed to one relation type and/or a set of papers, so evidence can be tied to
        the exact graph edge (subject, relation, object, paper) rather than just the entity pair.
        """
        if self.provenance.count() == 0:
            return []
        pair = {"$or": [
            {"$and": [{"subject": a}, {"object": b}]},
            {"$and": [{"subject": b}, {"object": a}]},
        ]}
        conds: List[Dict[str, Any]] = [pair]
        if relation:
            conds.append({"relation": relation})
        if paper_ids:  # an empty $in is invalid in Chroma, so only filter when there is something to match
            conds.append({"paper_id": {"$in": list(paper_ids)}})
        where = {"$and": conds} if len(conds) > 1 else pair
        # Chroma applies `limit` before we can sort by confidence, so over-fetch, sort, then slice.
        res = self.provenance.get(where=where, limit=max(limit * 10, 50))
        out = []
        for i, _id in enumerate(res.get("ids", [])):
            out.append({"id": _id, "text": res["documents"][i], "metadata": res["metadatas"][i] or {}})
        out.sort(key=lambda r: -float(r["metadata"].get("confidence", 0)))
        return out[:limit]

    def counts(self) -> Dict[str, int]:
        return {"chunks": self.chunks.count(), "provenance": self.provenance.count()}
