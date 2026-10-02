"""In-memory fakes so the whole stack (API, pipeline, Graph-RAG) is testable without Neo4j/LLM/Chroma."""
import hashlib
from collections import defaultdict

import numpy as np
import pytest

from backend.core.services import Services, set_services


class FakeEmbedder:
    dim = 32

    def _vec(self, t):
        seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_documents(self, texts, batch_size=64):
        return np.stack([self._vec(t) for t in texts]) if len(texts) else np.zeros((0, self.dim), np.float32)

    def embed_queries(self, texts):
        return self.embed_documents(texts)

    def embed_query(self, text):
        return self._vec(text).tolist()


class FakeNeo4j:
    def __init__(self):
        self.papers, self.entities, self.rel = {}, {}, {}
        self.mentions = defaultdict(set)
        self.version = 0

    def _bump(self):
        self.version += 1

    def upsert_paper(self, p):
        cur = self.papers.setdefault(p["paper_id"], {"processed": False})
        cur.update({k: v for k, v in p.items() if v is not None})
        self._bump()

    def paper_processed(self, pid):
        return bool(self.papers.get(pid, {}).get("processed"))

    def mark_paper_processed(self, pid, n_triplets, n_chunks):
        self.papers[pid].update(processed=True, n_triplets=n_triplets)

    def insert_triplets(self, triplets, paper_id, batch_size=200):
        n = 0
        for t in triplets:
            if not t["subject"] or not t["object"] or t["subject"] == t["object"]:
                continue
            for nm, ty, raw in ((t["subject"], t.get("subject_type"), t.get("subject_raw")),
                                (t["object"], t.get("object_type"), t.get("object_raw"))):
                e = self.entities.setdefault(nm, {"name": nm, "type": ty or "Other", "aliases": []})
                if raw and raw != nm and raw not in e["aliases"]:
                    e["aliases"].append(raw)
                self.mentions[nm].add(paper_id)
            r = self.rel.setdefault((t["subject"], t["object"], t["relation"]),
                                    {"papers": [], "confs": [], "year": None})
            if paper_id not in r["papers"]:
                r["papers"].append(paper_id)
                r["confs"].append(t["confidence"])
            y = self.papers.get(paper_id, {}).get("year")
            if y and (r["year"] is None or y < r["year"]):
                r["year"] = y
            n += 1
        self._bump()
        return n

    def fetch_entities(self):
        return [dict(e) for e in self.entities.values()]

    def fetch_edges(self):
        return [{"source": s, "target": o, "type": ty, "confidence": sum(r["confs"]) / len(r["confs"]),
                 "support": len(r["papers"]), "year": r["year"], "papers": list(r["papers"])}
                for (s, o, ty), r in self.rel.items()]

    def find_entity(self, term):
        t = term.strip().lower()
        for e in self.entities.values():
            if e["name"] == t or t in e["aliases"]:
                return {"name": e["name"], "type": e["type"]}
        hits = self.search_entities(term, 1)
        return hits[0] if hits else None

    def search_entities(self, term, limit=10):
        t = term.strip().lower()
        return [{"name": e["name"], "type": e["type"], "score": 1.0}
                for e in self.entities.values() if t in e["name"]][:limit]

    def get_papers(self, ids):
        return [dict(self.papers[i], paper_id=i) for i in ids if i in self.papers]

    def get_papers_for_entities(self, names, limit=100):
        by = defaultdict(set)
        for n in names:
            for pid in self.mentions.get(n, ()):
                by[pid].add(n)
        rows = [dict(self.papers[p], paper_id=p, entities=sorted(v)) for p, v in by.items()]
        rows.sort(key=lambda r: -len(r["entities"]))
        return rows[:limit]

    def get_edge_papers(self, a, b):
        return [{"source": s, "target": o, "type": ty, "confidence": 0.8, "support": len(r["papers"]),
                 "papers": r["papers"]} for (s, o, ty), r in self.rel.items() if {s, o} == {a, b}]

    def list_papers(self, limit=50, skip=0):
        return [{"paper_id": k, "title": v.get("title"), "processed": v["processed"]}
                for k, v in list(self.papers.items())[skip:skip + limit]]

    def stats(self):
        return {"entities": len(self.entities), "papers": len(self.papers), "relations": len(self.rel),
                "processed_papers": sum(p["processed"] for p in self.papers.values())}

    def clear_all(self):
        self.__init__()

    def close(self):
        pass


class FakeVectors:
    def __init__(self):
        self.chunks, self.prov = [], []

    def add_chunks(self, pid, chunks, meta):
        self.chunks += [{"text": c["text"], "metadata": {"paper_id": pid, "doi": meta.get("doi"),
                                                        "title": meta.get("title"), "year": meta.get("year")}}
                        for c in chunks]
        return len(chunks)

    def add_provenance(self, pid, trips, meta):
        rows = [t for t in trips if t.get("provenance_snippet")]
        self.prov += [{"text": t["provenance_snippet"], "metadata": {
            "paper_id": pid, "doi": meta.get("doi"), "title": meta.get("title"), "year": meta.get("year"),
            "subject": t["subject"], "object": t["object"], "relation": t["relation"],
            "confidence": t["confidence"]}} for t in rows]
        return len(rows)

    @staticmethod
    def _rank(rows, q, limit):
        qs = set(q.lower().split())
        scored = sorted(rows, key=lambda r: -len(qs & set(r["text"].lower().split())))
        return [dict(r, id=str(i), score=0.5) for i, r in enumerate(scored[:limit])]

    def search_chunks(self, q, limit=4, paper_ids=None):
        return self._rank(self.chunks, q, limit)

    def search_provenance(self, q, limit=4):
        return self._rank(self.prov, q, limit)

    def get_edge_provenance(self, a, b, limit=5, relation=None, paper_ids=None):
        rows = [r for r in self.prov if {r["metadata"]["subject"], r["metadata"]["object"]} == {a, b}]
        if relation:
            rows = [r for r in rows if r["metadata"]["relation"] == relation]
        if paper_ids:
            rows = [r for r in rows if r["metadata"]["paper_id"] in set(paper_ids)]
        return rows[:limit]

    def counts(self):
        return {"chunks": len(self.chunks), "provenance": len(self.prov)}


class FakeExtractor:
    """Returns canned triplets regardless of the text."""
    def __init__(self, triplets=None, exc=None):
        self.triplets, self.exc, self.calls = triplets or [], exc, 0

    def extract_triplets(self, text, title=None, min_confidence=None):
        self.calls += 1
        if self.exc:
            raise self.exc
        return [dict(t) for t in self.triplets]


class FakeLLM:
    def __init__(self, reply="Grounded answer [F1]."):
        self.reply, self.last_messages = reply, None

    def chat(self, messages, json_mode=False, temperature=0.1, max_tokens=1500):
        self.last_messages = messages
        return self.reply

    def is_available(self):
        return True


class FakeStore:
    def __init__(self):
        self.md = {}

    def save_markdown(self, pid, text):
        self.md[pid] = text

    def save_pdf(self, pid, content):
        from data_pipeline.storage.pdf_store import PDFStore
        if not content.startswith(b"%PDF"):
            raise ValueError("Uploaded content is not a valid PDF file")
        return f"/tmp/{pid}.pdf"


DEMO = [
    ("chocolate", "CONTAINS_COMPOUND", "cocoa", 0.95, "p1", 2016),
    ("cocoa", "CONTAINS_COMPOUND", "flavan-3-ol", 0.9, "p1", 2016),
    ("flavan-3-ol", "INHIBITS", "nf-kb", 0.85, "p2", 2018),
    ("nf-kb", "ACTIVATES", "microglial activation", 0.88, "p3", 2019),
    ("microglial activation", "PROMOTES", "neuroinflammation", 0.9, "p3", 2019),
    ("neuroinflammation", "CONTRIBUTES_TO", "alzheimer's disease", 0.9, "p4", 2020),
    ("curcumin", "INHIBITS", "nf-kb", 0.9, "p5", 2017),
    ("trem2", "REGULATES", "microglial activation", 0.86, "p6", 2021),
]


@pytest.fixture
def svc():
    from backend.core.services import Services
    from ai_engine.gnn.predict import LinkPredictionService

    neo, vec, emb = FakeNeo4j(), FakeVectors(), FakeEmbedder()
    years = {}
    for s, r, o, c, pid, y in DEMO:
        years[pid] = y
    for pid, y in years.items():
        neo.upsert_paper({"paper_id": pid, "title": f"Paper {pid}", "year": y, "doi": f"10.0000/{pid}",
                          "abstract": f"abstract of {pid} mentioning cocoa and nf-kb"})
    for s, r, o, c, pid, y in DEMO:
        t = {"subject": s, "relation": r, "object": o, "confidence": c, "provenance_snippet": f"{s} {r} {o}."}
        neo.insert_triplets([t], pid)
        vec.add_provenance(pid, [t], neo.papers[pid])
    for pid in years:
        neo.mark_paper_processed(pid, 1, 1)
    s = Services(neo4j=neo, embedder=emb, vectors=vec, llm=FakeLLM(), extractor=FakeExtractor(),
                 store=FakeStore(), link_predictor=LinkPredictionService(emb, model_path="/nonexistent.pt"))
    from backend.core.graph_analytics import GraphCache
    s._o["graph_cache"] = GraphCache(neo)
    from ai_engine.resolution.entity_linker import EntityResolver
    s._o["resolver"] = EntityResolver(emb)
    set_services(s)
    yield s
    set_services(None)


@pytest.fixture
def client(svc):
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)
