"""Regression: bridge evidence must belong to the exact edge (relation + paper) of the selected step.

Bug: evidence was looked up by (subject, object) only, so a sentence from a *different* relation/paper
between the same two entities (ACTIVATES / 2023 paper) was attached to the selected PROMOTES / demo:003 edge.
"""
import pytest

from tests.conftest import FakeEmbedder, FakeExtractor, FakeLLM, FakeNeo4j, FakeStore, FakeVectors

A, B = "microglial activation", "neuroinflammation"
PROMOTES_PAPER = "demo:003"
ACTIVATES_PAPER = "10.1038/s41392-023-01588-0"

EDGE_PROMOTES = {"subject": A, "relation": "PROMOTES", "object": B, "confidence": 0.9,
                 "provenance_snippet": "Microglial activation promotes neuroinflammation in rodent brain."}
EDGE_ACTIVATES = {"subject": A, "relation": "ACTIVATES", "object": B, "confidence": 0.8,
                  "provenance_snippet": "Microglia activate neuroinflammatory responses (2023 review)."}


def _meta(pid, year):
    return {"paper_id": pid, "title": f"Paper {pid}", "year": year, "doi": pid}


# --------------------------------------------------------------------------- bridge() level (fakes)
@pytest.fixture
def bridge_svc():
    from ai_engine.gnn.predict import LinkPredictionService
    from ai_engine.resolution.entity_linker import EntityResolver
    from backend.core.graph_analytics import GraphCache
    from backend.core.services import Services, set_services

    neo, vec, emb = FakeNeo4j(), FakeVectors(), FakeEmbedder()
    for pid, year, trip in ((PROMOTES_PAPER, 2019, EDGE_PROMOTES), (ACTIVATES_PAPER, 2023, EDGE_ACTIVATES)):
        neo.upsert_paper({"paper_id": pid, "title": f"Paper {pid}", "year": year, "doi": pid, "abstract": "x"})
        neo.insert_triplets([trip], pid)
        vec.add_provenance(pid, [trip], _meta(pid, year))
        neo.mark_paper_processed(pid, 1, 1)
    s = Services(neo4j=neo, embedder=emb, vectors=vec, llm=FakeLLM(), extractor=FakeExtractor(),
                 store=FakeStore(), link_predictor=LinkPredictionService(emb, model_path="/nonexistent.pt"))
    s._o["graph_cache"] = GraphCache(neo)
    s._o["resolver"] = EntityResolver(emb)
    set_services(s)
    yield s
    set_services(None)


def test_bridge_step_evidence_matches_selected_edge(bridge_svc):
    from backend.core.graph_rag import GraphRAG

    # precondition: the pair-only lookup really is ambiguous (this is the bug's raw material)
    pair_only = bridge_svc.vectors.get_edge_provenance(A, B)
    assert {r["metadata"]["relation"] for r in pair_only} == {"PROMOTES", "ACTIVATES"}

    res = GraphRAG(bridge_svc).bridge(A, B, k=1, max_hops=1)
    step = res["paths"][0]["steps"][0]
    # PROMOTES / demo:003 is the selected edge (higher confidence) ...
    assert (step["type"], step["papers"]) == ("PROMOTES", [PROMOTES_PAPER])
    # ... so its evidence must be that edge's provenance only
    assert step["evidence"], "selected edge has provenance, it must not come back empty"
    for ev in step["evidence"]:
        assert ev["metadata"]["relation"] == "PROMOTES"
        assert ev["metadata"]["paper_id"] == PROMOTES_PAPER
    blob = repr(step["evidence"])
    assert "ACTIVATES" not in blob and ACTIVATES_PAPER not in blob


def test_bridge_via_api_has_no_cross_edge_evidence(bridge_svc):
    from fastapi.testclient import TestClient
    from backend.main import app

    r = TestClient(app).get("/api/search/bridge", params={"a": A, "b": B, "k": 1, "max_hops": 1}).json()
    ev = r["paths"][0]["steps"][0]["evidence"]
    assert ev and all(e["metadata"]["relation"] == "PROMOTES" for e in ev)


def test_fake_vectors_old_call_signature_still_works(bridge_svc):
    v = bridge_svc.vectors
    assert len(v.get_edge_provenance(A, B)) == 2
    assert len(v.get_edge_provenance(B, A, limit=6)) == 2          # reverse direction, keyword limit
    assert len(v.get_edge_provenance(A, B, 5)) == 2                # positional limit


# --------------------------------------------------------------------------- real VectorClient + real Chroma
@pytest.fixture
def vc(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    from backend.config import settings
    from backend.core.vector_client import VectorClient

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))  # chroma_dir derives from this
    client = VectorClient(FakeEmbedder(), path=str(tmp_path / "chroma_db"))
    client.add_provenance(PROMOTES_PAPER, [EDGE_PROMOTES], _meta(PROMOTES_PAPER, 2019))
    client.add_provenance(ACTIVATES_PAPER, [EDGE_ACTIVATES], _meta(ACTIVATES_PAPER, 2023))
    return client


def _pairs(rows):
    return {(r["metadata"]["relation"], r["metadata"]["paper_id"]) for r in rows}


def test_chroma_relation_and_paper_filter_isolates_exact_edge(vc):
    rows = vc.get_edge_provenance(A, B, limit=2, relation="PROMOTES", paper_ids=[PROMOTES_PAPER])
    assert _pairs(rows) == {("PROMOTES", PROMOTES_PAPER)}
    rows = vc.get_edge_provenance(A, B, limit=2, relation="ACTIVATES", paper_ids=[ACTIVATES_PAPER])
    assert _pairs(rows) == {("ACTIVATES", ACTIVATES_PAPER)}


def test_chroma_right_relation_wrong_paper_returns_nothing(vc):
    assert vc.get_edge_provenance(A, B, relation="PROMOTES", paper_ids=[ACTIVATES_PAPER]) == []


def test_chroma_each_filter_works_alone(vc):
    assert _pairs(vc.get_edge_provenance(A, B, relation="PROMOTES")) == {("PROMOTES", PROMOTES_PAPER)}
    assert _pairs(vc.get_edge_provenance(A, B, paper_ids=[ACTIVATES_PAPER])) == {("ACTIVATES", ACTIVATES_PAPER)}
    both = vc.get_edge_provenance(A, B, paper_ids=[PROMOTES_PAPER, ACTIVATES_PAPER])
    assert len(both) == 2


def test_chroma_backward_compatible_and_empty_filters_ignored(vc):
    both = {("PROMOTES", PROMOTES_PAPER), ("ACTIVATES", ACTIVATES_PAPER)}
    assert _pairs(vc.get_edge_provenance(A, B)) == both                       # old 2-arg call
    assert _pairs(vc.get_edge_provenance(B, A, limit=6)) == both              # reverse + limit (graph.py style)
    assert _pairs(vc.get_edge_provenance(A, B, 5)) == both                    # positional limit
    # empty list / None / "" must not be turned into an (invalid) empty $in or a bogus relation filter
    assert _pairs(vc.get_edge_provenance(A, B, relation=None, paper_ids=[])) == both
    assert _pairs(vc.get_edge_provenance(A, B, relation="", paper_ids=None)) == both


def test_chroma_unknown_pair_and_empty_store(vc, tmp_path):
    assert vc.get_edge_provenance("nope", "nada", relation="PROMOTES", paper_ids=[PROMOTES_PAPER]) == []


def test_chroma_limit_applies_after_confidence_sort(vc):
    # Many low-confidence rows are inserted first; the best one last. Chroma's get(limit=) returns rows in
    # insertion order, so the old code (limit before sort) would have returned a weak row for limit=1.
    weak = [{"subject": A, "relation": "PROMOTES", "object": B, "confidence": 0.1 + i / 100,
             "provenance_snippet": f"weak statement number {i}."} for i in range(8)]
    vc.add_provenance("weak:paper", weak, _meta("weak:paper", 2010))
    best = {"subject": A, "relation": "PROMOTES", "object": B, "confidence": 0.99,
            "provenance_snippet": "The strongest statement."}
    vc.add_provenance("strong:paper", [best], _meta("strong:paper", 2024))

    top = vc.get_edge_provenance(A, B, limit=1)
    assert len(top) == 1 and top[0]["metadata"]["confidence"] == pytest.approx(0.99)
    top2 = vc.get_edge_provenance(A, B, limit=2)
    confs = [r["metadata"]["confidence"] for r in top2]
    assert len(top2) == 2 and confs == sorted(confs, reverse=True) and confs[0] == pytest.approx(0.99)
