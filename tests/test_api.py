import time


def test_search_finds_transitive_nodes_and_papers(client):
    r = client.get("/api/search", params={"q": "chocolate", "top_k": 10}).json()
    assert r["found"] and r["seed"] == "chocolate"
    ids = {n["id"] for n in r["nodes"]}
    assert {"cocoa", "flavan-3-ol", "nf-kb"} <= ids
    assert r["papers"] and "<mark" in "".join(p["snippet_html"] for p in r["papers"])
    assert r["link_prediction"]["method"] == "adamic_adar"


def test_search_unknown_gives_suggestions(client):
    r = client.get("/api/search", params={"q": "zzzz"}).json()
    assert r["found"] is False and r["nodes"] == []


def test_suggest(client):
    r = client.get("/api/search/suggest", params={"q": "coc"}).json()
    assert [s["name"] for s in r["suggestions"]] == ["cocoa"]


def test_bridge_path(client):
    r = client.get("/api/search/bridge", params={"a": "chocolate", "b": "alzheimer's disease", "max_hops": 7}).json()
    assert r["found"] and r["paths"][0]["nodes"][0] == "chocolate"
    assert r["paths"][0]["nodes"][-1] == "alzheimer's disease"
    assert r["paths"][0]["steps"][0]["evidence"]


def test_bridge_404(client):
    assert client.get("/api/search/bridge", params={"a": "chocolate", "b": "nonexistent"}).status_code == 404


def test_graph_endpoints(client):
    assert client.get("/api/graph/stats").json()["graph"]["entities"] == 9
    e = client.get("/api/graph/entity/cocoa").json()
    assert e["degree"] == 2
    ev = client.get("/api/graph/edge", params={"source": "cocoa", "target": "chocolate"}).json()
    assert ev["papers"] and ev["evidence"]
    assert client.get("/api/graph/edge", params={"source": "cocoa", "target": "trem2"}).status_code == 404
    p = client.get("/api/graph/predict", params={"name": "curcumin"}).json()
    assert p["status"]["method"] == "adamic_adar"


def test_chat_is_grounded_and_labels_context(client, svc):
    r = client.post("/api/chat", json={"question": "How is chocolate connected to flavan-3-ol?"}).json()
    assert r["grounded"] and r["answer"]
    assert r["facts"]
    sent = svc.llm.last_messages[-1]["content"]
    assert "[F1]" in sent and "PREDICTED LINKS" in sent


def test_chat_without_context(client, svc):
    svc.vectors.prov.clear(); svc.vectors.chunks.clear()
    r = client.post("/api/chat", json={"question": "what about qqq zzz?"}).json()
    assert r["grounded"] is False


def test_chat_llm_down_falls_back(client, svc):
    from ai_engine.extraction.llm_extractor import LLMUnavailable

    def boom(*a, **k):
        raise LLMUnavailable("down")
    svc.llm.chat = boom
    r = client.post("/api/chat", json={"question": "chocolate cocoa"}).json()
    assert r["llm_used"] is False and "FACTS" in r["answer"]


def test_ingest_upload_rejects_non_pdf(client):
    r = client.post("/api/ingest/upload", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400


def test_ingest_query_job_runs(client, svc, monkeypatch):
    from data_pipeline.scrapers.base_scraper import PaperRecord

    class FakeScraper:
        def search(self, q, limit):
            return [PaperRecord(paper_id="10.1/new", title="New paper", doi="10.1/new", year=2024,
                                abstract="Resveratrol activates SIRT1 in neurons. " * 10)]

        def download_pdf(self, rec):
            return None
    monkeypatch.setattr("backend.core.pipeline.get_scraper", lambda name: FakeScraper())
    svc._o["extractor"].triplets = [{"subject": "resveratrol", "relation": "ACTIVATES", "object": "sirt1",
                                     "confidence": 0.9, "subject_type": "Chemical", "object_type": "Protein",
                                     "provenance_snippet": "Resveratrol activates SIRT1 in neurons."}]
    jid = client.post("/api/ingest/query", json={"query": "resveratrol"}).json()["job_id"]
    for _ in range(50):
        j = client.get(f"/api/ingest/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j
    assert j["result"]["triplets"] == 1
    assert "sirt1" in svc.neo4j.entities


def test_admin_key_protects_write_endpoints(client, monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "admin_api_key", "s3cret")
    assert client.post("/api/ingest/query", json={"query": "abc"}).status_code == 401
    assert client.post("/api/graph/reload-model").status_code == 401
    assert client.post("/api/ingest/query", json={"query": "abc", "source": "nope"},
                       headers={"X-API-Key": "s3cret"}).status_code == 400   # passes auth, fails validation
    assert client.get("/api/search", params={"q": "cocoa"}).status_code == 200   # reads stay public


def test_rate_limit(client, monkeypatch):
    from backend.config import settings
    from backend.api import security
    security._hits.clear()
    monkeypatch.setattr(settings, "rate_limit_per_min", 3)
    codes = [client.get("/api/search/suggest", params={"q": "coc"}).status_code for _ in range(5)]
    assert codes[:3] == [200] * 3 and 429 in codes[3:]
    security._hits.clear()
