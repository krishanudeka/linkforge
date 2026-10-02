from backend.core.pipeline import IngestionPipeline

REC = {"paper_id": "10.9/x", "title": "T", "doi": "10.9/x", "year": 2024,
       "abstract": "Curcumin inhibits NF-kB in microglia and reduces inflammation markers in mice. " * 6}
TRIP = [{"subject": "Curcumin", "relation": "INHIBITS", "object": "NF-kB", "confidence": 0.9,
         "subject_type": "Chemical", "object_type": "Protein", "provenance_snippet": "Curcumin inhibits NF-kB"}]


def test_process_paper_from_abstract(svc):
    svc.extractor.triplets = TRIP
    res = IngestionPipeline(svc).process_paper(REC)
    assert res["status"] == "ok" and res["used"] == "abstract" and res["triplets"] >= 1
    assert svc.neo4j.paper_processed("10.9/x")
    assert svc.vectors.counts()["provenance"] > 8        # added on top of the 8 seeded


def test_skip_processed_unless_forced(svc):
    svc.extractor.triplets = TRIP
    p = IngestionPipeline(svc)
    p.process_paper(REC)
    calls = svc.extractor.calls
    assert p.process_paper(REC)["status"] == "skipped" and svc.extractor.calls == calls
    assert p.process_paper(REC, force=True)["status"] == "ok"


def test_llm_unavailable_leaves_paper_retryable(svc):
    from ai_engine.extraction.llm_extractor import LLMUnavailable
    svc._o["extractor"] = type(svc.extractor)(exc=LLMUnavailable("x"))
    res = IngestionPipeline(svc).process_paper(REC)
    assert res["status"] == "llm_unavailable" and not svc.neo4j.paper_processed("10.9/x")


def test_no_text(svc):
    res = IngestionPipeline(svc).process_paper({"paper_id": "e", "title": "empty", "abstract": ""})
    assert res["status"] == "no_text"


def test_resolver_merges_aliases(svc):
    svc.extractor.triplets = [
        {"subject": "NF-kB", "relation": "ACTIVATES", "object": "microglial activation", "confidence": .9,
         "subject_type": "Protein", "object_type": "Process", "provenance_snippet": "s"},
        {"subject": "NF-κB", "relation": "ACTIVATES", "object": "microglial activation", "confidence": .8,
         "subject_type": "Protein", "object_type": "Process", "provenance_snippet": "s"}]
    IngestionPipeline(svc).process_paper(REC)
    assert len([n for n in svc.neo4j.entities if n.startswith("nf")]) == 1
