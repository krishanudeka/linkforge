"""Load a small SYNTHETIC demo graph (no LLM, no internet).   python -m scripts.seed_demo [--reset] [--no-vectors]

The "papers" below are placeholders written for testing the UI and the algorithms; they are NOT real
publications and have no DOIs. Facts are loosely based on well-known biomedical themes
(cocoa flavanols, curcumin, neuroinflammation) purely so the demo graph looks sensible.
Years straddle 2022 so you can try the time-machine split.
"""
import argparse
from typing import Dict, List

PAPERS: List[Dict] = [
    {"paper_id": "demo:001", "title": "[DEMO] Cocoa composition overview", "year": 2016,
     "abstract": "Chocolate is made from cocoa, which derives from Theobroma cacao. Cocoa is rich in flavan-3-ols."},
    {"paper_id": "demo:002", "title": "[DEMO] Flavan-3-ols and inflammatory signalling", "year": 2018,
     "abstract": "Flavan-3-ols inhibit NF-kB signalling in cell models, reducing TNF-alpha and IL-6 release."},
    {"paper_id": "demo:003", "title": "[DEMO] NF-kB in microglial activation", "year": 2019,
     "abstract": "NF-kB drives microglial activation, which promotes neuroinflammation in rodent brain."},
    {"paper_id": "demo:004", "title": "[DEMO] Neuroinflammation and Alzheimer's disease", "year": 2020,
     "abstract": "Neuroinflammation contributes to Alzheimer's disease progression. Microglial activation is a hallmark."},
    {"paper_id": "demo:005", "title": "[DEMO] Curcumin as an anti-inflammatory", "year": 2017,
     "abstract": "Curcumin inhibits NF-kB and reduces TNF-alpha. Curcumin is a compound of turmeric."},
    {"paper_id": "demo:006", "title": "[DEMO] TREM2 and microglia", "year": 2021,
     "abstract": "TREM2 is expressed on microglia and regulates microglial activation. TREM2 variants raise Alzheimer's disease risk."},
    {"paper_id": "demo:007", "title": "[DEMO] Amyloid beta and microglia", "year": 2019,
     "abstract": "Amyloid beta accumulation activates microglia and is a pathological feature of Alzheimer's disease."},
    {"paper_id": "demo:008", "title": "[DEMO] Curcumin and amyloid beta aggregation", "year": 2023,
     "abstract": "Curcumin reduces amyloid beta aggregation in vitro."},
    {"paper_id": "demo:009", "title": "[DEMO] Flavanols and cognition trial", "year": 2023,
     "abstract": "Cocoa flavanols improved cognitive scores; flavan-3-ols reduce neuroinflammation markers."},
    {"paper_id": "demo:010", "title": "[DEMO] TREM2 signalling and NF-kB", "year": 2024,
     "abstract": "TREM2 modulates NF-kB signalling in microglia."},
]

# (subject, relation, object, confidence, paper_id, subject_type, object_type, snippet)
T = [
    ("chocolate", "CONTAINS_COMPOUND", "cocoa", 0.95, "demo:001", "Food", "Food", "Chocolate is made from cocoa."),
    ("cocoa", "DERIVED_FROM", "theobroma cacao", 0.95, "demo:001", "Food", "Organism", "cocoa, which derives from Theobroma cacao."),
    ("cocoa", "CONTAINS_COMPOUND", "flavan-3-ol", 0.92, "demo:001", "Food", "Chemical", "Cocoa is rich in flavan-3-ols."),
    ("flavan-3-ol", "INHIBITS", "nf-kb", 0.85, "demo:002", "Chemical", "Protein", "Flavan-3-ols inhibit NF-kB signalling in cell models."),
    ("flavan-3-ol", "DECREASES", "tnf-alpha", 0.8, "demo:002", "Chemical", "Protein", "reducing TNF-alpha and IL-6 release."),
    ("flavan-3-ol", "DECREASES", "il-6", 0.78, "demo:002", "Chemical", "Protein", "reducing TNF-alpha and IL-6 release."),
    ("nf-kb", "ACTIVATES", "microglial activation", 0.88, "demo:003", "Protein", "Process", "NF-kB drives microglial activation."),
    ("microglial activation", "PROMOTES", "neuroinflammation", 0.9, "demo:003", "Process", "Disease", "which promotes neuroinflammation in rodent brain."),
    ("neuroinflammation", "CONTRIBUTES_TO", "alzheimer's disease", 0.9, "demo:004", "Disease", "Disease", "Neuroinflammation contributes to Alzheimer's disease progression."),
    ("microglial activation", "ASSOCIATED_WITH", "alzheimer's disease", 0.82, "demo:004", "Process", "Disease", "Microglial activation is a hallmark."),
    ("curcumin", "INHIBITS", "nf-kb", 0.9, "demo:005", "Chemical", "Protein", "Curcumin inhibits NF-kB and reduces TNF-alpha."),
    ("curcumin", "DECREASES", "tnf-alpha", 0.85, "demo:005", "Chemical", "Protein", "Curcumin inhibits NF-kB and reduces TNF-alpha."),
    ("turmeric", "CONTAINS_COMPOUND", "curcumin", 0.93, "demo:005", "Food", "Chemical", "Curcumin is a compound of turmeric."),
    ("trem2", "REGULATES", "microglial activation", 0.86, "demo:006", "Protein", "Process", "TREM2 ... regulates microglial activation."),
    ("trem2", "ASSOCIATED_WITH", "alzheimer's disease", 0.84, "demo:006", "Protein", "Disease", "TREM2 variants raise Alzheimer's disease risk."),
    ("amyloid beta", "ACTIVATES", "microglial activation", 0.83, "demo:007", "Protein", "Process", "Amyloid beta accumulation activates microglia."),
    ("amyloid beta", "ASSOCIATED_WITH", "alzheimer's disease", 0.9, "demo:007", "Protein", "Disease", "a pathological feature of Alzheimer's disease."),
    ("curcumin", "DECREASES", "amyloid beta", 0.8, "demo:008", "Chemical", "Protein", "Curcumin reduces amyloid beta aggregation in vitro."),
    ("flavan-3-ol", "DECREASES", "neuroinflammation", 0.82, "demo:009", "Chemical", "Disease", "flavan-3-ols reduce neuroinflammation markers."),
    ("trem2", "MODULATES", "nf-kb", 0.75, "demo:010", "Protein", "Protein", "TREM2 modulates NF-kB signalling in microglia."),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="DELETE the whole graph first")
    ap.add_argument("--no-vectors", action="store_true", help="skip ChromaDB / embedding model")
    args = ap.parse_args(argv)

    from backend.core.services import get_services
    s = get_services()
    neo4j = s.neo4j
    if args.reset:
        neo4j.clear_all()
        print("Graph cleared.")
    by_paper: Dict[str, List[Dict]] = {}
    for sub, rel, obj, conf, pid, st, ot, snip in T:
        by_paper.setdefault(pid, []).append({
            "subject": sub, "relation": rel, "object": obj, "confidence": conf,
            "subject_type": st, "object_type": ot, "provenance_snippet": snip})
    for p in PAPERS:
        meta = dict(p, source="demo", url=None)
        neo4j.upsert_paper(meta)
        trips = by_paper.get(p["paper_id"], [])
        neo4j.insert_triplets(trips, p["paper_id"])
        if not args.no_vectors:
            s.vectors.add_chunks(p["paper_id"], [{"index": 0, "section": "abstract", "text": p["abstract"]}], meta)
            s.vectors.add_provenance(p["paper_id"], trips, meta)
        neo4j.mark_paper_processed(p["paper_id"], len(trips), 1)
    s.graph_cache.invalidate()
    print("Demo loaded:", neo4j.stats())
    print("Try:  curl 'http://localhost:8000/api/search?q=chocolate'   or bridge a=curcumin b=alzheimer's disease")


if __name__ == "__main__":
    main()
