from scripts import seed_demo


def test_seed_demo_builds_searchable_graph(svc):
    seed_demo.main(["--reset"])
    assert svc.neo4j.stats()["processed_papers"] == 10
    snap = svc.snapshot
    paths = snap.k_shortest_paths("curcumin", "alzheimer's disease", k=2)
    assert paths and paths[0]["nodes"][0] == "curcumin"
    # PPR is a probability distribution
    assert abs(snap.personalized_pagerank({"curcumin": 1.0}).sum() - 1.0) < 1e-6
    # time-machine: edges carry first-publication years on both sides of 2022
    years = {e["year"] for e in snap.edges}
    assert min(years) < 2022 <= max(years)
