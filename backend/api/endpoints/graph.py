from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.security import rate_limit, require_admin
from backend.core.graph_rag import GraphRAG
from backend.core.services import get_services

router = APIRouter(prefix="/graph", tags=["graph"], dependencies=[Depends(rate_limit)])


@router.get("/stats")
def stats():
    s = get_services()
    out = {"graph": s.neo4j.stats()}
    try:
        out["vectors"] = s.vectors.counts()
    except Exception as exc:  # noqa: BLE001
        out["vectors"] = {"error": str(exc)}
    try:
        out["link_prediction"] = s.link_predictor.status()
    except Exception as exc:  # noqa: BLE001
        out["link_prediction"] = {"method": "none", "reason": str(exc)}
    return out


@router.get("/entity/{name}")
def entity(name: str):
    s = get_services()
    rag = GraphRAG(s)
    canon = rag.resolve(name)
    if not canon:
        raise HTTPException(404, "Entity not found")
    snap = s.snapshot
    i = snap.index[canon]
    nbrs = []
    for j in snap.adj[i]:
        e = snap.best_edge(i, j)
        nbrs.append({"name": snap.names[j], "type": snap.types[j], "relation": e["type"],
                     "confidence": e["confidence"], "support": e["support"]})
    nbrs.sort(key=lambda n: -(n["confidence"] * n["support"]))
    aliases = next((e.get("aliases", []) for e in s.neo4j.fetch_entities() if e["name"] == canon), [])
    return {"name": canon, "type": snap.types[i], "aliases": aliases, "degree": len(nbrs),
            "neighbors": nbrs[:50]}


@router.get("/edge")
def edge(source: str, target: str):
    """Evidence for a link: relation records, papers (DOIs) and supporting sentences."""
    s = get_services()
    rows = s.neo4j.get_edge_papers(source, target)
    if not rows:
        raise HTTPException(404, "No such edge")
    pids = {p for r in rows for p in (r.get("papers") or [])}
    return {"relations": rows, "papers": s.neo4j.get_papers(pids),
            "evidence": s.vectors.get_edge_provenance(source, target, limit=6)}


@router.get("/predict")
def predict(name: str, top_k: int = Query(10, ge=1, le=50), min_score: float = Query(0.0, ge=0, le=1)):
    s = get_services()
    canon = GraphRAG(s).resolve(name)
    if not canon:
        raise HTTPException(404, "Entity not found")
    return {"entity": canon, "status": s.link_predictor.status(),
            "predictions": s.link_predictor.predict_for_entity(s.snapshot, canon, top_k, min_score)}


@router.post("/predictions/materialize", dependencies=[Depends(require_admin)])
def materialize(top_k_per_node: int = 5, min_score: float = 0.8):
    """Write strong predictions into Neo4j as :PREDICTED edges (for Neo4j Browser exploration)."""
    from ai_engine.gnn.predict import write_predictions_to_neo4j
    s = get_services()
    n = write_predictions_to_neo4j(s.neo4j, s.link_predictor, s.snapshot, top_k_per_node, min_score)
    return {"written": n}


@router.post("/reload-model", dependencies=[Depends(require_admin)])
def reload_model():
    s = get_services()
    s.link_predictor.reload()
    return s.link_predictor.status()
