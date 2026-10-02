from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.security import rate_limit
from backend.core.graph_rag import GraphRAG
from backend.core.services import get_services

router = APIRouter(prefix="/search", tags=["search"], dependencies=[Depends(rate_limit)])


@router.get("")
def search(q: str = Query(..., min_length=1), top_k: int = Query(20, ge=3, le=50),
           hops: int = Query(3, ge=1, le=5), predictions: bool = True):
    """Transitive search: PPR subgraph around the entity + papers with highlighted chains."""
    return GraphRAG(get_services()).search(q, top_k=top_k, max_hops=hops, with_predictions=predictions)


@router.get("/suggest")
def suggest(q: str = Query(..., min_length=1), limit: int = Query(8, ge=1, le=25)):
    return {"suggestions": get_services().neo4j.search_entities(q, limit)}


@router.get("/bridge")
def bridge(a: str = Query(..., min_length=1), b: str = Query(..., min_length=1),
           k: int = Query(3, ge=1, le=6), max_hops: int = Query(5, ge=1, le=7)):
    """Hypothesis / bridge search: A -> ? -> D (confidence-weighted k-shortest paths)."""
    res = GraphRAG(get_services()).bridge(a, b, k=k, max_hops=max_hops)
    if res.get("missing"):
        raise HTTPException(404, f"Entity not found in graph: {', '.join(res['missing'])}")
    return res
