from typing import Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.api.security import rate_limit
from backend.core.graph_rag import GraphRAG
from backend.core.services import get_services

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(rate_limit)])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1500)
    focus: Optional[List[str]] = Field(None, description="Entities currently selected in the UI")
    history: Optional[List[Dict[str, str]]] = None


@router.post("")
def chat(req: ChatRequest):
    """Graph-RAG answer grounded in Neo4j facts + ChromaDB excerpts, with DOIs."""
    return GraphRAG(get_services()).answer(req.question, req.focus, req.history)
