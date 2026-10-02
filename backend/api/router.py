from fastapi import APIRouter

from backend.api.endpoints import chatbot, graph, ingest, search

api_router = APIRouter(prefix="/api")
for _m in (search, graph, chatbot, ingest):
    api_router.include_router(_m.router)
