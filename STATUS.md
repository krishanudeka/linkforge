# LinkForge – build status

## Complete (all spec items)
Backend (API, pipeline, Graph-RAG, scheduler), data layer, AI engine, GNN code, scripts, tests, README, paper skeleton,
frontend (Vite/React/Tailwind/3D graph), Dockerfile, render.yaml, vercel.json, Colab notebook, optional Marker backend.

## Verified
- pytest: 17 tests pass (API, pipeline, Graph-RAG, PageRank, paths) using in-memory fakes.
- Frontend: `npm run build` succeeds; headless-browser smoke test with a mocked API renders search, 3D graph, paper list, chat box with no JS errors.

## NOT verified (needs your environment)
Real Neo4j Cypher, Groq/Ollama, scraper network calls, real ChromaDB + embedding model, PDF parsing on real papers,
GNN training/inference (torch), Marker, Colab notebook, Render/Vercel deploy.
