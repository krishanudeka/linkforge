# LinkForge – Graph-RAG & predictive knowledge discovery for scientific literature

Scrape open-access papers → parse PDFs → LLM triplet extraction → entity resolution → Neo4j + ChromaDB →
transitive search, bridge search, grounded chatbot, GNN link prediction. All free-tier / open-source.

## Quick start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt            # add requirements-gnn.txt only for GNN training/inference
cp .env.example .env                       # set NEO4J_* (AuraDB Free or Docker) and GROQ_API_KEY
docker compose up -d                       # only if you use local Neo4j instead of AuraDB
python -m scripts.init_db                  # constraints, indexes, folders
python -m scripts.seed_demo --reset        # synthetic demo graph (no LLM needed)
uvicorn backend.main:app --reload          # http://localhost:8000/docs
```
Try: `GET /api/search?q=chocolate`, `GET /api/search/bridge?a=curcumin&b=alzheimer's disease&max_hops=6`,
`POST /api/chat {"question": "How is curcumin connected to Alzheimer's disease?"}`.

## Real data
```bash
python -m scripts.run_pipeline --query "curcumin alzheimer" --source europe_pmc --limit 10
# or via API: POST /api/ingest/query {"query": "...", "source": "europe_pmc", "limit": 10}  then poll /api/ingest/jobs/{id}
# or upload: POST /api/ingest/upload (multipart PDF)
```
Papers without an open-access PDF fall back to title + abstract. If the LLM is unreachable the paper is left
unprocessed and retried on the next run.

## GNN link prediction (Colab / Kaggle)
```bash
python -m scripts.export_graph                              # -> storage_data/exports/graph_export.json
# on Colab: pip install -r requirements-gnn.txt
python -m ai_engine.gnn.train --source json --graph-json graph_export.json --split temporal --cutoff-year 2022
# copy ai_engine/models/linkforge_gnn.pt back, then POST /api/graph/reload-model
```
Without a trained model the API automatically uses the Adamic–Adar heuristic (`link_prediction.method` in
responses tells you which is active). Predicted links are always flagged `predicted: true`.

## Layout
```
backend/   main.py, api/ (router, endpoints, jobs), core/ (neo4j, vectors, analytics, pipeline, graph_rag, services), tasks/
ai_engine/ extraction/ (LLM + prompts), resolution/ (entity linker), gnn/ (dataset, model, train, predict)
data_pipeline/ scrapers/, parsers/, storage/
scripts/   init_db, run_pipeline, seed_demo, export_graph
tests/     fakes for Neo4j / Chroma / LLM – run with `pytest`
research_paper/  paper skeleton
```

## Tests
`pytest` runs the API, pipeline, Graph-RAG and algorithms against in-memory fakes (no Neo4j, LLM or model download).
**Not covered by automated tests:** real Cypher against Neo4j, Groq/Ollama, scraper network calls, real ChromaDB /
embedding model, PDF parsing on real papers, and the GNN code (torch was not installable where this was built).
Run `seed_demo` and a small `run_pipeline` against your own Neo4j first and expect small fixes.

## Frontend (React + Vite + Tailwind + react-force-graph-3d)
```bash
cd frontend && npm install && npm run dev      # http://localhost:5173 (proxies /api to :8000)
```
Production: deploy `frontend/` to Vercel with `VITE_API_URL=https://<your-render-url>`; backend via `render.yaml` / `Dockerfile`
(set NEO4J_*, GROQ_API_KEY, FRONTEND_URL). Features: search + autocomplete, 3D graph (solid = extracted, dashed = predicted),
click node → focus chat, click line → evidence + DOIs, paper list with highlighted entity chains, Bridge (A→?→D) tab,
Ingest tab (scrape or upload PDF), Graph-RAG chat sidebar. Colab training notebook: `notebooks/train_gnn_colab.ipynb`.
Optional Marker parser: `pip install marker-pdf` and `PDF_PARSER=marker` (untested; falls back to PyMuPDF on failure).

## Security
- Set `ADMIN_API_KEY` (and change the Neo4j password) before any public deployment; ingest, upload, model-reload and
  materialize endpoints then require an `X-API-Key` header. Reads (search/chat) are public but rate-limited per IP.
- Never commit `.env`. Paper text is untrusted input to the LLM (prompt-injection risk is reduced, not eliminated).
- PDFs are parsed with PyMuPDF: only accept uploads from people you trust, and keep dependencies updated.

## Notes
- PyMuPDF is the tested-by-syntax default; Marker is an optional, untested backend.
- `seed_demo` papers are synthetic placeholders, not real publications.
- API shapes: `/api/search` returns `nodes`, `links`
  (`predicted` flag → dashed lines) and `papers` (`snippet_html` with `<mark data-entity>` highlights).
