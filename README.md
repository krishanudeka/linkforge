# LinkForge — Graph-RAG & Predictive Knowledge Discovery for Scientific Literature

LinkForge is an end-to-end scientific literature discovery platform that transforms research papers into an interactive knowledge graph and combines **LLM-based knowledge extraction, Neo4j graph reasoning, ChromaDB semantic retrieval, link prediction, and Graph-RAG**.

Instead of simply returning relevant papers, LinkForge helps researchers explore:

- What scientific entities are connected?
- How are two concepts connected through multiple intermediate concepts?
- Which papers provide evidence for those connections?
- Which potentially missing relationships are worth investigating?

---

## Live Demo

- **Frontend:** [Open LinkForge](https://linkforge-silk.vercel.app)
- **Backend API:** [LinkForge API](https://linkforge-api-yil7.onrender.com)
- **API Documentation:** [Swagger / OpenAPI Docs](https://linkforge-api-yil7.onrender.com/docs)
- **Source Code:** [GitHub Repository](https://github.com/krishanudeka/linkforge)

---

## How LinkForge Works

```text
                       Scientific Papers
                              │
                              ▼
                  Europe PMC / Paper APIs
                              │
                              ▼
                    PDF / Abstract Parsing
                              │
                              ▼
                    Text Cleaning + Chunking
                              │
                              ▼
                         Groq LLM
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       Knowledge Triplets               Text Chunks
               │                             │
               ▼                             ▼
            Neo4j                        ChromaDB
        Knowledge Graph                Vector Store
               │                             │
               └──────────────┬──────────────┘
                              ▼
                           Graph-RAG
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
               Search      Bridge        Chatbot
                 │            │            │
                 └────────────┼────────────┘
                              ▼
                     React 3D Frontend
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                  Vercel              Browser

                         Backend
                            │
                         FastAPI
                            │
                          Render
```

---

## Key Features

### Scientific Literature Acquisition

- Europe PMC as the primary literature source
- PubMed and Semantic Scholar scraper modules
- Open-access paper retrieval
- PDF and metadata acquisition
- Title + abstract fallback when a usable PDF is unavailable

### Scientific Text Processing

- PDF parsing using PyMuPDF
- Optional Marker parser
- Text cleaning
- Sentence-aware chunking
- Configurable chunk size, overlap and maximum chunks per paper

### LLM Knowledge Extraction

- Groq LLM integration
- Subject–Relation–Object triplet extraction
- Controlled relation vocabulary
- Entity type extraction
- Confidence score extraction
- Verbatim provenance/evidence snippets
- DOI and paper-source association
- Extraction validation and duplicate removal

### Entity Resolution

LinkForge canonicalizes entities that appear under different names.

```text
diferuloylmethane → curcumin
NF-κB → nuclear factor kappa b
Aβ → amyloid beta
Alzheimer disease → Alzheimer's disease
```

Entity resolution combines:

- Normalization
- Curated synonym mappings
- Embedding similarity
- Jaro-Winkler similarity
- False-merge guards

---

## Knowledge Graph — Neo4j

Neo4j stores the structured scientific knowledge extracted from papers.

```text
(:Entity)-[:RELATION]->(:Entity)

(:Entity)-[:MENTIONED_IN]->(:Paper)
```

Example:

```text
Curcumin
   │
   ├── TREATS ─────────────► Alzheimer's disease
   │
   ├── INHIBITS ───────────► NF-kB
   │
   └── ASSOCIATED_WITH ────► Neuroinflammation
```

Each extracted relationship can retain:

- confidence
- supporting papers
- provenance
- source/DOI information

Every relationship originates from an LLM-extracted and validated subject–relation–object triplet.

---

## Graph Retrieval & Reasoning

### Personalized PageRank

Personalized PageRank ranks graph nodes according to their relevance to the user's query:

$$
p = (1-d)e + dMp
$$

where the transition matrix is weighted using graph relationship strength. It is used to identify the most relevant portion of the graph around a query entity.

```text
Query: curcumin

curcumin
   │
   ├── Alzheimer's disease
   ├── neurodegeneration
   ├── NF-kB
   ├── oxidative stress
   └── microglial activation
```

### Bridge Search

Bridge Search discovers paths connecting two scientific concepts:

```text
A → ? → ? → D
```

The graph uses confidence/support-derived edge costs and performs confidence-weighted k-shortest path search. Each result includes:

- path
- number of hops
- path score
- path cost
- evidence for each relationship

---

## Link Prediction

LinkForge distinguishes between **observed scientific relationships** and **predicted relationships**.

### Current deployed method: Adamic–Adar

When a trained GNN model is unavailable, LinkForge uses the Adamic–Adar heuristic:

$$
AA(x,y)=\sum_{z\in\Gamma(x)\cap\Gamma(y)}\frac{1}{\ln(\deg(z)+1)}
$$

It looks at common neighbours between two entities and gives greater weight to less-common neighbours.

Predicted edges are marked:

```json
{
  "predicted": true,
  "method": "adamic_adar"
}
```

The frontend displays these edges differently from extracted relationships.

> **Predicted edges are hypotheses, not established scientific facts.**

### Optional GraphSAGE Pipeline

The repository also contains an optional Graph Neural Network pipeline based on GraphSAGE / PyTorch Geometric. It supports:

- Graph export
- Dataset preparation
- Temporal splitting
- GraphSAGE training
- Link prediction
- Model checkpoint loading
- Runtime prediction

The currently deployed system uses **Adamic–Adar when a trained GNN checkpoint is unavailable**.

---

## Graph-RAG

LinkForge combines graph retrieval with semantic retrieval.

```text
                       User Question
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
              Neo4j                    ChromaDB
           Graph Context             Text Evidence
                 │                         │
                 └────────────┬────────────┘
                              ▼
                       Context Fusion
                              │
                              ▼
                           Groq LLM
                              │
                              ▼
                    Evidence-Grounded Answer
```

The chatbot can use:

- graph facts
- multi-hop paths
- relevant paper excerpts
- evidence snippets
- DOI/source information
- predicted links, explicitly labelled as unverified

The goal is to avoid answers based solely on the LLM's internal knowledge.

---

## ChromaDB

Neo4j answers: **what is connected to what?**
ChromaDB answers: **what does the source text actually say?**

Paper chunks and provenance snippets are embedded using `BAAI/bge-small-en-v1.5` and stored in ChromaDB, enabling:

- semantic search
- evidence retrieval
- provenance retrieval
- Graph-RAG context construction

```text
Neo4j     = structured graph knowledge
ChromaDB  = semantic text/evidence retrieval
```

---

## Chatbot

Example question:

```text
How is curcumin connected to Alzheimer's disease?
```

LinkForge retrieves graph facts, relevant graph paths, paper excerpts and evidence snippets, and provides this context to the LLM. The answer is expected to cite the retrieved facts and source papers rather than inventing unsupported mechanisms.

---

## Frontend

Built with React, Vite, Tailwind CSS, react-force-graph-3d and Three.js.

**Features**

- Search bar with autocomplete
- Interactive 3D knowledge graph
- Entity and relationship exploration
- Real vs predicted edge visualization
- Paper/result display
- Evidence snippets
- DOI/source links
- Bridge Search
- Ingestion interface
- PDF upload
- Graph-RAG chatbot

**Graph visualization**

```text
Solid edges   → extracted from scientific papers
Dashed edges  → predicted / unverified
```

---

## Backend

Built with **FastAPI**. Main endpoints:

```text
/api/search
/api/search/bridge
/api/graph
/api/chat
/api/ingest/query
/api/ingest/upload
/api/ingest/jobs/{id}
/api/graph/reload-model
```

The backend orchestrates:

```text
React
  ↓
FastAPI
  ↓
Neo4j · ChromaDB · Groq · Embedding Model · Graph Algorithms · Link Prediction
```

---

## Local Setup

### Requirements

- Python 3.11
- Node.js and npm
- Neo4j Aura or a local Neo4j instance
- Groq API key

### 1. Clone the repository

```bash
git clone https://github.com/krishanudeka/linkforge.git
cd linkforge
```

### 2. Create the Python environment

**Windows PowerShell**

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Development/testing:

```bash
python -m pip install -r requirements-dev.txt
```

Optional GNN training:

```bash
python -m pip install -r requirements-gnn.txt
```

### 4. Configure `.env`

**Windows**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS**

```bash
cp .env.example .env
```

Then fill in:

```env
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_USER=<your-user>
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=<your-database>

GROQ_API_KEY=<your-key>
GROQ_MODEL=openai/gpt-oss-20b
LLM_PROVIDER=groq

FRONTEND_URL=http://localhost:5173

ADMIN_API_KEY=<strong-random-key>
```

> Never commit `.env`.

### 5. Initialize Neo4j

```bash
python -m scripts.init_db
```

This creates the required constraints and indexes.

Optional synthetic demo data (not real publications):

```bash
python -m scripts.seed_demo --reset
```

### 6. Start the backend

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

- Backend: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### 7. Start the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173 (the Vite dev server proxies `/api` to the FastAPI backend).

---

## Ingesting Real Papers

### Command line

```bash
python -m scripts.run_pipeline \
  --query "curcumin alzheimer" \
  --source europe_pmc \
  --limit 10
```

### API

```http
POST /api/ingest/query
```

```json
{
  "query": "curcumin alzheimer",
  "source": "europe_pmc",
  "limit": 10
}
```

The API returns a job ID, which can be polled with:

```http
GET /api/ingest/jobs/{id}
```

### PDF upload

```http
POST /api/ingest/upload
```

When a usable open-access PDF is unavailable, LinkForge falls back to the paper title and abstract.

---

## Testing

```bash
python -m pytest -q
```

Current status: **28 / 28 tests passing**.

The suite covers API behaviour, pipeline logic, Graph-RAG behaviour, graph algorithms, provenance handling, and entity/relationship processing. Live Groq requests, real Neo4j instances and live paper APIs are not required.

---

## Docker

Build and run the backend:

```bash
docker build -t linkforge .
docker run -p 8000:8000 --env-file .env linkforge
```

To use a local Neo4j instead of Aura:

```bash
docker compose up -d
```

---

## Deployment

### Frontend — Vercel

Production: [https://linkforge-silk.vercel.app](https://linkforge-silk.vercel.app)

```env
VITE_API_URL=https://linkforge-api-yil7.onrender.com
```

### Backend — Render

Deployed using `Dockerfile` and `render.yaml`.

Production: [https://linkforge-api-yil7.onrender.com](https://linkforge-api-yil7.onrender.com) · [Docs](https://linkforge-api-yil7.onrender.com/docs)

Required environment variables:

```text
NEO4J_URI
NEO4J_USER
NEO4J_PASSWORD
NEO4J_DATABASE
GROQ_API_KEY
LLM_PROVIDER
FRONTEND_URL
ADMIN_API_KEY
CONTACT_EMAIL
```

> Render's free instances may sleep after inactivity, so the first request can have a cold-start delay.

### Database — Neo4j Aura

The deployed system uses **Neo4j Aura** to host the persistent knowledge graph (entities, papers and relationships).

---

## Security

- Never commit `.env`
- Never expose Neo4j credentials or Groq API keys
- Set `ADMIN_API_KEY` before public deployment
- Ingestion and other write operations require admin authorization when `ADMIN_API_KEY` is configured
- Search and chat are rate-limited
- Scientific paper text is treated as untrusted LLM input
- Only accept PDF uploads from trusted sources
- Keep dependencies updated

---

## Project Structure

```text
linkforge/
│
├── backend/
│   ├── api/
│   │   ├── endpoints/
│   │   ├── jobs.py
│   │   ├── router.py
│   │   └── security.py
│   │
│   ├── core/
│   │   ├── neo4j_client.py
│   │   ├── vector_client.py
│   │   ├── embeddings.py
│   │   ├── graph_analytics.py
│   │   ├── pipeline.py
│   │   ├── graph_rag.py
│   │   └── services.py
│   │
│   └── main.py
│
├── ai_engine/
│   ├── extraction/
│   │   ├── llm_extractor.py
│   │   └── prompts.py
│   │
│   ├── resolution/
│   │   └── entity_linker.py
│   │
│   └── gnn/
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       └── predict.py
│
├── data_pipeline/
│   ├── scrapers/
│   ├── parsers/
│   └── storage/
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/
│
├── scripts/
│   ├── init_db.py
│   ├── run_pipeline.py
│   ├── seed_demo.py
│   └── export_graph.py
│
├── tests/
│
├── research_paper/
│
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── requirements.txt
├── requirements-dev.txt
└── requirements-gnn.txt
```

---

## 📊 Example Queries

| Type | Example |
|---|---|
| Search | `curcumin` |
| Scientific concept search | `microglial activation` |
| Bridge Search | A = `curcumin`, B = `Alzheimer's disease` |
| Graph-RAG question | `How is curcumin connected to Alzheimer's disease?` |

---

## Example End-to-End Flow

For a search such as `curcumin`:

```text
1. Resolve "curcumin"
           ↓
2. Retrieve the relevant graph neighbourhood
           ↓
3. Personalized PageRank ranks connected nodes
           ↓
4. Retrieve supporting papers
           ↓
5. ChromaDB retrieves relevant evidence
           ↓
6. Adamic–Adar identifies candidate missing links
           ↓
7. Graph + evidence are combined
           ↓
8. React renders the 3D knowledge graph
```

For a chatbot question, the system combines Neo4j graph facts, graph paths, ChromaDB evidence and source/DOI metadata before generating the final answer.

---


## Future Work

- Train and formally evaluate GraphSAGE/GAT link prediction
- Expand the scientific corpus
- Evaluate extraction precision, recall and F1
- Evaluate Graph-RAG groundedness and citation accuracy
- Compare Adamic–Adar against GNN-based prediction
- Improve relation coverage and normalization
- Add additional scientific literature sources
- Improve large-scale vector storage and retrieval
- Expand UI/UX and scientific-domain support

---

## Project Summary

LinkForge combines scientific literature, LLM information extraction, entity resolution, knowledge graphs, vector search, graph algorithms, link prediction, Graph-RAG and interactive 3D visualization to transform large collections of papers into a **queryable, evidence-grounded knowledge discovery system**.

Developed as an academic research/project implementation focused on scientific knowledge discovery, graph reasoning, predictive link discovery and evidence-grounded question answering.
