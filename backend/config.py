"""Central configuration (Pydantic Settings). Values come from `.env` / environment."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Neo4j -----------------------------------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    neo4j_database: str = "neo4j"

    # ---- LLM -------------------------------------------------------------
    llm_provider: str = "groq"            # groq | ollama
    llm_fallback_provider: str = ""       # "" | groq | ollama
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    llm_min_interval_sec: float = 10.0
    llm_max_retries: int = 5
    llm_timeout_sec: float = 120.0

    # ---- PDF parsing -----------------------------------------------------
    pdf_parser: str = "pymupdf"           # pymupdf | marker (marker needs `pip install marker-pdf`; heavy)

    # ---- Embeddings ------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ---- Data acquisition ------------------------------------------------
    contact_email: str = "linkforge@example.com"
    ncbi_api_key: str = ""
    semantic_scholar_api_key: str = ""

    # ---- Extraction / resolution ----------------------------------------
    entity_merge_threshold: float = 0.85
    entity_alpha: float = 0.7
    min_triplet_confidence: float = 0.5
    max_chunks_per_paper: int = 30
    chunk_max_chars: int = 2500
    chunk_overlap_chars: int = 250

    # ---- Graph analytics -------------------------------------------------
    ppr_damping: float = 0.85
    graph_cache_ttl_sec: int = 300
    gnn_model_path: str = str(ROOT_DIR / "ai_engine" / "models" / "linkforge_gnn.pt")

    # ---- Storage ---------------------------------------------------------
    storage_dir: str = str(ROOT_DIR / "storage_data")
    r2_endpoint_url: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""

    # ---- Scheduler -------------------------------------------------------
    auto_scrape_enabled: bool = False
    auto_scrape_queries: str = ""         # ';'-separated
    auto_scrape_source: str = "europe_pmc"
    auto_scrape_limit: int = 10
    auto_scrape_interval_hours: float = 24.0

    # ---- API -------------------------------------------------------------
    frontend_url: str = "http://localhost:5173"
    admin_api_key: str = ""               # REQUIRED in production: protects ingest / model-reload / write endpoints
    rate_limit_per_min: int = 30          # per-IP limit on /api/chat and /api/search (0 = off)

    # ---- derived helpers -------------------------------------------------
    @property
    def pdf_dir(self) -> Path:
        return Path(self.storage_dir) / "pdfs"

    @property
    def markdown_dir(self) -> Path:
        return Path(self.storage_dir) / "parsed_markdown"

    @property
    def chroma_dir(self) -> Path:
        return Path(self.storage_dir) / "chroma_db"

    @property
    def export_dir(self) -> Path:
        return Path(self.storage_dir) / "exports"

    @property
    def scrape_queries(self) -> List[str]:
        return [q.strip() for q in self.auto_scrape_queries.split(";") if q.strip()]

    @property
    def cors_origins(self) -> List[str]:
        origins = {
            self.frontend_url.rstrip("/"),
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
        }
        return sorted(o for o in origins if o)


settings = Settings()
