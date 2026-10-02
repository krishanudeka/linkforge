"""Single shared, lazily-loaded sentence-embedding model (local, free).

Everything that needs vectors (ChromaDB, entity resolution, GNN node features)
goes through this class so the model is loaded once per process.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Sequence

import numpy as np

from backend.config import settings

logger = logging.getLogger("Embedder")

# BGE models are trained so that *queries* carry this instruction prefix, documents do not.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._lock = threading.Lock()

    # -- lazy load ----------------------------------------------------------
    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s ...", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    # -- API ---------------------------------------------------------------
    def embed_documents(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """L2-normalised float32 matrix (n, dim). Cosine similarity == dot product."""
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        vecs = self._load().encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self.embed_documents([prefix + t for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self.embed_queries([text])[0].tolist()
