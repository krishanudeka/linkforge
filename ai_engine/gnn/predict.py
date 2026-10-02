"""Link-prediction inference used by the API.

`LinkPredictionService` uses the trained GNN checkpoint when available (and torch is installed),
otherwise it transparently falls back to an Adamic–Adar heuristic. Every prediction carries a
`method` field ("gnn" | "adamic_adar") so the UI / chatbot can label it honestly.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ai_engine.gnn.dataset import build_node_features, undirected_edge_index
from backend.config import settings
from backend.core.graph_analytics import GraphSnapshot

logger = logging.getLogger("LinkPrediction")

try:  # torch is heavy/optional – the heuristic path must work without it
    import torch
    from ai_engine.gnn.model import LinkPredictionModel
    _TORCH_OK = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore
    _TORCH_OK = False


class LinkPredictionService:
    def __init__(self, embedder, model_path: Optional[str] = None):
        self.embedder = embedder
        self.model_path = Path(model_path or settings.gnn_model_path)
        self._model = None
        self._meta: Dict[str, Any] = {}
        self._reason = "not loaded"
        self._emb_cache: Dict[str, np.ndarray] = {}
        self._z = None
        self._z_key: Optional[tuple] = None
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if not _TORCH_OK:
            self._reason = "torch / torch-geometric not installed (pip install -r requirements-gnn.txt)"
            return
        if not self.model_path.exists():
            self._reason = f"no trained model at {self.model_path} (run: python -m ai_engine.gnn.train)"
            return
        try:
            try:
                ckpt = torch.load(self.model_path, map_location="cpu", weights_only=True)
            except TypeError:  # very old torch
                ckpt = torch.load(self.model_path, map_location="cpu")
            model = LinkPredictionModel.from_config(ckpt["config"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            if ckpt.get("embedding_model") != settings.embedding_model:
                self._reason = (f"model trained with embeddings '{ckpt.get('embedding_model')}' but "
                                f"EMBEDDING_MODEL='{settings.embedding_model}' – retrain or fix .env")
                return
            self._model = model
            self._meta = {k: ckpt.get(k) for k in ("metrics", "trained_at", "split", "graph", "config")}
            self._reason = "ok"
            logger.info("Loaded GNN link predictor from %s", self.model_path)
        except Exception as exc:  # noqa: BLE001
            self._reason = f"failed to load checkpoint: {exc}"
            logger.warning(self._reason)

    def reload(self) -> None:
        self._model, self._z, self._z_key = None, None, None
        self._load()

    @property
    def uses_gnn(self) -> bool:
        return self._model is not None

    def status(self) -> Dict[str, Any]:
        return {"method": "gnn" if self.uses_gnn else "adamic_adar", "gnn_available": self.uses_gnn,
                "reason": self._reason, "model": self._meta if self.uses_gnn else None}

    # ------------------------------------------------------------------ embeddings
    def _embeddings(self, snap: GraphSnapshot):
        key = (id(snap), snap.version, snap.n_nodes)
        with self._lock:
            if self._z is not None and self._z_key == key:
                return self._z
            x = build_node_features(snap.names, snap.types, self.embedder, self._emb_cache)
            pairs = np.array(list(snap.pair_edges.keys()), dtype=np.int64).reshape(-1, 2)
            ei = torch.as_tensor(undirected_edge_index(pairs), dtype=torch.long)
            with torch.no_grad():
                self._z = self._model.encode(torch.as_tensor(x, dtype=torch.float32), ei)
            self._z_key = key
            return self._z

    # ------------------------------------------------------------------ public API
    def predict_for_entity(self, snap: GraphSnapshot, name: str, top_k: int = 10,
                           min_score: float = 0.0) -> List[Dict[str, Any]]:
        if name not in snap.index:
            raise KeyError(name)
        if not self.uses_gnn:
            return [
                {"source": name, "target": t, "score": round(s, 4), "method": "adamic_adar", "predicted": True}
                for t, s in snap.adamic_adar_scores(name, top_k=top_k) if s >= min_score
            ]
        z = self._embeddings(snap)
        i = snap.index[name]
        with torch.no_grad():
            probs = torch.sigmoid(self._model.decoder.score_one_to_many(z, i)).numpy()
        probs[i] = -1.0
        for j in snap.adj[i]:          # exclude already-known neighbours
            probs[j] = -1.0
        order = np.argsort(-probs)[: top_k * 2]
        out = []
        for j in order:
            if probs[j] < max(min_score, 0.0):
                break
            out.append({"source": name, "target": snap.names[int(j)], "score": round(float(probs[j]), 4),
                        "method": "gnn", "predicted": True})
            if len(out) >= top_k:
                break
        return out

    def score_pair(self, snap: GraphSnapshot, a: str, b: str) -> Dict[str, Any]:
        if a not in snap.index or b not in snap.index:
            raise KeyError("entity not in graph")
        i, j = snap.index[a], snap.index[b]
        known = j in snap.adj[i]
        if self.uses_gnn:
            z = self._embeddings(snap)
            with torch.no_grad():
                score = float(torch.sigmoid(self._model.decode(z, torch.tensor([[i, j]]))).item())
            method = "gnn"
        else:
            hits = dict(snap.adamic_adar_scores(a, top_k=snap.n_nodes))
            score, method = float(hits.get(b, 0.0)), "adamic_adar"
        return {"source": a, "target": b, "score": round(score, 4), "method": method,
                "already_linked": known, "predicted": not known}

    def predict_between(self, snap: GraphSnapshot, names: List[str], top_k: int = 8,
                        min_score: float = 0.5) -> List[Dict[str, Any]]:
        """Most likely *missing* links among a set of displayed nodes (for dashed edges in the UI)."""
        ids = [snap.index[n] for n in names if n in snap.index]
        cands: Dict[tuple, float] = {}
        if self.uses_gnn:
            z = self._embeddings(snap)
            with torch.no_grad():
                sub = z[torch.tensor(ids)]
                P = torch.sigmoid(sub @ self._model.decoder.sym_w() @ sub.t()).numpy()
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    if ids[b] not in snap.adj[ids[a]]:
                        cands[(ids[a], ids[b])] = float(P[a, b])
            method = "gnn"
        else:
            idset = set(ids)
            for i in ids:
                for t, s in snap.adamic_adar_scores(snap.names[i], top_k=snap.n_nodes):
                    j = snap.index[t]
                    if j in idset and i < j:
                        cands[(i, j)] = max(cands.get((i, j), 0.0), s)
            method = "adamic_adar"
        ranked = sorted(cands.items(), key=lambda kv: -kv[1])
        return [{"source": snap.names[a], "target": snap.names[b], "score": round(s, 4),
                 "method": method, "predicted": True}
                for (a, b), s in ranked if s >= min_score][:top_k]


def write_predictions_to_neo4j(neo4j, service: LinkPredictionService, snap: GraphSnapshot,
                               top_k_per_node: int = 5, min_score: float = 0.8, max_nodes: int = 2000) -> int:
    """Materialise high-confidence predictions as (:Entity)-[:PREDICTED {score, method}]->(:Entity)."""
    neo4j._write("MATCH ()-[r:PREDICTED]->() DELETE r")
    rows = []
    hubs = sorted(range(snap.n_nodes), key=lambda i: -len(snap.adj[i]))[:max_nodes]
    for i in hubs:
        for p in service.predict_for_entity(snap, snap.names[i], top_k=top_k_per_node, min_score=min_score):
            rows.append({"a": p["source"], "b": p["target"], "score": p["score"], "method": p["method"]})
    for k in range(0, len(rows), 500):
        neo4j._write(
            """
            UNWIND $rows AS row
            MATCH (a:Entity {name: row.a}), (b:Entity {name: row.b})
            MERGE (a)-[r:PREDICTED]->(b) SET r.score = row.score, r.method = row.method
            """,
            rows=rows[k:k + 500],
        )
    return len(rows)
