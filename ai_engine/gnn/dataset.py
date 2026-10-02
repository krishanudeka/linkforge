"""Graph -> tensors for link prediction.

Works from three sources:
  * a live Neo4jClient (`load_graph_from_neo4j`)
  * an exported JSON file (`export_graph_json` / `load_graph_json`) – use this on Colab/Kaggle where
    your local Neo4j isn't reachable
  * plain python lists (tests)

Node features = [ sentence-embedding of the entity name | one-hot entity type ].
Because features do not depend on the node *index*, a trained model is inductive: it can score
entities that were added to the graph after training.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ai_engine.extraction.prompts import ENTITY_TYPES

logger = logging.getLogger("GNNDataset")

TYPE_TO_IDX = {t: i for i, t in enumerate(ENTITY_TYPES)}


# ============================================================================ graph container
@dataclass
class GraphData:
    names: List[str]
    types: List[str]
    pairs: np.ndarray        # (E, 2) int64, unique undirected pairs with u < v
    years: np.ndarray        # (E,) float64, NaN when unknown (= first year the link was published)

    @property
    def n_nodes(self) -> int:
        return len(self.names)

    @property
    def n_edges(self) -> int:
        return int(self.pairs.shape[0])


def build_graph_data(entities: Sequence[Dict], edges: Sequence[Dict]) -> GraphData:
    names, types, index = [], [], {}
    for e in entities:
        if e["name"] not in index:
            index[e["name"]] = len(names)
            names.append(e["name"])
            types.append(e.get("type", "Other"))
    pair_year: Dict[Tuple[int, int], float] = {}
    for ed in edges:
        for n in (ed["source"], ed["target"]):
            if n not in index:
                index[n] = len(names); names.append(n); types.append("Other")
        u, v = index[ed["source"]], index[ed["target"]]
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        y = ed.get("year")
        y = float(y) if y is not None else np.nan
        if key not in pair_year or (not np.isnan(y) and (np.isnan(pair_year[key]) or y < pair_year[key])):
            pair_year[key] = y
    keys = sorted(pair_year)
    pairs = np.array(keys, dtype=np.int64).reshape(-1, 2)
    years = np.array([pair_year[k] for k in keys], dtype=np.float64)
    return GraphData(names, types, pairs, years)


def load_graph_from_neo4j(neo4j) -> GraphData:
    return build_graph_data(neo4j.fetch_entities(), neo4j.fetch_edges())


def export_graph_json(graph: GraphData, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "names": graph.names, "types": graph.types, "pairs": graph.pairs.tolist(),
        "years": [None if np.isnan(y) else int(y) for y in graph.years],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_graph_json(path: str | Path) -> GraphData:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return GraphData(
        names=d["names"], types=d["types"],
        pairs=np.array(d["pairs"], dtype=np.int64).reshape(-1, 2),
        years=np.array([np.nan if y is None else y for y in d["years"]], dtype=np.float64),
    )


# ============================================================================ features
def build_node_features(names: Sequence[str], types: Sequence[str], embedder,
                        cache: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """(N, emb_dim + n_types) float32. `cache` (name -> embedding) avoids re-embedding known names."""
    cache = cache if cache is not None else {}
    missing = [n for n in dict.fromkeys(names) if n not in cache]
    if missing:
        for n, v in zip(missing, embedder.embed_documents(missing)):
            cache[n] = v
    emb = np.stack([cache[n] for n in names]).astype(np.float32) if len(names) else np.zeros((0, 0), np.float32)
    onehot = np.zeros((len(names), len(ENTITY_TYPES)), dtype=np.float32)
    for i, t in enumerate(types):
        onehot[i, TYPE_TO_IDX.get(t, TYPE_TO_IDX["Other"])] = 1.0
    return np.concatenate([emb, onehot], axis=1)


def undirected_edge_index(pairs: np.ndarray) -> np.ndarray:
    """(E,2) -> (2, 2E) with both directions, as message passing expects."""
    if pairs.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.concatenate([pairs.T, pairs[:, ::-1].T], axis=1).astype(np.int64)


# ============================================================================ splits & negatives
def sample_negatives(n_nodes: int, forbidden: Set[Tuple[int, int]], k: int,
                     rng: np.random.Generator, max_tries: int = 50) -> np.ndarray:
    """k random node pairs that are neither self-loops nor in `forbidden` (keys are (min,max))."""
    out: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    tries = 0
    while len(out) < k and tries < max_tries:
        need = (k - len(out)) * 2 + 16
        us = rng.integers(0, n_nodes, size=need)
        vs = rng.integers(0, n_nodes, size=need)
        for u, v in zip(us, vs):
            if u == v:
                continue
            key = (int(min(u, v)), int(max(u, v)))
            if key in forbidden or key in seen:
                continue
            seen.add(key); out.append(key)
            if len(out) >= k:
                break
        tries += 1
    return np.array(out, dtype=np.int64).reshape(-1, 2)


@dataclass
class EdgeSplit:
    train_pos: np.ndarray
    val_pos: np.ndarray
    test_pos: np.ndarray
    val_neg: np.ndarray
    test_neg: np.ndarray
    all_pairs: Set[Tuple[int, int]]
    mode: str
    cutoff_year: Optional[int] = None


def split_edges(graph: GraphData, mode: str = "random", cutoff_year: Optional[int] = None,
                val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> EdgeSplit:
    """
    random   : shuffle all edges, hold out val/test fractions.
    temporal : "time-machine" back-test. Edges first published <= cutoff_year form the known graph
               (minus a random validation hold-out); edges first published AFTER the cutoff are the
               test positives (restricted to nodes that already had >=1 link before the cutoff).
    """
    if graph.n_edges < 10:
        raise ValueError(f"Graph has only {graph.n_edges} edges – ingest more papers before training a GNN.")
    rng = np.random.default_rng(seed)
    pairs = graph.pairs
    all_pairs = {(int(u), int(v)) for u, v in pairs}

    if mode == "random":
        perm = rng.permutation(graph.n_edges)
        n_test = max(1, int(round(test_ratio * graph.n_edges)))
        n_val = max(1, int(round(val_ratio * graph.n_edges)))
        test_pos = pairs[perm[:n_test]]
        val_pos = pairs[perm[n_test:n_test + n_val]]
        train_pos = pairs[perm[n_test + n_val:]]
    elif mode == "temporal":
        if cutoff_year is None:
            raise ValueError("temporal split requires cutoff_year")
        is_new = graph.years > cutoff_year           # NaN > x is False -> unknown years count as old
        old, new = pairs[~is_new], pairs[is_new]
        if len(old) < 10 or len(new) < 1:
            raise ValueError(
                f"Cutoff {cutoff_year} leaves {len(old)} old / {len(new)} new edges – choose another cutoff.")
        deg = np.zeros(graph.n_nodes, dtype=np.int64)
        np.add.at(deg, old[:, 0], 1); np.add.at(deg, old[:, 1], 1)
        keep = (deg[new[:, 0]] > 0) & (deg[new[:, 1]] > 0)
        test_pos = new[keep]
        if len(test_pos) == 0:
            raise ValueError("No post-cutoff edges connect nodes that already existed before the cutoff.")
        perm = rng.permutation(len(old))
        n_val = max(1, int(round(val_ratio * len(old))))
        val_pos, train_pos = old[perm[:n_val]], old[perm[n_val:]]
    else:
        raise ValueError(f"unknown split mode '{mode}'")

    val_neg = sample_negatives(graph.n_nodes, all_pairs, len(val_pos), rng)
    test_neg = sample_negatives(graph.n_nodes, all_pairs, len(test_pos), rng)
    return EdgeSplit(train_pos, val_pos, test_pos, val_neg, test_neg, all_pairs, mode, cutoff_year)
