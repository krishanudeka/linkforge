"""In-memory graph algorithms over a snapshot of the Neo4j graph.

* Personalized PageRank (power iteration on a sparse column-stochastic matrix)
    p = (1 - d) e_A + d M p
* Hop-limited, connectivity-preserving local sub-graph extraction ranked by PPR
* Confidence-weighted k-shortest paths (Dijkstra / Yen via networkx)
    cost(e) = -log(conf_eff(e)) + hop_penalty,   conf_eff = 1 - (1 - conf) / (1 + ln(support))
  i.e. the design's  -log(mean confidence)  cost, where repeated support across papers makes an edge
  cheaper, plus a small per-hop penalty so zero-cost (conf = 1.0) edges can't create endless chains.

The snapshot is cached and rebuilt when the Neo4j write-version changes or the TTL expires.
"""
from __future__ import annotations

import heapq
import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import sparse

from backend.config import settings

logger = logging.getLogger("GraphAnalytics")

HOP_PENALTY = 0.05


def edge_strength(conf: float, support: int) -> float:
    """Relative reliability used for PPR transition weights and to pick the 'best' parallel edge."""
    return max(conf, 1e-3) * (1.0 + math.log(max(support, 1)))


def edge_cost(conf: float, support: int) -> float:
    conf = min(max(float(conf), 1e-3), 0.999)
    conf_eff = 1.0 - (1.0 - conf) / (1.0 + math.log(max(int(support), 1)))
    return -math.log(conf_eff) + HOP_PENALTY


@dataclass
class GraphSnapshot:
    names: List[str]
    types: List[str]
    index: Dict[str, int]
    edges: List[Dict[str, Any]]                           # raw directed relation edges
    alias_index: Dict[str, str]                           # lowercase alias/name -> canonical name
    adj: List[Dict[int, float]] = field(default_factory=list)            # undirected strength
    pair_edges: Dict[Tuple[int, int], List[Dict[str, Any]]] = field(default_factory=dict)
    A: Optional[sparse.csr_matrix] = None
    version: int = 0
    built_at: float = field(default_factory=time.time)

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, entities: Sequence[Dict[str, Any]], edges: Sequence[Dict[str, Any]],
              version: int = 0) -> "GraphSnapshot":
        names: List[str] = []
        types: List[str] = []
        index: Dict[str, int] = {}
        alias_index: Dict[str, str] = {}

        def add(name: str, etype: str = "Other") -> int:
            if name not in index:
                index[name] = len(names)
                names.append(name)
                types.append(etype or "Other")
            return index[name]

        for e in entities:
            add(e["name"], e.get("type", "Other"))
            alias_index[e["name"].lower()] = e["name"]
            for a in e.get("aliases", []) or []:
                alias_index.setdefault(a.lower(), e["name"])
        for ed in edges:                          # tolerate edges whose nodes weren't listed
            add(ed["source"]); add(ed["target"])

        n = len(names)
        adj: List[Dict[int, float]] = [dict() for _ in range(n)]
        pair_edges: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        rows, cols, vals = [], [], []
        for ed in edges:
            u, v = index[ed["source"]], index[ed["target"]]
            if u == v:
                continue
            conf = float(ed.get("confidence") or 0.5)
            sup = int(ed.get("support") or 1)
            pair_edges[(min(u, v), max(u, v))].append({
                "source": ed["source"], "target": ed["target"], "type": ed["type"],
                "confidence": round(conf, 4), "support": sup, "year": ed.get("year"),
                "papers": list(ed.get("papers") or []),
            })
            w = edge_strength(conf, sup)
            for a, b in ((u, v), (v, u)):
                adj[a][b] = adj[a].get(b, 0.0) + w
        for u in range(n):
            for v, w in adj[u].items():
                rows.append(u); cols.append(v); vals.append(w)
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
        return cls(names=names, types=types, index=index, edges=list(edges), alias_index=alias_index,
                   adj=adj, pair_edges=dict(pair_edges), A=A, version=version)

    # ---------------------------------------------------------------- basic info
    @property
    def n_nodes(self) -> int:
        return len(self.names)

    def degree(self, name: str) -> int:
        i = self.index.get(name)
        return len(self.adj[i]) if i is not None else 0

    def neighbors(self, name: str) -> List[str]:
        i = self.index.get(name)
        return [self.names[j] for j in self.adj[i]] if i is not None else []

    def best_edge(self, u: int, v: int) -> Optional[Dict[str, Any]]:
        cands = self.pair_edges.get((min(u, v), max(u, v)))
        if not cands:
            return None
        return max(cands, key=lambda e: edge_strength(e["confidence"], e["support"]))

    def edges_among(self, node_ids: Iterable[int]) -> List[Dict[str, Any]]:
        ids = set(node_ids)
        out = []
        for (a, b), lst in self.pair_edges.items():
            if a in ids and b in ids:
                out.extend(lst)
        return out

    def node_dict(self, i: int) -> Dict[str, Any]:
        return {"id": self.names[i], "label": self.types[i], "degree": len(self.adj[i])}

    # ---------------------------------------------------------------- entity linking
    def link_entities_in_text(self, text: str, max_ngram: int = 6) -> List[str]:
        """Greedy longest-match of known entity names/aliases inside free text."""
        import re
        toks = re.findall(r"[A-Za-z0-9α-ωΑ-Ω][A-Za-z0-9'’\-\+α-ωΑ-Ω]*", text.lower())
        found: List[str] = []
        i = 0
        while i < len(toks):
            hit = None
            for n in range(min(max_ngram, len(toks) - i), 0, -1):
                cand = " ".join(toks[i:i + n]).replace("’", "'")
                if n == 1 and len(cand) < 3:
                    continue
                if cand in self.alias_index:
                    hit = (self.alias_index[cand], n)
                    break
            if hit:
                if hit[0] not in found:
                    found.append(hit[0])
                i += hit[1]
            else:
                i += 1
        return found

    # ---------------------------------------------------------------- PPR
    def personalized_pagerank(self, seeds: Dict[str, float], damping: Optional[float] = None,
                              tol: float = 1e-9, max_iter: int = 200) -> np.ndarray:
        """p = (1-d) e + d M p,  M column-stochastic over the undirected, strength-weighted graph.
        Mass from dangling (isolated) nodes is returned to the seed distribution, so sum(p) == 1."""
        d = settings.ppr_damping if damping is None else damping
        n = self.n_nodes
        e = np.zeros(n)
        for name, w in seeds.items():
            if name in self.index:
                e[self.index[name]] += float(w)
        if e.sum() <= 0:
            raise KeyError("None of the seed entities exist in the graph")
        e /= e.sum()

        col_sum = np.asarray(self.A.sum(axis=0)).ravel()
        dangling = col_sum == 0
        inv = np.divide(1.0, col_sum, out=np.zeros_like(col_sum), where=~dangling)
        M = self.A @ sparse.diags(inv)
        p = e.copy()
        for _ in range(max_iter):
            p_new = (1 - d) * e + d * (M @ p + p[dangling].sum() * e)
            if np.abs(p_new - p).sum() < tol:
                p = p_new
                break
            p = p_new
        return p

    def local_subgraph(self, seed: str, top_k: int = 20, max_hops: int = 3,
                       extra_seeds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """The `top_k` most PPR-relevant nodes within `max_hops` of `seed`, grown greedily so that the
        result is always connected. Returns nodes (with hop + ppr), edges, and per-node parent path."""
        if seed not in self.index:
            raise KeyError(seed)
        seeds = {seed: 1.0}
        if extra_seeds:
            seeds.update(extra_seeds)
        ppr = self.personalized_pagerank(seeds)

        s = self.index[seed]
        selected: Dict[int, Dict[str, Any]] = {s: {"hop": 0, "parent": None}}
        heap: List[Tuple[float, int, int, int]] = []

        def push_neighbors(u: int, hop: int):
            if hop >= max_hops:
                return
            for v in self.adj[u]:
                if v not in selected:
                    heapq.heappush(heap, (-float(ppr[v]), v, u, hop + 1))

        push_neighbors(s, 0)
        while heap and len(selected) < top_k + 1:
            _, v, parent, hop = heapq.heappop(heap)
            if v in selected:
                continue
            selected[v] = {"hop": hop, "parent": parent}
            push_neighbors(v, hop)

        ids = list(selected)
        nodes = []
        for i in ids:
            nd = self.node_dict(i)
            nd.update(hop=selected[i]["hop"], ppr=float(ppr[i]), is_seed=(i == s))
            nodes.append(nd)
        return {
            "nodes": nodes,
            "links": self.edges_among(ids),
            "parents": {self.names[i]: (None if selected[i]["parent"] is None else self.names[selected[i]["parent"]])
                        for i in ids},
            "node_ids": ids,
            "ppr": ppr,
        }

    def path_to_seed(self, parents: Dict[str, Optional[str]], node: str) -> List[str]:
        chain = [node]
        while parents.get(chain[-1]) is not None:
            chain.append(parents[chain[-1]])
        return list(reversed(chain))

    # ---------------------------------------------------------------- shortest paths
    def _weighted_graph(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        for (u, v), lst in self.pair_edges.items():
            best = min(edge_cost(e["confidence"], e["support"]) for e in lst)
            G.add_edge(u, v, weight=best)
        return G

    def k_shortest_paths(self, source: str, target: str, k: int = 3, max_hops: int = 5
                         ) -> List[Dict[str, Any]]:
        if source not in self.index or target not in self.index:
            raise KeyError("source/target entity not in graph")
        if source == target:
            return []
        G = self._cached_weighted()
        s, t = self.index[source], self.index[target]
        out: List[Dict[str, Any]] = []
        try:
            gen = nx.shortest_simple_paths(G, s, t, weight="weight")
            for n_seen, path in enumerate(gen):
                if n_seen > 60 or len(out) >= k:
                    break
                if len(path) - 1 > max_hops:
                    continue
                steps, cost, conf_prod = [], 0.0, 1.0
                for a, b in zip(path, path[1:]):
                    e = self.best_edge(a, b)
                    cost += edge_cost(e["confidence"], e["support"])
                    conf_prod *= e["confidence"]
                    steps.append(e)
                out.append({
                    "nodes": [self.names[i] for i in path],
                    "edges": steps,
                    "hops": len(path) - 1,
                    "cost": round(cost, 4),
                    "score": round(conf_prod, 4),       # product of edge confidences
                })
        except nx.NetworkXNoPath:
            return []
        return out

    _wg: Optional[nx.Graph] = None

    def _cached_weighted(self) -> nx.Graph:
        if self._wg is None:
            self._wg = self._weighted_graph()
        return self._wg

    # ---------------------------------------------------------------- heuristic link prediction
    def adamic_adar_scores(self, name: str, top_k: int = 10, max_hops: int = 2) -> List[Tuple[str, float]]:
        """Fallback link scoring (no GNN needed): Adamic–Adar over common neighbours of non-adjacent nodes."""
        i = self.index[name]
        nbrs = set(self.adj[i])
        scores: Dict[int, float] = defaultdict(float)
        for z in nbrs:
            deg = len(self.adj[z])
            if deg < 2:
                continue
            w = 1.0 / math.log(deg + 1.0)
            for c in self.adj[z]:
                if c != i and c not in nbrs:
                    scores[c] += w
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.names[c], 1.0 - math.exp(-s)) for c, s in ranked]


class GraphCache:
    """Builds and caches `GraphSnapshot` from a Neo4jClient."""

    def __init__(self, neo4j):
        self.neo4j = neo4j
        self._snap: Optional[GraphSnapshot] = None
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        with self._lock:
            self._snap = None

    def get(self, force: bool = False) -> GraphSnapshot:
        with self._lock:
            snap = self._snap
            fresh = (snap is not None and snap.version == getattr(self.neo4j, "version", 0)
                     and time.time() - snap.built_at < settings.graph_cache_ttl_sec)
            if force or not fresh:
                t0 = time.time()
                entities = self.neo4j.fetch_entities()
                edges = self.neo4j.fetch_edges()
                snap = GraphSnapshot.build(entities, edges, version=getattr(self.neo4j, "version", 0))
                self._snap = snap
                logger.info("Graph snapshot built: %d nodes, %d relation edges (%.2fs)",
                            snap.n_nodes, len(snap.edges), time.time() - t0)
            return snap
