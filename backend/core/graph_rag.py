"""Query-time logic: transitive search, bridge search, grounded Graph-RAG answers."""
from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, List, Optional

from ai_engine.extraction.prompts import RAG_SYSTEM_PROMPT
from backend.core.services import Services, get_services

logger = logging.getLogger("GraphRAG")


def _edge_view(e: Dict[str, Any], predicted: bool = False) -> Dict[str, Any]:
    return {"source": e["source"], "target": e["target"], "type": e.get("type", "PREDICTED"),
            "confidence": e.get("confidence", e.get("score")), "support": e.get("support", 0),
            "papers": e.get("papers", []), "predicted": predicted}


def highlight_terms(text: str, terms: List[str]) -> str:
    """HTML-escape `text` and wrap graph entities in <mark data-entity=...> (longest first)."""
    if not text:
        return ""
    escaped = html.escape(text)
    pats = sorted({t for t in terms if t and len(t) > 2}, key=len, reverse=True)
    if not pats:
        return escaped
    rx = re.compile(r"(?<![\w-])(" + "|".join(re.escape(html.escape(t)) for t in pats) + r")(?![\w-])", re.I)
    return rx.sub(lambda m: f'<mark data-entity="{html.escape(m.group(1).lower())}">{m.group(1)}</mark>', escaped)


def snippet_around(text: str, terms: List[str], width: int = 280) -> str:
    text = (text or "").strip()
    if len(text) <= width:
        return text
    low = text.lower()
    pos = [low.find(t.lower()) for t in terms if t and low.find(t.lower()) >= 0]
    start = max(0, (min(pos) if pos else 0) - width // 4)
    piece = text[start:start + width].strip()
    return ("…" if start else "") + piece + ("…" if start + width < len(text) else "")


class GraphRAG:
    def __init__(self, services: Optional[Services] = None):
        self.s = services or get_services()

    # ------------------------------------------------------------------ helpers
    def resolve(self, term: str) -> Optional[str]:
        snap = self.s.snapshot
        t = term.strip().lower()
        if t in snap.alias_index:
            return snap.alias_index[t]
        hit = self.s.neo4j.find_entity(term)
        return hit["name"] if hit and hit["name"] in snap.index else None

    def _predictions(self, snap, seed: str, node_names: List[str], top_k: int = 6) -> List[Dict[str, Any]]:
        try:
            svc = self.s.link_predictor
            preds = svc.predict_between(snap, node_names, top_k=top_k, min_score=0.3)
            extra = svc.predict_for_entity(snap, seed, top_k=3, min_score=0.3)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Link prediction failed: %s", exc)
            return []
        seen, out = set(), []
        for p in preds + extra:
            key = frozenset((p["source"], p["target"]))
            if key not in seen:
                seen.add(key)
                out.append(p)
        return out

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: int = 20, max_hops: int = 3, with_predictions: bool = True,
               max_papers: int = 15) -> Dict[str, Any]:
        snap = self.s.snapshot
        seed = self.resolve(query)
        if seed is None:
            return {"query": query, "found": False, "suggestions": self.s.neo4j.search_entities(query, 8),
                    "nodes": [], "links": [], "papers": []}
        sub = snap.local_subgraph(seed, top_k=top_k, max_hops=max_hops)
        names = [n["id"] for n in sub["nodes"]]
        links = [_edge_view(e) for e in sub["links"]]
        preds = self._predictions(snap, seed, names) if with_predictions else []
        pred_links = []
        for p in preds:
            if p["target"] not in names:     # keep the canvas bounded: add predicted endpoints as nodes
                i = snap.index[p["target"]]
                nd = snap.node_dict(i)
                nd.update(hop=None, ppr=None, is_seed=False, predicted_only=True)
                sub["nodes"].append(nd)
                names.append(p["target"])
            pred_links.append({"source": p["source"], "target": p["target"], "type": "PREDICTED",
                               "confidence": p["score"], "support": 0, "papers": [], "predicted": True,
                               "method": p["method"]})

        # papers that mention the seed or its transitive neighbours; show the chain for each
        papers = []
        for row in self.s.neo4j.get_papers_for_entities(names, limit=max_papers * 2):
            ents = [e for e in row["entities"]]
            chain = None
            for e in sorted(ents, key=lambda x: -(dict((n["id"], n.get("ppr") or 0) for n in sub["nodes"]).get(x, 0))):
                if e != seed and e in sub["parents"]:
                    chain = snap.path_to_seed(sub["parents"], e)
                    break
            terms = [seed] + [e for e in ents if e != seed]
            abstract = row.get("abstract") or ""
            snip = snippet_around(abstract, terms)
            papers.append({
                "paper_id": row["paper_id"], "title": row["title"], "year": row["year"],
                "journal": row.get("journal"), "doi": row.get("doi"), "url": row.get("url"),
                "entities": ents, "path": chain,
                "snippet": snip, "snippet_html": highlight_terms(snip, terms),
            })
            if len(papers) >= max_papers:
                break
        return {"query": query, "found": True, "seed": seed,
                "nodes": sub["nodes"], "links": links + pred_links, "papers": papers,
                "link_prediction": self._lp_status()}

    def _lp_status(self) -> Dict[str, Any]:
        try:
            return self.s.link_predictor.status()
        except Exception as exc:  # noqa: BLE001
            return {"method": "none", "reason": str(exc)}

    # ------------------------------------------------------------------ bridge A -> ? -> D
    def bridge(self, a: str, b: str, k: int = 3, max_hops: int = 5) -> Dict[str, Any]:
        snap = self.s.snapshot
        ea, eb = self.resolve(a), self.resolve(b)
        missing = [q for q, e in ((a, ea), (b, eb)) if e is None]
        if missing:
            return {"found": False, "missing": missing, "paths": [], "nodes": [], "links": []}
        paths = snap.k_shortest_paths(ea, eb, k=k, max_hops=max_hops)
        nodes, links, seen = {}, {}, set()
        out_paths = []
        for p in paths:
            steps = []
            for e in p["edges"]:
                ev = _edge_view(e)
                ev["evidence"] = self.s.vectors.get_edge_provenance(
                    e["source"], e["target"], limit=2, relation=e["type"], paper_ids=e["papers"])
                steps.append(ev)
                links[(e["source"], e["target"], e["type"])] = _edge_view(e)
            for n in p["nodes"]:
                nodes[n] = snap.node_dict(snap.index[n])
            out_paths.append({"nodes": p["nodes"], "hops": p["hops"], "score": p["score"],
                              "cost": p["cost"], "steps": steps})
        score = None
        try:
            score = self.s.link_predictor.score_pair(snap, ea, eb)
        except Exception as exc:  # noqa: BLE001
            logger.debug("score_pair failed: %s", exc)
        hyp = None
        if score and not paths and not score.get("already_linked"):
            hyp = score
        return {"found": bool(paths), "source": ea, "target": eb, "paths": out_paths,
                "nodes": list(nodes.values()), "links": list(links.values()),
                "predicted_link": score, "hypothesis_only": hyp is not None}

    # ------------------------------------------------------------------ chat
    def build_context(self, question: str, focus: Optional[List[str]] = None) -> Dict[str, Any]:
        snap = self.s.snapshot
        ents = []
        for name in (focus or []):
            r = self.resolve(name)
            if r and r not in ents:
                ents.append(r)
        for e in snap.link_entities_in_text(question):
            if e not in ents:
                ents.append(e)
        facts: List[Dict[str, Any]] = []
        predicted: List[Dict[str, Any]] = []

        def add_fact(e: Dict[str, Any]):
            key = (e["source"], e["type"], e["target"])
            if all((f["source"], f["type"], f["target"]) != key for f in facts):
                facts.append(_edge_view(e))

        if len(ents) >= 2:
            for p in snap.k_shortest_paths(ents[0], ents[1], k=3, max_hops=5):
                for e in p["edges"]:
                    add_fact(e)
            try:
                sc = self.s.link_predictor.score_pair(snap, ents[0], ents[1])
                if sc["predicted"] and sc["score"] >= 0.3:
                    predicted.append(sc)
            except Exception:  # noqa: BLE001
                pass
        if ents and len(facts) < 8:
            sub = snap.local_subgraph(ents[0], top_k=10, max_hops=2)
            for e in sorted(sub["links"], key=lambda x: -(x["confidence"] * x["support"]))[:12]:
                add_fact(e)
            predicted.extend(self._predictions(snap, ents[0], [n["id"] for n in sub["nodes"]], top_k=3))
        facts = facts[:14]

        excerpts: List[Dict[str, Any]] = []
        seen = set()
        for hit in self.s.vectors.search_provenance(question, limit=5) + self.s.vectors.search_chunks(question, limit=4):
            if hit["text"] in seen:
                continue
            seen.add(hit["text"])
            m = hit["metadata"]
            excerpts.append({"text": hit["text"][:700], "doi": m.get("doi"), "title": m.get("title"),
                             "year": m.get("year"), "paper_id": m.get("paper_id"), "score": hit.get("score")})
        excerpts = excerpts[:8]
        return {"entities": ents, "facts": facts, "excerpts": excerpts, "predicted": predicted[:5]}

    @staticmethod
    def format_context(ctx: Dict[str, Any]) -> str:
        lines = ["FACTS (knowledge-graph edges):"]
        for i, f in enumerate(ctx["facts"], 1):
            lines.append(f"[F{i}] {f['source']} --{f['type']}--> {f['target']} "
                         f"(confidence {float(f['confidence'] or 0):.2f}, {f['support']} paper(s))")
        if not ctx["facts"]:
            lines.append("(none)")
        lines.append("\nEXCERPTS (paper text):")
        for i, x in enumerate(ctx["excerpts"], 1):
            ref = f"DOI:{x['doi']}" if x.get("doi") else (x.get("title") or "unknown source")
            lines.append(f"[S{i}] ({ref}, {x.get('year') or 'n.d.'}) {x['text']}")
        if not ctx["excerpts"]:
            lines.append("(none)")
        lines.append("\nPREDICTED LINKS (model hypotheses, NOT published findings):")
        for p in ctx["predicted"]:
            lines.append(f"- {p['source']} <-> {p['target']}: score {p['score']:.2f} ({p['method']})")
        if not ctx["predicted"]:
            lines.append("(none)")
        return "\n".join(lines)

    def answer(self, question: str, focus: Optional[List[str]] = None,
               history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        ctx = self.build_context(question, focus)
        if not ctx["facts"] and not ctx["excerpts"]:
            return {"answer": "I couldn't find anything relevant in the knowledge graph yet. "
                              "Try ingesting papers on this topic first.", "grounded": False, **ctx}
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        for h in (history or [])[-6:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"][:1500]})
        messages.append({"role": "user", "content": f"{self.format_context(ctx)}\n\nQUESTION: {question}"})
        from ai_engine.extraction.llm_extractor import LLMUnavailable
        try:
            text = self.s.llm.chat(messages, json_mode=False, temperature=0.2, max_tokens=700)
            llm_ok = True
        except LLMUnavailable as exc:
            logger.warning("LLM unavailable for chat: %s", exc)
            text = ("(LLM unavailable – showing retrieved evidence only)\n\n" + self.format_context(ctx))
            llm_ok = False
        return {"answer": text.strip(), "grounded": True, "llm_used": llm_ok, **ctx}
