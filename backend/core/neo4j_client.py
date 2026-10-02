"""Neo4j access layer: schema, batched writes, and all read queries used by the API.

Graph model
-----------
(:Entity {name, type, aliases[]})                       canonical, lower-cased name
(:Paper  {paper_id, doi, pmid, pmcid, title, year, journal, abstract, ...})
(:Entity)-[:MENTIONED_IN]->(:Paper)
(:Entity)-[:RELATION {type, confidence, support, papers[], conf_sum, first_year}]->(:Entity)

One RELATION edge exists per (subject, relation-type, object); evidence from many papers is
aggregated on it (`papers`, `support`, mean `confidence`).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from neo4j import GraphDatabase

from backend.config import settings

logger = logging.getLogger("Neo4jClient")

_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def lucene_escape(text: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", text)


def _clean_year(year: Any) -> Optional[int]:
    try:
        y = int(str(year)[:4])
        return y if 1800 < y < 2200 else None
    except (TypeError, ValueError):
        return None


class Neo4jClient:
    SCHEMA_STATEMENTS = [
        "CREATE CONSTRAINT entity_unique_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT paper_unique_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
        "CREATE INDEX paper_doi_index IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
        "CREATE INDEX paper_title_index IF NOT EXISTS FOR (p:Paper) ON (p.title)",
        "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.aliases]",
        "CREATE FULLTEXT INDEX paper_fulltext IF NOT EXISTS FOR (p:Paper) ON EACH [p.title, p.abstract]",
    ]

    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None, database: str | None = None):
        self.uri = uri or settings.neo4j_uri
        self.database = database or settings.neo4j_database
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(user or settings.neo4j_user, password or settings.neo4j_password),
        )
        # Bumped on every write so in-process caches (graph snapshot, GNN embeddings) know to refresh.
        self.version = 0

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> "Neo4jClient":
        """Verify connectivity and make sure constraints / indexes exist."""
        self.driver.verify_connectivity()
        self.ensure_schema()
        logger.info("Connected to Neo4j at %s (database=%s)", self.uri, self.database)
        return self

    def close(self) -> None:
        self.driver.close()

    def ensure_schema(self) -> None:
        for stmt in self.SCHEMA_STATEMENTS:
            try:
                self._write(stmt)
            except Exception as exc:  # e.g. fulltext unsupported on an exotic edition
                logger.warning("Schema statement failed (%s): %s", stmt[:60], exc)

    # ------------------------------------------------------------------ helpers
    def _read(self, cypher: str, **params) -> List[Dict[str, Any]]:
        def work(tx):
            return [r.data() for r in tx.run(cypher, **params)]

        with self.driver.session(database=self.database) as session:
            return session.execute_read(work)

    def _write(self, cypher: str, **params) -> None:
        def work(tx):
            tx.run(cypher, **params).consume()

        with self.driver.session(database=self.database) as session:
            session.execute_write(work)
        self.version += 1

    # ------------------------------------------------------------------ writes
    def upsert_paper(self, paper: Dict[str, Any]) -> None:
        params = {
            "paper_id": paper["paper_id"],
            "doi": paper.get("doi") or None,
            "pmid": paper.get("pmid") or None,
            "pmcid": paper.get("pmcid") or None,
            "title": paper.get("title") or None,
            "year": _clean_year(paper.get("year")),
            "journal": paper.get("journal") or None,
            "abstract": paper.get("abstract") or None,
            "source": paper.get("source") or None,
            "url": paper.get("url") or None,
            "authors": paper.get("authors") or None,
        }
        self._write(
            """
            MERGE (p:Paper {paper_id: $paper_id})
              ON CREATE SET p.created_at = timestamp(), p.processed = false
            SET p.doi      = coalesce($doi, p.doi),
                p.pmid     = coalesce($pmid, p.pmid),
                p.pmcid    = coalesce($pmcid, p.pmcid),
                p.title    = coalesce($title, p.title),
                p.year     = coalesce($year, p.year),
                p.journal  = coalesce($journal, p.journal),
                p.abstract = coalesce($abstract, p.abstract),
                p.source   = coalesce($source, p.source),
                p.url      = coalesce($url, p.url),
                p.authors  = coalesce($authors, p.authors)
            """,
            **params,
        )

    _INSERT_TRIPLETS = """
        MATCH (p:Paper {paper_id: $paper_id})
        UNWIND $triplets AS t
        MERGE (s:Entity {name: t.subject})
          ON CREATE SET s.aliases = [], s.type = t.subject_type, s.created_at = timestamp()
        MERGE (o:Entity {name: t.object})
          ON CREATE SET o.aliases = [], o.type = t.object_type, o.created_at = timestamp()
        SET s.type = CASE WHEN s.type IS NULL OR s.type = 'Other' THEN t.subject_type ELSE s.type END,
            o.type = CASE WHEN o.type IS NULL OR o.type = 'Other' THEN t.object_type ELSE o.type END,
            s.aliases = CASE
                WHEN t.subject_raw IS NULL OR t.subject_raw = s.name
                     OR t.subject_raw IN coalesce(s.aliases, []) THEN coalesce(s.aliases, [])
                ELSE coalesce(s.aliases, []) + t.subject_raw END,
            o.aliases = CASE
                WHEN t.object_raw IS NULL OR t.object_raw = o.name
                     OR t.object_raw IN coalesce(o.aliases, []) THEN coalesce(o.aliases, [])
                ELSE coalesce(o.aliases, []) + t.object_raw END
        MERGE (s)-[:MENTIONED_IN]->(p)
        MERGE (o)-[:MENTIONED_IN]->(p)
        MERGE (s)-[r:RELATION {type: t.relation}]->(o)
          ON CREATE SET r.papers = [], r.conf_sum = 0.0, r.support = 0, r.confidence = 0.0
        WITH p, t, r, (NOT (p.paper_id IN r.papers)) AS is_new
        SET r.papers = CASE WHEN is_new THEN r.papers + p.paper_id ELSE r.papers END,
            r.conf_sum = CASE WHEN is_new THEN r.conf_sum + t.confidence ELSE r.conf_sum END,
            r.first_year = CASE
                WHEN p.year IS NULL THEN r.first_year
                WHEN r.first_year IS NULL OR p.year < r.first_year THEN p.year
                ELSE r.first_year END
        WITH r
        SET r.support = size(r.papers),
            r.confidence = CASE WHEN size(r.papers) > 0 THEN r.conf_sum / size(r.papers) ELSE 0.0 END
    """

    def insert_triplets(self, triplets: List[Dict[str, Any]], paper_id: str, batch_size: int = 200) -> int:
        """Write canonicalised triplets for one (already upserted) paper. Returns #rows written.

        Each triplet dict: subject, relation, object, confidence, subject_type, object_type,
        subject_raw, object_raw.
        """
        rows = []
        for t in triplets:
            s, o = (t.get("subject") or "").strip(), (t.get("object") or "").strip()
            rel = (t.get("relation") or "").strip()
            if not s or not o or not rel or s == o:
                continue
            rows.append({
                "subject": s,
                "object": o,
                "relation": rel,
                "confidence": float(max(0.0, min(1.0, t.get("confidence", 0.5)))),
                "subject_type": t.get("subject_type") or "Other",
                "object_type": t.get("object_type") or "Other",
                "subject_raw": (t.get("subject_raw") or "").strip() or None,
                "object_raw": (t.get("object_raw") or "").strip() or None,
            })
        for i in range(0, len(rows), batch_size):
            self._write(self._INSERT_TRIPLETS, paper_id=paper_id, triplets=rows[i:i + batch_size])
        return len(rows)

    def insert_triplet(self, subject: str, relation: str, object_: str, confidence: float,
                       paper_meta: Dict[str, Any]) -> None:
        """Convenience single-triplet insert (keeps the original project API)."""
        paper = dict(paper_meta)
        paper.setdefault("paper_id", paper.get("doi") or paper.get("title") or "unknown")
        self.upsert_paper(paper)
        self.insert_triplets(
            [{
                "subject": subject.strip().lower(),
                "relation": relation.strip().upper().replace(" ", "_"),
                "object": object_.strip().lower(),
                "confidence": confidence,
            }],
            paper["paper_id"],
        )

    def mark_paper_processed(self, paper_id: str, n_triplets: int, n_chunks: int) -> None:
        self._write(
            """
            MATCH (p:Paper {paper_id: $pid})
            SET p.processed = true, p.processed_at = timestamp(),
                p.n_triplets = $n_triplets, p.n_chunks = $n_chunks
            """,
            pid=paper_id, n_triplets=n_triplets, n_chunks=n_chunks,
        )

    def clear_all(self) -> None:
        """DANGER: deletes every node/edge. Used by tests and `scripts/seed_demo.py --reset`."""
        self._write("MATCH (n) DETACH DELETE n")

    # ------------------------------------------------------------------ paper reads
    def paper_processed(self, paper_id: str) -> bool:
        rows = self._read("MATCH (p:Paper {paper_id: $pid}) RETURN coalesce(p.processed, false) AS done",
                          pid=paper_id)
        return bool(rows and rows[0]["done"])

    def get_papers(self, paper_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = list(dict.fromkeys(paper_ids))
        if not ids:
            return []
        return self._read(
            """
            MATCH (p:Paper) WHERE p.paper_id IN $ids
            RETURN p.paper_id AS paper_id, p.doi AS doi, p.title AS title, p.year AS year,
                   p.journal AS journal, p.abstract AS abstract, p.url AS url
            """,
            ids=ids,
        )

    def get_papers_for_entities(self, names: List[str], limit: int = 100) -> List[Dict[str, Any]]:
        """Papers that mention at least one of `names`, with which entities they mention."""
        if not names:
            return []
        return self._read(
            """
            MATCH (e:Entity)-[:MENTIONED_IN]->(p:Paper) WHERE e.name IN $names
            WITH p, collect(DISTINCT e.name) AS entities
            RETURN p.paper_id AS paper_id, p.doi AS doi, p.title AS title, p.year AS year,
                   p.journal AS journal, p.abstract AS abstract, p.url AS url, entities
            ORDER BY size(entities) DESC, p.year DESC
            LIMIT $limit
            """,
            names=names, limit=limit,
        )

    def list_papers(self, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        return self._read(
            """
            MATCH (p:Paper)
            RETURN p.paper_id AS paper_id, p.doi AS doi, p.title AS title, p.year AS year,
                   coalesce(p.processed, false) AS processed, coalesce(p.n_triplets, 0) AS n_triplets
            ORDER BY p.created_at DESC SKIP $skip LIMIT $limit
            """,
            limit=limit, skip=skip,
        )

    # ------------------------------------------------------------------ graph reads
    def fetch_entities(self) -> List[Dict[str, Any]]:
        return self._read(
            "MATCH (e:Entity) RETURN e.name AS name, coalesce(e.type, 'Other') AS type, "
            "coalesce(e.aliases, []) AS aliases"
        )

    def fetch_edges(self) -> List[Dict[str, Any]]:
        return self._read(
            """
            MATCH (s:Entity)-[r:RELATION]->(o:Entity)
            RETURN s.name AS source, o.name AS target, r.type AS type,
                   coalesce(r.confidence, 0.5) AS confidence, coalesce(r.support, 1) AS support,
                   r.first_year AS year, coalesce(r.papers, []) AS papers
            """
        )

    def get_edge_papers(self, source: str, target: str) -> List[Dict[str, Any]]:
        """All relation edges between two entities (either direction) with their paper ids."""
        return self._read(
            """
            MATCH (a:Entity)-[r:RELATION]->(b:Entity)
            WHERE (a.name = $x AND b.name = $y) OR (a.name = $y AND b.name = $x)
            RETURN a.name AS source, b.name AS target, r.type AS type, r.confidence AS confidence,
                   r.support AS support, r.papers AS papers
            """,
            x=source, y=target,
        )

    def find_entity(self, term: str) -> Optional[Dict[str, Any]]:
        """Resolve a user string to a canonical entity: exact name -> alias -> full-text best hit."""
        t = term.strip().lower()
        if not t:
            return None
        rows = self._read(
            """
            MATCH (e:Entity)
            WHERE e.name = $t OR $t IN coalesce(e.aliases, [])
            RETURN e.name AS name, coalesce(e.type, 'Other') AS type
            ORDER BY CASE WHEN e.name = $t THEN 0 ELSE 1 END LIMIT 1
            """,
            t=t,
        )
        if rows:
            return rows[0]
        hits = self.search_entities(term, limit=1)
        return hits[0] if hits else None

    def search_entities(self, term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text prefix search over canonical names + aliases (used for autocomplete)."""
        tokens = [tok for tok in re.split(r"\s+", term.strip().lower()) if tok]
        if not tokens:
            return []
        lucene = " ".join(f"{lucene_escape(tok)}*" for tok in tokens)
        try:
            return self._read(
                """
                CALL db.index.fulltext.queryNodes('entity_fulltext', $q) YIELD node, score
                RETURN node.name AS name, coalesce(node.type, 'Other') AS type, score
                ORDER BY score DESC LIMIT $limit
                """,
                q=lucene, limit=limit,
            )
        except Exception as exc:
            logger.warning("Full-text search failed (%s); falling back to CONTAINS.", exc)
            return self._read(
                """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $t OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS $t)
                RETURN e.name AS name, coalesce(e.type, 'Other') AS type, 1.0 AS score
                LIMIT $limit
                """,
                t=term.strip().lower(), limit=limit,
            )

    def stats(self) -> Dict[str, int]:
        rows = self._read(
            """
            CALL { MATCH (e:Entity) RETURN count(e) AS entities }
            CALL { MATCH (p:Paper) RETURN count(p) AS papers }
            CALL { MATCH ()-[r:RELATION]->() RETURN count(r) AS relations }
            CALL { MATCH (p:Paper) WHERE coalesce(p.processed, false) RETURN count(p) AS processed_papers }
            RETURN entities, papers, relations, processed_papers
            """
        )
        return rows[0] if rows else {"entities": 0, "papers": 0, "relations": 0, "processed_papers": 0}

    def get_transitive_subgraph(self, query_term: str, max_hops: int = 3, limit: int = 50) -> Dict[str, Any]:
        """Plain Cypher variable-length expansion (kept for parity with the original design;
        the API uses Personalized-PageRank subgraphs from `graph_analytics` instead)."""
        max_hops = max(1, min(int(max_hops), 5))
        rows = self._read(
            f"""
            MATCH path = (a:Entity {{name: $t}})-[:RELATION*1..{max_hops}]-(b:Entity)
            WITH path LIMIT $limit
            RETURN [n IN nodes(path) | {{id: n.name, label: coalesce(n.type, 'Other')}}] AS nodes,
                   [r IN relationships(path) | {{source: startNode(r).name, target: endNode(r).name,
                        type: r.type, confidence: coalesce(r.confidence, 1.0)}}] AS links
            """,
            t=query_term.strip().lower(), limit=limit,
        )
        nodes: Dict[str, Dict[str, Any]] = {}
        links: Dict[tuple, Dict[str, Any]] = {}
        for row in rows:
            for n in row["nodes"]:
                nodes[n["id"]] = n
            for l in row["links"]:
                links[(l["source"], l["target"], l["type"])] = l
        return {"nodes": list(nodes.values()), "links": list(links.values())}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = Neo4jClient().connect()
    client.insert_triplet(
        "theobroma cacao", "CONTAINS_COMPOUND", "flavan-3-ol", 0.95,
        {"paper_id": "10.1016/j.jnutbio.2024.108920", "doi": "10.1016/j.jnutbio.2024.108920",
         "title": "Cocoa Flavanols and Brain Health", "year": 2024,
         "journal": "Journal of Nutritional Biochemistry"},
    )
    print("Stats:", client.stats())
    client.close()
