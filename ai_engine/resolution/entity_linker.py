"""Entity canonicalisation / de-duplication.

Composite similarity (from the project design):

    S(e1, e2) = alpha * cos(v1, v2) + (1 - alpha) * JaroWinkler(e1, e2)      merge if S >= theta

Pipeline per surface form:
  1. normalise (unicode, greek letters, punctuation, articles)
  2. exact match on a punctuation-insensitive key (covers aliases we have seen before)
  3. curated synonym table (SARS-CoV-2 == COVID-19 == 2019-nCoV ...)
  4. nearest canonical entities by embedding -> composite score -> merge if >= theta,
     subject to guards that prevent classic false merges (IL-6 vs IL-8, tiny strings ...)
  5. otherwise register as a new canonical entity
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rapidfuzz.distance import JaroWinkler

from backend.config import settings

logger = logging.getLogger("EntityResolver")

GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "σ": "sigma", "ω": "omega", "Δ": "delta",
}

# canonical -> synonyms (all compared after normalisation)
SEED_SYNONYMS: Dict[str, List[str]] = {
    "sars-cov-2": ["covid-19", "covid19", "2019-ncov", "coronavirus disease 2019", "novel coronavirus",
                   "severe acute respiratory syndrome coronavirus 2", "sars cov 2"],
    "alzheimer's disease": ["alzheimer disease", "alzheimers disease", "alzheimer's dementia"],
    "parkinson's disease": ["parkinson disease", "parkinsons disease"],
    "tumor necrosis factor alpha": ["tnf-alpha", "tnf alpha", "tnf-a", "tnfa", "tnf"],
    "interleukin-6": ["il-6", "il6", "interleukin 6"],
    "interleukin-1 beta": ["il-1 beta", "il-1b", "il1b", "il-1beta", "interleukin-1beta", "interleukin 1 beta"],
    "nuclear factor kappa b": ["nf-kb", "nf-kappab", "nfkb", "nf-κb", "nf kappa b", "nuclear factor-kappa b",
                               "nuclear factor kappa-light-chain-enhancer of activated b cells"],
    "flavan-3-ol": ["flavan-3-ols", "flavanol", "flavanols", "flavan 3 ol", "flavan-3-ol compounds"],
    "theobroma cacao": ["cocoa tree", "cacao tree", "t. cacao"],
    "reactive oxygen species": ["ros"],
    "amyloid beta": ["amyloid-beta", "aβ", "abeta", "a-beta", "amyloid-β", "beta-amyloid", "amyloid β"],
    "curcumin": ["diferuloylmethane"],
}

# Words that change an entity's identity ("TNF" vs "TNF receptor"): if two candidates differ by one of
# these, they are never merged, however similar their embeddings are.
MODIFIER_TOKENS = {
    "receptor", "inhibitor", "antagonist", "agonist", "expression", "gene", "mrna", "protein",
    "pathway", "signaling", "signalling", "level", "activity", "deficiency", "syndrome",
    "type", "cell", "antibody", "ligand", "kinase", "phosphorylation", "knockout",
}

_ARTICLES = re.compile(r"^(?:the|a|an)\s+")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")


def normalize_name(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    for g, rep in GREEK.items():
        t = t.replace(g, rep)
    t = t.lower().replace("’", "'").replace("`", "'").replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip(" \t\r\n.,;:()[]{}\"'")
    t = _ARTICLES.sub("", t)
    return t


def match_key(text: str) -> str:
    """Punctuation-insensitive key: "Alzheimer's  disease" == "alzheimers disease"."""
    t = normalize_name(text).replace("'", "")
    return _NONALNUM.sub(" ", t).strip()


def _stem_tokens(text: str) -> set:
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in re.findall(r"[a-z0-9]+", text.lower())}


def _digit_signature(text: str) -> Tuple[str, ...]:
    return tuple(sorted(_DIGITS.findall(text)))


class EntityResolver:
    """Maintains the canonical-entity registry (+ embedding matrix) for the whole graph."""

    def __init__(self, embedder, threshold: Optional[float] = None, alpha: Optional[float] = None,
                 top_k: int = 5, use_seed_synonyms: bool = True):
        self.embedder = embedder
        self.threshold = settings.entity_merge_threshold if threshold is None else threshold
        self.alpha = settings.entity_alpha if alpha is None else alpha
        self.top_k = top_k

        self.names: List[str] = []                # canonical names (row i of the matrix)
        self.types: Dict[str, str] = {}
        self._key_to_name: Dict[str, str] = {}    # match_key -> canonical name
        self._index: Dict[str, int] = {}
        self._matrix: Optional[np.ndarray] = None
        self._size = 0
        self._seed_map: Dict[str, str] = {}       # match_key(synonym) -> canonical (normalised)
        if use_seed_synonyms:
            for canon, syns in SEED_SYNONYMS.items():
                c = normalize_name(canon)
                self._seed_map[match_key(c)] = c
                for s in syns:
                    self._seed_map[match_key(s)] = c

    # ------------------------------------------------------------------ registry
    def _append_vectors(self, vecs: np.ndarray) -> None:
        n = vecs.shape[0]
        if self._matrix is None:
            cap = max(1024, n * 2)
            self._matrix = np.zeros((cap, vecs.shape[1]), dtype=np.float32)
        if self._size + n > self._matrix.shape[0]:
            new_cap = max(self._matrix.shape[0] * 2, self._size + n)
            grown = np.zeros((new_cap, self._matrix.shape[1]), dtype=np.float32)
            grown[: self._size] = self._matrix[: self._size]
            self._matrix = grown
        self._matrix[self._size: self._size + n] = vecs
        self._size += n

    def register(self, canonical: str, etype: str = "Other", aliases: Iterable[str] = ()) -> None:
        """Add (or update) a canonical entity already known (e.g. loaded from Neo4j)."""
        if canonical in self._index:
            if etype and etype != "Other":
                self.types[canonical] = etype
            for a in aliases:
                self._key_to_name.setdefault(match_key(a), canonical)
            return
        vec = self.embedder.embed_documents([canonical])
        self._register_with_vec(canonical, etype, aliases, vec)

    def _register_with_vec(self, canonical, etype, aliases, vec) -> None:
        self._index[canonical] = len(self.names)
        self.names.append(canonical)
        self.types[canonical] = etype or "Other"
        self._key_to_name[match_key(canonical)] = canonical
        for a in aliases:
            self._key_to_name.setdefault(match_key(a), canonical)
        self._append_vectors(vec)

    def load_entities(self, entities: Sequence[Dict]) -> int:
        """Bulk-load [{name,type,aliases}] (from Neo4j) – embeds in batches."""
        fresh = [e for e in entities if e["name"] not in self._index]
        if fresh:
            vecs = self.embedder.embed_documents([e["name"] for e in fresh])
            for e, v in zip(fresh, vecs):
                self._register_with_vec(e["name"], e.get("type", "Other"), e.get("aliases", []), v[None, :])
        logger.info("Entity resolver holds %d canonical entities", len(self.names))
        return len(fresh)

    # ------------------------------------------------------------------ similarity
    def composite_score(self, a: str, b: str, cos: float) -> float:
        jw = JaroWinkler.similarity(a, b)
        return self.alpha * cos + (1.0 - self.alpha) * jw

    @staticmethod
    def _blocked(a: str, b: str) -> bool:
        """Hard guards against false merges."""
        if _digit_signature(a) != _digit_signature(b):   # il-6 vs il-8, type 1 vs type 2
            return True
        if min(len(a), len(b)) < 4:                       # short symbols must match exactly
            return True
        if (_stem_tokens(a) ^ _stem_tokens(b)) & MODIFIER_TOKENS:   # tnf vs tnf receptor
            return True
        return False

    def _best_candidate(self, name: str, vec: np.ndarray) -> Tuple[Optional[str], float]:
        if self._size == 0:
            return None, 0.0
        sims = self._matrix[: self._size] @ vec
        k = min(self.top_k, self._size)
        idx = np.argpartition(-sims, k - 1)[:k] if k < self._size else np.arange(self._size)
        best_name, best_score = None, 0.0
        for i in idx:
            cand = self.names[int(i)]
            if self._blocked(name, cand):
                continue
            score = self.composite_score(name, cand, float(sims[int(i)]))
            if score > best_score:
                best_name, best_score = cand, score
        return best_name, best_score

    # ------------------------------------------------------------------ resolution
    def _lookup_exact(self, name: str) -> Optional[str]:
        key = match_key(name)
        if key in self._key_to_name:
            return self._key_to_name[key]
        seed = self._seed_map.get(key)
        if seed:
            canon = self._key_to_name.get(match_key(seed))
            return canon or seed   # canonical may not be registered yet -> caller registers it
        return None

    def resolve_many(self, surface_forms: Sequence[str], types: Optional[Sequence[str]] = None
                     ) -> List[Tuple[str, str]]:
        """Map each surface form to (canonical_name, normalised_surface). Registers new entities."""
        types = list(types) if types is not None else ["Other"] * len(surface_forms)
        normed = [normalize_name(s) for s in surface_forms]

        # embed all not-yet-exactly-known forms in a single batch
        todo = [n for n in dict.fromkeys(normed) if n and self._lookup_exact(n) is None]
        vecs: Dict[str, np.ndarray] = {}
        if todo:
            for n, v in zip(todo, self.embedder.embed_documents(todo)):
                vecs[n] = v

        out: List[Tuple[str, str]] = []
        for raw, n, t in zip(surface_forms, normed, types):
            if not n:
                out.append(("", ""))
                continue
            hit = self._lookup_exact(n)
            if hit is not None:
                if hit not in self._index:   # seed canonical seen for the first time
                    self.register(hit, t)
                elif t and t != "Other" and self.types.get(hit, "Other") == "Other":
                    self.types[hit] = t
                self._key_to_name.setdefault(match_key(n), hit)
                out.append((hit, n))
                continue

            vec = vecs.get(n)
            if vec is None:  # appeared in `todo` earlier and was registered in this loop
                vec = self.embedder.embed_documents([n])[0]
            cand, score = self._best_candidate(n, vec)
            if cand is not None and score >= self.threshold:
                self._key_to_name.setdefault(match_key(n), cand)
                if t and t != "Other" and self.types.get(cand, "Other") == "Other":
                    self.types[cand] = t
                out.append((cand, n))
            else:
                self._register_with_vec(n, t, [], vec[None, :])
                out.append((n, n))
        return out

    def resolve(self, surface_form: str, etype: str = "Other") -> str:
        return self.resolve_many([surface_form], [etype])[0][0]

    def canonicalize_triplets(self, triplets: List[Dict]) -> List[Dict]:
        """Replace subject/object with canonical names; keep raw surface forms as aliases."""
        if not triplets:
            return []
        forms = [t["subject"] for t in triplets] + [t["object"] for t in triplets]
        types = [t.get("subject_type", "Other") for t in triplets] + [t.get("object_type", "Other") for t in triplets]
        resolved = self.resolve_many(forms, types)
        n = len(triplets)
        out = []
        for i, t in enumerate(triplets):
            (s_c, s_raw), (o_c, o_raw) = resolved[i], resolved[n + i]
            if not s_c or not o_c or s_c == o_c:
                continue
            nt = dict(t)
            nt.update(subject=s_c, object=o_c, subject_raw=s_raw, object_raw=o_raw,
                      subject_type=self.types.get(s_c, t.get("subject_type", "Other")),
                      object_type=self.types.get(o_c, t.get("object_type", "Other")))
            out.append(nt)
        return out
