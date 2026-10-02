"""Prompts + controlled vocabularies for LLM triplet extraction."""
from __future__ import annotations

ENTITY_TYPES = [
    "Chemical", "Protein", "Gene", "Disease", "Organism", "Process", "Food", "Anatomy", "Other",
]

# Canonical relation vocabulary. Keeping it small makes the graph dense enough for link prediction.
RELATION_TYPES = [
    "INHIBITS", "ACTIVATES", "UPREGULATES", "DOWNREGULATES", "REGULATES", "BINDS", "TARGETS",
    "CAUSES", "PREVENTS", "TREATS", "IMPROVES", "INCREASES_RISK_OF", "REDUCES_RISK_OF",
    "ASSOCIATED_WITH", "CONTAINS_COMPOUND", "PRODUCES", "DERIVED_FROM", "PART_OF",
    "EXPRESSED_IN", "METABOLIZED_BY", "BIOMARKER_OF", "INTERACTS_WITH",
]

# Free-form relations the LLM tends to invent -> canonical relation.
RELATION_ALIASES = {
    "INHIBIT": "INHIBITS", "SUPPRESSES": "INHIBITS", "SUPPRESS": "INHIBITS", "BLOCKS": "INHIBITS",
    "REDUCES": "DOWNREGULATES", "DECREASES": "DOWNREGULATES", "ATTENUATES": "DOWNREGULATES",
    "LOWERS": "DOWNREGULATES", "DOWN_REGULATES": "DOWNREGULATES", "DOWNREGULATE": "DOWNREGULATES",
    "ACTIVATE": "ACTIVATES", "STIMULATES": "ACTIVATES", "TRIGGERS": "ACTIVATES", "INDUCES": "ACTIVATES",
    "INCREASES": "UPREGULATES", "ENHANCES": "UPREGULATES", "ELEVATES": "UPREGULATES",
    "UP_REGULATES": "UPREGULATES", "UPREGULATE": "UPREGULATES", "PROMOTES": "UPREGULATES",
    "MODULATES": "REGULATES", "REGULATE": "REGULATES", "CONTROLS": "REGULATES",
    "BIND": "BINDS", "BINDS_TO": "BINDS", "TARGET": "TARGETS", "ACTS_ON": "TARGETS",
    "LEADS_TO": "CAUSES", "RESULTS_IN": "CAUSES", "CAUSE": "CAUSES", "CONTRIBUTES_TO": "CAUSES",
    "PREVENT": "PREVENTS", "PROTECTS_AGAINST": "PREVENTS", "PROTECTS": "PREVENTS",
    "TREAT": "TREATS", "ALLEVIATES": "TREATS", "AMELIORATES": "TREATS", "THERAPY_FOR": "TREATS",
    "IMPROVE": "IMPROVES", "BENEFITS": "IMPROVES",
    "RISK_FACTOR_FOR": "INCREASES_RISK_OF", "INCREASES_RISK": "INCREASES_RISK_OF",
    "DECREASES_RISK_OF": "REDUCES_RISK_OF", "REDUCES_RISK": "REDUCES_RISK_OF",
    "ASSOCIATED": "ASSOCIATED_WITH", "CORRELATES_WITH": "ASSOCIATED_WITH", "LINKED_TO": "ASSOCIATED_WITH",
    "RELATED_TO": "ASSOCIATED_WITH",
    "CONTAINS": "CONTAINS_COMPOUND", "RICH_IN": "CONTAINS_COMPOUND", "HAS_COMPOUND": "CONTAINS_COMPOUND",
    "PRODUCE": "PRODUCES", "SECRETES": "PRODUCES", "SYNTHESIZES": "PRODUCES", "RELEASES": "PRODUCES",
    "ISOLATED_FROM": "DERIVED_FROM", "EXTRACTED_FROM": "DERIVED_FROM", "FOUND_IN": "DERIVED_FROM",
    "COMPONENT_OF": "PART_OF", "SUBSET_OF": "PART_OF", "MEMBER_OF": "PART_OF",
    "EXPRESSED": "EXPRESSED_IN", "EXPRESSION_IN": "EXPRESSED_IN",
    "METABOLISED_BY": "METABOLIZED_BY", "MARKER_OF": "BIOMARKER_OF",
    "INTERACT_WITH": "INTERACTS_WITH", "INTERACTS": "INTERACTS_WITH",
}

GENERIC_ENTITY_STOPLIST = {
    "study", "studies", "result", "results", "data", "effect", "effects", "group", "groups",
    "analysis", "method", "methods", "model", "models", "it", "they", "this", "these", "we",
    "authors", "figure", "table", "patients", "subjects", "level", "levels", "treatment",
    "control", "samples", "sample", "mechanism", "mechanisms", "role", "factor", "factors",
}

SYSTEM_PROMPT = f"""You are a biomedical knowledge-graph extraction system.
Read a passage from a scientific paper and extract factual (subject, relation, object) triplets
that the passage EXPLICITLY states. Never infer facts that are not in the text.

Rules
1. Entities are specific concepts: chemicals, drugs, proteins, genes, diseases, organisms, foods,
   biological processes, anatomical structures. Do NOT use vague terms like "study", "results",
   "patients", "levels", "treatment".
2. Use the most standard, full name of each entity (e.g. "tumor necrosis factor alpha", not "it").
   Resolve pronouns and abbreviations using the text when possible.
3. "relation" MUST be one of: {", ".join(RELATION_TYPES)}.
   Pick the closest one; use ASSOCIATED_WITH only when nothing more specific fits.
4. "subject_type"/"object_type" MUST be one of: {", ".join(ENTITY_TYPES)}.
5. "confidence" (0.0-1.0) reflects how directly the text supports the triplet:
   0.9+ = explicit causal/mechanistic statement; 0.7-0.9 = clear but hedged; 0.5-0.7 = weak
   or correlational. Omit anything below 0.5.
6. "provenance_snippet" is ONE sentence copied verbatim from the passage that supports the triplet.
7. Direction matters: subject acts on / contains / produces the object.
8. Return at most 12 triplets. If nothing qualifies, return an empty list.
9. NEVER invent a relation outside the allowed relation list. If the text uses wording such as "contributes to", "ameliorates", or another unsupported relation, map it to the closest allowed relation; when uncertain, use ASSOCIATED_WITH.\n10. Return STRICTLY VALID JSON. Escape any double quotes inside provenance_snippet or other string values as JSON requires. Do not output raw/unescaped quotation marks inside a JSON string.\n
Output format: a single JSON object, nothing else:
{{"analysis": "<one short sentence on the key findings>", "triplets": [
  {{"subject": "...", "subject_type": "...", "relation": "...", "object": "...", "object_type": "...",
    "confidence": 0.0, "provenance_snippet": "..."}}
]}}"""

FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Passage:\nCocoa beans from Theobroma cacao are rich in flavan-3-ols such as epicatechin. "
            "In LPS-stimulated BV2 microglia, epicatechin significantly reduced TNF-alpha release and "
            "suppressed NF-kB activation, suggesting an anti-neuroinflammatory effect."
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"analysis": "Cocoa contains epicatechin, which dampens microglial inflammatory signalling.", '
            '"triplets": ['
            '{"subject": "theobroma cacao", "subject_type": "Organism", "relation": "CONTAINS_COMPOUND", '
            '"object": "flavan-3-ol", "object_type": "Chemical", "confidence": 0.93, '
            '"provenance_snippet": "Cocoa beans from Theobroma cacao are rich in flavan-3-ols such as epicatechin."}, '
            '{"subject": "epicatechin", "subject_type": "Chemical", "relation": "DOWNREGULATES", '
            '"object": "tumor necrosis factor alpha", "object_type": "Protein", "confidence": 0.9, '
            '"provenance_snippet": "In LPS-stimulated BV2 microglia, epicatechin significantly reduced TNF-alpha release and suppressed NF-kB activation, suggesting an anti-neuroinflammatory effect."}, '
            '{"subject": "epicatechin", "subject_type": "Chemical", "relation": "INHIBITS", '
            '"object": "nf-kb", "object_type": "Protein", "confidence": 0.9, '
            '"provenance_snippet": "In LPS-stimulated BV2 microglia, epicatechin significantly reduced TNF-alpha release and suppressed NF-kB activation, suggesting an anti-neuroinflammatory effect."}'
            "]}"
        ),
    },
    {
        "role": "user",
        "content": (
            "Passage:\nThe authors recruited 120 participants and measured fasting glucose at baseline. "
            "Samples were stored at -80 C until analysis."
        ),
    },
    {"role": "assistant", "content": '{"analysis": "Methods only; no relationships stated.", "triplets": []}'},
]


def build_messages(chunk: str, title: str | None = None):
    header = f"Paper title: {title}\n\n" if title else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{header}Passage:\n{chunk}"},
    ]



RAG_SYSTEM_PROMPT = """You are LinkForge, a scientific literature assistant that answers ONLY from the
provided knowledge-graph facts and paper excerpts.

Rules
- Ground every claim in the numbered FACTS [F#] and EXCERPTS [S#] below. Cite them inline, e.g. [F2], [S1].
- When you cite a paper, mention its DOI if available, e.g. (Smith 2024, DOI:10.1016/...).
- If the context does not contain the answer, say so plainly. Do not invent mechanisms, papers or DOIs.
- Items under PREDICTED LINKS are model-generated hypotheses, NOT published findings. If you mention
  them, label them clearly as "predicted, unverified" and give the score.
- Explain multi-hop paths step by step (A -> B -> C) in plain language. Be concise."""
