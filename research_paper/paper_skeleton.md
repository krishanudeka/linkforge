# LinkForge: Zero-Cost Graph-RAG for Transitive and Predictive Discovery in Scientific Literature
*Authors · Affiliation · Date*

## Abstract
(Problem: keyword search creates research silos. Method: LLM triplet extraction → resolved KG → PPR subgraphs,
confidence-weighted paths, GNN link prediction. Result: fill in after evaluation. Cost: $0.)

## 1. Introduction
- Silos in keyword search; Swanson-style literature-based discovery; contributions (3–4 bullets).

## 2. Related Work
Literature-based discovery · biomedical KG construction (SemMedDB, etc.) · Graph-RAG · GNN link prediction.

## 3. System
3.1 Acquisition & parsing · 3.2 Triplet extraction (prompt, validation, snippet verification) ·
3.3 Entity resolution: S = α·cos + (1−α)·JW, θ=0.85, α=0.7, guard rules · 3.4 Storage (Neo4j, ChromaDB) ·
3.5 Retrieval: PPR p=(1−d)e_A+dMp; edge cost W=−log(mean conf); k-shortest paths · 3.6 GNN (GraphSAGE/GAT, inner-product decoder) ·
3.7 Grounded chat with citations.

## 4. Experimental Setup
- Corpus (queries, sources, #papers, date range). LLM + version. Hardware (Colab T4).
- **Extraction quality:** hand-annotate N chunks → precision/recall/F1 of triplets.
- **Entity resolution:** annotated merge pairs → precision/recall; ablate α, θ, guards.
- **Time-machine link prediction:** train on edges first published < 2022 (`--split temporal --cutoff-year 2022`),
  evaluate on edges first published ≥ 2022 between existing nodes. Metrics: AUROC, AP, Hits@K.
  Baselines: Adamic–Adar, common neighbours, random, embedding-only (no graph).
- **Grounding:** sample chat answers; rate citation correctness / unsupported-claim rate.

## 5. Results
(Tables: extraction, resolution, link prediction; figure: example subgraph with predicted dashed edges.)

## 6. Discussion & Limitations
Extraction noise, abstract-only papers, novelty vs. plausibility of predictions, temporal leakage risks, LLM drift.

## 7. Conclusion & Future Work

## References
