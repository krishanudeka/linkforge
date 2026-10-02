"""Train the GNN link predictor.

Local (graph in Neo4j):
    python -m ai_engine.gnn.train --source neo4j --epochs 200
Colab / Kaggle (export first with `python scripts/export_graph.py`, upload the JSON):
    python -m ai_engine.gnn.train --source json --graph-json graph_export.json --device cuda
Time-machine back-test (train on <=2021 links, test on links first published after 2021):
    python -m ai_engine.gnn.train --source json --graph-json graph_export.json --split temporal --cutoff-year 2021
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from ai_engine.gnn.dataset import (
    EdgeSplit, GraphData, build_node_features, load_graph_from_neo4j, load_graph_json,
    sample_negatives, split_edges, undirected_edge_index,
)
from ai_engine.gnn.model import LinkPredictionModel

logger = logging.getLogger("GNNTrain")


def _t(a: np.ndarray, device, dtype=torch.long):
    return torch.as_tensor(a, dtype=dtype, device=device)


@torch.no_grad()
def evaluate(model, x, mp_pairs: np.ndarray, pos: np.ndarray, neg: np.ndarray, device) -> Dict[str, float]:
    model.eval()
    z = model.encode(x, _t(undirected_edge_index(mp_pairs), device))
    pairs = _t(np.concatenate([pos, neg]), device)
    logits = model.decode(z, pairs).cpu().numpy()
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    probs = 1.0 / (1.0 + np.exp(-logits))
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
    }


def train_model(graph: GraphData, x_np: np.ndarray, split: EdgeSplit, *, model_type: str = "sage",
                hidden: int = 128, out_dim: int = 64, layers: int = 2, heads: int = 4,
                dropout: float = 0.3, lr: float = 0.005, weight_decay: float = 1e-5,
                epochs: int = 200, patience: int = 25, supervision_frac: float = 0.3,
                device: str = "cpu", seed: int = 42, verbose: bool = True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=dev)
    model = LinkPredictionModel(x.shape[1], hidden, out_dim, layers, model_type, heads, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_pos = split.train_pos
    best = {"auc": -1.0, "state": None, "epoch": 0}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        # Hold part of the training edges out of message passing so the model can't just "see" the
        # edge it is asked to predict (prevents the classic leakage in link-prediction training).
        perm = rng.permutation(len(train_pos))
        n_sup = max(1, int(len(train_pos) * supervision_frac))
        sup, mp = train_pos[perm[:n_sup]], train_pos[perm[n_sup:]]
        neg = sample_negatives(graph.n_nodes, split.all_pairs, len(sup), rng)

        z = model.encode(x, _t(undirected_edge_index(mp), dev))
        pairs = _t(np.concatenate([sup, neg]), dev)
        labels = torch.cat([torch.ones(len(sup)), torch.zeros(len(neg))]).to(dev)
        loss = F.binary_cross_entropy_with_logits(model.decode(z, pairs), labels)
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 5 == 0 or epoch == 1:
            m = evaluate(model, x, split.train_pos, split.val_pos, split.val_neg, dev)
            history.append({"epoch": epoch, "loss": float(loss), **{f"val_{k}": v for k, v in m.items()}})
            if verbose and (epoch % 20 == 0 or epoch == 1):
                logger.info("epoch %3d  loss %.4f  val AUC %.4f  AP %.4f", epoch, float(loss), m["auc"], m["ap"])
            if m["auc"] > best["auc"]:
                best = {"auc": m["auc"], "epoch": epoch,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            elif epoch - best["epoch"] >= patience * 5:
                logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best["epoch"])
                break

    model.load_state_dict(best["state"])
    mp_test = np.concatenate([split.train_pos, split.val_pos])
    test = evaluate(model, x, mp_test, split.test_pos, split.test_neg, dev)
    val = evaluate(model, x, split.train_pos, split.val_pos, split.val_neg, dev)
    metrics = {"val": val, "test": test, "best_epoch": best["epoch"], "history": history}
    return model.cpu(), metrics


def save_checkpoint(model, metrics: dict, path: str | Path, *, embedding_model: str, split: EdgeSplit,
                    graph: GraphData) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": model.config,
        "embedding_model": embedding_model,
        "metrics": {"val": metrics["val"], "test": metrics["test"], "best_epoch": metrics["best_epoch"]},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": {"mode": split.mode, "cutoff_year": split.cutoff_year},
        "graph": {"nodes": graph.n_nodes, "edges": graph.n_edges},
    }, path)
    path.with_suffix(".json").write_text(json.dumps({
        "embedding_model": embedding_model, "config": model.config,
        "metrics": metrics, "split": {"mode": split.mode, "cutoff_year": split.cutoff_year},
        "graph": {"nodes": graph.n_nodes, "edges": graph.n_edges},
    }, indent=2), encoding="utf-8")
    return path


def main(argv=None):
    from backend.config import settings

    ap = argparse.ArgumentParser(description="Train the LinkForge GNN link predictor")
    ap.add_argument("--source", choices=["neo4j", "json"], default="neo4j")
    ap.add_argument("--graph-json", default=str(settings.export_dir / "graph_export.json"))
    ap.add_argument("--model", choices=["sage", "gat"], default="sage")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    ap.add_argument("--cutoff-year", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--embedding-model", default=settings.embedding_model)
    ap.add_argument("--out", default=settings.gnn_model_path)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    if args.source == "neo4j":
        from backend.core.neo4j_client import Neo4jClient
        client = Neo4jClient().connect()
        graph = load_graph_from_neo4j(client)
        client.close()
    else:
        graph = load_graph_json(args.graph_json)
    logger.info("Graph: %d nodes, %d undirected edges", graph.n_nodes, graph.n_edges)

    from backend.core.embeddings import Embedder
    embedder = Embedder(args.embedding_model)
    x = build_node_features(graph.names, graph.types, embedder)
    split = split_edges(graph, args.split, args.cutoff_year, seed=args.seed)
    logger.info("Split (%s): train %d / val %d / test %d positive edges", split.mode,
                len(split.train_pos), len(split.val_pos), len(split.test_pos))

    model, metrics = train_model(
        graph, x, split, model_type=args.model, hidden=args.hidden, out_dim=args.out_dim, layers=args.layers,
        heads=args.heads, dropout=args.dropout, lr=args.lr, epochs=args.epochs, patience=args.patience,
        device=device, seed=args.seed,
    )
    out = save_checkpoint(model, metrics, args.out, embedding_model=args.embedding_model, split=split, graph=graph)
    logger.info("VAL  AUC %.4f AP %.4f", metrics["val"]["auc"], metrics["val"]["ap"])
    logger.info("TEST AUC %.4f AP %.4f", metrics["test"]["auc"], metrics["test"]["ap"])
    logger.info("Saved checkpoint -> %s", out)


if __name__ == "__main__":
    main()
