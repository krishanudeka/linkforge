"""Export the graph as JSON for GNN training on Colab/Kaggle.   python -m scripts.export_graph"""
import argparse

from ai_engine.gnn.dataset import export_graph_json, load_graph_from_neo4j
from backend.config import settings
from backend.core.neo4j_client import Neo4jClient


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(settings.export_dir / "graph_export.json"))
    args = ap.parse_args(argv)
    client = Neo4jClient().connect()
    g = load_graph_from_neo4j(client)
    path = export_graph_json(g, args.out)
    print(f"Exported {g.n_nodes} nodes / {g.n_edges} undirected edges -> {path}")
    print("Upload it to Colab, then:  python -m ai_engine.gnn.train --source json --graph-json graph_export.json "
          "--split temporal")
    client.close()


if __name__ == "__main__":
    main()
