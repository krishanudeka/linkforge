"""Create Neo4j constraints / indexes and storage folders.   python -m scripts.init_db"""
from backend.config import settings
from backend.core.neo4j_client import Neo4jClient


def main():
    for d in (settings.pdf_dir, settings.markdown_dir, settings.chroma_dir, settings.export_dir):
        d.mkdir(parents=True, exist_ok=True)
    client = Neo4jClient().connect()
    print("Neo4j OK:", client.stats())
    client.close()


if __name__ == "__main__":
    main()
