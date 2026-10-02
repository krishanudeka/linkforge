"""Scrape + extract + store from the command line.

  python -m scripts.run_pipeline --query "curcumin alzheimer" --source europe_pmc --limit 10
  python -m scripts.run_pipeline --pdf-dir storage_data/pdfs      # (re)process local PDFs
"""
import argparse
import json
import logging
from pathlib import Path

from backend.core.pipeline import IngestionPipeline
from backend.core.services import get_services


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", action="append", help="search query (repeatable)")
    ap.add_argument("--source", default="europe_pmc", choices=["europe_pmc", "pubmed", "semantic_scholar"])
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--no-download", action="store_true", help="use abstracts only (no PDFs)")
    ap.add_argument("--pdf-dir", help="process every *.pdf in this folder")
    ap.add_argument("--force", action="store_true", help="re-process already processed papers")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.query and not args.pdf_dir:
        ap.error("give --query and/or --pdf-dir")

    pipe = IngestionPipeline(get_services())
    for q in args.query or []:
        res = pipe.ingest_query(q, args.source, args.limit, download=not args.no_download, force=args.force, progress=print)
        print(json.dumps({k: res[k] for k in ("query", "papers", "by_status", "triplets")}, indent=2))
    if args.pdf_dir:
        for pdf in sorted(Path(args.pdf_dir).glob("*.pdf")):
            r = pipe.process_paper({"paper_id": f"file:{pdf.stem}", "title": pdf.stem.replace("_", " "),
                                    "source": "local"}, pdf_path=str(pdf), force=args.force, progress=print)
            print(r["paper_id"], r["status"], r["triplets"])
    print("Graph:", get_services().neo4j.stats())


if __name__ == "__main__":
    main()
