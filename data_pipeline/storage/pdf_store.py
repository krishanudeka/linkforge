"""PDF / Markdown file manager: local disk by default, optional Cloudflare R2 mirror (S3 API)."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from backend.config import settings

logger = logging.getLogger("PDFStore")


def safe_name(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id).strip("_")[:120] or "paper"


class PDFStore:
    def __init__(self, pdf_dir: Optional[Path] = None, markdown_dir: Optional[Path] = None):
        self.pdf_dir = Path(pdf_dir or settings.pdf_dir)
        self.markdown_dir = Path(markdown_dir or settings.markdown_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self._s3 = None
        if settings.r2_endpoint_url and settings.r2_bucket:
            try:
                import boto3  # optional dependency: pip install boto3

                self._s3 = boto3.client(
                    "s3", endpoint_url=settings.r2_endpoint_url,
                    aws_access_key_id=settings.r2_access_key_id,
                    aws_secret_access_key=settings.r2_secret_access_key,
                )
            except ImportError:
                logger.warning("R2 configured but boto3 is not installed; using local disk only.")

    # ---- paths -----------------------------------------------------------
    def pdf_path(self, paper_id: str) -> Path:
        return self.pdf_dir / f"{safe_name(paper_id)}.pdf"

    def markdown_path(self, paper_id: str) -> Path:
        return self.markdown_dir / f"{safe_name(paper_id)}.md"

    # ---- pdf -------------------------------------------------------------
    def save_pdf(self, paper_id: str, content: bytes) -> Path:
        if not content.startswith(b"%PDF"):
            raise ValueError("Uploaded content is not a valid PDF file")
        path = self.pdf_path(paper_id)
        path.write_bytes(content)
        self._mirror(path, f"pdfs/{path.name}")
        return path

    def has_pdf(self, paper_id: str) -> bool:
        return self.pdf_path(paper_id).exists()

    def list_pdfs(self) -> List[Path]:
        return sorted(self.pdf_dir.glob("*.pdf"))

    # ---- markdown --------------------------------------------------------
    def save_markdown(self, paper_id: str, text: str) -> Path:
        path = self.markdown_path(paper_id)
        path.write_text(text, encoding="utf-8")
        return path

    def read_markdown(self, paper_id: str) -> Optional[str]:
        path = self.markdown_path(paper_id)
        return path.read_text(encoding="utf-8") if path.exists() else None

    # ---- optional R2 -----------------------------------------------------
    def _mirror(self, path: Path, key: str) -> None:
        if self._s3 is None:
            return
        try:
            self._s3.upload_file(str(path), settings.r2_bucket, key)
        except Exception as exc:  # never fail ingestion because the mirror is down
            logger.warning("R2 upload failed for %s: %s", key, exc)
