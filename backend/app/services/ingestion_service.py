# app/services/ingestion_service.py
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from app.core.loader import DocumentLoader
from app.core.chunker import Chunker
from app.core.embedder import Embedder
from app.core.vector_store import VectorStore
from app.core.logging import get_logger
logger=get_logger(__name__)


class IngestionService:
    """
    Orchestrates the full ingestion pipeline:
    Load → Chunk → Embed → Store.
    Each step is a separate class, making it testable and swappable.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.loader = DocumentLoader()
        self.chunker = Chunker()
        self.embedder = embedder
        self.vector_store = vector_store

    async def ingest_file(self, file_path: Path) -> dict:
        """
        Process a single file through the full pipeline.
        Returns metadata about the ingestion (chunk count, source).
        """
        logger.info(f"Ingesting file: {file_path}")

        # 1. Load
        documents = self.loader.load(file_path)
        if not documents:
            raise ValueError(f"No text extracted from {file_path}")

        # 2. Chunk
        chunks = self.chunker.split(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        # 3. Embed
        texts = [c.page_content for c in chunks]
        embeddings = self.embedder.embed_documents(texts)

        # 4. Store
        self.vector_store.upsert_documents(chunks, embeddings.tolist())

        return {
            "source": str(file_path),
            "chunks": len(chunks),
            "status": "completed",
        }