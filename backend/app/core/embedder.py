from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

class Embedder:
    """
    Wraps BGE-M3 — a multilingual, multi-granularity embedding model.
    Why BGE-M3?
    - Supports 100+ languages (production-ready for global documents)
    - 8192 token context window (handles long chunks)
    - 1024-dim dense vectors (good quality/cost balance)
    - Apache 2.0 license (fully open-source, self-hostable)
    """

    def __init__(self):
        settings = get_settings()
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device="cpu",  # switch to "cuda" for GPU acceleration
        )
        self.batch_size = settings.EMBEDDING_BATCH_SIZE

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of document chunks.
        Why batch: GPU utilization is far higher with batches of 32-64.
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # cosine similarity = dot product
            show_progress_bar=False,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query. BGE-M3 does NOT require instruction prefixes
        (unlike some older BGE models), simplifying the query path.
        """
        return self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
    

    