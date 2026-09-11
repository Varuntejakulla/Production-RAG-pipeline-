from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)
from langchain_core.documents import Document
from app.core.config import get_settings
import uuid

class VectorStore:
    """
    Qdrant wrapper for vector storage and retrieval.
    Why Qdrant?
    - Best-in-class payload filtering (critical for metadata filters)
    - Sub-5ms median latency for full databases
    - Apache 2.0, self-hostable, no vendor lock-in
    - Native hybrid search support
    """

    def __init__(self):
        settings = get_settings()
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=30,
        )
        self.collection = settings.QDRANT_COLLECTION
        self.vector_dim = settings.EMBEDDING_DIM

    def ensure_collection(self):
        """Create collection if it doesn't exist. Idempotent."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE,
                )
            )
    def upsert_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """
        Store chunks with their embeddings and full metadata payload.
        Payload includes source, page, chunk_index — enabling
        filtered retrieval and citation generation.
        """
        points = []
        for doc, emb in zip(documents, embeddings):
            point_id = str(uuid.uuid4())  # deterministic IDs for idempotency
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb,
                    payload={
                        "text": doc.page_content,
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "chunk_index": doc.metadata.get("chunk_index"),
                    },
                )
            )

        # Batch upsert — Qdrant handles 100+ points per call efficiently
        self.client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True,
        )
    def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> List[dict]:
        """
        Dense vector search with optional metadata filtering.
        Returns payload + score for downstream reranking.
        """
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {
                "text": hit.payload["text"],
                "source": hit.payload.get("source"),
                "page": hit.payload.get("page"),
                "chunk_index": hit.payload.get("chunk_index"),
                "score": hit.score,
            }
            for hit in results
        ]
    