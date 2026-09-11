# app/core/reranker.py
from typing import List
from sentence_transformers import CrossEncoder
from app.core.config import get_settings


class Reranker:
    """
    Cross-encoder reranking of retrieved candidates.
    Why rerank?
    - Bi-encoder retrieval (embedding similarity) is fast but coarse
    - Cross-encoder reads query+document together, producing
      much more accurate relevance scores
    - Applied only to top-K candidates, so latency is acceptable
    """

    def __init__(self):
        settings = get_settings()
        self.model = CrossEncoder(
            settings.RERANKER_MODEL,
            max_length=512,
        )
        self.top_k = settings.TOP_K_RERANK

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        """
        Score each (query, document) pair with the cross-encoder,
        then return the top-k highest-scoring chunks.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[: self.top_k]