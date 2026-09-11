from typing import List
from rank_bm25 import BM25Okapi
from app.core.vector_store import VectorStore
from app.core.embedder import Embedder
from app.core.config import get_settings

class HybridRetriever:

    """
    Combines sparse (BM25) and dense (vector) retrieval using
    weighted score fusion. Why hybrid?
    - BM25 excels at exact keyword matches (names, codes, jargon)
    - Vector search excels at semantic similarity
    - Fusion improves recall and robustness for diverse queries
    """

    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.settings = get_settings()
        self._bm25: BM25Okapi | None = None
        self._corpus: List[str] = []
        self._corpus_meta: List[dict] = []

    def build_bm25_index(self, documents: List[dict]):
        """
        Build a BM25 index from the full corpus.
        In production, you'd persist this or rebuild incrementally.
        For simplicity, we build in-memory on startup.
        """
        self._corpus = [doc["text"] for doc in documents]
        self._corpus_meta = documents
        tokenized = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int | None = None) -> List[dict]:
        """
        Run hybrid retrieval and fuse scores.
        Steps:
        1. Dense: embed query, search Qdrant
        2. Sparse: tokenize query, score BM25 corpus
        3. Fuse: weighted sum of normalized scores
        """
        k = top_k or self.settings.TOP_K_RETRIEVAL
        vw = self.settings.VECTOR_WEIGHT
        bw = self.settings.BM25_WEIGHT

        # --- Dense retrieval ---
        query_vec = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_vec, top_k=k)

        # --- Sparse retrieval (BM25) ---
        sparse_results = []
        if self._bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokenized_query)
            # Get top-k indices
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:k]

            for idx in top_indices:
                if bm25_scores[idx] > 0:  # ignore zero-score docs
                    sparse_results.append({
                        "text": self._corpus[idx],
                        **self._corpus_meta[idx],
                        "score": float(bm25_scores[idx]),
                    })

        # --- Score fusion ---
        # Normalize scores to [0,1] before weighting
        def _normalize(results, key="score"):
            if not results:
                return results
            scores = [r[key] for r in results]
            mn, mx = min(scores), max(scores)
            span = mx - mn if mx != mn else 1.0
            for r in results:
                r[key] = (r[key] - mn) / span
            return results

        dense_results = _normalize(dense_results)
        sparse_results = _normalize(sparse_results)

        # Build fused score dict keyed by (source, page, chunk_index)
        fused: dict[tuple, dict] = {}
        for r in dense_results:
            key = (r.get("source"), r.get("page"), r.get("chunk_index"))
            fused[key] = {**r, "fused_score": vw * r["score"]}
        for r in sparse_results:
            key = (r.get("source"), r.get("page"), r.get("chunk_index"))
            if key in fused:
                fused[key]["fused_score"] += bw * r["score"]
            else:
                fused[key] = {**r, "fused_score": bw * r["score"]}

        # Sort by fused score, return top-k
        ranked = sorted(fused.values(), key=lambda x: x["fused_score"], reverse=True)
        return ranked[:k]