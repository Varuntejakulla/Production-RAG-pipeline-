# app/services/retrieval_service.py
from typing import List
from app.core.retriever import HybridRetriever
from app.core.reranker import Reranker


class RetrievalService:
    """
    Orchestrates retrieval + reranking.
    """
    def __init__(self, retriever: HybridRetriever, reranker: Reranker):
        self.retriever = retriever
        self.reranker = reranker

    async def retrieve(self, query: str) -> List[dict]:
        candidates = self.retriever.retrieve(query)
        reranked = self.reranker.rerank(query, candidates)
        return reranked