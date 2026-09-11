# app/api/dependencies.py
from functools import lru_cache
from app.core.embedder import Embedder
from app.core.vector_store import VectorStore
from app.core.retriever import HybridRetriever
from app.core.reranker import Reranker
from app.core.llm_client import LLMClient
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.services.cache_service import CacheService


# Singleton instances — created once, shared across requests
_embedder: Embedder | None = None
_vector_store: VectorStore | None = None
_retriever: HybridRetriever | None = None
_reranker: Reranker | None = None
_llm: LLMClient | None = None
_cache: CacheService | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        _vector_store.ensure_collection()
    return _vector_store


def get_retrieval_service() -> RetrievalService:
    global _retriever, _reranker
    if _retriever is None:
        _retriever = HybridRetriever(get_vector_store(), get_embedder())
        # In production, build BM25 index from stored documents
        # (or use a persistent BM25 index)
    if _reranker is None:
        _reranker = Reranker()
    return RetrievalService(_retriever, _reranker)


def get_generation_service() -> GenerationService:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return GenerationService(_llm)


def get_cache_service() -> CacheService:
    global _cache
    if _cache is None:
        _cache = CacheService()
    return _cache


def get_ingestion_service() -> IngestionService:
    return IngestionService(get_embedder(), get_vector_store())