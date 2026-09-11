from pydantic_settings import BaseSettings,SettingsConfigDict
from functools import  lru_cache

class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- App ---
    APP_NAME: str = "RAG Backend"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # --- Qdrant ---
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "documents"

    # --- Redis (cache + job status) ---
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600          # 1 hour semantic cache

    # --- Embedding ---
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024              # BGE-M3 dense vector dimension
    EMBEDDING_BATCH_SIZE: int = 32

    # --- Chunking ---
    CHUNK_SIZE: int = 512                  # tokens
    CHUNK_OVERLAP: int = 64                # tokens

    # --- Retrieval ---
    TOP_K_RETRIEVAL: int = 20              # candidates from hybrid search
    TOP_K_RERANK: int = 5                  # final chunks after reranking
    BM25_WEIGHT: float = 0.3               # weight for sparse in fusion
    VECTOR_WEIGHT: float = 0.7             # weight for dense in fusion

    # --- Reranker ---
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM ---
    LLM_PROVIDER: str = "ollama"           # "ollama" | "openai_compatible"
    LLM_MODEL: str = "llama3.2"
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings — instantiated once, reused everywhere.
    lru_cache prevents re-reading .env on every request.
    """
    return Settings()