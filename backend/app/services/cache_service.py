# app/services/cache_service.py
import hashlib
import json
import redis.asyncio as redis
from app.core.config import get_settings


class CacheService:
    """
    Redis-backed semantic cache for query answers.
    Why cache?
    - Repeated/near-duplicate queries are common in production
    - Caching eliminates redundant retrieval + LLM calls
    - Redis provides sub-millisecond lookups
    """

    def __init__(self):
        settings = get_settings()
        self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.ttl = settings.CACHE_TTL_SECONDS

    @staticmethod
    def _cache_key(query: str) -> str:
        """Normalize query and hash for consistent key generation."""
        normalized = query.strip().lower()
        return f"rag:query:{hashlib.sha256(normalized.encode()).hexdigest()}"

    async def get(self, query: str) -> dict | None:
        """Return cached answer if it exists."""
        key = self._cache_key(query)
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set(self, query: str, answer: str, sources: list[dict]):
        """Cache answer with source metadata."""
        key = self._cache_key(query)
        payload = json.dumps({
            "answer": answer,
            "sources": sources,
        })
        await self.redis.setex(key, self.ttl, payload)

    async def close(self):
        await self.redis.close()