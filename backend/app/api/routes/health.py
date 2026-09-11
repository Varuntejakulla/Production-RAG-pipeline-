# app/api/routes/health.py
"""
Health & readiness probes.

- /health          — liveness: is the process up? (fast, no deps)
- /health/ready    — readiness: can we serve real traffic?
                     Checks Qdrant + Redis reachability.
- /health/startup  — startup probe for Kubernetes slow-start containers.

Liveness must NEVER touch external services — if Qdrant is down, the
process is still alive and should not be restarted by the orchestrator.
Only readiness should fail in that case, so traffic is routed elsewhere.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Response, status

from app.core.config import get_settings
from app.core.logging import get_logger

router = APIRouter(prefix="/health", tags=["health"])
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared probe helpers
# ---------------------------------------------------------------------------

async def _probe_qdrant(timeout: float = 2.0) -> dict[str, Any]:
    """
    Run a lightweight Qdrant call inside a thread with a hard timeout.

    `get_collections()` is cheap, authoritative, and doesn't require
    the target collection to exist yet (so it works on first boot).
    """
    def _sync() -> dict[str, Any]:
        from app.core.vector_store import VectorStore
        vs = VectorStore()
        collections = vs.client.get_collections()
        return {"ok": True, "collections": len(collections.collections)}

    try:
        return await asyncio.wait_for(asyncio.to_thread(_sync), timeout=timeout)
    except asyncio.TimeoutError:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


async def _probe_redis(timeout: float = 2.0) -> dict[str, Any]:
    """Ping Redis with a hard timeout."""
    from app.services.cache_service import CacheService

    cache = CacheService()
    try:
        pong = await asyncio.wait_for(cache.redis.ping(), timeout=timeout)
        return {"ok": bool(pong), "response": pong}
    except asyncio.TimeoutError:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        await cache.close()


# ---------------------------------------------------------------------------
# Liveness — process is up
# ---------------------------------------------------------------------------

@router.get("", summary="Liveness probe")
async def liveness() -> dict[str, Any]:
    """
    Cheap liveness check. Never touches external services.
    K8s uses this to decide whether to restart the container.
    """
    settings = get_settings()
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "timestamp": int(time.time()),
    }


# ---------------------------------------------------------------------------
# Readiness — can we serve real traffic?
# ---------------------------------------------------------------------------

@router.get("/ready", summary="Readiness probe")
async def readiness(response: Response) -> dict[str, Any]:
    """
    Deep readiness: verifies downstream dependencies.

    Returns 200 if all critical deps are reachable; 503 otherwise.
    Response body carries per-dependency status so dashboards can pinpoint
    which dependency is failing.
    """
    settings = get_settings()

    # Run both probes in parallel — halve worst-case latency
    qdrant_task = asyncio.create_task(_probe_qdrant())
    redis_task = asyncio.create_task(_probe_redis())
    qdrant_res, redis_res = await asyncio.gather(qdrant_task, redis_task)

    # Qdrant is required for both ingest and query → hard failure
    # Redis is a cache → soft dependency; readiness still passes if it's down
    ready = qdrant_res["ok"]

    payload = {
        "status": "ready" if ready else "not_ready",
        "service": settings.APP_NAME,
        "checks": {
            "qdrant": qdrant_res,
            "redis": redis_res,
        },
        "timestamp": int(time.time()),
    }

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("readiness_failed", extra={"checks": payload["checks"]})

    return payload


# ---------------------------------------------------------------------------
# Startup probe — for slow-starting containers
# ---------------------------------------------------------------------------

@router.get("/startup", summary="Startup probe")
async def startup() -> dict[str, Any]:
    """
    Kubernetes startup probe.

    The lifespan has already loaded the embedding model and ensured the
    Qdrant collection exists by the time this is reachable, so returning
    `started: true` is sufficient. Kept separate from readiness so slow
    model loads don't cause K8s to prematurely restart the pod.
    """
    return {"status": "started", "timestamp": int(time.time())}