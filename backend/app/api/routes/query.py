# app/api/routes/query.py
"""
Query endpoints — the read path of the RAG pipeline.

Flow:
    cache lookup → hybrid retrieval → cross-encoder rerank → LLM → cache write

Two endpoints:
    POST /query          — synchronous JSON response with answer + sources
    POST /query/stream   — Server-Sent Events streaming of the answer tokens

Both share the same retrieval + generation services; only the delivery differs.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    get_cache_service,
    get_generation_service,
    get_retrieval_service,
)
from app.core.exceptions import CacheError, LLMError, RetrievalError
from app.core.logging import get_logger, get_current_request_id
from app.models.schemas import QueryRequest, QueryResponse, SourceInfo

router = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_sources(chunks: list[dict]) -> list[SourceInfo]:
    """
    Convert raw chunk dicts into the public SourceInfo model.

    Only expose source + page (safe, useful for citations).
    Internal fields like `score`, `rerank_score`, `chunk_index` stay private.
    """
    return [
        SourceInfo(source=c.get("source"), page=c.get("page"))
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@router.post("", response_model=QueryResponse, summary="Ask a question")
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline.

    Steps:
      1. Cache lookup — return immediately on hit.
      2. Retrieve candidates via hybrid search.
      3. Rerank candidates with a cross-encoder.
      4. Generate an answer grounded in the top-K chunks.
      5. Cache the answer for future identical/near-identical queries.
    """
    request_id = get_current_request_id()
    cache = get_cache_service()

    # --- 1. Cache lookup (non-fatal if Redis is down) ---
    try:
        cached = await cache.get(request.query)
    except CacheError as e:
        # Cache failure must never break the request — degrade to a miss.
        logger.warning(
            "cache_lookup_failed",
            extra={"internal": e.internal, "query": request.query},
        )
        cached = None

    if cached:
        logger.info(
            "query_cache_hit",
            extra={"query_len": len(request.query)},
        )
        return QueryResponse(
            answer=cached["answer"],
            sources=[SourceInfo(**s) for s in cached.get("sources", [])],
            cached=True,
        )

    # --- 2 & 3. Retrieve + rerank ---
    try:
        retriever = get_retrieval_service()
        chunks = await retriever.retrieve(request.query)
    except RetrievalError:
        raise  # already a domain exception with the right status code
    except Exception as e:
        # Wrap unexpected retrieval failures so clients see a clean message
        raise RetrievalError(
            internal=f"{type(e).__name__}: {e}",
            details={"request_id": request_id},
        ) from e

    # --- Empty retrieval: short-circuit with an honest abstention ---
    if not chunks:
        logger.info(
            "query_no_chunks",
            extra={"query_len": len(request.query)},
        )
        return QueryResponse(
            answer="I don't have enough information to answer this question.",
            sources=[],
            cached=False,
        )

    # --- 4. Generate answer ---
    try:
        generator = get_generation_service()
        answer = await generator.generate(request.query, chunks)
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(
            internal=f"{type(e).__name__}: {e}",
            details={"request_id": request_id},
        ) from e

    # --- 5. Cache the result ---
    sources = _to_sources(chunks)
    try:
        await cache.set(
            request.query,
            answer,
            [s.model_dump() for s in sources],
        )
    except CacheError as e:
        # A cache-write failure must never affect the response
        logger.warning(
            "cache_write_failed",
            extra={"internal": e.internal},
        )

    logger.info(
        "query_completed",
        extra={
            "query_len": len(request.query),
            "num_sources": len(sources),
            "answer_len": len(answer),
        },
    )

    return QueryResponse(answer=answer, sources=sources, cached=False)


# ---------------------------------------------------------------------------
# POST /query/stream
# ---------------------------------------------------------------------------

@router.post("/stream", summary="Ask a question (streaming SSE)")
async def query_stream(request: QueryRequest, http_request: Request) -> StreamingResponse:
    """
    Streaming variant of /query.

    Retrieval and reranking happen synchronously (they're fast), then the
    LLM answer streams token-by-token over Server-Sent Events.

    SSE frame format:
        data: <token>\\n\\n
        ...
        data: [DONE]\\n\\n

    Each frame's `data:` line carries raw text. If the LLM emits a newline,
    SSE would break the frame — so we JSON-encode each token. The client
    does `JSON.parse(data)` to recover the exact text.
    """
    import json

    request_id = get_current_request_id()
    logger.info("query_stream_started", extra={"query_len": len(request.query)})

    # --- Retrieve + rerank before opening the stream ---
    retriever = get_retrieval_service()
    chunks = await retriever.retrieve(request.query)

    # --- Empty retrieval: single short SSE frame and done ---
    if not chunks:
        async def _empty():
            payload = json.dumps(
                {"type": "answer",
                 "text": "I don't have enough information to answer this question."}
            )
            yield f"data: {payload}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            _empty(),
            media_type="text/event-stream",
            headers={"X-Request-ID": request_id, "Cache-Control": "no-cache"},
        )

    # --- Source metadata: send upfront so the UI can render citations early ---
    sources = _to_sources(chunks)

    generator = get_generation_service()

    async def event_stream():
        # 1) Announce the sources
        meta = {"type": "sources", "sources": [s.model_dump() for s in sources]}
        yield f"data: {json.dumps(meta)}\n\n"

        # 2) Stream the tokens
        collected: list[str] = []
        try:
            async for token in generator.generate_stream(request.query, chunks):
                # Client-side disconnect → stop early to save LLM cycles
                if await http_request.is_disconnected():
                    logger.info("query_stream_client_disconnected")
                    return
                collected.append(token)
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        except LLMError as e:
            logger.warning("query_stream_llm_error", extra={"internal": e.internal})
            err = {"type": "error", "code": e.code, "message": e.message}
            yield f"data: {json.dumps(err)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # 3) Optionally cache the full answer for future non-streaming hits
        try:
            cache = get_cache_service()
            await cache.set(
                request.query,
                "".join(collected),
                [s.model_dump() for s in sources],
            )
        except CacheError as e:
            logger.warning("cache_write_failed", extra={"internal": e.internal})

        logger.info(
            "query_stream_completed",
            extra={"answer_len": sum(len(t) for t in collected)},
        )

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "X-Request-ID": request_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )