# app/core/exceptions.py
"""
Centralized exception handling for the RAG backend.

Design goals:
1. Every error returned to the client has a consistent JSON shape.
2. Internal details (stack traces, file paths, provider keys) never leak.
3. Every error is correlated with a request_id for log tracing.
4. Domain-specific exceptions map to the correct HTTP status code.
5. Unhandled exceptions are caught, logged with full traceback, and
   returned as a generic 500 — never exposing internals.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import get_settings
from app.core.logging import get_logger
logger = get_logger(__name__)



# ---------------------------------------------------------------------------
# 1. Correlation ID helper
# ---------------------------------------------------------------------------

def get_request_id(request: Request) -> str:
    """
    Retrieve or generate a request ID for correlation.
    If an upstream proxy (nginx, ALB, Cloudflare) sends X-Request-ID,
    we reuse it so tracing spans the whole request lifecycle.
    Otherwise, we generate a UUID4.
    """
    return request.headers.get("X-Request-ID") or str(uuid.uuid4())


# ---------------------------------------------------------------------------
# 2. Base application exception
# ---------------------------------------------------------------------------

class RAGException(Exception):
    """
    Base exception for all domain-level failures.

    Attributes:
        code:        Machine-readable error code (e.g., "DOC_NOT_FOUND").
        message:     Safe, human-readable message to return to the client.
        status_code: HTTP status code to use in the response.
        details:     Optional structured metadata (e.g., job_id, file).
        internal:    Optional internal exception/message for logs only.
                     NEVER returned to the client.
    """

    code: str = "INTERNAL_ERROR"
    message: str = "An internal error occurred."
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        internal: str | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.details = details or {}
        self.internal = internal
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# 3. Domain-specific exceptions
# ---------------------------------------------------------------------------

class ValidationError(RAGException):
    """Client sent a malformed or semantically invalid request."""
    code = "VALIDATION_ERROR"
    message = "The request is invalid."
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class DocumentNotFoundError(RAGException):
    """Requested document / chunk / job does not exist."""
    code = "DOCUMENT_NOT_FOUND"
    message = "The requested document was not found."
    status_code = status.HTTP_404_NOT_FOUND


class UnsupportedFileTypeError(RAGException):
    """Uploaded file has an extension we cannot process."""
    code = "UNSUPPORTED_FILE_TYPE"
    message = "The uploaded file type is not supported."
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


class FileTooLargeError(RAGException):
    """Uploaded file exceeds the configured max size."""
    code = "FILE_TOO_LARGE"
    message = "The uploaded file is too large."
    status_code = status.HTTP_413_CONTENT_TOO_LARGE


class IngestionError(RAGException):
    """Failure during document ingestion (parse/chunk/embed/store)."""
    code = "INGESTION_FAILED"
    message = "Document ingestion failed."
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class EmbeddingError(RAGException):
    """Embedding model failed to produce vectors."""
    code = "EMBEDDING_FAILED"
    message = "Failed to generate embeddings."
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class VectorStoreError(RAGException):
    """Qdrant (or other vector DB) failure."""
    code = "VECTOR_STORE_ERROR"
    message = "Vector store operation failed."
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class RetrievalError(RAGException):
    """Retrieval or reranking failure."""
    code = "RETRIEVAL_FAILED"
    message = "Failed to retrieve relevant documents."
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class LLMError(RAGException):
    """LLM provider failure (timeout, unavailable, bad response)."""
    code = "LLM_FAILED"
    message = "The language model is currently unavailable."
    status_code = status.HTTP_502_BAD_GATEWAY


class LLMTimeoutError(LLMError):
    """LLM took too long to respond."""
    code = "LLM_TIMEOUT"
    message = "The language model timed out."
    status_code = status.HTTP_504_GATEWAY_TIMEOUT


class CacheError(RAGException):
    """Redis / cache failure. Non-fatal — usually logged and ignored."""
    code = "CACHE_ERROR"
    message = "Cache operation failed."
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class RateLimitError(RAGException):
    """Client exceeded a rate limit."""
    code = "RATE_LIMITED"
    message = "Too many requests. Please slow down."
    status_code = status.HTTP_429_TOO_MANY_REQUESTS


# ---------------------------------------------------------------------------
# 4. Response envelope
# ---------------------------------------------------------------------------

def _error_envelope(
    request_id: str,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Uniform error payload shape.
    Frontend can rely on `error.code` for programmatic handling
    and `error.message` for user-facing display.
    """
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id,
        }
    }


# ---------------------------------------------------------------------------
# 5. Handlers
# ---------------------------------------------------------------------------

async def _handle_rag_exception(request: Request, exc: RAGException) -> JSONResponse:
    """
    Handle all our custom domain exceptions.
    Logs full context (including `internal`) but returns only safe fields.
    """
    request_id = get_request_id(request)

    logger.warning(
        "domain_exception",
        extra={
            "request_id": request_id,
            "code": exc.code,
            "path": str(request.url.path),
            "method": request.method,
            "details": exc.details,
            "internal": exc.internal,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=_error_envelope(
            request_id=request_id,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        ),
        headers={"X-Request-ID": request_id},
    )


async def _handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    FastAPI/Pydantic validation failures (bad JSON, missing fields, wrong types).
    We surface the field-level errors because they help developers fix the request,
    but we strip `ctx` which can contain non-serializable objects.
    """
    request_id = get_request_id(request)

    # Pydantic v2 error list — sanitize each entry
    sanitized_errors = []
    for err in exc.errors():
        sanitized_errors.append({
            "loc": list(err.get("loc", [])),
            "msg": err.get("msg", "invalid value"),
            "type": err.get("type", "value_error"),
        })

    logger.info(
        "validation_error",
        extra={
            "request_id": request_id,
            "path": str(request.url.path),
            "errors": sanitized_errors,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_error_envelope(
            request_id=request_id,
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            details={"fields": sanitized_errors},
        ),
        headers={"X-Request-ID": request_id},
    )


async def _handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """
    Handles FastAPI's HTTPException (raised manually via `raise HTTPException(...)`).
    We wrap it in our envelope for consistency with domain exceptions.
    """
    request_id = get_request_id(request)

    logger.info(
        "http_exception",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "detail": exc.detail,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=_error_envelope(
            request_id=request_id,
            code=f"HTTP_{exc.status_code}",
            message=str(exc.detail) if exc.detail else "Request failed.",
        ),
        headers={"X-Request-ID": request_id, **(exc.headers or {})},
    )


async def _handle_unhandled_exception(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all for anything not explicitly handled.
    This is the last line of defense — it MUST NOT leak internals.
    Full traceback is logged; client only sees a generic 500 with a request_id
    they can quote to support.
    """
    request_id = get_request_id(request)
    settings = get_settings()

    # `logger.exception` automatically includes the traceback
    logger.exception(
        "unhandled_exception",
        extra={
            "request_id": request_id,
            "path": str(request.url.path),
            "method": request.method,
            "exception_type": type(exc).__name__,
        },
    )

    # In DEBUG, expose the error type/message to speed up local development.
    # In production, return the safest possible message.
    if settings.DEBUG:
        message = f"{type(exc).__name__}: {exc}"
        details = {"debug": True}
    else:
        message = "An unexpected error occurred. Please contact support."
        details = {}

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_envelope(
            request_id=request_id,
            code="INTERNAL_ERROR",
            message=message,
            details=details,
        ),
        headers={"X-Request-ID": request_id},
    )


# ---------------------------------------------------------------------------
# 6. Registration
# ---------------------------------------------------------------------------

def register_exception_handlers(app: FastAPI) -> None:
    """
    Wire all handlers into the FastAPI app.

    Order matters conceptually, not literally — FastAPI dispatches by
    exception type hierarchy, so the most specific type wins.
    """
    app.add_exception_handler(RAGException, _handle_rag_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(Exception, _handle_unhandled_exception)