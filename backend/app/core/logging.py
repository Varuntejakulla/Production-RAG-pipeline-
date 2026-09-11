# app/core/logging.py
"""
Structured logging for the RAG backend.

What this gives you over `basicConfig`:
- A `request_id` attached to every log line (via ContextVar), so you can
  correlate a client-reported ID with server logs end-to-end.
- Pretty, human-readable output in DEBUG; JSON-lines in production for
  ingestion by Loki / ELK / Datadog / CloudWatch.
- Per-module loggers via `get_logger(__name__)`.
- Third-party noise suppression so your signals aren't drowned out.
- Idempotent `setup_logging()` so hot-reload and tests don't stack handlers.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any


# ---------------------------------------------------------------------------
# 1. Request ID propagation
# ---------------------------------------------------------------------------
# A ContextVar is async-safe: each request's task gets its own isolated value,
# so concurrent requests never clobber each other's IDs. Callers never need to
# pass request_id around manually.
# ---------------------------------------------------------------------------
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> None:
    """Bind the current request's ID. Called from middleware / exception handlers."""
    _request_id_ctx.set(request_id)


def get_current_request_id() -> str:
    """Read the current request's ID."""
    return _request_id_ctx.get()


# ---------------------------------------------------------------------------
# 2. Filter: inject `request_id` into every LogRecord automatically
# ---------------------------------------------------------------------------
class RequestIdFilter(logging.Filter):
    """
    Guarantees every record has a `request_id` attribute, pulled from the
    ContextVar unless the caller explicitly passed one via `extra=`.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = _request_id_ctx.get()
        return True


# ---------------------------------------------------------------------------
# 3. Formatters
# ---------------------------------------------------------------------------
class DevFormatter(logging.Formatter):
    """Human-readable format for local development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(request_id)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class JSONFormatter(logging.Formatter):
    """
    One JSON object per line. Any `extra=` fields on the log call are
    merged into the payload, with non-serializable values falling back
    to `repr()` so a bad extra never crashes a log call.
    """

    # Attributes that belong to the LogRecord itself — never re-emit these.
    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "asctime", "taskName", "request_id",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                + f".{int(record.msecs):03d}Z"
            ),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 4. Public entry point
# ---------------------------------------------------------------------------
_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger once.

    - Uses JSON output in production (DEBUG=False).
    - Uses pretty output in development (DEBUG=True).
    - Silences noisy third-party libraries.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Imported here to avoid a circular import at module load.
    from app.core.config import get_settings
    settings = get_settings()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(DevFormatter() if settings.DEBUG else JSONFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Quiet the chatty libraries so our logs stay signal-rich.
    for noisy in (
        "httpx", "httpcore", "urllib3",
        "qdrant_client",
        "sentence_transformers", "transformers", "huggingface_hub",
        "uvicorn.access",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Let uvicorn's own logs flow through our root handler.
    for uvi in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(uvi)
        lg.handlers.clear()
        lg.propagate = True

    _CONFIGURED = True


# ---------------------------------------------------------------------------
# 5. Logger factory
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Preferred acquisition in every module:

        from app.core.logging import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)