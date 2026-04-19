"""Structured request logging middleware.

Emits one INFO log line per HTTP request with fields suitable for
structured log aggregation (CloudWatch Insights, Loki, Datadog, ...).

Why this exists
---------------
Before this middleware, a failed request that never produced an HTTP
response — because the handler hung past an infrastructure-layer
timeout, because a worker crashed, because the client disconnected
mid-request — would leave NO trace in the gateway logs.  Operators
investigating "my request hung" had no record of the request even
arriving, its duration, status, or model.  The ``CONN_CLOSED`` and
``EXCEPTION:<Type>`` status markers below are specifically designed
to surface those silent-failure modes.

Each record carries
-------------------
    path        request path (e.g. /api/v1/chat/completions)
    method      HTTP method (GET, POST, ...)
    status      HTTP status code, or the sentinel values:
                  ``CONN_CLOSED``         — handler was cancelled
                                             without producing a response
                                             (client disconnect / server
                                             shutdown / infrastructure
                                             TCP reset)
                  ``EXCEPTION:<TypeName>`` — an uncaught exception
                                             propagated past the handler
                                             into the middleware
    elapsed_s   wall-clock duration in seconds (3 decimal places)
    req_bytes   Content-Length header value from the request
    model       the ``model`` field extracted from the JSON request
                body where possible (best-effort; ``-`` when missing)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)


# Paths excluded from request logging — high-volume probe endpoints
# that carry no useful signal.  Kept as a frozenset so callers can't
# mutate it at runtime by accident.
_QUIET_PATHS = frozenset({"/health", "/metrics"})

# Only peek inside bodies up to this size (bytes) when extracting the
# ``model`` field.  The proxy accepts very large request bodies (100+KB
# prompts), but we only care about the small top-level JSON object that
# carries the model ID.  Limiting what we parse caps the worst-case cost
# of a badly-formed giant JSON attempt.
_MODEL_PEEK_MAX_BYTES = 64 * 1024  # 64 KB


def _peek_model_from_body(body: bytes) -> str | None:
    """Extract the top-level ``model`` string from a JSON request body.

    Never raises.  Returns None if the body isn't JSON, isn't an object,
    doesn't contain a ``model`` key, or exceeds the peek size cap.
    """
    if not body or len(body) > _MODEL_PEEK_MAX_BYTES:
        return None
    try:
        parsed: Any = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(parsed, dict):
        value = parsed.get("model")
        if isinstance(value, str):
            return value
    return None


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Emit one structured log record per HTTP request.

    Logs at INFO regardless of status code so operators see every call.
    Connection-level failures that prevent response construction are
    captured in a ``finally`` block and logged with sentinel statuses,
    ensuring no request is ever silently lost.

    Body-reading technique
    ----------------------
    To expose the ``model`` field in the log line we must read the
    request body BEFORE forwarding to the inner app.  Starlette's
    ``BaseHTTPMiddleware`` creates a fresh Request inside ``call_next``,
    so merely reading ``request._body`` after ``call_next`` returns
    gives nothing.  We instead:

      1. Drain the body here.
      2. Replace ``request._receive`` with a callable that replays the
         drained bytes plus a terminal ``more_body=False`` marker.
      3. Let FastAPI read the body via its own Request — which uses the
         replayed receive and sees the original bytes unchanged.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path in _QUIET_PATHS:
            return await call_next(request)

        body_bytes = await _drain_and_replay_body(request)

        start = time.perf_counter()
        response: Response | None = None
        exc: BaseException | None = None
        try:
            response = await call_next(request)
            return response
        except BaseException as e:
            exc = e
            raise
        finally:
            elapsed = time.perf_counter() - start
            if response is not None:
                status: str = str(response.status_code)
            elif exc is not None:
                status = f"EXCEPTION:{type(exc).__name__}"
            else:
                # Reached when the request coroutine was cancelled
                # before the handler produced a Response (client
                # disconnect, server shutdown, asyncio.CancelledError).
                status = "CONN_CLOSED"

            try:
                req_bytes = int(request.headers.get("content-length", "0"))
            except ValueError:
                req_bytes = 0
            if req_bytes == 0 and body_bytes:
                # Some clients omit Content-Length but send a body anyway
                # (chunked transfer encoding); fall back to the drained size.
                req_bytes = len(body_bytes)

            logger.info(
                "request path=%s method=%s status=%s elapsed_s=%.3f req_bytes=%d model=%s",
                request.url.path,
                request.method,
                status,
                elapsed,
                req_bytes,
                _peek_model_from_body(body_bytes) or "-",
            )


async def _drain_and_replay_body(request: Request) -> bytes:
    """Read the full request body, then replace the request's receive
    callable so the inner app sees the body unchanged.

    Returns the drained body bytes (possibly empty).
    """
    body = await request.body()

    replayed = {"done": False}

    async def replay_receive() -> dict:
        if replayed["done"]:
            # Mimic the ASGI protocol for "no more body" — the inner
            # app should never hit this because it reads the body once.
            return {"type": "http.disconnect"}
        replayed["done"] = True
        return {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }

    # Swap the receive callable on the outer Request so FastAPI's
    # internal Request (constructed from the same scope + receive in
    # BaseHTTPMiddleware.call_next) uses the replayed body.
    request._receive = replay_receive  # type: ignore[attr-defined]
    return body


def install_request_log_middleware(app: FastAPI) -> None:
    """Register the request-log middleware on a FastAPI app."""
    app.add_middleware(RequestLogMiddleware)
