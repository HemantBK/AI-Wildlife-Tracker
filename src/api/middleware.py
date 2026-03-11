"""
API Middleware
Request logging, rate limiting, request ID injection, and timing.
"""

import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ─── Request Logging Middleware ──────────────────────────────────


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with timing and status info.
    Injects X-Request-ID header into every response.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = request_id

        # Time the request
        start_time = time.time()
        method = request.method
        path = request.url.path

        logger.info(f"[{request_id}] {method} {path} - Started")

        try:
            response = await call_next(request)
            elapsed = time.time() - start_time

            logger.info(
                f"[{request_id}] {method} {path} - "
                f"Status: {response.status_code} - "
                f"Duration: {elapsed:.3f}s"
            )

            # Inject headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{elapsed:.3f}s"

            return response

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{request_id}] {method} {path} - Error: {str(e)} - Duration: {elapsed:.3f}s"
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(e),
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )


# ─── Rate Limiting Middleware ────────────────────────────────────


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter.
    Tracks requests per IP per minute window.

    Config:
        max_requests: Maximum requests per window (default: 10)
        window_seconds: Time window in seconds (default: 60)
    """

    def __init__(self, app, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # {ip: [timestamp, timestamp, ...]}
        self.request_log: dict[str, list[float]] = defaultdict(list)

    def _clean_old_entries(self, ip: str, now: float):
        """Remove entries outside the current window."""
        cutoff = now - self.window_seconds
        self.request_log[ip] = [ts for ts in self.request_log[ip] if ts > cutoff]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries and check limit
        self._clean_old_entries(client_ip, now)

        if len(self.request_log[client_ip]) >= self.max_requests:
            remaining_time = self.window_seconds - (now - self.request_log[client_ip][0])
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.max_requests} requests per {self.window_seconds}s. "
                    f"Try again in {remaining_time:.0f}s.",
                },
                headers={
                    "Retry-After": str(int(remaining_time)),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Record this request
        self.request_log[client_ip].append(now)
        remaining = self.max_requests - len(self.request_log[client_ip])

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


# ─── Timeout Middleware ──────────────────────────────────────────


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Enforces a maximum request duration.
    Uses asyncio timeout to prevent long-running requests.
    """

    def __init__(self, app, timeout_seconds: int = 30):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import asyncio

        try:
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds,
            )
            return response
        except asyncio.TimeoutError:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.error(f"[{request_id}] Request timed out after {self.timeout_seconds}s")
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Request timeout",
                    "detail": f"Request exceeded {self.timeout_seconds}s limit.",
                    "request_id": request_id,
                },
            )
