"""Gated HTTP endpoint to mint access keys (bootstrap secret + rate limit)."""
from __future__ import annotations

import hashlib
import hmac
import json
from collections import defaultdict, deque
from time import monotonic
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from src.auth.access_keys import issue_access_key
from src.config import get_logger

logger = get_logger(__name__)


class AdminRateLimiter:
    """Sliding 60s window per client IP."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_ip: str) -> bool:
        now = monotonic()
        q = self._hits[client_ip]
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= self._max:
            return False
        q.append(now)
        return True


def _secrets_equal(received: str, expected: str) -> bool:
    """Length-independent comparison via SHA-256 digests."""
    ra = hashlib.sha256(received.encode("utf-8")).digest()
    eb = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(ra, eb)


def make_admin_issue_handler(db_path: str, admin_secret: str, limiter: AdminRateLimiter):
    async def admin_issue_access_key(request: Request) -> JSONResponse:
        ip = request.client.host if request.client else "unknown"
        if not limiter.allow(ip):
            logger.warning("admin.access_keys.rate_limited", client_ip=ip)
            return JSONResponse(
                {"error": "too_many_requests", "error_description": "Rate limit exceeded"},
                status_code=429,
            )

        auth_header = request.headers.get("authorization") or ""
        scheme, _, value = auth_header.partition(" ")
        token = value.strip() if scheme.lower() == "bearer" else ""
        if not token or not _secrets_equal(token, admin_secret):
            logger.warning("admin.access_keys.unauthorized", client_ip=ip)
            return JSONResponse(
                {"error": "invalid_client", "error_description": "Invalid or missing credentials"},
                status_code=401,
            )

        try:
            body: dict[str, Any] = await request.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        label = body.get("label")
        if label is not None and not isinstance(label, str):
            return JSONResponse(
                {"error": "invalid_request", "error_description": "label must be a string"},
                status_code=400,
            )
        if label == "":
            label = None

        key_id, full_token = await issue_access_key(db_path, label=label)
        logger.info("admin.access_keys.issued", key_id=key_id, client_ip=ip)
        return JSONResponse({"key_id": key_id, "token": full_token})

    return admin_issue_access_key
