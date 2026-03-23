"""
rag/cache.py — Optional Redis cache for embeddings and ranking results.
Avoids re-embedding identical texts and caches ranking results per JD.
Gracefully degrades when Redis is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from typing import Any, List, Optional

from src.config import get_logger, settings

logger = get_logger(__name__)


class CacheClient:
    """Thin Redis wrapper with pickle serialisation and silent fallback."""

    def __init__(self):
        self._client = None
        self._available = False
        if settings.redis.enabled:
            self._init_client()

    def _init_client(self) -> None:
        try:
            import redis
            self._client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=False,
            )
            self._client.ping()
            self._available = True
            logger.info("cache.connected", host=settings.redis.host)
        except Exception as exc:
            logger.warning("cache.unavailable", error=str(exc))
            self._available = False

    # ── Generic get/set ───────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        if not self._available:
            return None
        try:
            raw = self._client.get(key)
            return pickle.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("cache.get_error", key=key, error=str(exc))
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self._available:
            return False
        try:
            raw = pickle.dumps(value)
            self._client.set(key, raw, ex=ttl or settings.redis.ttl_seconds)
            return True
        except Exception as exc:
            logger.warning("cache.set_error", key=key, error=str(exc))
            return False

    def delete(self, key: str) -> None:
        if not self._available:
            return
        try:
            self._client.delete(key)
        except Exception:
            pass

    def delete_pattern(self, pattern: str) -> int:
        if not self._available:
            return 0
        try:
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
        except Exception:
            pass
        return 0

    # ── Embedding cache ───────────────────────────────────────────────────────

    @staticmethod
    def _emb_key(text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()[:24]
        return f"emb:{h}"

    def get_embedding(self, text: str) -> Optional[List[float]]:
        return self.get(self._emb_key(text))

    def set_embedding(self, text: str, embedding: List[float]) -> None:
        self.set(self._emb_key(text), embedding, ttl=86400 * 7)  # 7 days

    # ── Ranking cache ─────────────────────────────────────────────────────────

    @staticmethod
    def _rank_key(jd_id: str) -> str:
        return f"rank:{jd_id}"

    def get_ranking(self, jd_id: str) -> Optional[list]:
        return self.get(self._rank_key(jd_id))

    def set_ranking(self, jd_id: str, result: list) -> None:
        self.set(self._rank_key(jd_id), result, ttl=settings.redis.ttl_seconds)

    def invalidate_ranking(self, jd_id: str) -> None:
        self.delete(self._rank_key(jd_id))

    def invalidate_all_rankings(self) -> int:
        return self.delete_pattern("rank:*")

    # ── Resume cache ──────────────────────────────────────────────────────────

    @staticmethod
    def _resume_key(resume_id: str) -> str:
        return f"resume:{resume_id}"

    def get_resume_doc(self, resume_id: str) -> Optional[Any]:
        return self.get(self._resume_key(resume_id))

    def set_resume_doc(self, resume_id: str, doc: Any) -> None:
        self.set(self._resume_key(resume_id), doc, ttl=86400)  # 1 day

    def invalidate_resume(self, resume_id: str) -> None:
        self.delete(self._resume_key(resume_id))
        # Also blow away cached rankings since candidate pool changed
        self.invalidate_all_rankings()

    def health_check(self) -> dict:
        if not self._available:
            return {"status": "disabled_or_unavailable"}
        try:
            self._client.ping()
            info = self._client.info("memory")
            return {
                "status": "healthy",
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": self._client.info("clients").get("connected_clients"),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


cache = CacheClient()