"""Distributed rate limiting: Redis sliding window with memory fallback.

Single-process token buckets can be bypassed the moment the API runs behind
2+ workers or replicas. This limiter counts hits in a Redis sorted set per
key (ZADD + ZREMRANGEBYSCORE + ZCARD + EXPIRE in one pipeline), so every
replica shares the same budget. When Redis is absent or unreachable — field
laptops, air-gapped single nodes — it degrades to an in-process bucket with
identical semantics and a throttled warning, never an open gate by accident
(the local budget still enforces per-process; documented limitation).
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)

_WINDOW_S = 60.0

#: Cap on distinct keys held by the in-process fallback. The fallback is keyed
#: per client IP; under IP spoofing the dict would otherwise grow without bound.
_MEMORY_MAX_KEYS = 20_000


class RateLimiter:
    """per_minute hits per key per rolling 60 s window."""

    def __init__(self, per_minute: int, redis_url: str = "") -> None:
        self.per_minute = max(0, per_minute)
        self._redis_url = redis_url
        self._client: Any = None
        self._memory: dict[str, list[float]] = {}
        self._warned_at = 0.0
        self._cap_warned_at = 0.0

    def _redis(self) -> Any:
        if self._client is None and self._redis_url:
            try:
                import redis

                self._client = redis.Redis.from_url(
                    self._redis_url, socket_connect_timeout=1, socket_timeout=1
                )
                self._client.ping()
            except Exception as exc:
                self._client = False
                self._warn(exc)
        return self._client or None

    def _warn(self, exc: Exception) -> None:
        now = time.time()
        if now - self._warned_at > 60:
            self._warned_at = now
            log.warning("ratelimit: redis unavailable (%s); per-process budget", exc)

    def allowed(self, key: str) -> bool:
        """Consume one hit; False means the budget is exhausted (HTTP 429)."""
        if self.per_minute <= 0:
            return True
        client = self._redis()
        if client is not None:
            try:
                return self._allowed_redis(client, key)
            except Exception as exc:
                self._client = False
                self._warn(exc)
        return self._allowed_memory(key)

    def _allowed_redis(self, client: Any, key: str) -> bool:
        now = time.time()
        member = f"{now:.6f}"
        pipe = client.pipeline(transaction=True)
        pipe.zremrangebyscore(key, 0, now - _WINDOW_S)
        pipe.zadd(key, {member: now})
        pipe.zcard(key)
        pipe.expire(key, 65)
        _, _, count, _ = pipe.execute()
        return int(count) <= self.per_minute

    def _allowed_memory(self, key: str) -> bool:
        now = time.time()
        if len(self._memory) >= _MEMORY_MAX_KEYS:
            # Bound memory under key flood (e.g. spoofed client IPs). Evict the
            # oldest batch rather than scanning every key on every request: the
            # scan would itself become the hot path exactly when it matters.
            for k in list(self._memory)[: max(1, _MEMORY_MAX_KEYS // 10)]:
                self._memory.pop(k, None)
            self._warn_cap()
        hits = [h for h in self._memory.get(key, []) if now - h < _WINDOW_S]
        if len(hits) >= self.per_minute:
            return False
        hits.append(now)
        self._memory[key] = hits[-self.per_minute :]
        return True

    def _warn_cap(self) -> None:
        now = time.time()
        if now - self._cap_warned_at > 60:
            self._cap_warned_at = now
            log.warning("ratelimit: in-process bucket key cap reached; evicting oldest")
