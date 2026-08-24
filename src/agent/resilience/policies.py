"""Retry policy and circuit breaker for every outbound dependency.

Two distinct failure classes are handled differently:

  * transient  (5xx, timeout, rate limit, connection reset) -> retry with
    exponential backoff + full jitter, then fail over to the next provider.
  * permanent  (4xx auth, malformed request, quota exhausted for the day) ->
    do NOT retry, fail over immediately. Retrying these only burns money.

The circuit breaker stops the agent from paying the timeout tax on a dependency
that is already known to be down: after `failure_threshold` consecutive
failures the breaker opens for `reset_timeout_s`, during which calls fail fast
and the router advances to the next provider without a network round trip.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar

from ..obs import metrics

T = TypeVar("T")


class TransientError(RuntimeError):
    """Worth retrying."""


class PermanentError(RuntimeError):
    """Not worth retrying - fail over now."""


class CircuitOpenError(RuntimeError):
    """Dependency is in a known-bad state; skipped without calling it."""


_TRANSIENT_MARKERS = (
    "timeout", "timed out", "deadline exceeded", "503", "502", "504", "500",
    "unavailable", "connection reset", "connection aborted", "temporarily",
    "rate limit", "429", "resource_exhausted", "resource exhausted",
    "overloaded", "internal error", "backendtimeout", "econnreset",
)
_PERMANENT_MARKERS = (
    "api key not valid", "invalid api key", "unauthorized", "401", "403",
    "permission denied", "not found", "404", "invalid argument",
    "unsupported", "billing", "consumer_suspended",
)


def classify(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    for marker in _PERMANENT_MARKERS:
        if marker in text:
            return "permanent"
    for marker in _TRANSIENT_MARKERS:
        if marker in text:
            return "transient"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transient"
    return "unknown"


@dataclass
class RetryPolicy:
    attempts: int = 3
    base_delay_s: float = 0.4
    max_delay_s: float = 6.0
    jitter: bool = True

    def delay_for(self, attempt: int) -> float:
        raw = min(self.max_delay_s, self.base_delay_s * (2 ** attempt))
        return random.uniform(0, raw) if self.jitter else raw


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    reset_timeout_s: float = 30.0
    _failures: int = field(default=0, init=False)
    _opened_at: Optional[float] = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if time.time() - self._opened_at >= self.reset_timeout_s:
                return "half_open"
            return "open"

    def guard(self) -> None:
        if self.state == "open":
            metrics.incr("llm.circuit_open", dependency=self.name)
            raise CircuitOpenError(
                f"circuit for '{self.name}' is open; retry in "
                f"{self.reset_timeout_s - (time.time() - (self._opened_at or 0)):.0f}s"
            )

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.time()


_BREAKERS: dict[str, CircuitBreaker] = {}
_BREAKERS_LOCK = threading.Lock()


def breaker(name: str, **kwargs) -> CircuitBreaker:
    with _BREAKERS_LOCK:
        if name not in _BREAKERS:
            _BREAKERS[name] = CircuitBreaker(name=name, **kwargs)
        return _BREAKERS[name]


def breaker_states() -> dict[str, str]:
    with _BREAKERS_LOCK:
        return {name: cb.state for name, cb in _BREAKERS.items()}


def reset_breakers() -> None:
    """Clear all breaker state. Used by tests and by the /health reset path."""
    with _BREAKERS_LOCK:
        _BREAKERS.clear()


def call_with_resilience(
    fn: Callable[[], T],
    *,
    dependency: str,
    policy: RetryPolicy | None = None,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Run `fn` under a circuit breaker plus classified retry."""
    policy = policy or RetryPolicy()
    cb = breaker(dependency)
    cb.guard()

    last: BaseException | None = None
    for attempt in range(policy.attempts):
        try:
            result = fn()
            cb.record_success()
            return result
        except CircuitOpenError:
            raise
        except BaseException as exc:  # noqa: BLE001 - deliberate: we classify below
            last = exc
            kind = classify(exc)
            cb.record_failure()
            if kind == "permanent" or attempt == policy.attempts - 1:
                break
            if on_retry:
                on_retry(attempt, exc)
            time.sleep(policy.delay_for(attempt))

    assert last is not None
    if classify(last) == "permanent":
        raise PermanentError(str(last)) from last
    raise TransientError(str(last)) from last
