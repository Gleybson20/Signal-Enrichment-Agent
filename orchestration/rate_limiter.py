"""
rate_limiter.py
---------------
Token-bucket rate limiter that enforces both Requests Per Minute (RPM)
and Tokens Per Minute (TPM) limits set by LLM API providers.

Design decisions:
- Dual-bucket: one bucket for requests, one for tokens. A single call can
  block on either, matching how OpenAI enforces limits independently.
- Sliding window (not fixed-window) avoids the thundering herd that happens
  when a fixed window resets and all queued requests fire at once.
- Thread-safe via threading.Lock — ready for concurrent batch workers.
- sleep_fn is injectable so tests run at wall-clock speed without mocking time.
"""

from __future__ import annotations
import logging
import time
from collections import deque
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_RPM = 500
DEFAULT_TPM = 30_000


class RateLimiter:
    """
    Sliding-window rate limiter for RPM + TPM.

    Usage:
        limiter = RateLimiter(rpm=500, tpm=30_000)

        # Before each API call:
        limiter.acquire(estimated_tokens=200)

        # After each call, record the actual token count:
        limiter.record(actual_tokens=185)
    """

    def __init__(
        self,
        rpm: int = DEFAULT_RPM,
        tpm: int = DEFAULT_TPM,
        window_seconds: float = 60.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rpm = rpm
        self._tpm = tpm
        self._window = window_seconds
        self._sleep = sleep_fn
        self._request_timestamps: deque[float] = deque()
        self._token_timestamps: deque[tuple[float, int]] = deque()

        import threading
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int = 0) -> None:
        """
        Block until there is capacity to make one request consuming
        approximately `estimated_tokens` tokens.

        Call this *before* the API call.
        """
        while True:
            with self._lock:
                now = time.monotonic()
                self._evict_old(now)

                requests_in_window = len(self._request_timestamps)
                tokens_in_window = sum(t for _, t in self._token_timestamps)

                rpm_ok = requests_in_window < self._rpm
                tpm_ok = (tokens_in_window + estimated_tokens) <= self._tpm

                if rpm_ok and tpm_ok:
                    self._request_timestamps.append(now)
                    if estimated_tokens:
                        self._token_timestamps.append((now, estimated_tokens))
                    return
                wait = self._next_free_slot(
                    now, requests_in_window, tokens_in_window, estimated_tokens
                )

            logger.debug(
                "Rate limit reached (rpm=%d/%d, tpm=%d/%d). Sleeping %.2fs.",
                requests_in_window,
                self._rpm,
                tokens_in_window,
                self._tpm,
                wait,
            )
            self._sleep(wait)

    def record(self, actual_tokens: int) -> None:
        """
        Update the token bucket with the *actual* tokens used after the call.

        If you passed an estimate to acquire(), this corrects the accounting.
        Call this *after* the API call returns.
        """
        with self._lock:
            now = time.monotonic()
            if self._token_timestamps and self._token_timestamps[-1][1] != actual_tokens:
                ts, _ = self._token_timestamps.pop()
                self._token_timestamps.append((ts, actual_tokens))

    def wait_if_needed(self, tokens: int = 0) -> None:
        """
        Convenience: acquire + record in one call when the token count is known
        upfront (e.g. for estimated-token workflows).
        """
        self.acquire(estimated_tokens=tokens)

    def _evict_old(self, now: float) -> None:
        """Remove entries older than the sliding window."""
        cutoff = now - self._window
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()
        while self._token_timestamps and self._token_timestamps[0][0] < cutoff:
            self._token_timestamps.popleft()

    def _next_free_slot(
        self,
        now: float,
        requests_in_window: int,
        tokens_in_window: int,
        estimated_tokens: int,
    ) -> float:
        """
        Estimate how many seconds until at least one limit clears.
        Returns the smallest sufficient wait time.
        """
        waits: list[float] = []

        if requests_in_window >= self._rpm and self._request_timestamps:
            oldest_req = self._request_timestamps[0]
            waits.append((oldest_req + self._window) - now)

        if (tokens_in_window + estimated_tokens) > self._tpm and self._token_timestamps:
            needed = (tokens_in_window + estimated_tokens) - self._tpm
            cumulative = 0
            for ts, count in self._token_timestamps:
                cumulative += count
                if cumulative >= needed:
                    waits.append((ts + self._window) - now)
                    break
        return max(0.1, min(waits) if waits else 1.0) + 0.05

    @property
    def current_rpm(self) -> int:
        with self._lock:
            self._evict_old(time.monotonic())
            return len(self._request_timestamps)

    @property
    def current_tpm(self) -> int:
        with self._lock:
            self._evict_old(time.monotonic())
            return sum(t for _, t in self._token_timestamps)
