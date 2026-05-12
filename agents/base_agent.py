"""
base_agent.py
-------------
Abstract base class that defines the contract every enrichment agent must follow.

Design decisions:
- ABC enforces the interface at import time, not at runtime.
- Generic[T] allows each subclass to declare its own output schema type,
  enabling strict typing through the entire call chain.
- run() is the only public entry point; validate() and log_result() are
  implementation details called internally by the final template method _execute().
- Retry logic lives here so individual agents never re-implement it.
"""

from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class AgentResult(Generic[T]):
    """Typed envelope returned by every agent."""

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    model: str = ""
    raw_response: Optional[str] = None


@dataclass
class AgentConfig:
    """Runtime configuration injected into every agent at construction time."""

    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 512
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    timeout_seconds: int = 30
    prompt_version: str = "v1"
    extra: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC, Generic[T]):
    """
    Template-method base for all enrichment agents.

    Subclasses must implement:
        - _build_prompt(record) → str
        - _call_llm(prompt) → str
        - _parse_response(raw) → T
        - validate(result) → bool
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)

    def run(self, record: dict[str, Any]) -> AgentResult[T]:
        """
        Execute the enrichment for a single record.

        Handles retries with exponential backoff internally.
        Callers receive a typed AgentResult regardless of success/failure.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            start = time.monotonic()
            try:
                result = self._execute(record)
                result.latency_ms = (time.monotonic() - start) * 1000
                self.log_result(record, result)
                return result

            except Exception as exc:
                last_error = exc
                wait = self.config.retry_backoff_base ** (attempt - 1)
                self._logger.warning(
                    "Attempt %d/%d failed for record %s: %s. Retrying in %.1fs.",
                    attempt,
                    self.config.max_retries,
                    record.get("review_id", "?"),
                    exc,
                    wait,
                )
                if attempt < self.config.max_retries:
                    time.sleep(wait)

        error_msg = f"All {self.config.max_retries} attempts failed: {last_error}"
        self._logger.error(error_msg)
        return AgentResult(success=False, error=error_msg, model=self.config.model)

    def _execute(self, record: dict[str, Any]) -> AgentResult[T]:
        """Orchestrates the steps for a single enrichment call."""
        prompt = self._build_prompt(record)
        raw_response = self._call_llm(prompt)
        parsed = self._parse_response(raw_response)

        if not self.validate(parsed):
            raise ValueError(f"Validation failed for parsed output: {parsed!r}")
        
        tokens = getattr(self, "_last_token_count", 0)

        return AgentResult(
            success=True,
            data=parsed,
            tokens_used=tokens,
            model=self.config.model,
            raw_response=raw_response,
        )
    @abstractmethod
    def _build_prompt(self, record: dict[str, Any]) -> str:
        """
        Combine the versioned prompt template with record-specific variables.

        Returns a ready-to-send prompt string.
        """

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """
        Send the prompt to the LLM provider and return the raw text response.

        Sets self._last_token_count before returning so AgentResult is accurate.
        """

    @abstractmethod
    def _parse_response(self, raw: str) -> T:
        """
        Parse and deserialize the raw LLM response into the typed output schema.

        Raises ValueError if the response cannot be parsed.
        """

    @abstractmethod
    def validate(self, result: T) -> bool:
        """
        Domain-level validation of the parsed output.

        Example: a sentiment agent checks the value is one of
        {"positive", "negative", "neutral"} before accepting it.
        """

    def log_result(self, record: dict[str, Any], result: AgentResult[T]) -> None:
        """
        Structured log entry for every agent execution.

        Format is intentionally machine-parseable (JSON-like key=value)
        so it can be ingested by log aggregators (Datadog, Loki, CloudWatch).
        """
        self._logger.info(
            "agent=%s record=%s success=%s tokens=%d latency_ms=%.1f model=%s",
            self.__class__.__name__,
            record.get("review_id", "?"),
            result.success,
            result.tokens_used,
            result.latency_ms,
            result.model,
        )
        if not result.success:
            self._logger.error(
                "agent=%s record=%s error=%s",
                self.__class__.__name__,
                record.get("review_id", "?"),
                result.error,
            )
