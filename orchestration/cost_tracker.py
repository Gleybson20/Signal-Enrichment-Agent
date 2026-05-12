"""
cost_tracker.py
---------------
Tracks token usage and dollar cost for every LLM call in the enrichment pipeline.

Design decisions:
- Pricing table is stored in-module and versioned by model name so it's easy to
  update when OpenAI changes prices without touching orchestration logic.
- All writes are append-only to a JSONL file so the log survives partial runs.
- Thread-safe via a simple lock — safe for future async/concurrent batches.
- Totals are accumulated in memory and also persisted so dashboards can be built
  from the JSONL without re-summing.
"""

from __future__ import annotations
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PRICING_USD_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o":               {"input": 0.005,   "output": 0.015},
    "gpt-4o-mini":          {"input": 0.000150, "output": 0.000600},
    "gpt-4-turbo":          {"input": 0.010,   "output": 0.030},
    "gpt-3.5-turbo":        {"input": 0.000500, "output": 0.001500},
    "gemini-1.5-pro":       {"input": 0.007,   "output": 0.021},
    "gemini-1.5-flash":     {"input": 0.000350, "output": 0.001050},
}

UNKNOWN_PRICING = {"input": 0.0, "output": 0.0}
@dataclass
class CallRecord:
    """One LLM call logged by the tracker."""

    timestamp: str
    agent: str
    model: str
    record_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    batch_id: Optional[str] = None


@dataclass
class CostSummary:
    """Aggregated totals for a processing run."""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    by_agent: dict[str, dict] = field(default_factory=dict)
    by_model: dict[str, dict] = field(default_factory=dict)
class CostTracker:
    """
    Thread-safe cost tracker for LLM calls.

    Usage:
        tracker = CostTracker(log_path=Path("logs/cost.jsonl"))

        tracker.log(
            agent="SentimentAgent",
            model="gpt-4o",
            record_id="rev_001",
            input_tokens=80,
            output_tokens=62,
        )

        summary = tracker.summary()
        print(f"Total cost so far: ${summary.total_cost_usd:.6f}")
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        budget_usd: Optional[float] = None,
    ) -> None:
        """
        Args:
            log_path:   If provided, each call record is appended to this JSONL file.
            budget_usd: If provided, log a warning whenever cumulative cost exceeds this.
        """
        self._log_path = log_path
        self._budget_usd = budget_usd
        self._lock = threading.Lock()
        self._summary = CostSummary()

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        agent: str,
        model: str,
        record_id: str,
        input_tokens: int,
        output_tokens: int,
        batch_id: Optional[str] = None,
    ) -> float:
        """
        Record a single LLM call and return the computed cost in USD.

        When only total tokens are known (e.g. the provider doesn't split
        input/output), pass total as input_tokens and 0 as output_tokens.
        """
        cost = self._compute_cost(model, input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens

        call = CallRecord(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            agent=agent,
            model=model,
            record_id=record_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            batch_id=batch_id,
        )

        with self._lock:
            self._accumulate(call)
            self._write(call)
            self._check_budget()

        return cost

    def summary(self) -> CostSummary:
        """Return a snapshot of the current aggregated totals."""
        with self._lock:
            import copy
            return copy.deepcopy(self._summary)

    def reset(self) -> None:
        """Reset in-memory totals. Does NOT truncate the log file."""
        with self._lock:
            self._summary = CostSummary()

    @staticmethod
    def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = PRICING_USD_PER_1K.get(model, UNKNOWN_PRICING)
        if pricing == UNKNOWN_PRICING:
            logger.warning("No pricing data for model '%s'. Cost will show as $0.00.", model)
        cost = (input_tokens / 1_000) * pricing["input"] + (
            output_tokens / 1_000
        ) * pricing["output"]
        return round(cost, 8)

    def _accumulate(self, call: CallRecord) -> None:
        s = self._summary
        s.total_calls += 1
        s.total_input_tokens += call.input_tokens
        s.total_output_tokens += call.output_tokens
        s.total_tokens += call.total_tokens
        s.total_cost_usd = round(s.total_cost_usd + call.cost_usd, 8)

        if call.agent not in s.by_agent:
            s.by_agent[call.agent] = {"calls": 0, "tokens": 0, "cost_usd": 0.0}
        s.by_agent[call.agent]["calls"] += 1
        s.by_agent[call.agent]["tokens"] += call.total_tokens
        s.by_agent[call.agent]["cost_usd"] = round(
            s.by_agent[call.agent]["cost_usd"] + call.cost_usd, 8
        )

        if call.model not in s.by_model:
            s.by_model[call.model] = {"calls": 0, "tokens": 0, "cost_usd": 0.0}
        s.by_model[call.model]["calls"] += 1
        s.by_model[call.model]["tokens"] += call.total_tokens
        s.by_model[call.model]["cost_usd"] = round(
            s.by_model[call.model]["cost_usd"] + call.cost_usd, 8
        )

    def _write(self, call: CallRecord) -> None:
        if not self._log_path:
            return
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(call)) + "\n")
        except OSError as exc:
            logger.error("Failed to write cost log: %s", exc)

    def _check_budget(self) -> None:
        if self._budget_usd and self._summary.total_cost_usd > self._budget_usd:
            logger.warning(
                "BUDGET EXCEEDED: accumulated cost $%.4f exceeds budget $%.4f",
                self._summary.total_cost_usd,
                self._budget_usd,
            )
