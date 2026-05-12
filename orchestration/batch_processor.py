"""
batch_processor.py
------------------
Orchestrates the enrichment of a full Silver dataset using all three agents.

Responsibilities:
1. Split records into fixed-size batches.
2. For each batch, run SentimentAgent → CategoryAgent → EntityAgent sequentially.
3. Track cost via CostTracker.
4. Save state via Checkpoint so failures are resumable.
5. Respect rate limits via RateLimiter.
6. Write enriched records to Gold via DuckDBWriter.

Design decisions:
- Sequential (not parallel) within a single run to stay within RPM limits.
  Parallel mode can be added later by wrapping _process_batch() in a thread pool.
- Agents are injected at construction so the processor can be tested with mocks.
- Partial failures inside a batch are logged but don't abort the batch; the record
  gets written with NULL enrichment fields rather than being silently dropped.
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from agents.category_agent import CategoryAgent
from agents.entity_agent import EntityAgent
from agents.sentiment_agent import SentimentAgent
from iod.duckdb_reader import DuckDBReader
from iod.duckdb_writer import DuckDBWriter
from orchestration.checkpoint import Checkpoint
from orchestration.cost_tracker import CostTracker
from orchestration.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

@dataclass
class BatchProcessorConfig:
    batch_size: int = 50
    run_sentiment: bool = True
    run_category: bool = True
    run_entity: bool = True
    checkpoint_dir: str = ".checkpoints"
    cost_log_path: str = "logs/cost.jsonl"
    budget_usd: Optional[float] = None   # Hard stop when exceeded


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class BatchProcessor:
    """
    Orchestrates the full enrichment pipeline for a dataset.

    Usage:
        processor = BatchProcessor(
            config=BatchProcessorConfig(batch_size=100),
            reader=DuckDBReader(...),
            writer=DuckDBWriter(...),
            sentiment_agent=SentimentAgent.from_env(),
            category_agent=CategoryAgent.from_env(),
            entity_agent=EntityAgent.from_env(),
            rate_limiter=RateLimiter(rpm=500, tpm=30_000),
        )
        summary = processor.run()
    """

    def __init__(
        self,
        config: BatchProcessorConfig,
        reader: DuckDBReader,
        writer: DuckDBWriter,
        sentiment_agent: Optional[SentimentAgent] = None,
        category_agent: Optional[CategoryAgent] = None,
        entity_agent: Optional[EntityAgent] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self._config = config
        self._reader = reader
        self._writer = writer
        self._sentiment = sentiment_agent
        self._category = category_agent
        self._entity = entity_agent
        self._limiter = rate_limiter or RateLimiter()
        self._tracker = CostTracker(
            log_path=__import__("pathlib").Path(config.cost_log_path),
            budget_usd=config.budget_usd,
        )

    def run(self, run_id: Optional[str] = None) -> dict:
        """
        Enrich all pending Silver records and write them to Gold.

        Returns a summary dict with totals useful for logging / CI output.
        """
        run_id = run_id or str(uuid.uuid4())[:8]
        logger.info("Starting enrichment run '%s'.", run_id)

        records = self._reader.fetch_pending()
        if not records:
            logger.info("No pending records found. Exiting.")
            return {"run_id": run_id, "records_processed": 0}

        batches = self._split_into_batches(records)
        total_batches = len(batches)
        logger.info(
            "%d records → %d batches of %d.",
            len(records),
            total_batches,
            self._config.batch_size,
        )

        from pathlib import Path

        checkpoint = Checkpoint(
            path=Path(self._config.checkpoint_dir) / f"{run_id}.json",
            run_id=run_id,
        )
        checkpoint.initialise(total_batches=total_batches)

        records_processed = 0
        batches_processed = 0

        for batch_idx, batch in enumerate(batches):
            if checkpoint.already_done(batch_idx):
                logger.info("Batch %d/%d already done — skipping.", batch_idx + 1, total_batches)
                continue

            logger.info(
                "Processing batch %d/%d (%d records).",
                batch_idx + 1,
                total_batches,
                len(batch),
            )

            enriched_batch = self._process_batch(batch, run_id=run_id)
            self._writer.upsert_many(enriched_batch)

            checkpoint.save(batch_idx, records_in_batch=len(batch))
            records_processed += len(batch)
            batches_processed += 1

        checkpoint.complete()
        cost_summary = self._tracker.summary()

        summary = {
            "run_id": run_id,
            "records_processed": records_processed,
            "batches_processed": batches_processed,
            "total_cost_usd": cost_summary.total_cost_usd,
            "total_tokens": cost_summary.total_tokens,
            "cost_by_agent": cost_summary.by_agent,
        }
        logger.info("Run '%s' complete: %s", run_id, summary)
        return summary

    def _process_batch(self, batch: list[dict], run_id: str) -> list[dict]:
        """Enrich every record in the batch; never raise — log failures instead."""
        enriched_records = []

        for record in batch:
            try:
                enriched = self._enrich_record(record, run_id=run_id)
            except Exception as exc:
                logger.error(
                    "Unexpected error enriching record '%s': %s. Storing with null fields.",
                    record.get("review_id"),
                    exc,
                )
                enriched = self._null_enrichment(record)

            enriched_records.append(enriched)

        return enriched_records

    def _enrich_record(self, record: dict, run_id: str) -> dict:
        """Run all enabled agents on a single record sequentially."""
        enriched = dict(record)

        if self._config.run_sentiment and self._sentiment:
            self._limiter.acquire(estimated_tokens=150)
            enriched = self._sentiment.enrich(enriched)
            actual = enriched.get("sentiment_tokens_used", 0)
            self._limiter.record(actual_tokens=actual)
            self._tracker.log(
                agent="SentimentAgent",
                model=enriched.get("sentiment_model", "unknown"),
                record_id=str(record.get("review_id", "")),
                input_tokens=int(actual * 0.6),
                output_tokens=int(actual * 0.4),
                batch_id=run_id,
            )

        if self._config.run_category and self._category:
            self._limiter.acquire(estimated_tokens=150)
            enriched = self._category.enrich(enriched)
            actual = enriched.get("category_tokens_used", 0)
            self._limiter.record(actual_tokens=actual)
            self._tracker.log(
                agent="CategoryAgent",
                model=enriched.get("category_model", "unknown"),
                record_id=str(record.get("review_id", "")),
                input_tokens=int(actual * 0.6),
                output_tokens=int(actual * 0.4),
                batch_id=run_id,
            )

        if self._config.run_entity and self._entity:
            self._limiter.acquire(estimated_tokens=150)
            enriched = self._entity.enrich(enriched)
            actual = enriched.get("entity_tokens_used", 0)
            self._limiter.record(actual_tokens=actual)
            self._tracker.log(
                agent="EntityAgent",
                model=enriched.get("entity_model", "unknown"),
                record_id=str(record.get("review_id", "")),
                input_tokens=int(actual * 0.6),
                output_tokens=int(actual * 0.4),
                batch_id=run_id,
            )

        enriched["enriched_at"] = datetime.now(tz=timezone.utc).isoformat()
        return enriched

    @staticmethod
    def _null_enrichment(record: dict) -> dict:
        """Return a record with all enrichment fields set to null — never drop data."""
        enriched = dict(record)
        enriched.update(
            {
                "sentiment": None,
                "sentiment_confidence": None,
                "category": None,
                "subcategory": None,
                "entities_brands": [],
                "entities_locations": [],
                "entities_persons": [],
                "enriched_at": datetime.now(tz=timezone.utc).isoformat(),
                "enrichment_error": True,
            }
        )
        return enriched

    def _split_into_batches(self, records: list[dict]) -> list[list[dict]]:
        size = self._config.batch_size
        return [records[i : i + size] for i in range(0, len(records), size)]
