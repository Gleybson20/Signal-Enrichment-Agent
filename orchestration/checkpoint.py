"""
checkpoint.py
-------------
Durable, atomic checkpoint for batch processing pipelines.

Saves the index of the last successfully processed batch to disk so a
restart can resume from exactly the right position without reprocessing
or losing data.

Design decisions:
- Writes to a temp file then renames — this is atomic on all POSIX systems,
  so a crash mid-write never leaves a corrupt checkpoint file.
- JSON format keeps the file human-readable for debugging.
- The Checkpoint object is decoupled from the batch processor so it can
  be tested independently and reused in other pipelines.
"""

from __future__ import annotations
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CHECKPOINT_VERSION = 1
@dataclass
class CheckpointState:
    version: int
    run_id: str
    last_completed_batch: int
    total_records_processed: int
    total_batches: int
    started_at: str
    updated_at: str
    metadata: dict
class Checkpoint:
    """
    Manages checkpoint persistence for a single processing run.

    Usage:
        cp = Checkpoint(path=Path(".checkpoints/run_abc.json"), run_id="run_abc")
        cp.initialise(total_batches=200)

        for batch_idx, batch in enumerate(batches):
            if cp.already_done(batch_idx):
                continue

            process(batch)
            cp.save(batch_idx, records_in_batch=len(batch))

        cp.complete()
    """

    def __init__(self, path: Path, run_id: str) -> None:
        self._path = path
        self._run_id = run_id
        self._state: Optional[CheckpointState] = None
        path.parent.mkdir(parents=True, exist_ok=True)

    def initialise(
        self,
        total_batches: int,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Prepare the checkpoint for a new run.

        If a checkpoint already exists for this run_id and overwrite=False,
        the existing state is loaded and the run resumes from where it stopped.
        """
        if self._path.exists() and not overwrite:
            self._state = self._load()
            logger.info(
                "Resuming run '%s' from batch %d / %d.",
                self._run_id,
                self._state.last_completed_batch + 1,
                self._state.total_batches,
            )
        else:
            now = datetime.now(tz=timezone.utc).isoformat()
            self._state = CheckpointState(
                version=_CHECKPOINT_VERSION,
                run_id=self._run_id,
                last_completed_batch=-1,
                total_records_processed=0,
                total_batches=total_batches,
                started_at=now,
                updated_at=now,
                metadata=metadata or {},
            )
            self._persist()
            logger.info("Initialised checkpoint for run '%s'.", self._run_id)

    def already_done(self, batch_idx: int) -> bool:
        """Return True if this batch was completed in a previous run."""
        if self._state is None:
            raise RuntimeError("Call initialise() before querying the checkpoint.")
        done = batch_idx <= self._state.last_completed_batch
        if done:
            logger.debug("Skipping batch %d (already completed).", batch_idx)
        return done

    def save(self, batch_idx: int, records_in_batch: int) -> None:
        """Mark batch_idx as completed and persist to disk immediately."""
        if self._state is None:
            raise RuntimeError("Call initialise() before saving.")
        self._state.last_completed_batch = batch_idx
        self._state.total_records_processed += records_in_batch
        self._state.updated_at = datetime.now(tz=timezone.utc).isoformat()
        self._persist()
        logger.debug(
            "Checkpoint saved: batch %d complete. Total records: %d.",
            batch_idx,
            self._state.total_records_processed,
        )

    def complete(self) -> None:
        """Mark the entire run as finished."""
        if self._state is None:
            return
        self._state.metadata["completed"] = True
        self._state.updated_at = datetime.now(tz=timezone.utc).isoformat()
        self._persist()
        logger.info(
            "Run '%s' completed. %d records across %d batches.",
            self._run_id,
            self._state.total_records_processed,
            self._state.last_completed_batch + 1,
        )

    def delete(self) -> None:
        """Remove the checkpoint file (e.g. after a clean successful run)."""
        if self._path.exists():
            self._path.unlink()
            logger.info("Checkpoint file deleted: %s", self._path)

    @property
    def state(self) -> Optional[CheckpointState]:
        return self._state

    @property
    def resume_from(self) -> int:
        """Return the first batch index that needs processing."""
        if self._state is None:
            return 0
        return max(0, self._state.last_completed_batch + 1)

    def _persist(self) -> None:
        """
        Atomic write: write to temp file, then rename.
        rename() is atomic on POSIX; on Windows it replaces atomically in Python 3.3+.
        """
        data = json.dumps(asdict(self._state), indent=2, ensure_ascii=False)
        dir_ = self._path.parent
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp_path, self._path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def _load(self) -> CheckpointState:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if raw.get("version") != _CHECKPOINT_VERSION:
            logger.warning(
                "Checkpoint version mismatch (file=%s, expected=%s). Starting fresh.",
                raw.get("version"),
                _CHECKPOINT_VERSION,
            )
            raise ValueError("Checkpoint version mismatch")
        if raw.get("run_id") != self._run_id:
            raise ValueError(
                f"Checkpoint run_id mismatch: file has '{raw['run_id']}', "
                f"expected '{self._run_id}'."
            )
        return CheckpointState(**raw)
