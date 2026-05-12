"""
duckdb_writer.py
----------------
Writes enriched records to the Gold layer of a DuckDB database.

Design decisions:
- Uses INSERT OR REPLACE (upsert) so the pipeline is idempotent: running it
  twice on the same records updates them rather than duplicating rows.
- Accepts a list of plain dicts — no coupling to any dataclass schema —
  so the writer works regardless of which agents ran.
- Schema is auto-created on first write if it doesn't exist, removing the
  need for a separate migration step.
- DuckDB does not support concurrent writers to the same file, so all writes
  happen inside a single connection that is opened and closed per batch.
  For multi-process setups, consider MotherDuck or a shared write queue.
"""

from __future__ import annotations
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_GOLD_TABLE = "gold.reviews"

GOLD_SCHEMA = """
    review_id            VARCHAR PRIMARY KEY,
    product_id           VARCHAR,
    review_text          VARCHAR,
    date                 DATE,
    sentiment            VARCHAR,
    sentiment_confidence DOUBLE,
    category             VARCHAR,
    subcategory          VARCHAR,
    entities_brands      JSON,
    entities_locations   JSON,
    entities_persons     JSON,
    enriched_at          TIMESTAMPTZ,
    model_used           VARCHAR,
    tokens_used          INTEGER,
    enrichment_cost_usd  DOUBLE,
    enrichment_error     BOOLEAN
"""


class DuckDBWriter:
    """
    Writes enriched records to the DuckDB Gold layer using upsert semantics.

    Usage:
        writer = DuckDBWriter(db_path=Path("data/warehouse.duckdb"))
        writer.upsert_many(enriched_records)
    """

    def __init__(
        self,
        db_path: Path,
        gold_table: str = DEFAULT_GOLD_TABLE,
        id_column: str = "review_id",
    ) -> None:
        self._db_path = db_path
        self._gold_table = gold_table
        self._id_column = id_column
        self._schema, self._table = self._parse_table_ref(gold_table)

    def upsert_many(self, records: list[dict]) -> int:
        """
        Upsert a list of enriched records into the Gold table.

        Returns the number of rows written.
        """
        if not records:
            return 0

        df = self._prepare_dataframe(records)

        with self._connection() as con:
            self._ensure_schema(con)
            self._ensure_table(con)
            con.register("__batch__", df)
            con.execute(
                f"INSERT OR REPLACE INTO {self._gold_table} SELECT * FROM __batch__"
            )
            con.unregister("__batch__")

        logger.info(
            "Upserted %d records into '%s'.", len(records), self._gold_table
        )
        return len(records)

    def upsert_one(self, record: dict) -> None:
        """Convenience wrapper for single-record upserts (e.g. real-time mode)."""
        self.upsert_many([record])

    def count(self) -> int:
        """Return the total number of rows in the Gold table."""
        with self._connection() as con:
            try:
                return con.execute(
                    f"SELECT COUNT(*) FROM {self._gold_table}"
                ).fetchone()[0]
            except duckdb.CatalogException:
                return 0
            
    def _prepare_dataframe(self, records: list[dict]) -> pd.DataFrame:
        """
        Normalise records into a DataFrame.

        - JSON list fields (entities_*) are serialised to JSON strings so
          DuckDB stores them in the JSON column type.
        - Missing columns are filled with None so INSERT is always schema-safe.
        """
        normalised = []
        for record in records:
            row = dict(record)
            for key in ("entities_brands", "entities_locations", "entities_persons"):
                value = row.get(key, [])
                row[key] = json.dumps(value if isinstance(value, list) else [])

            total_tokens = (
                row.pop("sentiment_tokens_used", 0)
                + row.pop("category_tokens_used", 0)
                + row.pop("entity_tokens_used", 0)
            )
            row.setdefault("tokens_used", total_tokens)
            model = (
                row.pop("sentiment_model", None)
                or row.pop("category_model", None)
                or row.pop("entity_model", None)
                or "unknown"
            )
            row.setdefault("model_used", model)

            for extra in ("sentiment_model", "category_model", "entity_model"):
                row.pop(extra, None)

            normalised.append(row)

        return pd.DataFrame(normalised)

    def _ensure_schema(self, con: duckdb.DuckDBPyConnection) -> None:
        if self._schema:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")

    def _ensure_table(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {self._gold_table} ({GOLD_SCHEMA})"
        )

    @contextmanager
    def _connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        con = duckdb.connect(str(self._db_path), read_only=False)
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    @staticmethod
    def _parse_table_ref(table_ref: str) -> tuple[Optional[str], str]:
        """Split 'schema.table' into ('schema', 'table'), or (None, 'table')."""
        parts = table_ref.split(".", maxsplit=1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, parts[0]
