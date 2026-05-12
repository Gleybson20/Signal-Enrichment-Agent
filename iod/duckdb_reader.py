"""
duckdb_reader.py
----------------
Reads pending (unenriched) records from the Silver layer of a DuckDB database.

Design decisions:
- Uses a context manager so connections are always closed even on exceptions.
- "Pending" is determined by the absence of the `enriched_at` column value
  (or the column itself), which makes the filter schema-agnostic.
- Returns plain dicts, not dataclasses, so the caller doesn't need to import
  anything from this module to use the data.
- `batch_size` in fetch_pending is a safety valve — for very large tables,
  callers should process in pages rather than loading everything into RAM.
"""

from __future__ import annotations
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
import duckdb

logger = logging.getLogger(__name__)

DEFAULT_SILVER_TABLE = "silver.reviews"


class DuckDBReader:
    """
    Reads pending records from the DuckDB Silver layer.

    Usage:
        reader = DuckDBReader(db_path=Path("data/warehouse.duckdb"))
        records = reader.fetch_pending(limit=1000)
    """

    def __init__(
        self,
        db_path: Path,
        silver_table: str = DEFAULT_SILVER_TABLE,
        text_column: str = "review_text",
        id_column: str = "review_id",
    ) -> None:
        self._db_path = db_path
        self._silver_table = silver_table
        self._text_column = text_column
        self._id_column = id_column

    def fetch_pending(
        self,
        limit: Optional[int] = None,
        gold_table: str = "gold.reviews",
    ) -> list[dict]:
        """
        Return records from the Silver table that have not yet been written to Gold.

        A record is considered "pending" if its ID is absent from the Gold table.
        This is more reliable than checking a flag column because it handles the
        case where a partial Gold write left some records behind.

        Args:
            limit:      Cap the number of records returned (useful for dry-runs).
            gold_table: Name of the Gold table used to determine what's already done.

        Returns:
            List of dicts, one per pending record.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            SELECT s.*
            FROM {self._silver_table} s
            LEFT JOIN {gold_table} g
                ON s.{self._id_column} = g.{self._id_column}
            WHERE g.{self._id_column} IS NULL
              AND s.{self._text_column} IS NOT NULL
              AND TRIM(s.{self._text_column}) != ''
            ORDER BY s.{self._id_column}
            {limit_clause}
        """

        with self._connection() as con:
            try:
                result = con.execute(query).fetchdf()
            except duckdb.CatalogException:
                logger.warning(
                    "Gold table '%s' not found. Treating all Silver records as pending.",
                    gold_table,
                )
                fallback_query = f"""
                    SELECT *
                    FROM {self._silver_table}
                    WHERE {self._text_column} IS NOT NULL
                      AND TRIM({self._text_column}) != ''
                    ORDER BY {self._id_column}
                    {limit_clause}
                """
                result = con.execute(fallback_query).fetchdf()

        records = result.to_dict(orient="records")
        logger.info("Fetched %d pending records from '%s'.", len(records), self._silver_table)
        return records

    def count_pending(self, gold_table: str = "gold.reviews") -> int:
        """Quick count without fetching data — useful for monitoring."""
        query = f"""
            SELECT COUNT(*) AS n
            FROM {self._silver_table} s
            LEFT JOIN {gold_table} g
                ON s.{self._id_column} = g.{self._id_column}
            WHERE g.{self._id_column} IS NULL
        """
        with self._connection() as con:
            try:
                return con.execute(query).fetchone()[0]
            except duckdb.CatalogException:
                return con.execute(
                    f"SELECT COUNT(*) FROM {self._silver_table}"
                ).fetchone()[0]

    def fetch_by_ids(self, ids: list[str]) -> list[dict]:
        """Fetch specific records by ID — useful for re-enrichment workflows."""
        if not ids:
            return []
        placeholders = ", ".join(f"'{i}'" for i in ids)
        query = f"""
            SELECT *
            FROM {self._silver_table}
            WHERE {self._id_column} IN ({placeholders})
            ORDER BY {self._id_column}
        """
        with self._connection() as con:
            result = con.execute(query).fetchdf()
        return result.to_dict(orient="records")

    @contextmanager
    def _connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        con = duckdb.connect(str(self._db_path), read_only=True)
        try:
            yield con
        finally:
            con.close()
