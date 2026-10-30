"""Reusable utilities for dataset path resolution, hashing, and loading."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from aarohan.config import resolve_project_path


def resolve_dataset_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a dataset path from configuration relative to project root.

    Args:
        config: Parsed configuration dictionary.
        key: Configuration key containing the path string (e.g. 'raw_train_path').

    Returns:
        Absolute Path object.

    Raises:
        KeyError: If key is not found in configuration.
    """
    if key not in config:
        raise KeyError(f"Configuration missing required path key: '{key}'")
    return resolve_project_path(config[key])


def check_file_exists(file_path: Path | str) -> bool:
    """Check if the specified path points to an existing regular file."""
    path = Path(file_path)
    return path.exists() and path.is_file()


def compute_sha256(file_path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculate the cryptographic SHA-256 checksum using chunked stream reads.

    Args:
        file_path: Path to file to hash.
        chunk_size: Byte size of chunks to stream (default 8 MB).

    Returns:
        Hexadecimal SHA-256 digest string.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_parquet_metadata(file_path: Path | str) -> pq.FileMetaData:
    """Read Parquet file metadata without reading data contents.

    Args:
        file_path: Path to the Parquet file.

    Returns:
        pyarrow.parquet.FileMetaData object.
    """
    return pq.read_metadata(file_path)


def get_arrow_schema(file_path: Path | str) -> pa.Schema:
    """Read the Arrow schema from a Parquet file without loading row data.

    Args:
        file_path: Path to the Parquet file.

    Returns:
        pyarrow.Schema object.
    """
    return pq.read_schema(file_path)


def create_duckdb_connection(
    threads: int = 2, max_memory: str = "512MB"
) -> duckdb.DuckDBPyConnection:
    """Create a configured in-memory DuckDB connection.

    Args:
        threads: Maximum CPU execution threads (conservative default: 2).
        max_memory: Maximum memory ceiling (conservative default: '512MB').

    Returns:
        Configured duckdb connection object.
    """
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {int(threads)};")
    con.execute(f"SET max_memory = '{max_memory}';")
    return con


@contextmanager
def duckdb_cursor(
    threads: int = 2, max_memory: str = "512MB"
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context manager for DuckDB connections ensuring reliable resource cleanup.

    Yields:
        Configured duckdb connection.
    """
    con = create_duckdb_connection(threads=threads, max_memory=max_memory)
    try:
        yield con
    finally:
        con.close()


def iter_parquet_batches(
    file_path: Path | str,
    columns: list[str] | None = None,
    batch_size: int = 5000,
) -> Iterator[pa.RecordBatch]:
    """Yield RecordBatches from a Parquet file in controlled, memory-efficient batches.

    Args:
        file_path: Path to the Parquet file.
        columns: Optional list of column names to project.
        batch_size: Number of rows per batch (default 5,000).

    Yields:
        pyarrow.RecordBatch instances.
    """
    parquet_file = pq.ParquetFile(file_path)
    yield from parquet_file.iter_batches(batch_size=batch_size, columns=columns)
