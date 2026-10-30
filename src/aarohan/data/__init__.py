"""Data ingestion, validation, and metadata management for Aarohan."""

from aarohan.data.dataset import (
    DatasetMetadata,
    ValidationCheck,
    ValidationResult,
    ValidationStatus,
)
from aarohan.data.load import (
    check_file_exists,
    compute_sha256,
    create_duckdb_connection,
    duckdb_cursor,
    get_arrow_schema,
    iter_parquet_batches,
    read_parquet_metadata,
    resolve_dataset_path,
)
from aarohan.data.validate import run_validation_pipeline

__all__ = [
    "DatasetMetadata",
    "ValidationCheck",
    "ValidationResult",
    "ValidationStatus",
    "check_file_exists",
    "compute_sha256",
    "create_duckdb_connection",
    "duckdb_cursor",
    "get_arrow_schema",
    "iter_parquet_batches",
    "read_parquet_metadata",
    "resolve_dataset_path",
    "run_validation_pipeline",
]
