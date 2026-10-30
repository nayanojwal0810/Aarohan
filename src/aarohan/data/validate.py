"""Data validation engine for Aarohan retail demand panels."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa

from aarohan.config import (
    get_project_root,
    load_config,
    resolve_project_path,
)
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
    get_arrow_schema,
    iter_parquet_batches,
    read_parquet_metadata,
    resolve_dataset_path,
)

logger = logging.getLogger("aarohan.data.validate")


def _types_match(actual_type: pa.DataType, expected_type_str: str) -> bool:
    """Normalize Arrow logical types and verify compatibility."""
    expected_norm = (
        expected_type_str.lower().replace(" ", "").replace("element", "item")
    )
    actual_norm = str(actual_type).lower().replace(" ", "").replace("element", "item")

    if expected_norm == actual_norm:
        return True

    if expected_norm in ("int64", "long") and actual_type == pa.int64():
        return True
    if expected_norm in ("int32", "integer") and actual_type == pa.int32():
        return True
    if expected_norm in ("double", "float64") and actual_type == pa.float64():
        return True
    if expected_norm in ("string", "utf8") and actual_type in (
        pa.string(),
        pa.large_string(),
    ):
        return True
    if "list" in expected_norm and pa.types.is_list(actual_type):
        if "double" in expected_norm and actual_type.value_type == pa.float64():
            return True
        if "int64" in expected_norm and actual_type.value_type == pa.int64():
            return True

    return False


def _is_numeric_type(arrow_type: pa.DataType | None) -> bool:
    """Return True if Arrow type is numeric (float or integer)."""
    if arrow_type is None:
        return False
    return pa.types.is_floating(arrow_type) or pa.types.is_integer(arrow_type)


def _is_string_or_date_type(arrow_type: pa.DataType | None) -> bool:
    """Return True if Arrow type is string, large_string, or date."""
    if arrow_type is None:
        return False
    return (
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_date(arrow_type)
    )


def _is_valid_hourly_sales_type(arrow_type: pa.DataType | None) -> bool:
    """Return True if Arrow type is a list of numeric values."""
    if arrow_type is None:
        return False
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return _is_numeric_type(arrow_type.value_type)
    return False


def _is_valid_hourly_stock_type(arrow_type: pa.DataType | None) -> bool:
    """Return True if Arrow type is a list of integer values."""
    if arrow_type is None:
        return False
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return pa.types.is_integer(arrow_type.value_type)
    return False


def _get_relative_path_str(file_path: Path) -> str:
    """Get project-relative path as a forward-slash string."""
    root_dir = get_project_root()
    try:
        return file_path.resolve().relative_to(root_dir.resolve()).as_posix()
    except (ValueError, RuntimeError):
        return file_path.as_posix()


def validate_file_level(
    file_path: Path,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
    expected_cols: int | None = None,
) -> tuple[list[ValidationCheck], DatasetMetadata | None]:
    """Validate file presence, size, checksum, and Parquet metadata."""
    checks: list[ValidationCheck] = []
    rel_path_str = _get_relative_path_str(file_path)

    if not check_file_exists(file_path):
        checks.append(
            ValidationCheck(
                name="file_existence",
                status=ValidationStatus.FAIL.value,
                expected="Existing regular file",
                actual="File not found",
                explanation=f"Dataset file does not exist at {rel_path_str}",
            )
        )
        return checks, None

    checks.append(
        ValidationCheck(
            name="file_existence",
            status=ValidationStatus.PASS.value,
            expected="Existing regular file",
            actual=str(file_path.name),
            explanation=f"File verified at {rel_path_str}",
        )
    )

    actual_sha = compute_sha256(file_path)
    if expected_sha256 is not None:
        if actual_sha.lower() == expected_sha256.lower():
            checks.append(
                ValidationCheck(
                    name="sha256_checksum",
                    status=ValidationStatus.PASS.value,
                    expected=expected_sha256,
                    actual=actual_sha,
                    explanation=(
                        "Cryptographic SHA-256 digest matches official benchmark"
                    ),
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="sha256_checksum",
                    status=ValidationStatus.FAIL.value,
                    expected=expected_sha256,
                    actual=actual_sha,
                    explanation="Checksum mismatch against official benchmark",
                )
            )

    try:
        meta = read_parquet_metadata(file_path)
        schema = get_arrow_schema(file_path)
    except Exception as exc:
        checks.append(
            ValidationCheck(
                name="parquet_readability",
                status=ValidationStatus.FAIL.value,
                expected="Valid readable Parquet file",
                actual=f"Read failure: {exc}",
                explanation="Parquet file metadata or schema is corrupt/unreadable",
            )
        )
        return checks, None

    if expected_rows is not None:
        status = (
            ValidationStatus.PASS.value
            if meta.num_rows == expected_rows
            else ValidationStatus.FAIL.value
        )
        checks.append(
            ValidationCheck(
                name="row_count",
                status=status,
                expected=expected_rows,
                actual=meta.num_rows,
                explanation=f"Parquet row count: {meta.num_rows:,}",
                counts={"row_count": meta.num_rows},
            )
        )

    if expected_cols is not None:
        status = (
            ValidationStatus.PASS.value
            if meta.num_columns == expected_cols
            else ValidationStatus.FAIL.value
        )
        checks.append(
            ValidationCheck(
                name="column_count",
                status=status,
                expected=expected_cols,
                actual=meta.num_columns,
                explanation=f"Parquet column count: {meta.num_columns}",
                counts={"column_count": meta.num_columns},
            )
        )

    file_size_bytes = file_path.stat().st_size
    dataset_meta = DatasetMetadata(
        name=file_path.stem,
        path=rel_path_str,
        file_size_bytes=file_size_bytes,
        file_size_mb=round(file_size_bytes / (1024 * 1024), 2),
        sha256=actual_sha,
        num_rows=meta.num_rows,
        num_columns=meta.num_columns,
        num_row_groups=meta.num_row_groups,
        column_names=schema.names,
        schema={f.name: str(f.type) for f in schema},
        min_dt="",
        max_dt="",
        num_days=0,
        num_stores=0,
        num_products=0,
        num_series=0,
    )

    return checks, dataset_meta


def validate_schema(
    file_path: Path, expected_schema: dict[str, str]
) -> list[ValidationCheck]:
    """Validate that actual Parquet columns and Arrow types match."""
    checks: list[ValidationCheck] = []
    try:
        schema = get_arrow_schema(file_path)
    except Exception as exc:
        checks.append(
            ValidationCheck(
                name="schema_inspection",
                status=ValidationStatus.FAIL.value,
                expected="Readable schema",
                actual=f"Error: {exc}",
                explanation="Failed to read Arrow schema from Parquet file",
            )
        )
        return checks

    actual_cols = set(schema.names)
    expected_cols = set(expected_schema.keys())

    missing_cols = sorted(expected_cols - actual_cols)
    unexpected_cols = sorted(actual_cols - expected_cols)

    if missing_cols:
        checks.append(
            ValidationCheck(
                name="schema_missing_columns",
                status=ValidationStatus.FAIL.value,
                expected="All declared columns present",
                actual=f"Missing {len(missing_cols)} columns: {missing_cols}",
                explanation="Required dataset columns are absent",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="schema_missing_columns",
                status=ValidationStatus.PASS.value,
                expected="All declared columns present",
                actual="No missing columns",
                explanation="All required schema columns are present",
            )
        )

    if unexpected_cols:
        checks.append(
            ValidationCheck(
                name="schema_unexpected_columns",
                status=ValidationStatus.FAIL.value,
                expected="No extraneous columns",
                actual=(
                    f"Found {len(unexpected_cols)} unexpected columns: "
                    f"{unexpected_cols}"
                ),
                explanation="Dataset contains undeclared columns",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="schema_unexpected_columns",
                status=ValidationStatus.PASS.value,
                expected="No extraneous columns",
                actual="Zero extraneous columns",
                explanation="No unexpected columns detected",
            )
        )

    type_mismatches: dict[str, dict[str, str]] = {}
    for col_name, exp_type in expected_schema.items():
        if col_name in actual_cols:
            actual_type = schema.field(col_name).type
            if not _types_match(actual_type, exp_type):
                type_mismatches[col_name] = {
                    "expected": exp_type,
                    "actual": str(actual_type),
                }

    if type_mismatches:
        checks.append(
            ValidationCheck(
                name="schema_data_types",
                status=ValidationStatus.FAIL.value,
                expected="All column types matching specification",
                actual=f"{len(type_mismatches)} type mismatches",
                explanation=f"Column type mismatches found: {type_mismatches}",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="schema_data_types",
                status=ValidationStatus.PASS.value,
                expected="All column types matching specification",
                actual="All types match",
                explanation="All column types match the Arrow schema specification",
            )
        )

    return checks


def validate_general_integrity(
    con: duckdb.DuckDBPyConnection, file_path: Path
) -> list[ValidationCheck]:
    """Execute SQL aggregate integrity checks using DuckDB."""
    checks: list[ValidationCheck] = []
    parquet_sql = f"read_parquet('{file_path.as_posix()}')"

    try:
        schema = get_arrow_schema(file_path)
    except Exception as exc:
        checks.append(
            ValidationCheck(
                name="integrity_query_execution",
                status=ValidationStatus.FAIL.value,
                expected="Readable Parquet file",
                actual=f"Read error: {exc}",
                explanation="Cannot execute SQL integrity checks on unreadable file",
            )
        )
        return checks

    actual_cols = set(schema.names)
    scalar_cols = [
        f.name for f in schema if f.name not in ("hours_sale", "hours_stock_status")
    ]
    float_cols = [
        f.name
        for f in schema
        if f.type == pa.float64() and f.name not in ("hours_sale", "hours_stock_status")
    ]

    # Null checks across scalar columns
    if scalar_cols:
        null_exprs = [
            f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)" for c in scalar_cols
        ]
        total_nulls = con.execute(
            f"SELECT {' + '.join(null_exprs)} FROM {parquet_sql}"
        ).fetchone()[0]

        status = (
            ValidationStatus.PASS.value
            if total_nulls == 0
            else ValidationStatus.FAIL.value
        )
        checks.append(
            ValidationCheck(
                name="missing_values_scalar",
                status=status,
                expected=0,
                actual=int(total_nulls),
                explanation=f"Total null values across scalar fields: {total_nulls}",
                counts={"null_count": int(total_nulls)},
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="missing_values_scalar",
                status=ValidationStatus.FAIL.value,
                expected="Scalar columns available for null check",
                actual="No scalar columns found",
                explanation="Dataset lacks scalar columns to check for nulls",
            )
        )

    # Non-finite values in float columns
    if float_cols:
        non_finite_exprs = [
            f"SUM(CASE WHEN isnan({c}) OR isinf({c}) THEN 1 ELSE 0 END)"
            for c in float_cols
        ]
        non_finite_total = con.execute(
            f"SELECT {' + '.join(non_finite_exprs)} FROM {parquet_sql}"
        ).fetchone()[0]

        status = (
            ValidationStatus.PASS.value
            if non_finite_total == 0
            else ValidationStatus.FAIL.value
        )
        checks.append(
            ValidationCheck(
                name="non_finite_values",
                status=status,
                expected=0,
                actual=int(non_finite_total),
                explanation=(
                    f"Non-finite (NaN/Inf) values across numeric fields: "
                    f"{non_finite_total}"
                ),
                counts={"non_finite_count": int(non_finite_total)},
            )
        )

    # Negative recorded sales
    if "sale_amount" in actual_cols:
        sale_field = schema.field("sale_amount")
        if not _is_numeric_type(sale_field.type):
            checks.append(
                ValidationCheck(
                    name="negative_sales",
                    status=ValidationStatus.FAIL.value,
                    expected="Numeric sale_amount (double)",
                    actual=f"Invalid type: {sale_field.type}",
                    explanation=("Cannot check negative sales on non-numeric column"),
                )
            )
        else:
            neg_sales = con.execute(
                f"SELECT COUNT(*) FROM {parquet_sql} WHERE sale_amount < 0"
            ).fetchone()[0]
            status = (
                ValidationStatus.PASS.value
                if neg_sales == 0
                else ValidationStatus.FAIL.value
            )
            checks.append(
                ValidationCheck(
                    name="negative_sales",
                    status=status,
                    expected=0,
                    actual=int(neg_sales),
                    explanation=f"Negative sale_amount occurrences: {neg_sales}",
                    counts={"negative_sales_count": int(neg_sales)},
                )
            )
    else:
        checks.append(
            ValidationCheck(
                name="negative_sales",
                status=ValidationStatus.FAIL.value,
                expected="sale_amount column present",
                actual="Column missing",
                explanation="Cannot check negative sales: sale_amount column missing",
            )
        )

    # Natural key duplicates on (store_id, product_id, dt)
    if {"store_id", "product_id", "dt"}.issubset(actual_cols):
        key_types_valid = not any(
            pa.types.is_nested(schema.field(c).type)
            for c in ("store_id", "product_id", "dt")
        )
        if not key_types_valid:
            checks.append(
                ValidationCheck(
                    name="natural_key_duplicates",
                    status=ValidationStatus.FAIL.value,
                    expected="Scalar store_id, product_id, dt columns",
                    actual="Nested types found in natural key columns",
                    explanation="Cannot group by nested columns for key duplicates",
                )
            )
        else:
            dup_groups = con.execute(f"""
                SELECT COUNT(*) FROM (
                    SELECT store_id, product_id, dt, COUNT(*) as cnt
                    FROM {parquet_sql}
                    GROUP BY store_id, product_id, dt
                    HAVING cnt > 1
                )
            """).fetchone()[0]

            status = (
                ValidationStatus.PASS.value
                if dup_groups == 0
                else ValidationStatus.FAIL.value
            )
            checks.append(
                ValidationCheck(
                    name="natural_key_duplicates",
                    status=status,
                    expected=0,
                    actual=int(dup_groups),
                    explanation=(
                        f"Duplicate natural key (store_id, product_id, dt) "
                        f"groups: {dup_groups}"
                    ),
                    counts={"duplicate_groups": int(dup_groups)},
                )
            )
    else:
        checks.append(
            ValidationCheck(
                name="natural_key_duplicates",
                status=ValidationStatus.FAIL.value,
                expected="store_id, product_id, and dt columns present",
                actual="Missing natural key columns",
                explanation="Cannot check key duplicates: required columns missing",
            )
        )

    # Binary flags validity
    if {"holiday_flag", "activity_flag"}.issubset(actual_cols):
        hf_t = schema.field("holiday_flag").type
        af_t = schema.field("activity_flag").type
        flags_valid = (pa.types.is_integer(hf_t) or pa.types.is_boolean(hf_t)) and (
            pa.types.is_integer(af_t) or pa.types.is_boolean(af_t)
        )
        if not flags_valid:
            checks.append(
                ValidationCheck(
                    name="binary_flags_validity",
                    status=ValidationStatus.FAIL.value,
                    expected="Integer or boolean holiday_flag and activity_flag",
                    actual=f"Invalid types: holiday={hf_t}, activity={af_t}",
                    explanation=("Cannot validate binary flags on non-integer columns"),
                )
            )
        else:
            invalid_flags = con.execute(f"""
                SELECT COUNT(*) FROM {parquet_sql}
                WHERE holiday_flag NOT IN (0, 1) OR activity_flag NOT IN (0, 1)
            """).fetchone()[0]

            status = (
                ValidationStatus.PASS.value
                if invalid_flags == 0
                else ValidationStatus.FAIL.value
            )
            checks.append(
                ValidationCheck(
                    name="binary_flags_validity",
                    status=status,
                    expected=0,
                    actual=int(invalid_flags),
                    explanation=(
                        "Invalid holiday_flag or activity_flag non-binary values: "
                        f"{invalid_flags}"
                    ),
                    counts={"invalid_flags_count": int(invalid_flags)},
                )
            )
    else:
        checks.append(
            ValidationCheck(
                name="binary_flags_validity",
                status=ValidationStatus.FAIL.value,
                expected="holiday_flag and activity_flag columns present",
                actual="Missing binary flag columns",
                explanation="Cannot check binary flags: required columns missing",
            )
        )

    # Date format check (ISO-8601 YYYY-MM-DD)
    if "dt" in actual_cols:
        dt_field = schema.field("dt")
        if not _is_string_or_date_type(dt_field.type):
            checks.append(
                ValidationCheck(
                    name="date_format_check",
                    status=ValidationStatus.FAIL.value,
                    expected="String column with YYYY-MM-DD format",
                    actual=f"Invalid dt column type: {dt_field.type}",
                    explanation=(
                        "Cannot validate ISO date format on non-string column"
                    ),
                )
            )
        else:
            invalid_dates = con.execute(f"""
                SELECT COUNT(*) FROM {parquet_sql}
                WHERE try_strptime(CAST(dt AS VARCHAR), '%Y-%m-%d') IS NULL
            """).fetchone()[0]

            status = (
                ValidationStatus.PASS.value
                if invalid_dates == 0
                else ValidationStatus.FAIL.value
            )
            checks.append(
                ValidationCheck(
                    name="date_format_check",
                    status=status,
                    expected=0,
                    actual=int(invalid_dates),
                    explanation=(
                        f"Dates failing ISO YYYY-MM-DD parsing: {invalid_dates}"
                    ),
                    counts={"invalid_dates_count": int(invalid_dates)},
                )
            )
    else:
        checks.append(
            ValidationCheck(
                name="date_format_check",
                status=ValidationStatus.FAIL.value,
                expected="dt column present",
                actual="Column missing",
                explanation="Cannot check date format: dt column missing",
            )
        )

    return checks


def validate_entity_and_time_series(
    con: duckdb.DuckDBPyConnection,
    file_path: Path,
    expected_size: dict[str, Any],
    expected_entities: dict[str, Any],
    metadata: DatasetMetadata | None = None,
) -> list[ValidationCheck]:
    """Validate entity counts, date range, panel completeness, and continuity."""
    checks: list[ValidationCheck] = []
    parquet_sql = f"read_parquet('{file_path.as_posix()}')"

    try:
        schema = get_arrow_schema(file_path)
    except Exception as exc:
        checks.append(
            ValidationCheck(
                name="entity_series_query",
                status=ValidationStatus.FAIL.value,
                expected="Readable Parquet file",
                actual=f"Read error: {exc}",
                explanation="Cannot check entity series on unreadable file",
            )
        )
        return checks

    actual_cols = set(schema.names)
    required_cols = {"dt", "store_id", "product_id"}
    if not required_cols.issubset(actual_cols):
        missing = sorted(required_cols - actual_cols)
        checks.append(
            ValidationCheck(
                name="entity_series_required_columns",
                status=ValidationStatus.FAIL.value,
                expected="dt, store_id, product_id present",
                actual=f"Missing: {missing}",
                explanation="Cannot validate entity structure: columns missing",
            )
        )
        return checks

    exp_stores = expected_entities.get("stores", 898)
    exp_prods = expected_entities.get("products", 865)
    exp_series = expected_size.get("series", 50000)
    exp_start = expected_size.get("start_date")
    exp_end = expected_size.get("end_date")
    exp_days = expected_size.get("dates")

    store_field = schema.field("store_id")
    prod_field = schema.field("product_id")
    dt_field = schema.field("dt")

    is_store_valid = not pa.types.is_nested(store_field.type) and (
        pa.types.is_integer(store_field.type)
        or pa.types.is_string(store_field.type)
        or pa.types.is_large_string(store_field.type)
    )
    is_prod_valid = not pa.types.is_nested(prod_field.type) and (
        pa.types.is_integer(prod_field.type)
        or pa.types.is_string(prod_field.type)
        or pa.types.is_large_string(prod_field.type)
    )
    dt_type_valid = _is_string_or_date_type(dt_field.type)

    min_dt = None
    max_dt = None
    num_days = None
    num_stores = None
    num_products = None
    num_series = None

    if is_store_valid and is_prod_valid and dt_type_valid:
        try:
            stats = con.execute(f"""
                SELECT
                    MIN(dt) as min_dt,
                    MAX(dt) as max_dt,
                    COUNT(DISTINCT dt) as num_days,
                    COUNT(DISTINCT store_id) as num_stores,
                    COUNT(DISTINCT product_id) as num_products,
                    COUNT(DISTINCT store_id || '_' || product_id) as num_series
                FROM {parquet_sql}
            """).fetchone()
            min_dt, max_dt, num_days, num_stores, num_products, num_series = stats
        except Exception as exc:
            logger.warning("Error running entity and time series query: %s", exc)
    elif is_store_valid and is_prod_valid:
        try:
            stats = con.execute(f"""
                SELECT
                    COUNT(DISTINCT store_id) as num_stores,
                    COUNT(DISTINCT product_id) as num_products,
                    COUNT(DISTINCT store_id || '_' || product_id) as num_series
                FROM {parquet_sql}
            """).fetchone()
            num_stores, num_products, num_series = stats
        except Exception as exc:
            logger.warning("Error running entity store/product query: %s", exc)

    if metadata is not None:
        metadata.min_dt = str(min_dt) if min_dt is not None else ""
        metadata.max_dt = str(max_dt) if max_dt is not None else ""
        metadata.num_days = int(num_days) if num_days is not None else 0
        metadata.num_stores = int(num_stores) if num_stores is not None else 0
        metadata.num_products = int(num_products) if num_products is not None else 0
        metadata.num_series = int(num_series) if num_series is not None else 0

    if is_store_valid:
        if num_stores is not None:
            checks.append(
                ValidationCheck(
                    name="unique_stores",
                    status=(
                        ValidationStatus.PASS.value
                        if num_stores == exp_stores
                        else ValidationStatus.FAIL.value
                    ),
                    expected=exp_stores,
                    actual=int(num_stores),
                    explanation=f"Unique store count: {num_stores}",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="unique_stores",
                    status=ValidationStatus.FAIL.value,
                    expected=exp_stores,
                    actual="Query unavailable",
                    explanation="Failed to query unique store count",
                )
            )
    else:
        checks.append(
            ValidationCheck(
                name="unique_stores",
                status=ValidationStatus.FAIL.value,
                expected=exp_stores,
                actual=f"Invalid type: {store_field.type}",
                explanation=(
                    f"Cannot count unique stores: store_id has invalid type "
                    f"({store_field.type})"
                ),
            )
        )

    if is_prod_valid:
        if num_products is not None:
            checks.append(
                ValidationCheck(
                    name="unique_products",
                    status=(
                        ValidationStatus.PASS.value
                        if num_products == exp_prods
                        else ValidationStatus.FAIL.value
                    ),
                    expected=exp_prods,
                    actual=int(num_products),
                    explanation=f"Unique product count: {num_products}",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="unique_products",
                    status=ValidationStatus.FAIL.value,
                    expected=exp_prods,
                    actual="Query unavailable",
                    explanation="Failed to query unique product count",
                )
            )
    else:
        checks.append(
            ValidationCheck(
                name="unique_products",
                status=ValidationStatus.FAIL.value,
                expected=exp_prods,
                actual=f"Invalid type: {prod_field.type}",
                explanation=(
                    f"Cannot count unique products: product_id has invalid type "
                    f"({prod_field.type})"
                ),
            )
        )

    if is_store_valid and is_prod_valid:
        if num_series is not None:
            checks.append(
                ValidationCheck(
                    name="unique_series",
                    status=(
                        ValidationStatus.PASS.value
                        if num_series == exp_series
                        else ValidationStatus.FAIL.value
                    ),
                    expected=exp_series,
                    actual=int(num_series),
                    explanation=f"Unique store-product series: {num_series:,}",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="unique_series",
                    status=ValidationStatus.FAIL.value,
                    expected=exp_series,
                    actual="Query unavailable",
                    explanation="Failed to query unique series count",
                )
            )
    else:
        checks.append(
            ValidationCheck(
                name="unique_series",
                status=ValidationStatus.FAIL.value,
                expected=exp_series,
                actual="Invalid key types",
                explanation=(
                    "Cannot count unique series: store_id or product_id has "
                    "invalid type"
                ),
            )
        )

    if exp_start and exp_end:
        if not dt_type_valid:
            checks.append(
                ValidationCheck(
                    name="date_boundaries",
                    status=ValidationStatus.FAIL.value,
                    expected=f"{exp_start} to {exp_end}",
                    actual=f"Invalid dt column type: {dt_field.type}",
                    explanation="Cannot check date boundaries on non-string column",
                )
            )
        elif min_dt is not None and max_dt is not None:
            date_bound_ok = (str(min_dt) == str(exp_start)) and (
                str(max_dt) == str(exp_end)
            )
            checks.append(
                ValidationCheck(
                    name="date_boundaries",
                    status=(
                        ValidationStatus.PASS.value
                        if date_bound_ok
                        else ValidationStatus.FAIL.value
                    ),
                    expected=f"{exp_start} to {exp_end}",
                    actual=f"{min_dt} to {max_dt}",
                    explanation=f"Date range span: {min_dt} to {max_dt}",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="date_boundaries",
                    status=ValidationStatus.FAIL.value,
                    expected=f"{exp_start} to {exp_end}",
                    actual="Query unavailable",
                    explanation="Failed to query date boundaries",
                )
            )

    if exp_days is not None:
        if not dt_type_valid:
            checks.append(
                ValidationCheck(
                    name="date_continuity",
                    status=ValidationStatus.FAIL.value,
                    expected=exp_days,
                    actual=f"Invalid dt column type: {dt_field.type}",
                    explanation="Cannot check date continuity on non-string column",
                )
            )
        elif min_dt is not None and max_dt is not None and num_days is not None:
            try:
                start_date = datetime.date.fromisoformat(str(min_dt))
                end_date = datetime.date.fromisoformat(str(max_dt))
                expected_calendar_span = (end_date - start_date).days + 1
                continuous = (num_days == exp_days) and (
                    num_days == expected_calendar_span
                )
                checks.append(
                    ValidationCheck(
                        name="date_continuity",
                        status=(
                            ValidationStatus.PASS.value
                            if continuous
                            else ValidationStatus.FAIL.value
                        ),
                        expected=exp_days,
                        actual=int(num_days),
                        explanation=(
                            f"Calendar date count: {num_days} days "
                            f"(calendar span: {expected_calendar_span})"
                        ),
                    )
                )
            except (ValueError, TypeError) as exc:
                checks.append(
                    ValidationCheck(
                        name="date_continuity",
                        status=ValidationStatus.FAIL.value,
                        expected=exp_days,
                        actual=f"Invalid date format: {min_dt} to {max_dt}",
                        explanation=f"Failed to parse date range: {exc}",
                    )
                )
        else:
            checks.append(
                ValidationCheck(
                    name="date_continuity",
                    status=ValidationStatus.FAIL.value,
                    expected=exp_days,
                    actual="Query unavailable",
                    explanation="Failed to query date continuity",
                )
            )

    if is_store_valid and is_prod_valid:
        try:
            series_days_dist = con.execute(f"""
                SELECT MIN(cnt), MAX(cnt)
                FROM (
                    SELECT store_id, product_id, COUNT(*) as cnt
                    FROM {parquet_sql}
                    GROUP BY store_id, product_id
                )
            """).fetchone()
            min_series_days, max_series_days = series_days_dist

            balanced = (min_series_days == exp_days) and (max_series_days == exp_days)
            checks.append(
                ValidationCheck(
                    name="series_observation_balance",
                    status=(
                        ValidationStatus.PASS.value
                        if balanced
                        else ValidationStatus.FAIL.value
                    ),
                    expected=f"Exactly {exp_days} observations per series",
                    actual=f"Min: {min_series_days}, Max: {max_series_days}",
                    explanation=(
                        f"Series observation balance across "
                        f"{num_series or 0:,} series: "
                        f"min={min_series_days}, max={max_series_days}"
                    ),
                    counts={
                        "min_days_per_series": (
                            int(min_series_days) if min_series_days is not None else 0
                        ),
                        "max_days_per_series": (
                            int(max_series_days) if max_series_days is not None else 0
                        ),
                    },
                )
            )
        except Exception as exc:
            checks.append(
                ValidationCheck(
                    name="series_observation_balance",
                    status=ValidationStatus.FAIL.value,
                    expected=f"Exactly {exp_days} observations per series",
                    actual=f"Query error: {exc}",
                    explanation="Failed to calculate series observation balance",
                )
            )
    else:
        checks.append(
            ValidationCheck(
                name="series_observation_balance",
                status=ValidationStatus.FAIL.value,
                expected=f"Exactly {exp_days} observations per series",
                actual="Invalid key column types",
                explanation=(
                    "Cannot calculate series balance: store_id or product_id has "
                    "invalid type"
                ),
            )
        )

    if dt_type_valid:
        try:
            daily_rows_dist = con.execute(f"""
                SELECT MIN(cnt), MAX(cnt)
                FROM (
                    SELECT dt, COUNT(*) as cnt
                    FROM {parquet_sql}
                    GROUP BY dt
                )
            """).fetchone()
            min_daily_rows, max_daily_rows = daily_rows_dist

            daily_balanced = (min_daily_rows == exp_series) and (
                max_daily_rows == exp_series
            )
            checks.append(
                ValidationCheck(
                    name="daily_record_balance",
                    status=(
                        ValidationStatus.PASS.value
                        if daily_balanced
                        else ValidationStatus.FAIL.value
                    ),
                    expected=f"Exactly {exp_series} observations per calendar day",
                    actual=f"Min: {min_daily_rows}, Max: {max_daily_rows}",
                    explanation=(
                        f"Daily record counts across {num_days or 0} days: "
                        f"min={min_daily_rows}, max={max_daily_rows}"
                    ),
                    counts={
                        "min_daily_records": (
                            int(min_daily_rows) if min_daily_rows is not None else 0
                        ),
                        "max_daily_records": (
                            int(max_daily_rows) if max_daily_rows is not None else 0
                        ),
                    },
                )
            )
        except Exception as exc:
            checks.append(
                ValidationCheck(
                    name="daily_record_balance",
                    status=ValidationStatus.FAIL.value,
                    expected=f"Exactly {exp_series} observations per calendar day",
                    actual=f"Query error: {exc}",
                    explanation="Failed to calculate daily record balance",
                )
            )
    else:
        checks.append(
            ValidationCheck(
                name="daily_record_balance",
                status=ValidationStatus.FAIL.value,
                expected=f"Exactly {exp_series} observations per calendar day",
                actual=f"Invalid dt column type: {dt_field.type}",
                explanation=(
                    "Cannot calculate daily record balance on non-string column"
                ),
            )
        )

    return checks


def validate_hourly_data(
    file_path: Path,
    batch_size: int = 5000,
    tolerance: float = 1e-5,
    daytime_start: int = 6,
    daytime_end: int = 22,
) -> tuple[list[ValidationCheck], dict[str, Any]]:
    """Validate nested hourly arrays and reconciliation using streaming batches."""
    checks: list[ValidationCheck] = []
    diagnostics: dict[str, Any] = {}

    cols_to_read = [
        "sale_amount",
        "hours_sale",
        "stock_hour6_22_cnt",
        "hours_stock_status",
    ]

    try:
        schema = get_arrow_schema(file_path)
    except Exception as exc:
        for chk_name in (
            "hourly_array_lengths",
            "hourly_sales_validity",
            "hourly_stock_status_binary",
            "daily_hourly_sales_reconciliation",
            "stockout_count_reconciliation",
        ):
            checks.append(
                ValidationCheck(
                    name=chk_name,
                    status=ValidationStatus.FAIL.value,
                    expected="Readable file with hourly data",
                    actual=f"Read error: {exc}",
                    explanation="Failed to inspect schema for hourly validation",
                )
            )
        return checks, diagnostics

    actual_cols = set(schema.names)
    missing_cols = [c for c in cols_to_read if c not in actual_cols]
    if missing_cols:
        for chk_name in (
            "hourly_array_lengths",
            "hourly_sales_validity",
            "hourly_stock_status_binary",
            "daily_hourly_sales_reconciliation",
            "stockout_count_reconciliation",
        ):
            checks.append(
                ValidationCheck(
                    name=chk_name,
                    status=ValidationStatus.FAIL.value,
                    expected=f"Columns present: {cols_to_read}",
                    actual=f"Missing: {missing_cols}",
                    explanation="Required columns absent for hourly validation",
                )
            )
        return checks, diagnostics

    sale_field = schema.field("sale_amount")
    hours_sale_field = schema.field("hours_sale")
    stock_cnt_field = schema.field("stock_hour6_22_cnt")
    hours_stock_field = schema.field("hours_stock_status")

    is_sale_amt_numeric = _is_numeric_type(sale_field.type)
    is_hours_sale_list = pa.types.is_list(
        hours_sale_field.type
    ) or pa.types.is_large_list(hours_sale_field.type)
    is_hours_sale_valid = is_hours_sale_list and _is_numeric_type(
        hours_sale_field.type.value_type
    )
    is_stock_cnt_int = pa.types.is_integer(stock_cnt_field.type)
    is_hours_stock_list = pa.types.is_list(
        hours_stock_field.type
    ) or pa.types.is_large_list(hours_stock_field.type)
    is_hours_stock_valid = is_hours_stock_list and pa.types.is_integer(
        hours_stock_field.type.value_type
    )

    total_rows = 0
    invalid_array_length_count = 0
    null_parent_sales_rows = 0
    null_parent_stock_rows = 0
    null_parent_array_count = 0
    null_child_sales_rows = 0
    null_child_stock_rows = 0
    non_finite_hourly_sales_count = 0
    negative_hourly_sales_count = 0
    non_binary_stock_status_count = 0

    sales_mismatch_count = 0
    max_sales_diff = 0.0
    sum_sales_diff = 0.0
    finite_sales_diff_count = 0

    stock_count_mismatches = 0
    max_stock_diff = 0.0
    daytime_no_stockout_rows = 0
    daytime_full_stockout_rows = 0
    full_day_stockout_rows = 0

    stockout_hours_total = 0
    clean_stockout_hours_total = 0
    positive_sale_stockout_hours = 0
    sales_during_stockout_volume = 0.0

    for batch in iter_parquet_batches(
        file_path, columns=cols_to_read, batch_size=batch_size
    ):
        n = batch.num_rows
        total_rows += n

        sale_col = batch.column("sale_amount")
        stock_cnt_col = batch.column("stock_hour6_22_cnt")
        hours_sale_col = batch.column("hours_sale")
        hours_stock_col = batch.column("hours_stock_status")

        if isinstance(hours_sale_col, pa.ChunkedArray):
            hours_sale_col = hours_sale_col.combine_chunks()
        if isinstance(hours_stock_col, pa.ChunkedArray):
            hours_stock_col = hours_stock_col.combine_chunks()

        # Parent null and length verification for hours_sale
        if is_hours_sale_list:
            sale_parent_nulls = hours_sale_col.is_null().to_numpy(zero_copy_only=False)
            null_parent_sales_rows += int(sale_parent_nulls.sum())
            sale_lens = (
                hours_sale_col.value_lengths()
                .fill_null(0)
                .to_numpy(zero_copy_only=False)
            )
            bad_sale_len = (sale_lens != 24) | sale_parent_nulls
        else:
            sale_parent_nulls = np.zeros(n, dtype=bool)
            bad_sale_len = np.ones(n, dtype=bool)

        # Parent null and length verification for hours_stock_status
        if is_hours_stock_list:
            stock_parent_nulls = hours_stock_col.is_null().to_numpy(
                zero_copy_only=False
            )
            null_parent_stock_rows += int(stock_parent_nulls.sum())
            stock_lens = (
                hours_stock_col.value_lengths()
                .fill_null(0)
                .to_numpy(zero_copy_only=False)
            )
            bad_stock_len = (stock_lens != 24) | stock_parent_nulls
        else:
            stock_parent_nulls = np.zeros(n, dtype=bool)
            bad_stock_len = np.ones(n, dtype=bool)

        parent_null_mask = sale_parent_nulls | stock_parent_nulls
        null_parent_array_count += int(parent_null_mask.sum())
        bad_len_mask = bad_sale_len | bad_stock_len
        invalid_array_length_count += int(bad_len_mask.sum())

        # Sales element validation and reconciliation
        sales_flat: np.ndarray | None = None
        valid_sales_row: np.ndarray | None = None
        if is_hours_sale_valid and is_sale_amt_numeric:
            sale_amounts = sale_col.to_numpy(zero_copy_only=False)
            if not bad_sale_len.any():
                child_sale_null_total = hours_sale_col.values.null_count
                if child_sale_null_total > 0:
                    child_sale_null_bits = (
                        hours_sale_col.values.is_null()
                        .to_numpy(zero_copy_only=False)
                        .reshape(n, 24)
                    )
                    child_sale_null_rows = child_sale_null_bits.any(axis=1)
                    null_child_sales_rows += int(child_sale_null_rows.sum())
                else:
                    child_sale_null_bits = np.zeros((n, 24), dtype=bool)
                    child_sale_null_rows = np.zeros(n, dtype=bool)

                sales_flat = hours_sale_col.values.to_numpy(
                    zero_copy_only=False
                ).reshape(n, 24)
                nan_mask = np.isnan(sales_flat) & ~child_sale_null_bits
                inf_mask = np.isinf(sales_flat)
                non_finite_mask = nan_mask | inf_mask
                row_has_non_finite = non_finite_mask.any(axis=1)
                non_finite_hourly_sales_count += int(row_has_non_finite.sum())

                negative_mask = (sales_flat < 0) & np.isfinite(sales_flat)
                row_has_negative = negative_mask.any(axis=1)
                negative_hourly_sales_count += int(row_has_negative.sum())

                valid_sales_row = (
                    (~child_sale_null_rows)
                    & (~row_has_non_finite)
                    & np.isfinite(sale_amounts)
                )
                if valid_sales_row.any():
                    valid_sums = sales_flat[valid_sales_row].sum(axis=1)
                    valid_diffs = np.abs(sale_amounts[valid_sales_row] - valid_sums)
                    tol_mismatches = valid_diffs > tolerance
                    batch_max = float(valid_diffs.max())
                    if batch_max > max_sales_diff:
                        max_sales_diff = batch_max
                    sum_sales_diff += float(valid_diffs.sum())
                    finite_sales_diff_count += int(valid_sales_row.sum())
                    sales_mismatch_count += int((~valid_sales_row).sum()) + int(
                        tol_mismatches.sum()
                    )
                else:
                    sales_mismatch_count += n
            else:
                # Row-by-row fallback for sales
                valid_sales_flags = np.zeros(n, dtype=bool)
                sales_flat = np.zeros((n, 24), dtype=np.float64)
                for i in range(n):
                    s_p_null = bool(sale_parent_nulls[i])
                    s_val = hours_sale_col[i].as_py() if not s_p_null else None
                    if s_val is not None and len(s_val) == 24:
                        if any(x is None for x in s_val):
                            null_child_sales_rows += 1
                            sales_mismatch_count += 1
                        else:
                            s_arr = np.array(s_val, dtype=np.float64)
                            nan_any = np.isnan(s_arr).any()
                            inf_any = np.isinf(s_arr).any()
                            if nan_any or inf_any:
                                non_finite_hourly_sales_count += 1
                                sales_mismatch_count += 1
                            else:
                                if (s_arr < 0).any():
                                    negative_hourly_sales_count += 1
                                amt = sale_amounts[i]
                                if amt is not None and np.isfinite(amt):
                                    h_sum = float(s_arr.sum())
                                    diff = abs(float(amt) - h_sum)
                                    if diff > tolerance:
                                        sales_mismatch_count += 1
                                    if diff > max_sales_diff:
                                        max_sales_diff = diff
                                    sum_sales_diff += diff
                                    finite_sales_diff_count += 1
                                    valid_sales_flags[i] = True
                                    sales_flat[i] = s_arr
                                else:
                                    sales_mismatch_count += 1
                    else:
                        sales_mismatch_count += 1
                valid_sales_row = valid_sales_flags
        else:
            sales_mismatch_count += n

        # Stock element validation and daytime reconciliation
        stock_flat: np.ndarray | None = None
        valid_stock_row: np.ndarray | None = None
        clean_stock_counts: np.ndarray | None = None
        if is_hours_stock_valid and is_stock_cnt_int:
            stock_counts = stock_cnt_col.to_numpy(zero_copy_only=False)
            if not bad_stock_len.any():
                child_stock_null_total = hours_stock_col.values.null_count
                if child_stock_null_total > 0:
                    child_stock_null_bits = (
                        hours_stock_col.values.is_null()
                        .to_numpy(zero_copy_only=False)
                        .reshape(n, 24)
                    )
                    child_stock_null_rows = child_stock_null_bits.any(axis=1)
                    null_child_stock_rows += int(child_stock_null_rows.sum())
                    stock_flat_raw = hours_stock_col.values.to_numpy(
                        zero_copy_only=False
                    ).reshape(n, 24)
                    non_binary_mask = (
                        ~np.isin(stock_flat_raw, [0, 1])
                    ) & ~child_stock_null_bits
                    stock_flat = np.nan_to_num(stock_flat_raw, nan=-1.0).astype(
                        np.int64
                    )
                    row_has_non_binary = (non_binary_mask | child_stock_null_bits).any(
                        axis=1
                    )
                else:
                    child_stock_null_rows = np.zeros(n, dtype=bool)
                    stock_flat = hours_stock_col.values.to_numpy().reshape(n, 24)
                    non_binary_mask = ~np.isin(stock_flat, [0, 1])
                    row_has_non_binary = non_binary_mask.any(axis=1)

                non_binary_stock_status_count += int(row_has_non_binary.sum())

                valid_stock_row = (
                    (~child_stock_null_rows)
                    & (~non_binary_mask.any(axis=1))
                    & np.isfinite(stock_counts)
                )
                if valid_stock_row.any():
                    daytime_slice = stock_flat[
                        valid_stock_row, daytime_start:daytime_end
                    ]
                    daytime_sums = daytime_slice.sum(axis=1)
                    stock_diffs = np.abs(stock_counts[valid_stock_row] - daytime_sums)
                    tol_stock_mismatches = stock_diffs != 0
                    batch_stock_max = float(stock_diffs.max())
                    if batch_stock_max > max_stock_diff:
                        max_stock_diff = batch_stock_max
                    stock_count_mismatches += int((~valid_stock_row).sum()) + int(
                        tol_stock_mismatches.sum()
                    )
                else:
                    stock_count_mismatches += n
                clean_stock_counts = stock_counts
            else:
                # Row-by-row fallback for stock
                valid_stock_flags = np.zeros(n, dtype=bool)
                stock_flat = np.zeros((n, 24), dtype=np.int64)
                for i in range(n):
                    st_p_null = bool(stock_parent_nulls[i])
                    st_val = hours_stock_col[i].as_py() if not st_p_null else None
                    if st_val is not None and len(st_val) == 24:
                        if any(x is None for x in st_val):
                            null_child_stock_rows += 1
                            stock_count_mismatches += 1
                        else:
                            st_arr = np.array(st_val, dtype=np.int64)
                            if not np.isin(st_arr, [0, 1]).all():
                                non_binary_stock_status_count += 1
                                stock_count_mismatches += 1
                            else:
                                cnt = stock_counts[i]
                                if cnt is not None and np.isfinite(cnt):
                                    d_sum = int(st_arr[daytime_start:daytime_end].sum())
                                    s_diff = abs(int(cnt) - d_sum)
                                    if s_diff != 0:
                                        stock_count_mismatches += 1
                                    if s_diff > max_stock_diff:
                                        max_stock_diff = float(s_diff)
                                    valid_stock_flags[i] = True
                                    stock_flat[i] = st_arr
                                else:
                                    stock_count_mismatches += 1
                    else:
                        stock_count_mismatches += 1
                valid_stock_row = valid_stock_flags
                clean_stock_counts = stock_counts
        else:
            stock_count_mismatches += n

        # Diagnostics on valid clean rows
        if (
            stock_flat is not None
            and valid_stock_row is not None
            and clean_stock_counts is not None
        ):
            if valid_stock_row.any():
                v_counts = clean_stock_counts[valid_stock_row]
                v_stock = stock_flat[valid_stock_row]
                daytime_no_stockout_rows += int((v_counts == 0).sum())
                daytime_full_stockout_rows += int((v_counts == 16).sum())
                full_day_stockout_rows += int((v_stock.sum(axis=1) == 24).sum())

                stockout_mask = v_stock == 1
                stockout_hours_total += int(stockout_mask.sum())

                if (
                    is_hours_sale_valid
                    and sales_flat is not None
                    and valid_sales_row is not None
                ):
                    clean_mask = valid_sales_row & valid_stock_row
                    if clean_mask.any():
                        c_sales = sales_flat[clean_mask]
                        c_stock = stock_flat[clean_mask]
                        c_st_mask = c_stock == 1
                        clean_stockout_hours_total += int(c_st_mask.sum())
                        pos_in_st = c_st_mask & (c_sales > 0)
                        positive_sale_stockout_hours += int(pos_in_st.sum())
                        sales_during_stockout_volume += float(c_sales[pos_in_st].sum())

    # 1. Array length check
    if not is_hours_sale_list or not is_hours_stock_list:
        checks.append(
            ValidationCheck(
                name="hourly_array_lengths",
                status=ValidationStatus.FAIL.value,
                expected="List arrays for hours_sale and hours_stock_status",
                actual=(
                    f"hours_sale: {hours_sale_field.type}, "
                    f"hours_stock_status: {hours_stock_field.type}"
                ),
                explanation="Cannot validate array lengths on non-list columns",
            )
        )
    else:
        len_status = (
            ValidationStatus.PASS.value
            if (invalid_array_length_count == 0 and null_parent_array_count == 0)
            else ValidationStatus.FAIL.value
        )
        checks.append(
            ValidationCheck(
                name="hourly_array_lengths",
                status=len_status,
                expected="Exactly 24 elements per array with non-null parents",
                actual=(
                    f"{invalid_array_length_count} invalid length rows, "
                    f"{null_parent_array_count} null parent rows "
                    f"({null_parent_sales_rows} sales nulls, "
                    f"{null_parent_stock_rows} stock nulls)"
                    if (invalid_array_length_count or null_parent_array_count)
                    else "All 24 elements with non-null parents"
                ),
                explanation=(
                    "Array length validation across hours_sale and hours_stock_status"
                ),
                counts={
                    "invalid_length_rows": invalid_array_length_count,
                    "null_parent_sales_rows": null_parent_sales_rows,
                    "null_parent_stock_rows": null_parent_stock_rows,
                    "null_parent_rows": null_parent_array_count,
                },
            )
        )

    # 2. Hourly sales element validity
    if not is_hours_sale_valid:
        checks.append(
            ValidationCheck(
                name="hourly_sales_validity",
                status=ValidationStatus.FAIL.value,
                expected="List of numeric values",
                actual=f"Invalid type: {hours_sale_field.type}",
                explanation=(
                    "Cannot validate hourly sales elements on non-numeric list column"
                ),
            )
        )
    else:
        sales_elem_status = (
            ValidationStatus.PASS.value
            if (
                negative_hourly_sales_count == 0
                and non_finite_hourly_sales_count == 0
                and null_child_sales_rows == 0
                and null_parent_sales_rows == 0
            )
            else ValidationStatus.FAIL.value
        )
        checks.append(
            ValidationCheck(
                name="hourly_sales_validity",
                status=sales_elem_status,
                expected="Finite, non-negative, and non-null values",
                actual=(
                    f"Negative: {negative_hourly_sales_count}, "
                    f"Non-finite: {non_finite_hourly_sales_count}, "
                    f"Child nulls: {null_child_sales_rows}, "
                    f"Parent nulls: {null_parent_sales_rows}"
                ),
                explanation=(
                    "Hourly sales values must be finite, non-negative real numbers"
                ),
                counts={
                    "negative_sales_rows": negative_hourly_sales_count,
                    "non_finite_sales_rows": non_finite_hourly_sales_count,
                    "null_child_sales_rows": null_child_sales_rows,
                    "null_parent_sales_rows": null_parent_sales_rows,
                },
            )
        )

    # 3. Stockout status binary validity
    if not is_hours_stock_valid:
        checks.append(
            ValidationCheck(
                name="hourly_stock_status_binary",
                status=ValidationStatus.FAIL.value,
                expected="List of integer values",
                actual=f"Invalid type: {hours_stock_field.type}",
                explanation=("Cannot validate stock status on non-integer list column"),
            )
        )
    else:
        stock_status_val = (
            ValidationStatus.PASS.value
            if (
                non_binary_stock_status_count == 0
                and null_child_stock_rows == 0
                and null_parent_stock_rows == 0
            )
            else ValidationStatus.FAIL.value
        )
        checks.append(
            ValidationCheck(
                name="hourly_stock_status_binary",
                status=stock_status_val,
                expected="Strictly binary values in {0, 1} with zero nulls",
                actual=(
                    f"Non-binary: {non_binary_stock_status_count}, "
                    f"Child nulls: {null_child_stock_rows}, "
                    f"Parent nulls: {null_parent_stock_rows}"
                ),
                explanation=(
                    "Hourly stockout flags must be strictly binary integers {0, 1}"
                ),
                counts={
                    "non_binary_rows": non_binary_stock_status_count,
                    "null_child_stock_rows": null_child_stock_rows,
                    "null_parent_stock_rows": null_parent_stock_rows,
                },
            )
        )

    # 4. Daily and hourly sales reconciliation
    if not (is_sale_amt_numeric and is_hours_sale_valid):
        checks.append(
            ValidationCheck(
                name="daily_hourly_sales_reconciliation",
                status=ValidationStatus.FAIL.value,
                expected="Numeric sale_amount and list of numeric hours_sale",
                actual=(
                    f"sale_amount: {sale_field.type}, "
                    f"hours_sale: {hours_sale_field.type}"
                ),
                explanation="Cannot reconcile sales on invalid column types",
            )
        )
    else:
        mean_sales_diff = (
            sum_sales_diff / finite_sales_diff_count
            if finite_sales_diff_count > 0
            else 0.0
        )
        rec_status = (
            ValidationStatus.PASS.value
            if sales_mismatch_count == 0
            else ValidationStatus.FAIL.value
        )
        checks.append(
            ValidationCheck(
                name="daily_hourly_sales_reconciliation",
                status=rec_status,
                expected=(
                    f"sale_amount == sum(hours_sale) within tolerance {tolerance}"
                ),
                actual=(
                    f"{sales_mismatch_count} mismatches "
                    f"(max diff: {max_sales_diff:.2e}, "
                    f"mean diff: {mean_sales_diff:.2e})"
                ),
                explanation=("Daily recorded sales reconcile with sum of hourly sales"),
                counts={
                    "mismatch_count": sales_mismatch_count,
                    "max_diff": max_sales_diff,
                    "mean_diff": mean_sales_diff,
                    "within_tolerance_rows": max(0, total_rows - sales_mismatch_count),
                },
            )
        )

    # 5. Daytime stockout count reconciliation
    if not (is_stock_cnt_int and is_hours_stock_valid):
        checks.append(
            ValidationCheck(
                name="stockout_count_reconciliation",
                status=ValidationStatus.FAIL.value,
                expected=(
                    "Integer stock_hour6_22_cnt and list of integer hours_stock_status"
                ),
                actual=(
                    f"stock_hour6_22_cnt: {stock_cnt_field.type}, "
                    f"hours_stock_status: {hours_stock_field.type}"
                ),
                explanation="Cannot reconcile stock counts on invalid column types",
            )
        )
    else:
        stock_rec_status = (
            ValidationStatus.PASS.value
            if stock_count_mismatches == 0
            else ValidationStatus.FAIL.value
        )
        checks.append(
            ValidationCheck(
                name="stockout_count_reconciliation",
                status=stock_rec_status,
                expected="stock_hour6_22_cnt == sum(hours_stock_status[6:22])",
                actual=(
                    f"{stock_count_mismatches} mismatches (max diff: {max_stock_diff})"
                ),
                explanation=(
                    "Daytime stockout count aligns with sum of hours 6 through 21"
                ),
                counts={
                    "mismatch_count": stock_count_mismatches,
                    "max_diff": max_stock_diff,
                },
            )
        )

    if is_hours_sale_valid and is_hours_stock_valid:
        cooccurrence_available = True
        pct_pos_sales_stockout: float | None = (
            round((positive_sale_stockout_hours / clean_stockout_hours_total * 100), 4)
            if clean_stockout_hours_total > 0
            else 0.0
        )
        pos_sale_st_hrs_val: int | None = positive_sale_stockout_hours
        sales_vol_val: float | None = round(sales_during_stockout_volume, 3)
    else:
        cooccurrence_available = False
        pct_pos_sales_stockout = None
        pos_sale_st_hrs_val = None
        sales_vol_val = None

    diagnostics["stockout_diagnostics"] = {
        "total_rows": total_rows,
        "daytime_no_stockout_rows": daytime_no_stockout_rows,
        "daytime_no_stockout_pct": (
            round(daytime_no_stockout_rows / total_rows * 100, 2)
            if total_rows > 0
            else 0.0
        ),
        "daytime_full_stockout_rows": daytime_full_stockout_rows,
        "full_day_stockout_rows": full_day_stockout_rows,
        "total_stockout_hours": stockout_hours_total,
        "cooccurrence_available": cooccurrence_available,
        "positive_sale_stockout_hours": pos_sale_st_hrs_val,
        "pct_positive_sale_stockout_hours": pct_pos_sales_stockout,
        "sales_volume_during_stockout": sales_vol_val,
    }

    return checks, diagnostics


def validate_anomalies(
    con: duckdb.DuckDBPyConnection, file_path: Path
) -> list[ValidationCheck]:
    """Validate documented anomalies such as discount range."""
    checks: list[ValidationCheck] = []
    parquet_sql = f"read_parquet('{file_path.as_posix()}')"

    try:
        schema = get_arrow_schema(file_path)
    except Exception as exc:
        checks.append(
            ValidationCheck(
                name="anomalies_schema_read",
                status=ValidationStatus.FAIL.value,
                expected="Readable Parquet schema",
                actual=f"Read error: {exc}",
                explanation="Cannot validate anomalies on unreadable file",
            )
        )
        return checks

    if "discount" not in schema.names:
        checks.append(
            ValidationCheck(
                name="discount_non_negative",
                status=ValidationStatus.FAIL.value,
                expected="discount column present",
                actual="Column missing",
                explanation="Cannot check discount range: discount column missing",
            )
        )
        checks.append(
            ValidationCheck(
                name="discount_ceiling_anomaly",
                status=ValidationStatus.FAIL.value,
                expected="discount column present",
                actual="Column missing",
                explanation="Cannot check discount ceiling: discount column missing",
            )
        )
        return checks

    disc_field = schema.field("discount")
    if not _is_numeric_type(disc_field.type):
        checks.append(
            ValidationCheck(
                name="discount_non_negative",
                status=ValidationStatus.FAIL.value,
                expected="Numeric discount column",
                actual=f"Invalid type: {disc_field.type}",
                explanation="Cannot check discount range on non-numeric column",
            )
        )
        checks.append(
            ValidationCheck(
                name="discount_ceiling_anomaly",
                status=ValidationStatus.FAIL.value,
                expected="Numeric discount column",
                actual=f"Invalid type: {disc_field.type}",
                explanation="Cannot check discount ceiling on non-numeric column",
            )
        )
        return checks

    res = con.execute(f"""
        SELECT
            MIN(discount) as min_disc,
            MAX(discount) as max_disc,
            SUM(CASE WHEN discount < 0.0 THEN 1 ELSE 0 END) as neg_disc,
            SUM(CASE WHEN discount > 1.0 THEN 1 ELSE 0 END) as gt_1_disc,
            SUM(CASE WHEN discount == 1.0 THEN 1 ELSE 0 END) as eq_1_disc,
            SUM(CASE WHEN discount < 1.0 AND discount >= 0.0 THEN 1 ELSE 0 END)
                as discounted_rows
        FROM {parquet_sql}
    """).fetchone()

    min_disc, max_disc, neg_disc, gt_1_disc, eq_1_disc, discounted_rows = res

    # Negative discount is a critical failure
    if neg_disc is not None and neg_disc > 0:
        checks.append(
            ValidationCheck(
                name="discount_non_negative",
                status=ValidationStatus.FAIL.value,
                expected="discount >= 0.0",
                actual=f"{neg_disc} negative discount records (min: {min_disc})",
                explanation="Negative discount rate values detected",
                counts={"negative_discount_count": int(neg_disc)},
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="discount_non_negative",
                status=ValidationStatus.PASS.value,
                expected="discount >= 0.0",
                actual="Zero negative discount values",
                explanation="Discount rates are non-negative across all records",
            )
        )

    # Discount greater than 1.0 produces a WARNING per specification
    if gt_1_disc is not None and gt_1_disc > 0:
        checks.append(
            ValidationCheck(
                name="discount_ceiling_anomaly",
                status=ValidationStatus.WARNING.value,
                expected="discount <= 1.0 (conventional promotional range)",
                actual=f"{gt_1_disc} records with discount > 1.0 (max: {max_disc})",
                explanation=(
                    f"Found {gt_1_disc} records with discount > 1.0 "
                    f"(maximum {max_disc}). discount > 1.0 is an observed "
                    "anomaly relative to the conventional discount range. "
                    "Its cause is not established by this validation. Raw "
                    "values are preserved without clipping or correction."
                ),
                counts={
                    "discount_gt_1_count": int(gt_1_disc),
                    "max_discount": float(max_disc),
                },
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="discount_ceiling_anomaly",
                status=ValidationStatus.PASS.value,
                expected="discount <= 1.0",
                actual=f"Max discount: {max_disc}",
                explanation="All discount values are <= 1.0",
            )
        )

    return checks


def validate_cross_partition_consistency(
    con: duckdb.DuckDBPyConnection,
    train_path: Path,
    eval_path: Path,
    train_valid: bool = True,
    eval_valid: bool = True,
) -> ValidationResult:
    """Validate consistency across training and evaluation partitions."""
    result = ValidationResult(dataset_name="cross_partition")

    if (
        not train_valid
        or not eval_valid
        or not train_path.is_file()
        or not eval_path.is_file()
    ):
        result.add_check(
            ValidationCheck(
                name="temporal_partition_separation",
                status=ValidationStatus.FAIL.value,
                expected="train_max_dt < eval_min_dt",
                actual="Partition missing or invalid",
                explanation="Cannot verify separation: partition failure",
            )
        )
        result.add_check(
            ValidationCheck(
                name="cross_partition_series_consistency",
                status=ValidationStatus.FAIL.value,
                expected="100% series overlap between train and evaluation",
                actual="Partition missing or invalid",
                explanation="Cannot verify series overlap: partition failure",
            )
        )
        return result

    try:
        train_schema = get_arrow_schema(train_path)
        eval_schema = get_arrow_schema(eval_path)
    except Exception as exc:
        result.add_check(
            ValidationCheck(
                name="cross_partition_schema_read",
                status=ValidationStatus.FAIL.value,
                expected="Readable schemas",
                actual=f"Read error: {exc}",
                explanation="Failed to inspect schemas for cross-partition checks",
            )
        )
        return result

    req_cols = {"store_id", "product_id", "dt"}
    if not req_cols.issubset(train_schema.names) or not req_cols.issubset(
        eval_schema.names
    ):
        result.add_check(
            ValidationCheck(
                name="cross_partition_required_columns",
                status=ValidationStatus.FAIL.value,
                expected="store_id, product_id, dt in both partitions",
                actual="Missing required key columns in one or both files",
                explanation="Cannot run cross-partition checks: columns missing",
            )
        )
        return result

    train_sql = f"read_parquet('{train_path.as_posix()}')"
    eval_sql = f"read_parquet('{eval_path.as_posix()}')"

    # Validate dt column types for temporal separation
    train_dt_t = train_schema.field("dt").type
    eval_dt_t = eval_schema.field("dt").type
    if not _is_string_or_date_type(train_dt_t) or not _is_string_or_date_type(
        eval_dt_t
    ):
        result.add_check(
            ValidationCheck(
                name="temporal_partition_separation",
                status=ValidationStatus.FAIL.value,
                expected="String/date dt columns in both partitions",
                actual=f"train dt: {train_dt_t}, eval dt: {eval_dt_t}",
                explanation=(
                    "Cannot verify temporal separation on non-string date columns"
                ),
            )
        )
    else:
        train_max_dt = con.execute(f"SELECT MAX(dt) FROM {train_sql}").fetchone()[0]
        eval_min_dt = con.execute(f"SELECT MIN(dt) FROM {eval_sql}").fetchone()[0]

        temporal_order_ok = (
            train_max_dt is not None
            and eval_min_dt is not None
            and str(train_max_dt) < str(eval_min_dt)
        )
        result.add_check(
            ValidationCheck(
                name="temporal_partition_separation",
                status=(
                    ValidationStatus.PASS.value
                    if temporal_order_ok
                    else ValidationStatus.FAIL.value
                ),
                expected=(
                    f"train_max_dt < eval_min_dt ({train_max_dt} < {eval_min_dt})"
                ),
                actual=f"train max: {train_max_dt}, eval min: {eval_min_dt}",
                explanation=(
                    "Training partition strictly precedes evaluation partition "
                    "without temporal overlap"
                ),
            )
        )

    # Validate key columns for cross-partition consistency
    keys_nested = any(
        pa.types.is_nested(s.field(c).type)
        for s in (train_schema, eval_schema)
        for c in ("store_id", "product_id")
    )
    if keys_nested:
        result.add_check(
            ValidationCheck(
                name="cross_partition_series_consistency",
                status=ValidationStatus.FAIL.value,
                expected="Scalar store_id and product_id columns",
                actual="Nested types in store_id or product_id",
                explanation="Cannot verify series overlap on nested key columns",
            )
        )
    else:
        train_store_t = train_schema.field("store_id").type
        eval_store_t = eval_schema.field("store_id").type
        train_prod_t = train_schema.field("product_id").type
        eval_prod_t = eval_schema.field("product_id").type

        if train_store_t != eval_store_t or train_prod_t != eval_prod_t:
            result.add_check(
                ValidationCheck(
                    name="cross_partition_series_consistency",
                    status=ValidationStatus.FAIL.value,
                    expected=(
                        "Compatible store_id and product_id types across partitions"
                    ),
                    actual=(
                        f"store_id: ({train_store_t} vs {eval_store_t}), "
                        f"product_id: ({train_prod_t} vs {eval_prod_t})"
                    ),
                    explanation=(
                        "Cannot verify series overlap: store_id or product_id types "
                        "are incompatible between training and evaluation partitions"
                    ),
                )
            )
        else:
            try:
                con.execute(
                    f"CREATE OR REPLACE TEMP VIEW v_train_sp AS "
                    f"SELECT DISTINCT store_id, product_id FROM {train_sql}"
                )
                con.execute(
                    f"CREATE OR REPLACE TEMP VIEW v_eval_sp AS "
                    f"SELECT DISTINCT store_id, product_id FROM {eval_sql}"
                )

                train_cnt = con.execute("SELECT COUNT(*) FROM v_train_sp").fetchone()[0]
                eval_cnt = con.execute("SELECT COUNT(*) FROM v_eval_sp").fetchone()[0]
                common_cnt = con.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT * FROM v_train_sp INTERSECT SELECT * FROM v_eval_sp"
                    ")"
                ).fetchone()[0]
                eval_only = con.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT * FROM v_eval_sp EXCEPT SELECT * FROM v_train_sp"
                    ")"
                ).fetchone()[0]
                train_only = con.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT * FROM v_train_sp EXCEPT SELECT * FROM v_eval_sp"
                    ")"
                ).fetchone()[0]

                entity_consistent = (
                    train_cnt == eval_cnt == common_cnt
                    and eval_only == 0
                    and train_only == 0
                )
                result.add_check(
                    ValidationCheck(
                        name="cross_partition_series_consistency",
                        status=(
                            ValidationStatus.PASS.value
                            if entity_consistent
                            else ValidationStatus.FAIL.value
                        ),
                        expected="100% series overlap between train and evaluation",
                        actual=(
                            f"Common: {common_cnt:,}, "
                            f"Train-only: {train_only}, "
                            f"Eval-only: {eval_only}"
                        ),
                        explanation=(
                            "Training and evaluation partitions contain identical "
                            "store-product series"
                            if entity_consistent
                            else "Mismatch in store-product series across partitions"
                        ),
                        counts={
                            "train_series": int(train_cnt),
                            "eval_series": int(eval_cnt),
                            "common_series": int(common_cnt),
                        },
                    )
                )
            except Exception as exc:
                result.add_check(
                    ValidationCheck(
                        name="cross_partition_series_consistency",
                        status=ValidationStatus.FAIL.value,
                        expected="100% series overlap between train and evaluation",
                        actual=f"Query error: {exc}",
                        explanation="Failed to execute cross-partition series query",
                    )
                )

    return result


def calculate_evaluation_cohort(
    con: duckdb.DuckDBPyConnection,
    eval_path: Path,
    eval_valid: bool = True,
) -> dict[str, Any]:
    """Calculate the evaluation cohort breakdown for stock_hour6_22_cnt == 0."""
    fallback_cohort = {
        "total_rows": 0,
        "daytime_no_stockout_rows": 0,
        "pct_daytime_no_stockout": 0.0,
        "daytime_with_stockout_rows": 0,
        "pct_daytime_with_stockout": 0.0,
    }

    if not eval_valid or not eval_path.is_file():
        return fallback_cohort

    try:
        schema = get_arrow_schema(eval_path)
    except Exception:
        return fallback_cohort

    if "stock_hour6_22_cnt" not in schema.names:
        return fallback_cohort

    stock_cnt_f = schema.field("stock_hour6_22_cnt")
    if not pa.types.is_integer(stock_cnt_f.type):
        return fallback_cohort

    eval_sql = f"read_parquet('{eval_path.as_posix()}')"
    res = con.execute(f"""
        SELECT
            COUNT(*) as total_rows,
            SUM(CASE WHEN stock_hour6_22_cnt == 0 THEN 1 ELSE 0 END)
                as daytime_no_stockout,
            SUM(CASE WHEN stock_hour6_22_cnt > 0 THEN 1 ELSE 0 END)
                as daytime_with_stockout
        FROM {eval_sql}
    """).fetchone()

    total_rows, no_stockout, with_stockout = res
    if total_rows is None or total_rows == 0:
        return fallback_cohort

    no_st = int(no_stockout) if no_stockout is not None else 0
    with_st = int(with_stockout) if with_stockout is not None else 0
    pct_no_stockout = round(no_st / total_rows * 100, 2)

    return {
        "total_rows": int(total_rows),
        "daytime_no_stockout_rows": no_st,
        "pct_daytime_no_stockout": pct_no_stockout,
        "daytime_with_stockout_rows": with_st,
        "pct_daytime_with_stockout": round(100.0 - pct_no_stockout, 2),
    }


def run_validation_pipeline(
    config: dict[str, Any] | None = None,
    reports_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute full data ingestion and validation pipeline safely."""
    if config is None:
        config = load_config("data")

    logger.info("Initializing Aarohan Data Validation Pipeline")
    root_dir = get_project_root()
    target_reports_dir = Path(reports_dir) if reports_dir else root_dir / "reports"
    target_manifest_path = (
        Path(manifest_path)
        if manifest_path
        else root_dir / "artifacts" / "metadata" / "dataset_manifest.json"
    )

    if "raw_train_path" in config:
        train_path = resolve_dataset_path(config, "raw_train_path")
    else:
        train_path = resolve_project_path(
            config.get("raw_data", {}).get("train_path", "data/raw/train.parquet")
        )

    if "raw_eval_path" in config:
        eval_path = resolve_dataset_path(config, "raw_eval_path")
    else:
        eval_path = resolve_project_path(
            config.get("raw_data", {}).get("eval_path", "data/raw/eval.parquet")
        )

    expected_size = config.get("dataset_size", {})
    expected_entities = config.get("expected_entities", {})
    expected_schema = config.get("expected_schema", {})
    checksums = config.get("checksums", {})
    hourly_cfg = config.get("hourly_settings", {})
    proc_cfg = config.get("processing", {})

    batch_size = proc_cfg.get("batch_size", 5000)
    threads = proc_cfg.get("duckdb_threads", 2)
    max_memory = proc_cfg.get("duckdb_memory_limit", "512MB")
    tolerance = hourly_cfg.get("sales_reconciliation_tolerance", 1e-5)
    daytime_start = hourly_cfg.get("daytime_stockout_start", 6)
    daytime_end = hourly_cfg.get("daytime_stockout_end", 22)

    con = create_duckdb_connection(threads=threads, max_memory=max_memory)

    try:
        # 1. Training partition validation
        logger.info("Validating training partition: %s", train_path)
        train_result = ValidationResult(dataset_name="train")

        train_file_checks, train_meta = validate_file_level(
            train_path,
            expected_sha256=checksums.get("train_sha256"),
            expected_rows=expected_size.get("train", {}).get("rows"),
            expected_cols=expected_size.get("train", {}).get("columns"),
        )
        for c in train_file_checks:
            train_result.add_check(c)
        train_result.metadata = train_meta

        if train_meta is not None:
            for c in validate_schema(train_path, expected_schema):
                train_result.add_check(c)
            for c in validate_general_integrity(con, train_path):
                train_result.add_check(c)
            for c in validate_entity_and_time_series(
                con,
                train_path,
                expected_size.get("train", {}),
                expected_entities,
                metadata=train_meta,
            ):
                train_result.add_check(c)
            hourly_checks, hourly_diag = validate_hourly_data(
                train_path,
                batch_size=batch_size,
                tolerance=tolerance,
                daytime_start=daytime_start,
                daytime_end=daytime_end,
            )
            for c in hourly_checks:
                train_result.add_check(c)
            train_result.diagnostics.update(hourly_diag)

            for c in validate_anomalies(con, train_path):
                train_result.add_check(c)

        # 2. Evaluation partition validation
        logger.info("Validating evaluation partition: %s", eval_path)
        eval_result = ValidationResult(dataset_name="eval")

        eval_file_checks, eval_meta = validate_file_level(
            eval_path,
            expected_sha256=checksums.get("eval_sha256"),
            expected_rows=expected_size.get("eval", {}).get("rows"),
            expected_cols=expected_size.get("eval", {}).get("columns"),
        )
        for c in eval_file_checks:
            eval_result.add_check(c)
        eval_result.metadata = eval_meta

        if eval_meta is not None:
            for c in validate_schema(eval_path, expected_schema):
                eval_result.add_check(c)
            for c in validate_general_integrity(con, eval_path):
                eval_result.add_check(c)
            for c in validate_entity_and_time_series(
                con,
                eval_path,
                expected_size.get("eval", {}),
                expected_entities,
                metadata=eval_meta,
            ):
                eval_result.add_check(c)
            hourly_checks, hourly_diag = validate_hourly_data(
                eval_path,
                batch_size=batch_size,
                tolerance=tolerance,
                daytime_start=daytime_start,
                daytime_end=daytime_end,
            )
            for c in hourly_checks:
                eval_result.add_check(c)
            eval_result.diagnostics.update(hourly_diag)

            for c in validate_anomalies(con, eval_path):
                eval_result.add_check(c)

        # 3. Cross-partition validation & Cohort evaluation
        logger.info("Performing cross-partition consistency checks")
        train_is_readable = train_meta is not None
        eval_is_readable = eval_meta is not None
        cross_result = validate_cross_partition_consistency(
            con,
            train_path,
            eval_path,
            train_valid=train_is_readable,
            eval_valid=eval_is_readable,
        )
        cohort_diag = calculate_evaluation_cohort(
            con, eval_path, eval_valid=(eval_meta is not None)
        )
        eval_result.diagnostics["evaluation_cohort"] = cohort_diag

        # Determine overall pipeline status
        all_results = [train_result, eval_result, cross_result]
        all_statuses = [r.status for r in all_results]

        if ValidationStatus.FAIL.value in all_statuses:
            overall_status = ValidationStatus.FAIL.value
        elif ValidationStatus.WARNING.value in all_statuses:
            overall_status = ValidationStatus.WARNING.value
        else:
            overall_status = ValidationStatus.PASS.value

        pipeline_summary = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "overall_status": overall_status,
            "train": train_result.to_dict(),
            "eval": eval_result.to_dict(),
            "cross_partition": cross_result.to_dict(),
            "evaluation_cohort": cohort_diag,
            "system_info": {
                "duckdb_threads": threads,
                "duckdb_memory_limit": max_memory,
                "batch_size": batch_size,
            },
        }

        # Generate and save report deliverables
        _generate_output_deliverables(
            root_dir,
            pipeline_summary,
            config,
            reports_dir=target_reports_dir,
            manifest_path=target_manifest_path,
        )

        logger.info("Validation completed with overall status: %s", overall_status)
        return pipeline_summary

    finally:
        con.close()


def _generate_output_deliverables(
    root_dir: Path,
    summary: dict[str, Any],
    config: dict[str, Any],
    reports_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> None:
    """Write machine-readable and human-readable reporting artifacts."""
    val_reports_dir = reports_dir if reports_dir is not None else root_dir / "reports"
    val_manifest_path = (
        manifest_path
        if manifest_path is not None
        else root_dir / "artifacts" / "metadata" / "dataset_manifest.json"
    )

    # 1. Machine-readable validation summary: reports/tables/data_validation.json
    val_json_path = val_reports_dir / "tables" / "data_validation.json"
    val_json_path.parent.mkdir(parents=True, exist_ok=True)
    val_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote machine-readable validation summary: %s", val_json_path)

    # 2. Dataset manifest: artifacts/metadata/dataset_manifest.json
    val_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    train_meta = summary.get("train", {}).get("metadata")
    eval_meta = summary.get("eval", {}).get("metadata")

    train_checks = summary.get("train", {}).get("checks", [])
    disc_counts: dict[str, Any] = {}
    for c in train_checks:
        if c.get("name") == "discount_ceiling_anomaly":
            disc_counts = c.get("counts", {})
            break

    tr_gt1 = disc_counts.get("discount_gt_1_count", 0)
    tr_max_d = disc_counts.get("max_discount")
    tr_diag = (
        summary.get("train", {}).get("diagnostics", {}).get("stockout_diagnostics", {})
    )
    ev_diag = (
        summary.get("eval", {}).get("diagnostics", {}).get("stockout_diagnostics", {})
    )
    tr_pos_pct = tr_diag.get("pct_positive_sale_stockout_hours")
    ev_pos_pct = ev_diag.get("pct_positive_sale_stockout_hours")

    known_limitations = [
        (
            "Sales values are globally normalized demand index values; "
            "original physical unit counts cannot be reconstructed."
        ),
    ]
    if tr_gt1 > 0:
        known_limitations.append(
            f"{tr_gt1} training records feature discount values > 1.0 "
            f"(maximum {tr_max_d}), which is an observed anomaly "
            "relative to the conventional discount range with an unknown cause. "
            "Raw values are preserved without clipping or correction."
        )
    else:
        known_limitations.append(
            "Values exceeding 1.0 in discount rate may occur as observed "
            "anomalies relative to the conventional discount range with an "
            "unknown cause."
        )

    if (
        tr_pos_pct is not None
        and ev_pos_pct is not None
        and (tr_pos_pct > 0 or ev_pos_pct > 0)
    ):
        known_limitations.append(
            f"Positive sales co-occur with stockout flags in approximately "
            f"{tr_pos_pct:.2f}% of training stockout hours and "
            f"{ev_pos_pct:.2f}% of evaluation stockout hours. Their "
            "within-hour timing and cause are not established by this validation."
        )
    else:
        known_limitations.append(
            "Positive sales may co-occur with stockout flags within the same hour; "
            "their within-hour timing and cause are not established by this validation."
        )

    known_limitations.append(
        "The project's selected primary evaluation cohort uses "
        "stock_hour6_22_cnt == 0 representing daytime stockout-free "
        "series-days; daytime availability does not guarantee "
        "24-hour stockout absence."
    )

    measured_stores = (
        train_meta.get("num_stores")
        if isinstance(train_meta, dict)
        else getattr(train_meta, "num_stores", None)
    ) or (
        eval_meta.get("num_stores")
        if isinstance(eval_meta, dict)
        else getattr(eval_meta, "num_stores", None)
    )
    measured_products = (
        train_meta.get("num_products")
        if isinstance(train_meta, dict)
        else getattr(train_meta, "num_products", None)
    ) or (
        eval_meta.get("num_products")
        if isinstance(eval_meta, dict)
        else getattr(eval_meta, "num_products", None)
    )
    measured_series = (
        train_meta.get("num_series")
        if isinstance(train_meta, dict)
        else getattr(train_meta, "num_series", None)
    ) or (
        eval_meta.get("num_series")
        if isinstance(eval_meta, dict)
        else getattr(eval_meta, "num_series", None)
    )
    if (
        measured_stores is not None
        and measured_products is not None
        and measured_series is not None
    ):
        entity_counts = {
            "stores": int(measured_stores),
            "products": int(measured_products),
            "store_product_series": int(measured_series),
        }
    else:
        entity_counts = config.get("expected_entities", {})

    manifest_data = {
        "dataset_name": "FreshRetailNet-50K",
        "provider": "Dingdong Inc.",
        "source_url": "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "files": {
            "train": train_meta,
            "eval": eval_meta,
        },
        "entity_counts": entity_counts,
        "known_limitations": known_limitations,
    }
    val_manifest_path.write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote dataset manifest: %s", val_manifest_path)

    # 3. Data dictionary: docs/data_dictionary.md
    dict_path = root_dir / "docs" / "data_dictionary.md"
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    dict_content = _build_data_dictionary_md()
    dict_path.write_text(dict_content, encoding="utf-8")
    logger.info("Wrote data dictionary: %s", dict_path)

    # 4. Human-readable data profile: reports/data_profile.md
    profile_path = val_reports_dir / "data_profile.md"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_content = _build_data_profile_md(summary)
    profile_path.write_text(profile_content, encoding="utf-8")
    logger.info("Wrote human-readable data profile: %s", profile_path)


def _build_data_dictionary_md() -> str:
    """Generate Markdown data dictionary documenting all 19 columns."""
    lines = [
        "# Aarohan: Data Dictionary",
        "",
        (
            "This document details all 19 columns in the FreshRetailNet-50K "
            "dataset used by the Aarohan demand planning system."
        ),
        "",
        "---",
        "",
        "## Column Specifications",
        "",
        (
            "| Column Name | Data Type | Status | Business Meaning | "
            "Interpretation Notes |"
        ),
        "| :--- | :--- | :--- | :--- | :--- |",
        (
            "| `city_id` | `int64` | Observed | City entity identifier | "
            "Categorical ID denoting metropolitan delivery area "
            "(18 unique cities). |"
        ),
        (
            "| `store_id` | `int64` | Observed | Retail store/warehouse identifier | "
            "Frontline fulfilment station ID (898 unique stores). |"
        ),
        (
            "| `management_group_id` | `int64` | Observed | "
            "Operational cluster identifier | Administrative store management cluster "
            "(7 unique groups). |"
        ),
        (
            "| `first_category_id` | `int64` | Observed | "
            "L1 Product hierarchy category | Broad merchandise department "
            "(32 categories, e.g. Vegetables, Meat). |"
        ),
        (
            "| `second_category_id` | `int64` | Observed | "
            "L2 Product hierarchy category | Sub-department category (84 categories). |"
        ),
        (
            "| `third_category_id` | `int64` | Observed | "
            "L3 Product hierarchy category | Granular product sub-category "
            "(233 categories). |"
        ),
        (
            "| `product_id` | `int64` | Observed | SKU item identifier | "
            "Stock keeping unit identifier (865 unique perishable items). |"
        ),
        (
            "| `dt` | `string` | Observed | Calendar date | "
            "Date of transaction in `YYYY-MM-DD` ISO-8601 format. |"
        ),
        (
            "| `sale_amount` | `double` | Observed | Daily sales volume | "
            "Daily recorded sales quantity after global normalization by Dingdong Inc. "
            "Values are non-negative normalized demand index values, not physical unit "
            "counts. Reconciles with `sum(hours_sale)` within float precision "
            "($10^{-5}$). |"
        ),
        (
            "| `hours_sale` | `list<double>` | Observed | "
            "24-hour sales distribution | Array of exactly 24 non-negative "
            "floating-point values representing sales for each hour from 00:00 "
            "to 23:00. |"
        ),
        (
            "| `stock_hour6_22_cnt` | `int32` | Derived | "
            "Daytime stockout hour count | Count of daytime hours (06:00 through "
            "21:00 inclusive) with recorded stockout. Integer range $[0, 16]$. "
            "Equals $\\sum_{h=6}^{21} \\text{hours\\_stock\\_status}_h$. |"
        ),
        (
            "| `hours_stock_status` | `list<int64>` | Observed | "
            "24-hour stockout flags | Array of exactly 24 binary integers. "
            "`1` indicates out-of-stock during that hour; `0` indicates item was "
            "in stock. |"
        ),
        (
            "| `discount` | `double` | Observed | Daily discount rate | "
            "Daily discount rate as a promotional factor (selling price / list price). "
            "1.0 indicates full retail price; values below 1.0 represent discounts. "
            "Values exceeding 1.0 may occur as observed anomalies. |"
        ),
        (
            "| `holiday_flag` | `int32` | Derived | Statutory holiday indicator | "
            "Binary flag (`1` for statutory holidays, `0` for regular days). |"
        ),
        (
            "| `activity_flag` | `int32` | Derived | Marketing campaign indicator | "
            "Binary flag (`1` if a promotional campaign was active, `0` otherwise). |"
        ),
        (
            "| `precpt` | `double` | Observed | Daily precipitation | "
            "Realized rainfall measurement in millimeters (mm). |"
        ),
        (
            "| `avg_temperature` | `double` | Observed | Daily average temperature | "
            "Ambient outdoor temperature in degrees Celsius (°C). |"
        ),
        (
            "| `avg_humidity` | `double` | Observed | "
            "Daily average relative humidity | Relative humidity percentage "
            "($[0, 100]$%). |"
        ),
        (
            "| `avg_wind_level` | `double` | Observed | Daily average wind level | "
            "Meteorological wind intensity index. |"
        ),
        "",
        "---",
        "",
        "## Critical Modeling Notes",
        "",
        (
            "1. **Normalized Scale:** `sale_amount` and `hours_sale` are globally "
            "normalized demand index values. Raw physical item counts cannot be "
            "reconstructed. Predictions represent normalized demand index values."
        ),
        (
            "2. **Stockout Semantics:** `hours_stock_status = 1` indicates unavailable "
            "stock. Positive sales may co-occur with stockout flags within the same "
            "hour; their within-hour timing and cause are not established by this "
            "validation."
        ),
        (
            "3. **Evaluation Filter:** `stock_hour6_22_cnt == 0` designates the "
            "project's selected primary evaluation cohort free of daytime stockouts. "
            "Daytime availability does not guarantee 24-hour stockout absence across "
            "overnight non-trading hours (22:00 to 06:00)."
        ),
    ]
    return "\n".join(lines) + "\n"


def _chk_field(
    partition: dict[str, Any], check_name: str, field: str, default: Any = "N/A"
) -> Any:
    """Helper to safely fetch a field from a check result dict."""
    for c in partition.get("checks", []):
        if c.get("name") == check_name:
            return c.get(field, default)
    return default


def _build_data_profile_md(summary: dict[str, Any]) -> str:
    """Generate professional analytical data profile report from validation summary."""
    train = summary.get("train", {})
    eval_part = summary.get("eval", {})
    cross = summary.get("cross_partition", {})
    cohort = summary.get("evaluation_cohort", {})

    train_meta = train.get("metadata") or {}
    eval_meta = eval_part.get("metadata") or {}

    train_diag = train.get("diagnostics", {}).get("stockout_diagnostics", {})
    eval_diag = eval_part.get("diagnostics", {}).get("stockout_diagnostics", {})

    def get_chk(partition: dict[str, Any], name: str) -> dict[str, Any]:
        for c in partition.get("checks", []):
            if c.get("name") == name:
                return c
        return {}

    def schema_partition_ok(partition: dict[str, Any]) -> bool:
        missing_chk = get_chk(partition, "schema_missing_columns")
        unexp_chk = get_chk(partition, "schema_unexpected_columns")
        types_chk = get_chk(partition, "schema_data_types")
        return (
            missing_chk.get("status") == ValidationStatus.PASS.value
            and unexp_chk.get("status") == ValidationStatus.PASS.value
            and types_chk.get("status") == ValidationStatus.PASS.value
        )

    def schema_discrepancy(partition: dict[str, Any]) -> str:
        issues = []
        for chk_name in (
            "schema_missing_columns",
            "schema_unexpected_columns",
            "schema_data_types",
        ):
            c = get_chk(partition, chk_name)
            if c.get("status") == ValidationStatus.FAIL.value:
                issues.append(str(c.get("actual", "Discrepancy detected")))
        return "; ".join(issues) if issues else "All column types verified"

    schema_ok = schema_partition_ok(train) and schema_partition_ok(eval_part)

    tr_null_chk = get_chk(train, "missing_values_scalar")
    ev_null_chk = get_chk(eval_part, "missing_values_scalar")
    tr_null_cnt = (
        tr_null_chk.get("counts", {}).get("null_count") if tr_null_chk else None
    )
    ev_null_cnt = (
        ev_null_chk.get("counts", {}).get("null_count") if ev_null_chk else None
    )

    tr_fin_chk = get_chk(train, "non_finite_values")
    ev_fin_chk = get_chk(eval_part, "non_finite_values")
    tr_non_fin = (
        tr_fin_chk.get("counts", {}).get("non_finite_count") if tr_fin_chk else None
    )
    ev_non_fin = (
        ev_fin_chk.get("counts", {}).get("non_finite_count") if ev_fin_chk else None
    )

    tr_neg_chk = get_chk(train, "negative_sales")
    ev_neg_chk = get_chk(eval_part, "negative_sales")
    tr_neg_sales = (
        tr_neg_chk.get("counts", {}).get("negative_sales_count") if tr_neg_chk else None
    )
    ev_neg_sales = (
        ev_neg_chk.get("counts", {}).get("negative_sales_count") if ev_neg_chk else None
    )

    tr_dup_chk = get_chk(train, "natural_key_duplicates")
    ev_dup_chk = get_chk(eval_part, "natural_key_duplicates")
    tr_dup = (
        tr_dup_chk.get("counts", {}).get("duplicate_groups") if tr_dup_chk else None
    )
    ev_dup = (
        ev_dup_chk.get("counts", {}).get("duplicate_groups") if ev_dup_chk else None
    )

    tr_flags_chk = get_chk(train, "binary_flags_validity")
    ev_flags_chk = get_chk(eval_part, "binary_flags_validity")
    tr_flags_ok = (
        tr_flags_chk.get("status") == ValidationStatus.PASS.value
        if tr_flags_chk
        else None
    )
    ev_flags_ok = (
        ev_flags_chk.get("status") == ValidationStatus.PASS.value
        if ev_flags_chk
        else None
    )

    temp_sep_chk = get_chk(cross, "temporal_partition_separation")
    series_bal_ok = (
        get_chk(train, "series_observation_balance").get("status")
        == ValidationStatus.PASS.value
        and get_chk(eval_part, "series_observation_balance").get("status")
        == ValidationStatus.PASS.value
    )
    daily_bal_ok = (
        get_chk(train, "daily_record_balance").get("status")
        == ValidationStatus.PASS.value
        and get_chk(eval_part, "daily_record_balance").get("status")
        == ValidationStatus.PASS.value
    )

    tr_len_ok = (
        get_chk(train, "hourly_array_lengths").get("status")
        == ValidationStatus.PASS.value
    )
    ev_len_ok = (
        get_chk(eval_part, "hourly_array_lengths").get("status")
        == ValidationStatus.PASS.value
    )

    tr_sales_elem_ok = (
        get_chk(train, "hourly_sales_validity").get("status")
        == ValidationStatus.PASS.value
    )
    ev_sales_elem_ok = (
        get_chk(eval_part, "hourly_sales_validity").get("status")
        == ValidationStatus.PASS.value
    )

    tr_stock_elem_ok = (
        get_chk(train, "hourly_stock_status_binary").get("status")
        == ValidationStatus.PASS.value
    )
    ev_stock_elem_ok = (
        get_chk(eval_part, "hourly_stock_status_binary").get("status")
        == ValidationStatus.PASS.value
    )

    tr_rec = get_chk(train, "daily_hourly_sales_reconciliation").get("counts")
    ev_rec = get_chk(eval_part, "daily_hourly_sales_reconciliation").get("counts")
    if tr_rec is not None and ev_rec is not None:
        tr_max_diff = float(tr_rec.get("max_diff", 0.0))
        ev_max_diff = float(ev_rec.get("max_diff", 0.0))
        overall_max_diff = max(tr_max_diff, ev_max_diff)
        total_sales_mismatches = tr_rec.get("mismatch_count", 0) + ev_rec.get(
            "mismatch_count", 0
        )
        sales_rec_available = True
    else:
        sales_rec_available = False
        overall_max_diff = 0.0
        total_sales_mismatches = 0

    tr_st_rec = get_chk(train, "stockout_count_reconciliation").get("counts")
    ev_st_rec = get_chk(eval_part, "stockout_count_reconciliation").get("counts")
    if tr_st_rec is not None and ev_st_rec is not None:
        total_stock_mismatches = tr_st_rec.get("mismatch_count", 0) + ev_st_rec.get(
            "mismatch_count", 0
        )
        max_stock_diff = max(
            float(tr_st_rec.get("max_diff", 0.0)),
            float(ev_st_rec.get("max_diff", 0.0)),
        )
        stock_rec_available = True
    else:
        stock_rec_available = False
        total_stock_mismatches = 0
        max_stock_diff = 0.0

    tr_rows = train_meta.get("num_rows", 0)
    ev_rows = eval_meta.get("num_rows", 0)
    total_rows = tr_rows + ev_rows
    tr_cols = train_meta.get("num_columns", 0)
    ev_cols = eval_meta.get("num_columns", 0)
    tr_groups = train_meta.get("num_row_groups", 0)
    ev_groups = eval_meta.get("num_row_groups", 0)
    tr_days = train_meta.get("num_days", 0)
    ev_days = eval_meta.get("num_days", 0)
    tr_series = train_meta.get("num_series", 0)
    ev_series = eval_meta.get("num_series", 0)
    tr_stores = train_meta.get("num_stores", 0)
    ev_stores = eval_meta.get("num_stores", 0)
    tr_products = train_meta.get("num_products", 0)
    ev_products = eval_meta.get("num_products", 0)

    tr_disc_chk = get_chk(train, "discount_ceiling_anomaly")
    ev_disc_chk = get_chk(eval_part, "discount_ceiling_anomaly")
    tr_gt1_count = (
        tr_disc_chk.get("counts", {}).get("discount_gt_1_count")
        if tr_disc_chk
        else None
    )
    tr_max_disc = (
        tr_disc_chk.get("counts", {}).get("max_discount") if tr_disc_chk else None
    )
    ev_gt1_count = (
        ev_disc_chk.get("counts", {}).get("discount_gt_1_count")
        if ev_disc_chk
        else None
    )
    ev_max_disc = (
        ev_disc_chk.get("counts", {}).get("max_discount") if ev_disc_chk else None
    )
    tr_disc_non_neg_chk = get_chk(train, "discount_non_negative")
    ev_disc_non_neg_chk = get_chk(eval_part, "discount_non_negative")
    tr_disc_non_neg = (
        tr_disc_non_neg_chk.get("status") == ValidationStatus.PASS.value
        if tr_disc_non_neg_chk
        else None
    )
    ev_disc_non_neg = (
        ev_disc_non_neg_chk.get("status") == ValidationStatus.PASS.value
        if ev_disc_non_neg_chk
        else None
    )

    tr_st_hrs = train_diag.get("total_stockout_hours", 0)
    ev_st_hrs = eval_diag.get("total_stockout_hours", 0)
    tr_pos_st_hrs = train_diag.get("positive_sale_stockout_hours")
    ev_pos_st_hrs = eval_diag.get("positive_sale_stockout_hours")
    tr_pos_st_pct = train_diag.get("pct_positive_sale_stockout_hours")
    ev_pos_st_pct = eval_diag.get("pct_positive_sale_stockout_hours")
    tr_pos_st_vol = train_diag.get("sales_volume_during_stockout")
    ev_pos_st_vol = eval_diag.get("sales_volume_during_stockout")

    tr_daytime_no_st = train_diag.get("daytime_no_stockout_rows", 0)
    ev_daytime_no_st = eval_diag.get("daytime_no_stockout_rows", 0)
    tr_daytime_no_st_pct = train_diag.get("daytime_no_stockout_pct", 0.0)
    ev_daytime_no_st_pct = eval_diag.get("daytime_no_stockout_pct", 0.0)

    tr_daytime_full = train_diag.get("daytime_full_stockout_rows", 0)
    ev_daytime_full = eval_diag.get("daytime_full_stockout_rows", 0)
    tr_daytime_full_pct = round(tr_daytime_full / tr_rows * 100, 2) if tr_rows else 0.0
    ev_daytime_full_pct = round(ev_daytime_full / ev_rows * 100, 2) if ev_rows else 0.0

    tr_full_24 = train_diag.get("full_day_stockout_rows", 0)
    ev_full_24 = eval_diag.get("full_day_stockout_rows", 0)
    tr_full_24_pct = round(tr_full_24 / tr_rows * 100, 2) if tr_rows else 0.0
    ev_full_24_pct = round(ev_full_24 / ev_rows * 100, 2) if ev_rows else 0.0

    lines = [
        "# Aarohan: Dataset Profile and Validation Report",
        "",
        "## 1. Dataset Overview",
        "",
        (
            "Aarohan utilizes the FreshRetailNet-50K retail demand dataset "
            "published by Dingdong Inc. under CC BY 4.0. The dataset contains "
            "daily panel observations across store-product series in perishable "
            "grocery retail, annotated with 24-hour stockout status flags and "
            "hourly sales distributions."
        ),
        "",
        "---",
        "",
        "## 2. Dataset Coverage",
        "",
        "### Partition File Specifications",
        "",
        "| Partition | File Path | File Size | SHA-256 Checksum |",
        "| :--- | :--- | :--- | :--- |",
        (
            f"| Training | `{train_meta.get('path', 'N/A')}` | "
            f"{train_meta.get('file_size_mb', 0)} MB "
            f"({train_meta.get('file_size_bytes', 0):,} B) | "
            f"`{train_meta.get('sha256', 'N/A')}` |"
        ),
        (
            f"| Evaluation | `{eval_meta.get('path', 'N/A')}` | "
            f"{eval_meta.get('file_size_mb', 0)} MB "
            f"({eval_meta.get('file_size_bytes', 0):,} B) | "
            f"`{eval_meta.get('sha256', 'N/A')}` |"
        ),
        "",
        "### Dataset Dimensions",
        "",
        "| Dimension | Training Partition | Evaluation Partition | Combined Total |",
        "| :--- | :--- | :--- | :--- |",
        f"| Row Count | {tr_rows:,} | {ev_rows:,} | {total_rows:,} |",
        f"| Column Count | {tr_cols} | {ev_cols} | {max(tr_cols, ev_cols)} |",
        (f"| Row Groups | {tr_groups:,} | {ev_groups:,} | {tr_groups + ev_groups:,} |"),
        (
            f"| Unique Dates | {tr_days} days | {ev_days} days | "
            f"{tr_days + ev_days} continuous days |"
        ),
        (
            f"| Unique Series | {tr_series:,} | {ev_series:,} | "
            f"{max(tr_series, ev_series):,} series |"
        ),
        (
            f"| Unique Stores | {tr_stores:,} | {ev_stores:,} | "
            f"{max(tr_stores, ev_stores):,} stores |"
        ),
        (
            f"| Unique SKUs | {tr_products:,} | {ev_products:,} | "
            f"{max(tr_products, ev_products):,} products |"
        ),
        "",
        "---",
        "",
        "## 3. Data Structure",
        "",
    ]

    if schema_ok:
        lines.extend(
            [
                (
                    "All 19 required columns conform to expected logical Arrow "
                    "types:\n"
                    "- Entity Identifiers: `city_id` (int64), `store_id` (int64), "
                    "`management_group_id` (int64), `first_category_id` (int64), "
                    "`second_category_id` (int64), `third_category_id` (int64), "
                    "`product_id` (int64)\n"
                    "- Temporal and Sales: `dt` (string), `sale_amount` (double), "
                    "`hours_sale` (list<double>)\n"
                    "- Stockout Indicators: `stock_hour6_22_cnt` (int32), "
                    "`hours_stock_status` (list<int64>)\n"
                    "- Covariates: `discount` (double), `holiday_flag` (int32), "
                    "`activity_flag` (int32), `precpt` (double), "
                    "`avg_temperature` (double), `avg_humidity` (double), "
                    "`avg_wind_level` (double)"
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    "Schema discrepancies were detected during column structure "
                    "validation:\n"
                    f"- Training partition: {schema_discrepancy(train)}\n"
                    f"- Evaluation partition: {schema_discrepancy(eval_part)}"
                ),
            ]
        )

    if tr_null_cnt is not None and ev_null_cnt is not None:
        scalar_null_line = (
            "No missing scalar values exist across all columns in either partition."
            if tr_null_cnt == 0 and ev_null_cnt == 0
            else (
                f"Scalar missing values detected: {tr_null_cnt} in training, "
                f"{ev_null_cnt} in evaluation."
            )
        )
    elif tr_null_cnt is not None:
        scalar_null_line = (
            f"Scalar missing values: {tr_null_cnt} in training; evaluation unavailable."
        )
    elif ev_null_cnt is not None:
        scalar_null_line = (
            f"Scalar missing values: training unavailable; {ev_null_cnt} in evaluation."
        )
    else:
        scalar_null_line = (
            "Scalar missing values check unavailable (could not be calculated)."
        )

    if tr_non_fin is not None and ev_non_fin is not None:
        non_fin_line = (
            "All floating-point numerical values are finite across both "
            "partitions (zero NaN or Infinity occurrences)."
            if tr_non_fin == 0 and ev_non_fin == 0
            else (
                f"Non-finite values detected: {tr_non_fin} in training, "
                f"{ev_non_fin} in evaluation."
            )
        )
    elif tr_non_fin is not None:
        non_fin_line = (
            f"Non-finite values: {tr_non_fin} in training; evaluation unavailable."
        )
    elif ev_non_fin is not None:
        non_fin_line = (
            f"Non-finite values: training unavailable; {ev_non_fin} in evaluation."
        )
    else:
        non_fin_line = "Non-finite values check unavailable (could not be calculated)."

    if tr_neg_sales is not None and ev_neg_sales is not None:
        neg_sales_line = (
            "No negative sales values exist in either partition."
            if tr_neg_sales == 0 and ev_neg_sales == 0
            else (
                f"Negative sales values detected: {tr_neg_sales} in training, "
                f"{ev_neg_sales} in evaluation."
            )
        )
    elif tr_neg_sales is not None:
        neg_sales_line = (
            f"Negative sales values: {tr_neg_sales} in training; "
            "evaluation unavailable."
        )
    elif ev_neg_sales is not None:
        neg_sales_line = (
            f"Negative sales values: training unavailable; "
            f"{ev_neg_sales} in evaluation."
        )
    else:
        neg_sales_line = "Negative sales check unavailable (could not be calculated)."

    if tr_dup is not None and ev_dup is not None:
        dup_line = (
            "Zero duplicate records exist on the primary natural key "
            "`(store_id, product_id, dt)` across all observations in both "
            "partitions."
            if tr_dup == 0 and ev_dup == 0
            else (
                f"Natural key duplicates detected: {tr_dup} in training, "
                f"{ev_dup} in evaluation."
            )
        )
    elif tr_dup is not None:
        dup_line = (
            f"Natural key duplicates: {tr_dup} in training; evaluation unavailable."
        )
    elif ev_dup is not None:
        dup_line = (
            f"Natural key duplicates: training unavailable; {ev_dup} in evaluation."
        )
    else:
        dup_line = "Natural key duplicate check unavailable (could not be calculated)."

    if tr_flags_ok is True and ev_flags_ok is True:
        flags_line = (
            "Holiday and marketing activity indicators strictly conform to "
            "binary values {0, 1}."
        )
    elif tr_flags_ok is None and ev_flags_ok is None:
        flags_line = (
            "Indicator flag validity check unavailable (could not be calculated)."
        )
    else:
        flags_line = "Indicator flag validity discrepancies detected."

    if sales_rec_available:
        sales_rec_line = (
            "- Sales Reconciliation: Daily recorded sales reconcile with the "
            "sum of hourly sales within floating-point tolerance (1e-5) across "
            f"all records in both partitions (maximum absolute difference: "
            f"{overall_max_diff:.2e})."
            if total_sales_mismatches == 0
            else (
                "- Sales Reconciliation: Discrepancies observed between recorded "
                f"daily sales and sum of hourly sales ({total_sales_mismatches} "
                f"mismatches, maximum difference: {overall_max_diff:.2e})."
            )
        )
    else:
        sales_rec_line = (
            "- Sales Reconciliation: Sales reconciliation check unavailable."
        )

    if stock_rec_available:
        stock_rec_line = (
            "- Daytime Stockout Count Reconciliation: Daytime stockout hour "
            "counts (`stock_hour6_22_cnt`) reconcile exactly with the sum of "
            "hourly stockout indicators across hours 06:00 through 21:00 inclusive "
            "(`hours_stock_status[6:22]`) across all records in both partitions."
            if total_stock_mismatches == 0
            else (
                "- Daytime Stockout Count Reconciliation: Discrepancies observed "
                f"in daytime stockout hour counts ({total_stock_mismatches} "
                f"mismatches, maximum difference: {max_stock_diff})."
            )
        )
    else:
        stock_rec_line = (
            "- Daytime Stockout Count Reconciliation: Daytime stockout count "
            "reconciliation check unavailable."
        )

    if (
        tr_pos_st_pct is not None
        and ev_pos_st_pct is not None
        and tr_pos_st_hrs is not None
        and ev_pos_st_hrs is not None
        and tr_pos_st_vol is not None
        and ev_pos_st_vol is not None
    ):
        cooccur_line = (
            "- Positive Sales During Stockout Flags: "
            f"Training {tr_pos_st_pct:.2f}% of stockout hours ({tr_pos_st_hrs:,} "
            f"hours, volume: {tr_pos_st_vol:,.2f} normalized demand index values); "
            f"Evaluation {ev_pos_st_pct:.2f}% of stockout hours ({ev_pos_st_hrs:,} "
            f"hours, volume: {ev_pos_st_vol:,.2f} normalized demand index values). "
            "Positive sales co-occurring with stockout flags are an observed "
            "diagnostic. Their within-hour timing and cause are not established "
            "by this validation."
        )
    else:
        cooccur_line = (
            "- Positive Sales During Stockout Flags: Diagnostic unavailable. "
            "Positive sales and stockout co-occurrence requires valid information "
            "from both hourly sales and hourly stockout arrays."
        )

    if tr_disc_non_neg is True and ev_disc_non_neg is True:
        disc_non_neg_line = (
            "- Non-Negative Discount Rates: Discount rates are non-negative "
            "across all records in both partitions."
        )
    elif tr_disc_non_neg is None and ev_disc_non_neg is None:
        disc_non_neg_line = (
            "- Non-Negative Discount Rates: Discount non-negative check unavailable."
        )
    else:
        disc_non_neg_line = "- Negative discount rate values detected."

    if tr_gt1_count is not None:
        tr_gt1_line = (
            f"- Training Partition Discount Ceiling: {tr_gt1_count} records "
            f"feature discount values exceeding 1.0 (maximum observed discount: "
            f"{tr_max_disc}). discount > 1.0 is an observed anomaly relative to "
            "the conventional discount range with an unknown cause. Raw values "
            "are preserved without clipping or correction."
            if tr_gt1_count > 0
            else (
                "- Training Partition Discount Ceiling: All discount values are "
                "within the conventional range (<= 1.0)."
            )
        )
    else:
        tr_gt1_line = (
            "- Training Partition Discount Ceiling: Discount ceiling check unavailable."
        )

    if ev_gt1_count is not None:
        ev_gt1_line = (
            f"- Evaluation Partition Discount Ceiling: {ev_gt1_count} records "
            f"feature discount values exceeding 1.0 (maximum observed discount: "
            f"{ev_max_disc})."
            if ev_gt1_count > 0
            else (
                "- Evaluation Partition Discount Ceiling: All discount values "
                "conform to the conventional promotional range (<= 1.0)."
            )
        )
    else:
        ev_gt1_line = (
            "- Evaluation Partition Discount Ceiling: Discount ceiling check "
            "unavailable."
        )

    if cohort.get("total_rows", 0) > 0:
        cohort_lines = [
            f"- Total Evaluation Records: {cohort.get('total_rows', 0):,}",
            (
                "- Daytime Stockout-Free Cohort: "
                f"{cohort.get('daytime_no_stockout_rows', 0):,} records "
                f"({cohort.get('pct_daytime_no_stockout', 0.0)}%)"
            ),
            (
                "- Daytime Stockout-Affected Cohort: "
                f"{cohort.get('daytime_with_stockout_rows', 0):,} records "
                f"({cohort.get('pct_daytime_with_stockout', 0.0)}%)"
            ),
        ]
    else:
        cohort_lines = [
            "- Primary Evaluation Cohort: Evaluation cohort calculation unavailable."
        ]

    item1_line = (
        "1. Demand Index Scale: Recorded sales values represent globally "
        "normalized demand index figures rather than raw physical unit "
        "quantities. Original physical unit sales cannot be reconstructed "
        "from the public dataset. Demand forecasting and evaluation operate "
        "in this normalized space. Observed sales reflect censored transaction "
        "volume during periods of product availability; stockout status flags "
        "indicate product unavailability; and customer demand remains unobserved "
        "and latent when stockouts occur."
    )
    item2_line = (
        "2. Daytime Operating Scope: The primary evaluation cohort "
        "(`stock_hour6_22_cnt == 0`) captures stockout absence across daytime "
        "operating hours (06:00 through 21:00 inclusive); daytime availability "
        "does not imply complete stockout absence during overnight fulfillment "
        "and restocking windows (22:00 through 06:00)."
    )
    if tr_gt1_count is not None and tr_gt1_count > 0:
        item3_line = (
            f"3. Discount Anomalies: {tr_gt1_count} training records feature "
            f"discount values exceeding 1.0 (reaching {tr_max_disc}), which are "
            "observed anomalies with an unknown cause. Raw observed values are "
            "preserved without clipping, truncation, or synthetic adjustment."
        )
    else:
        item3_line = (
            "3. Discount Anomalies: Discount values exceeding 1.0 may occur "
            "as observed anomalies relative to the conventional promotional "
            "range with an unknown cause. Raw values are preserved without "
            "clipping or correction."
        )
    if tr_pos_st_pct is not None and ev_pos_st_pct is not None:
        item4_line = (
            f"4. Sales and Stockout Co-occurrence: Positive sales co-occur with "
            f"stockout flags in approximately {tr_pos_st_pct:.2f}% of training "
            f"stockout hours and {ev_pos_st_pct:.2f}% of evaluation stockout "
            "hours. Their within-hour timing and cause are not established by "
            "this validation."
        )
    else:
        item4_line = (
            "4. Sales and Stockout Co-occurrence: Co-occurrence diagnostic was "
            "unavailable for one or both partitions. Positive sales and stockout "
            "co-occurrence assessment requires complete, valid arrays in both "
            "sales and stockout status."
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Data Completeness and Integrity",
            "",
            scalar_null_line,
            non_fin_line,
            neg_sales_line,
            dup_line,
            flags_line,
            "",
            "---",
            "",
            "## 5. Temporal Coverage",
            "",
            (
                f"- Training Window: `{train_meta.get('min_dt', 'N/A')}` to "
                f"`{train_meta.get('max_dt', 'N/A')}` ({tr_days} calendar days)."
            ),
            (
                f"- Evaluation Window: `{eval_meta.get('min_dt', 'N/A')}` to "
                f"`{eval_meta.get('max_dt', 'N/A')}` ({ev_days} calendar days)."
            ),
            (
                "- Temporal Partition Ordering: The training partition strictly "
                "precedes the evaluation partition without temporal overlap "
                f"(`{train_meta.get('max_dt', 'N/A')}` < "
                f"`{eval_meta.get('min_dt', 'N/A')}`)."
                if temp_sep_chk.get("status") == ValidationStatus.PASS.value
                else (
                    "- Temporal Partition Ordering: Temporal separation between "
                    "partitions could not be established or exhibited overlap."
                )
            ),
            (
                "- Series Continuity and Balance: Every store-product series "
                "contains an unbroken daily observation record without missing "
                f"calendar dates ({tr_days} days in training, {ev_days} days in "
                "evaluation). Daily record counts are exactly balanced at "
                f"{tr_series:,} observations per day."
                if series_bal_ok and daily_bal_ok
                else (
                    "- Series Continuity and Balance: Discrepancies were observed "
                    "in series observation counts or daily balance."
                )
            ),
            "",
            "---",
            "",
            "## 6. Hourly Sales and Stock Availability",
            "",
            (
                "- Nested Array Structure: Every record in both partitions contains "
                "exactly 24 hourly observations for both sales and stock status "
                "with non-null parent arrays."
                if tr_len_ok and ev_len_ok
                else (
                    "- Nested Array Structure: Discrepancies detected in array "
                    f"lengths or parent nulls (training: "
                    f"{get_chk(train, 'hourly_array_lengths').get('actual')}, "
                    f"evaluation: "
                    f"{get_chk(eval_part, 'hourly_array_lengths').get('actual')})."
                )
            ),
            (
                "- Hourly Sales Elements: Hourly sales arrays contain exclusively "
                "finite, non-negative numerical values with zero missing child "
                "elements across all 24 hours."
                if tr_sales_elem_ok and ev_sales_elem_ok
                else (
                    "- Hourly Sales Elements: Discrepancies detected in "
                    "hourly sales values."
                )
            ),
            (
                "- Hourly Stockout Indicators: Hourly stockout flags are strictly "
                "binary integers in {0, 1} across all hours in both partitions."
                if tr_stock_elem_ok and ev_stock_elem_ok
                else (
                    "- Hourly Stockout Indicators: Non-binary or null stockout flags "
                    "detected."
                )
            ),
            sales_rec_line,
            stock_rec_line,
            "",
            "### Stockout Availability Diagnostics",
            "",
            (
                f"- Daytime Stockout-Free Series-Days (`stock_hour6_22_cnt == 0`): "
                f"Training {tr_daytime_no_st:,} rows ({tr_daytime_no_st_pct}%), "
                f"Evaluation {ev_daytime_no_st:,} rows ({ev_daytime_no_st_pct}%)."
            ),
            (
                f"- Full Daytime Stockout Series-Days (`stock_hour6_22_cnt == 16`): "
                f"Training {tr_daytime_full:,} rows ({tr_daytime_full_pct}%), "
                f"Evaluation {ev_daytime_full:,} rows ({ev_daytime_full_pct}%)."
            ),
            (
                f"- Full 24-Hour Stockout Series-Days: "
                f"Training {tr_full_24:,} rows ({tr_full_24_pct}%), "
                f"Evaluation {ev_full_24:,} rows ({ev_full_24_pct}%)."
            ),
            (
                f"- Total Stockout Hours: "
                f"Training {tr_st_hrs:,} hours, Evaluation {ev_st_hrs:,} hours."
            ),
            cooccur_line,
            "",
            "---",
            "",
            "## 7. Observed Data Anomalies",
            "",
            disc_non_neg_line,
            tr_gt1_line,
            ev_gt1_line,
            "",
            "---",
            "",
            "## 8. Evaluation Cohort",
            "",
            (
                "The project's selected primary evaluation cohort is defined by "
                "`stock_hour6_22_cnt == 0` (series-days free of daytime "
                "stockouts):"
            ),
            *cohort_lines,
            "",
            "---",
            "",
            "## 9. Data Limitations and Modeling Considerations",
            "",
            item1_line,
            item2_line,
            item3_line,
            item4_line,
        ]
    )

    return "\n".join(lines) + "\n"
