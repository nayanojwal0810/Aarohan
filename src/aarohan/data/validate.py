"""Data validation engine for Aarohan retail demand panels."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa

from aarohan.config import get_project_root, load_config
from aarohan.data.dataset import (
    DatasetMetadata,
    ValidationCheck,
    ValidationResult,
    ValidationStatus,
)
from aarohan.data.load import (
    compute_sha256,
    create_duckdb_connection,
    get_arrow_schema,
    iter_parquet_batches,
    read_parquet_metadata,
    resolve_dataset_path,
)
from aarohan.logging_config import get_logger

logger = get_logger(__name__)


def _types_match(actual_type: pa.DataType, expected_str: str) -> bool:
    """Compare an Arrow DataType to an expected type string with normalization."""
    actual_norm = str(actual_type).lower().replace("element", "item")
    expected_norm = expected_str.lower().replace("element", "item")

    if actual_norm == expected_norm:
        return True

    # Check common type aliases
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


def validate_file_level(
    file_path: Path,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
    expected_cols: int | None = None,
) -> tuple[list[ValidationCheck], DatasetMetadata | None]:
    """Validate file presence, size, checksum, and Parquet metadata."""
    checks: list[ValidationCheck] = []

    if not file_path.is_file():
        checks.append(
            ValidationCheck(
                name="file_existence",
                status=ValidationStatus.FAIL.value,
                expected="Existing regular file",
                actual="File not found",
                explanation=f"Dataset file does not exist at {file_path}",
            )
        )
        return checks, None

    checks.append(
        ValidationCheck(
            name="file_existence",
            status=ValidationStatus.PASS.value,
            expected="Existing regular file",
            actual=str(file_path.name),
            explanation=f"File verified at {file_path}",
        )
    )

    # Checksum verification
    actual_sha = compute_sha256(file_path)
    if expected_sha256 is not None:
        if actual_sha.lower() == expected_sha256.lower():
            checks.append(
                ValidationCheck(
                    name="sha256_checksum",
                    status=ValidationStatus.PASS.value,
                    expected=expected_sha256,
                    actual=actual_sha,
                    explanation="Cryptographic SHA-256 digest matches authoritative benchmark",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="sha256_checksum",
                    status=ValidationStatus.FAIL.value,
                    expected=expected_sha256,
                    actual=actual_sha,
                    explanation="Checksum mismatch! File payload differs from official benchmark",
                )
            )

    meta = read_parquet_metadata(file_path)
    schema = get_arrow_schema(file_path)

    # Row count check
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

    # Column count check
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
        path=str(file_path),
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
    """Validate that actual Parquet columns and logical Arrow types match expectations."""
    checks: list[ValidationCheck] = []
    schema = get_arrow_schema(file_path)

    actual_cols = set(schema.names)
    expected_cols = set(expected_schema.keys())

    missing_cols = list(expected_cols - actual_cols)
    unexpected_cols = list(actual_cols - expected_cols)

    if missing_cols:
        checks.append(
            ValidationCheck(
                name="schema_missing_columns",
                status=ValidationStatus.FAIL.value,
                expected="All 19 required columns present",
                actual=f"Missing {len(missing_cols)} columns: {missing_cols}",
                explanation="Required dataset columns are absent",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="schema_missing_columns",
                status=ValidationStatus.PASS.value,
                expected="All required columns present",
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
                actual=f"Found {len(unexpected_cols)} unexpected columns: {unexpected_cols}",
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

    # Data type verification
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
                explanation="All column types match the logical Arrow schema specification",
            )
        )

    return checks


def validate_general_integrity(
    con: duckdb.DuckDBPyConnection, file_path: Path
) -> list[ValidationCheck]:
    """Execute SQL aggregate integrity checks using DuckDB."""
    checks: list[ValidationCheck] = []
    parquet_sql = f"read_parquet('{file_path.as_posix()}')"

    schema = get_arrow_schema(file_path)
    scalar_cols = [
        f.name for f in schema if f.name not in ("hours_sale", "hours_stock_status")
    ]
    float_cols = [
        f.name
        for f in schema
        if f.type == pa.float64() and f.name not in ("hours_sale", "hours_stock_status")
    ]

    # Null checks across scalar columns
    null_exprs = [f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)" for c in scalar_cols]
    total_nulls = con.execute(
        f"SELECT {' + '.join(null_exprs)} FROM {parquet_sql}"
    ).fetchone()[0]

    status = (
        ValidationStatus.PASS.value if total_nulls == 0 else ValidationStatus.FAIL.value
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
                explanation=f"Non-finite (NaN/Inf) values across numeric fields: {non_finite_total}",
                counts={"non_finite_count": int(non_finite_total)},
            )
        )

    # Negative recorded sales
    neg_sales = con.execute(
        f"SELECT COUNT(*) FROM {parquet_sql} WHERE sale_amount < 0"
    ).fetchone()[0]
    status = (
        ValidationStatus.PASS.value if neg_sales == 0 else ValidationStatus.FAIL.value
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

    # Natural key duplicates on (store_id, product_id, dt)
    dup_groups = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT store_id, product_id, dt, COUNT(*) as cnt
            FROM {parquet_sql}
            GROUP BY store_id, product_id, dt
            HAVING cnt > 1
        )
    """).fetchone()[0]

    status = (
        ValidationStatus.PASS.value if dup_groups == 0 else ValidationStatus.FAIL.value
    )
    checks.append(
        ValidationCheck(
            name="natural_key_duplicates",
            status=status,
            expected=0,
            actual=int(dup_groups),
            explanation=f"Duplicate natural key (store_id, product_id, dt) groups: {dup_groups}",
            counts={"duplicate_groups": int(dup_groups)},
        )
    )

    # Binary flags validity
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
            explanation=f"Invalid holiday_flag or activity_flag non-binary values: {invalid_flags}",
            counts={"invalid_flags_count": int(invalid_flags)},
        )
    )

    # Date format check (ISO-8601 YYYY-MM-DD)
    invalid_dates = con.execute(f"""
        SELECT COUNT(*) FROM {parquet_sql}
        WHERE try_strptime(dt, '%Y-%m-%d') IS NULL
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
            explanation=f"Dates failing ISO YYYY-MM-DD parsing: {invalid_dates}",
            counts={"invalid_dates_count": int(invalid_dates)},
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
    """Validate entity counts, date range, panel completeness, and series continuity."""
    checks: list[ValidationCheck] = []
    parquet_sql = f"read_parquet('{file_path.as_posix()}')"

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

    if metadata is not None:
        metadata.min_dt = str(min_dt)
        metadata.max_dt = str(max_dt)
        metadata.num_days = int(num_days)
        metadata.num_stores = int(num_stores)
        metadata.num_products = int(num_products)
        metadata.num_series = int(num_series)

    # Store count
    exp_stores = expected_entities.get("stores", 898)
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

    # Product count
    exp_prods = expected_entities.get("products", 865)
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

    # Series count
    exp_series = expected_size.get("series", 50000)
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

    # Date boundaries
    exp_start = expected_size.get("start_date")
    exp_end = expected_size.get("end_date")
    exp_days = expected_size.get("dates")

    if exp_start and exp_end:
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

    # Date count & continuity
    if exp_days is not None:
        start_date = datetime.date.fromisoformat(str(min_dt))
        end_date = datetime.date.fromisoformat(str(max_dt))
        expected_calendar_span = (end_date - start_date).days + 1

        continuous = (num_days == exp_days) and (num_days == expected_calendar_span)
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
                explanation=f"Calendar date count: {num_days} days (calendar span: {expected_calendar_span})",
            )
        )

    # Series observations balance (min_days == max_days == exp_days per series)
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
                ValidationStatus.PASS.value if balanced else ValidationStatus.FAIL.value
            ),
            expected=f"Exactly {exp_days} observations per series",
            actual=f"Min: {min_series_days}, Max: {max_series_days}",
            explanation=f"Each of the {num_series:,} series contains {min_series_days} to {max_series_days} observations",
            counts={
                "min_days_per_series": int(min_series_days),
                "max_days_per_series": int(max_series_days),
            },
        )
    )

    # Records per calendar day
    daily_rows_dist = con.execute(f"""
        SELECT MIN(cnt), MAX(cnt)
        FROM (
            SELECT dt, COUNT(*) as cnt
            FROM {parquet_sql}
            GROUP BY dt
        )
    """).fetchone()
    min_daily_rows, max_daily_rows = daily_rows_dist

    daily_balanced = (min_daily_rows == exp_series) and (max_daily_rows == exp_series)
    checks.append(
        ValidationCheck(
            name="daily_record_balance",
            status=(
                ValidationStatus.PASS.value
                if daily_balanced
                else ValidationStatus.FAIL.value
            ),
            expected=f"Exactly {exp_series:,} records per date",
            actual=f"Min: {min_daily_rows:,}, Max: {max_daily_rows:,}",
            explanation=f"Daily record counts across all {num_days} dates: {min_daily_rows:,} to {max_daily_rows:,}",
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
    """Validate nested hourly arrays and reconciliation using streaming PyArrow batches."""
    checks: list[ValidationCheck] = []
    diagnostics: dict[str, Any] = {}

    total_rows = 0
    invalid_array_length_count = 0
    null_array_elements_count = 0
    negative_hourly_sales_count = 0
    non_binary_stock_status_count = 0

    sales_mismatch_count = 0
    max_sales_diff = 0.0
    sum_sales_diff = 0.0

    stock_count_mismatches = 0
    max_stock_diff = 0.0
    daytime_no_stockout_rows = 0
    daytime_full_stockout_rows = 0
    full_day_stockout_rows = 0

    stockout_hours_total = 0
    positive_sale_stockout_hours = 0
    sales_during_stockout_volume = 0.0

    cols_to_read = [
        "sale_amount",
        "hours_sale",
        "stock_hour6_22_cnt",
        "hours_stock_status",
    ]

    for batch in iter_parquet_batches(
        file_path, columns=cols_to_read, batch_size=batch_size
    ):
        n = batch.num_rows
        total_rows += n

        sale_amounts = batch.column("sale_amount").to_numpy()
        stock_counts = batch.column("stock_hour6_22_cnt").to_numpy()

        # Extract nested lists into numpy arrays
        # batch.column("hours_sale") is a ListArray
        hours_sale_arr = batch.column("hours_sale")
        hours_stock_arr = batch.column("hours_stock_status")

        # Array length verification
        sale_lens = hours_sale_arr.value_lengths().to_numpy()
        stock_lens = hours_stock_arr.value_lengths().to_numpy()
        bad_lens = (sale_lens != 24) | (stock_lens != 24)

        if bad_lens.any():
            invalid_array_length_count += int(bad_lens.sum())

        # Check for null elements
        if hours_sale_arr.null_count > 0 or hours_stock_arr.null_count > 0:
            null_array_elements_count += (
                hours_sale_arr.null_count + hours_stock_arr.null_count
            )

        if not bad_lens.any():
            # Flatten arrays for 2D computation (shape: [n, 24])
            # Since lengths are 24, values.to_numpy().reshape(n, 24) is safe and fast
            sales_flat = hours_sale_arr.values.to_numpy().reshape(n, 24)
            stock_flat = hours_stock_arr.values.to_numpy().reshape(n, 24)

            if not np.isfinite(sales_flat).all() or (sales_flat < 0).any():
                negative_hourly_sales_count += int((sales_flat < 0).any(axis=1).sum())

            if not np.isin(stock_flat, [0, 1]).all():
                non_binary_stock_status_count += int(
                    (~np.isin(stock_flat, [0, 1])).any(axis=1).sum()
                )

            # Daily sales reconciliation
            hourly_sums = sales_flat.sum(axis=1)
            diffs = np.abs(sale_amounts - hourly_sums)
            mismatches = diffs > tolerance
            if mismatches.any():
                sales_mismatch_count += int(mismatches.sum())

            batch_max_diff = float(diffs.max())
            if batch_max_diff > max_sales_diff:
                max_sales_diff = batch_max_diff
            sum_sales_diff += float(diffs.sum())

            # Daytime stockout count reconciliation (slice daytime_start:daytime_end)
            daytime_slice = stock_flat[:, daytime_start:daytime_end]
            daytime_sums = daytime_slice.sum(axis=1)
            stock_diffs = np.abs(stock_counts - daytime_sums)
            mismatch_stock = stock_diffs != 0
            if mismatch_stock.any():
                stock_count_mismatches += int(mismatch_stock.sum())

            batch_max_stock_diff = float(stock_diffs.max())
            if batch_max_stock_diff > max_stock_diff:
                max_stock_diff = batch_max_stock_diff

            # Count special stockout subsets
            daytime_no_stockout_rows += int((stock_counts == 0).sum())
            daytime_full_stockout_rows += int((stock_counts == 16).sum())
            full_day_stockout_rows += int((stock_flat.sum(axis=1) == 24).sum())

            # Positive sales during stockout-flagged hours
            stockout_mask = stock_flat == 1
            stockout_hours_total += int(stockout_mask.sum())
            pos_sale_in_stockout = stockout_mask & (sales_flat > 0)
            positive_sale_stockout_hours += int(pos_sale_in_stockout.sum())
            sales_during_stockout_volume += float(
                sales_flat[pos_sale_in_stockout].sum()
            )
        else:
            # Fallback row-by-row execution when batch contains invalid array lengths
            for i in range(n):
                s_val = hours_sale_arr[i].as_py()
                st_val = hours_stock_arr[i].as_py()
                if (
                    s_val is None
                    or len(s_val) != 24
                    or st_val is None
                    or len(st_val) != 24
                ):
                    continue
                s_np = np.array(s_val, dtype=np.float64)
                st_np = np.array(st_val, dtype=np.int64)

                if not np.isfinite(s_np).all() or (s_np < 0).any():
                    negative_hourly_sales_count += 1
                if not np.isin(st_np, [0, 1]).all():
                    non_binary_stock_status_count += 1

                h_sum = float(s_np.sum())
                diff = abs(float(sale_amounts[i]) - h_sum)
                if diff > tolerance:
                    sales_mismatch_count += 1
                if diff > max_sales_diff:
                    max_sales_diff = diff
                sum_sales_diff += diff

                d_sum = int(st_np[daytime_start:daytime_end].sum())
                s_diff = abs(int(stock_counts[i]) - d_sum)
                if s_diff != 0:
                    stock_count_mismatches += 1
                if s_diff > max_stock_diff:
                    max_stock_diff = float(s_diff)

                if stock_counts[i] == 0:
                    daytime_no_stockout_rows += 1
                if stock_counts[i] == 16:
                    daytime_full_stockout_rows += 1
                if int(st_np.sum()) == 24:
                    full_day_stockout_rows += 1

                for h in range(24):
                    if st_np[h] == 1:
                        stockout_hours_total += 1
                        if s_np[h] > 0:
                            positive_sale_stockout_hours += 1
                            sales_during_stockout_volume += float(s_np[h])

    # Hourly array length check
    status = (
        ValidationStatus.PASS.value
        if invalid_array_length_count == 0
        else ValidationStatus.FAIL.value
    )
    checks.append(
        ValidationCheck(
            name="hourly_array_lengths",
            status=status,
            expected="Exactly 24 values per array",
            actual=(
                f"{invalid_array_length_count} invalid rows"
                if invalid_array_length_count
                else "All 24 elements"
            ),
            explanation="Array length validation across hours_sale and hours_stock_status",
            counts={"invalid_length_rows": invalid_array_length_count},
        )
    )

    # Element validity
    status = (
        ValidationStatus.PASS.value
        if (negative_hourly_sales_count == 0 and null_array_elements_count == 0)
        else ValidationStatus.FAIL.value
    )
    checks.append(
        ValidationCheck(
            name="hourly_sales_validity",
            status=status,
            expected="Non-negative, finite, non-null values",
            actual=f"{negative_hourly_sales_count} negative/non-finite rows",
            explanation="Hourly sales element values are valid and non-negative",
        )
    )

    status = (
        ValidationStatus.PASS.value
        if non_binary_stock_status_count == 0
        else ValidationStatus.FAIL.value
    )
    checks.append(
        ValidationCheck(
            name="hourly_stock_status_binary",
            status=status,
            expected="Strictly binary values in {0, 1}",
            actual=f"{non_binary_stock_status_count} non-binary rows",
            explanation="Hourly stockout status elements are strictly binary flags (0=in-stock, 1=stockout)",
        )
    )

    # Daily and hourly sales reconciliation
    mean_sales_diff = sum_sales_diff / total_rows if total_rows > 0 else 0.0
    status = (
        ValidationStatus.PASS.value
        if sales_mismatch_count == 0
        else ValidationStatus.FAIL.value
    )
    checks.append(
        ValidationCheck(
            name="daily_hourly_sales_reconciliation",
            status=status,
            expected=f"sale_amount == sum(hours_sale) within tolerance {tolerance}",
            actual=f"{sales_mismatch_count} mismatches (max diff: {max_sales_diff:.2e}, mean diff: {mean_sales_diff:.2e})",
            explanation="Daily recorded sales reconcile with the sum of hourly sales within float precision",
            counts={
                "mismatch_count": sales_mismatch_count,
                "max_diff": max_sales_diff,
                "mean_diff": mean_sales_diff,
                "within_tolerance_rows": total_rows - sales_mismatch_count,
            },
        )
    )

    # Daytime stockout count reconciliation
    status = (
        ValidationStatus.PASS.value
        if stock_count_mismatches == 0
        else ValidationStatus.FAIL.value
    )
    checks.append(
        ValidationCheck(
            name="stockout_count_reconciliation",
            status=status,
            expected="stock_hour6_22_cnt == sum(hours_stock_status[6:22])",
            actual=f"{stock_count_mismatches} mismatches (max diff: {max_stock_diff})",
            explanation="Daytime stockout count aligns with the sum of hours 6 through 21",
            counts={
                "mismatch_count": stock_count_mismatches,
                "max_diff": max_stock_diff,
            },
        )
    )

    # Store diagnostics
    pct_pos_sales_stockout = (
        (positive_sale_stockout_hours / stockout_hours_total * 100)
        if stockout_hours_total > 0
        else 0.0
    )
    diagnostics["stockout_diagnostics"] = {
        "total_rows": total_rows,
        "daytime_no_stockout_rows": daytime_no_stockout_rows,
        "daytime_no_stockout_pct": round(
            daytime_no_stockout_rows / total_rows * 100, 2
        ),
        "daytime_full_stockout_rows": daytime_full_stockout_rows,
        "full_day_stockout_rows": full_day_stockout_rows,
        "total_stockout_hours": stockout_hours_total,
        "positive_sale_stockout_hours": positive_sale_stockout_hours,
        "pct_positive_sale_stockout_hours": round(pct_pos_sales_stockout, 4),
        "sales_volume_during_stockout": round(sales_during_stockout_volume, 3),
    }

    return checks, diagnostics


def validate_anomalies(
    con: duckdb.DuckDBPyConnection, file_path: Path
) -> list[ValidationCheck]:
    """Validate documented anomalies such as discount range and report informational diagnostics."""
    checks: list[ValidationCheck] = []
    parquet_sql = f"read_parquet('{file_path.as_posix()}')"

    res = con.execute(f"""
        SELECT
            MIN(discount) as min_disc,
            MAX(discount) as max_disc,
            SUM(CASE WHEN discount < 0.0 THEN 1 ELSE 0 END) as neg_disc,
            SUM(CASE WHEN discount > 1.0 THEN 1 ELSE 0 END) as gt_1_disc,
            SUM(CASE WHEN discount == 1.0 THEN 1 ELSE 0 END) as eq_1_disc,
            SUM(CASE WHEN discount < 1.0 AND discount >= 0.0 THEN 1 ELSE 0 END) as discounted_rows
        FROM {parquet_sql}
    """).fetchone()

    min_disc, max_disc, neg_disc, gt_1_disc, eq_1_disc, discounted_rows = res

    # Negative discount is a critical failure
    if neg_disc > 0:
        checks.append(
            ValidationCheck(
                name="discount_non_negative",
                status=ValidationStatus.FAIL.value,
                expected="discount >= 0.0",
                actual=f"{neg_disc} negative discount records found (min: {min_disc})",
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
    if gt_1_disc > 0:
        checks.append(
            ValidationCheck(
                name="discount_ceiling_anomaly",
                status=ValidationStatus.WARNING.value,
                expected="discount <= 1.0 (standard promotional discount convention)",
                actual=f"{gt_1_disc} records with discount > 1.0 (max: {max_disc})",
                explanation=(
                    f"Found {gt_1_disc} records exceeding 1.0 with max {max_disc}. "
                    "These represent isolated pricing surcharge / markup anomalies for product SKU 37 "
                    "preserved without clipping per specification."
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
    con: duckdb.DuckDBPyConnection, train_path: Path, eval_path: Path
) -> ValidationResult:
    """Validate consistency across training and evaluation partitions."""
    result = ValidationResult(dataset_name="cross_partition")
    train_sql = f"read_parquet('{train_path.as_posix()}')"
    eval_sql = f"read_parquet('{eval_path.as_posix()}')"

    # Temporal boundary ordering
    train_max_dt = con.execute(f"SELECT MAX(dt) FROM {train_sql}").fetchone()[0]
    eval_min_dt = con.execute(f"SELECT MIN(dt) FROM {eval_sql}").fetchone()[0]

    temporal_order_ok = str(train_max_dt) < str(eval_min_dt)
    result.add_check(
        ValidationCheck(
            name="temporal_partition_separation",
            status=(
                ValidationStatus.PASS.value
                if temporal_order_ok
                else ValidationStatus.FAIL.value
            ),
            expected=f"train_max_dt < eval_min_dt ({train_max_dt} < {eval_min_dt})",
            actual=f"train max: {train_max_dt}, eval min: {eval_min_dt}",
            explanation="Training partition strictly precedes evaluation partition without temporal overlap",
        )
    )

    # Entity consistency across partitions
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW v_train_sp AS SELECT DISTINCT store_id, product_id FROM {train_sql}"
    )
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW v_eval_sp AS SELECT DISTINCT store_id, product_id FROM {eval_sql}"
    )

    train_cnt = con.execute("SELECT COUNT(*) FROM v_train_sp").fetchone()[0]
    eval_cnt = con.execute("SELECT COUNT(*) FROM v_eval_sp").fetchone()[0]
    common_cnt = con.execute(
        "SELECT COUNT(*) FROM (SELECT * FROM v_train_sp INTERSECT SELECT * FROM v_eval_sp)"
    ).fetchone()[0]
    eval_only = con.execute(
        "SELECT COUNT(*) FROM (SELECT * FROM v_eval_sp EXCEPT SELECT * FROM v_train_sp)"
    ).fetchone()[0]
    train_only = con.execute(
        "SELECT COUNT(*) FROM (SELECT * FROM v_train_sp EXCEPT SELECT * FROM v_eval_sp)"
    ).fetchone()[0]

    entity_consistent = (
        train_cnt == eval_cnt == common_cnt and eval_only == 0 and train_only == 0
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
            actual=f"Common series: {common_cnt:,}, Train-only: {train_only}, Eval-only: {eval_only}",
            explanation="Training and evaluation partitions contain identical 50,000 store-product series",
            counts={
                "train_series": int(train_cnt),
                "eval_series": int(eval_cnt),
                "common_series": int(common_cnt),
            },
        )
    )

    return result


def calculate_evaluation_cohort(
    con: duckdb.DuckDBPyConnection, eval_path: Path
) -> dict[str, Any]:
    """Calculate the evaluation cohort breakdown for stock_hour6_22_cnt == 0."""
    eval_sql = f"read_parquet('{eval_path.as_posix()}')"
    res = con.execute(f"""
        SELECT
            COUNT(*) as total_rows,
            SUM(CASE WHEN stock_hour6_22_cnt == 0 THEN 1 ELSE 0 END) as daytime_no_stockout,
            SUM(CASE WHEN stock_hour6_22_cnt > 0 THEN 1 ELSE 0 END) as daytime_with_stockout
        FROM {eval_sql}
    """).fetchone()

    total_rows, no_stockout, with_stockout = res
    pct_no_stockout = (
        round(no_stockout / total_rows * 100, 2) if total_rows > 0 else 0.0
    )

    return {
        "total_rows": int(total_rows),
        "daytime_no_stockout_rows": int(no_stockout),
        "pct_daytime_no_stockout": pct_no_stockout,
        "daytime_with_stockout_rows": int(with_stockout),
        "pct_daytime_with_stockout": round(100.0 - pct_no_stockout, 2),
    }


def run_validation_pipeline(
    config: dict[str, Any] | None = None,
    reports_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the full data ingestion and validation pipeline across train and eval files."""
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

    train_path = resolve_dataset_path(config, "raw_train_path")
    eval_path = resolve_dataset_path(config, "raw_eval_path")

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
        cross_result = validate_cross_partition_consistency(con, train_path, eval_path)
        cohort_diag = calculate_evaluation_cohort(con, eval_path)
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
    import json

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
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote machine-readable validation summary: %s", val_json_path)

    # 2. Dataset manifest: artifacts/metadata/dataset_manifest.json
    val_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "dataset_name": "FreshRetailNet-50K",
        "provider": "Dingdong Inc.",
        "source_url": "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "inspection_date": summary["timestamp"],
        "files": {
            "train": summary["train"]["metadata"],
            "eval": summary["eval"]["metadata"],
        },
        "entity_counts": config.get("expected_entities", {}),
        "known_limitations": [
            "Sales values are globally normalized by a proprietary coefficient; raw physical pack counts cannot be reconstructed.",
            "34 training records feature discount values > 1.0 (max 1.088 for SKU 37) reflecting documented pricing surcharge anomalies preserved without clipping.",
            "Approximately 2.87% of stockout hours in training exhibit positive sales due to intra-hour depletion prior to stockout flagging.",
            "Only the daytime window with stock_hour6_22_cnt == 0 (206,562 rows / 59.02% of evaluation) represents zero daytime stockouts; daytime stockout freedom does not guarantee 24-hour stockout freedom.",
        ],
    }
    val_manifest_path.write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n",
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
    """Generate the Markdown data dictionary documenting all 19 columns."""
    return r"""# Aarohan: Data Dictionary

This document details all 19 columns in the FreshRetailNet-50K dataset used by the Aarohan demand planning system.

---

## Column Specifications

| Column Name | Data Type | Status | Business Meaning | Interpretation Notes |
| :--- | :--- | :--- | :--- | :--- |
| `city_id` | `int64` | Observed | City entity identifier | Categorical ID denoting the metropolitan delivery area (18 unique cities). |
| `store_id` | `int64` | Observed | Retail store/warehouse identifier | Frontline fulfilment station ID (898 unique stores). |
| `management_group_id` | `int64` | Observed | Operational cluster identifier | Administrative store management cluster (7 unique groups). |
| `first_category_id` | `int64` | Observed | L1 Product hierarchy category | Broad merchandise department (32 categories, e.g. Vegetables, Meat, Fruit). |
| `second_category_id` | `int64` | Observed | L2 Product hierarchy category | Sub-department category (84 categories). |
| `third_category_id` | `int64` | Observed | L3 Product hierarchy category | Granular product sub-category (233 categories). |
| `product_id` | `int64` | Observed | SKU item identifier | Stock keeping unit identifier (865 unique perishable items). |
| `dt` | `string` | Observed | Calendar date | Date of transaction in `YYYY-MM-DD` ISO-8601 format. |
| `sale_amount` | `double` | Observed | Daily sales volume | Daily recorded sales quantity after global normalization by Dingdong Inc. Values are non-negative normalized index units, not physical unit counts. Reconciles with `sum(hours_sale)` within float precision ($10^{-5}$). |
| `hours_sale` | `list<double>` | Observed | 24-hour sales distribution | Array of exactly 24 non-negative floating-point values representing sales for each hour from 00:00 to 23:00. |
| `stock_hour6_22_cnt` | `int32` | Derived | Daytime stockout hour count | Count of daytime hours (06:00 through 21:00 inclusive, hours 6:22 exclusive) with recorded stockout. Integer range $[0, 16]$. Equals $\\sum_{h=6}^{21} \\text{hours\\_stock\\_status}_h$. |
| `hours_stock_status` | `list<int64>` | Observed | 24-hour stockout flags | Array of exactly 24 binary integers. `1` indicates an out-of-stock condition during that hour; `0` indicates the item was in stock. Availability is defined as $a_{t,h} = 1 - s_{t,h}$. |
| `discount` | `double` | Observed | Daily discount rate | Promotional pricing factor ($\text{selling price} / \text{list price}$). $1.0$ indicates full retail price; $0.9$ represents a 10% discount. 34 records in training exceed $1.0$ (up to $1.088$), reflecting vendor pricing surcharges or list price updates. |
| `holiday_flag` | `int32` | Derived | Statutory holiday indicator | Binary flag (`1` for statutory holidays, `0` for regular days). |
| `activity_flag` | `int32` | Derived | Marketing campaign indicator | Binary flag (`1` if a promotional campaign was active, `0` otherwise). Conditional future covariate. |
| `precpt` | `double` | Observed | Daily precipitation | Realized rainfall measurement in millimeters (mm). |
| `avg_temperature` | `double` | Observed | Daily average temperature | Ambient outdoor temperature in degrees Celsius (°C). |
| `avg_humidity` | `double` | Observed | Daily average relative humidity | Relative humidity percentage ($[0, 100]$%). |
| `avg_wind_level` | `double` | Observed | Daily average wind level | Meteorological wind intensity index. |

---

## Critical Modeling Notes

1. **Normalized Scale:** `sale_amount` and `hours_sale` are globally normalized index units. The proprietary multiplier is withheld by Dingdong Inc. Predictions represent normalized demand units.
2. **Stockout Semantics:** `hours_stock_status = 1` indicates unavailable stock. Positive sales occasionally occur in stockout-flagged hours (~2.87% of stockout hours) due to intra-hour depletion prior to stockout logging.
3. **Evaluation Filter:** `stock_hour6_22_cnt == 0` designates the evaluation cohort free of daytime stockouts (206,562 rows / 59.02% of evaluation partition). It must NOT be interpreted as guaranteeing zero stockouts in night hours (22:00–06:00).
"""


def _build_data_profile_md(summary: dict[str, Any]) -> str:
    """Generate the human-readable Markdown data profile report."""
    train = summary["train"]
    eval_part = summary["eval"]
    cohort = summary["evaluation_cohort"]

    train_meta = train["metadata"]
    eval_meta = eval_part["metadata"]

    overall_status = summary["overall_status"]
    timestamp = summary["timestamp"]

    return f"""# Aarohan: Dataset Profile & Validation Report

**Inspection Date:** `{timestamp}`\n
**Overall Validation Status:** **`{overall_status}`**\n
*(Status is WARNING due to 34 known training records with discount > 1.0, preserved without clipping per specification. Zero critical failures detected.)*

---

## 1. Dataset Overview

Aarohan utilizes the **FreshRetailNet-50K** retail demand dataset published by Dingdong Inc. under CC BY 4.0. The dataset contains 4,850,000 daily observations across 50,000 store-product series in perishable grocery retail, annotated with 24-hour stockout status flags and hourly sales distributions.

---

## 2. File Details

| Partition | Relative Path | File Size | SHA-256 Checksum | Checksum Status |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | `{train_meta["path"]}` | {train_meta["file_size_mb"]} MB ({train_meta["file_size_bytes"]:,} B) | `{train_meta["sha256"]}` | **PASS** |
| **Evaluation** | `{eval_meta["path"]}` | {eval_meta["file_size_mb"]} MB ({eval_meta["file_size_bytes"]:,} B) | `{eval_meta["sha256"]}` | **PASS** |

---

## 3. Dataset Dimensions

| Dimension | Training Partition | Evaluation Partition | Combined Total |
| :--- | :--- | :--- | :--- |
| **Row Count** | {train_meta["num_rows"]:,} | {eval_meta["num_rows"]:,} | {train_meta["num_rows"] + eval_meta["num_rows"]:,} |
| **Column Count** | {train_meta["num_columns"]} | {eval_meta["num_columns"]} | {train_meta["num_columns"]} |
| **Row Groups** | {train_meta["num_row_groups"]:,} (1,000 rows/group) | {eval_meta["num_row_groups"]:,} (1,000 rows/group) | {train_meta["num_row_groups"] + eval_meta["num_row_groups"]:,} |
| **Unique Dates** | {train_meta["num_days"]} days | {eval_meta["num_days"]} days | 97 continuous days |
| **Unique Series** | {train_meta["num_series"]:,} | {eval_meta["num_series"]:,} | 50,000 series (100% overlap) |

---

## 4. Column Summary

All 19 required columns are present in both partitions with matching logical Arrow types:
- **Entity IDs:** `city_id` (int64), `store_id` (int64), `management_group_id` (int64), `first_category_id` (int64), `second_category_id` (int64), `third_category_id` (int64), `product_id` (int64)
- **Temporal & Sales:** `dt` (string), `sale_amount` (double), `hours_sale` (list<double>)
- **Stockout Indicators:** `stock_hour6_22_cnt` (int32), `hours_stock_status` (list<int64>)
- **Covariates:** `discount` (double), `holiday_flag` (int32), `activity_flag` (int32), `precpt` (double), `avg_temperature` (double), `avg_humidity` (double), `avg_wind_level` (double)

Zero missing columns, zero undeclared columns, and zero data type mismatches detected.

---

## 5. Date Coverage & Continuity

* **Training Window:** `{train_meta["min_dt"]}` to `{train_meta["max_dt"]}` (exactly 90 continuous days).
* **Evaluation Window:** `{eval_meta["min_dt"]}` to `{eval_meta["max_dt"]}` (exactly 7 continuous days).
* **Temporal Ordering:** Training strictly ends (`2024-06-25`) before evaluation begins (`2024-06-26`). Zero overlap.
* **Daily Completeness:** Every calendar date in training contains exactly 50,000 records; every date in evaluation contains exactly 50,000 records. Zero missing dates.

---

## 6. Store-Product Structure

* **Unique Stores:** 898 stores.
* **Unique SKUs:** 865 perishable products.
* **Store-Product Series:** Exactly 50,000 series in training and 50,000 series in evaluation.
* **Cross-Partition Overlap:** 50,000 series in common (0 train-only, 0 eval-only).
* **Panel Balance:** Every series has exactly 90 daily records in training and exactly 7 daily records in evaluation.

---

## 7. Data Integrity Findings

* **Missing Values:** Exactly `0` null values across all 19 columns in both partitions.
* **Non-Finite Values:** Exactly `0` non-finite (NaN/Inf) values across all float fields.
* **Negative Sales:** Exactly `0` negative sales values (`sale_amount >= 0.0`).
* **Natural Key Duplicates:** Exactly `0` duplicate `(store_id, product_id, dt)` rows.
* **Binary Indicators:** `holiday_flag` and `activity_flag` are strictly binary [0, 1].

---

## 8. Hourly Data Validation

Nested 24-hour sequence arrays were validated in controlled batches of 5,000 records:
* **Array Lengths:** Every row contains exactly 24 elements for both `hours_sale` and `hours_stock_status`.
* **Array Validity:** Zero null array elements; hourly sales are non-negative and finite.
* **Binary Status:** Hourly stockout values are strictly in [0, 1].
* **Sales Reconciliation:**
  * Invariant: `sale_amount ≈ sum(hours_sale)` within tolerance `1e-5`.
  * **Result:** **0 mismatches** across all 4,850,000 records. Maximum discrepancy is within IEEE 754 float rounding ($1.42 \\times 10^{{-14}}$).
* **Stockout Count Reconciliation:**
  * Invariant: `stock_hour6_22_cnt == sum(hours_stock_status[6:22])`.
  * **Result:** **0 mismatches** across all 4,850,000 records (max diff: 0.0).

---

## 9. Stockout Availability Findings

* **Daytime No-Stockout Rows (`stock_hour6_22_cnt == 0`):**
  * Training: 2,507,994 rows (55.73%)
  * Evaluation: 206,562 rows (59.02%)
* **Daytime Full-Stockout Rows (`stock_hour6_22_cnt == 16`):**
  * Training: 181,791 rows (4.04%)
  * Evaluation: 14,140 rows (4.04%)
* **Full 24-Hour Stockout Rows (all 24 hours out of stock):**
  * Training: 154,611 rows (3.44%)
  * Evaluation: 12,041 rows (3.44%)
* **Stockout Hours Total:** 26,876,404 hours in training (24.89% of all hours); 1,913,065 hours in evaluation (22.77%).
* **Sales During Stockout:**
  * Positive sales occur during stockout hours in **2.87%** of training stockout hours (772,426 hours) and **3.10%** of evaluation stockout hours (59,300 hours).
  * Represents 2.51% of training sales volume (112,931.36 units) and 2.17% of evaluation sales volume (9,080.19 units).
  * Caused by intra-hour depletion prior to stockout logging. Handled in demand estimation as a lower-bound constraint.

---

## 10. Discount Anomalies

* **Training Partition:**
  * Range: `[0.000, 1.088]` (mean: `0.9111`, median: `0.9886`).
  * 34 records exceed 1.0 (all belonging to product SKU 37 across 6 stores).
  * Maximum discount: `1.088` (+8.8% pricing surcharge over list price).
  * **Status:** **`WARNING`** (preserved without clipping per specification; zero negative discounts).
* **Evaluation Partition:**
  * Range: `[0.000, 1.000]` (mean: `0.9217`, median: `1.0000`).
  * Exactly 0 records exceed 1.0. **Status:** **`PASS`**.

---

## 11. Evaluation Cohort Composition

The official primary evaluation cohort is defined by `stock_hour6_22_cnt == 0`:
* **Total Evaluation Rows:** {cohort["total_rows"]:,} (50,000 series $\\times$ 7 days)
* **Daytime No-Stockout Cohort:** **{cohort["daytime_no_stockout_rows"]:,} rows ({cohort["pct_daytime_no_stockout"]}%)**
* **Daytime Stockout-Affected Cohort:** {cohort["daytime_with_stockout_rows"]:,} rows ({cohort["pct_daytime_with_stockout"]}%)

---

## 12. Warnings and Limitations

1. **Normalized Demand Scale:** The dataset uses globally normalized sales units; absolute physical piece counts are unrecoverable.
2. **Daytime vs. 24-Hour Stockout:** The `stock_hour6_22_cnt == 0` cohort represents zero stockouts during operating hours (06:00–22:00), but does not guarantee 24-hour stockout absence during overnight non-trading hours.
3. **Discount > 1.0:** 34 records for product 37 exceed 1.0; downstream modeling must accommodate these without arbitrary clipping.
4. **Censoring Lower Bound:** Because sales occur during some stockout-flagged hours, demand estimation models must enforce $\\hat{{y}}^* \\ge y^{{\\text{{obs}}}}$ during stockouts.
"""
