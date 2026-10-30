"""Unit tests for Aarohan data ingestion, validation, and reporting pipeline.

Tests use synthetic Parquet fixtures constructed via PyArrow in temporary
directories (tmp_path). Covers all integrity, schema, continuity, hourly
sequences, anomalies, cohort checks, and safe pipeline failure handling.
"""

from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aarohan.config import get_project_root
from aarohan.data.dataset import ValidationStatus
from aarohan.data.load import (
    check_file_exists,
    create_duckdb_connection,
    read_parquet_metadata,
)
from aarohan.data.validate import (
    calculate_evaluation_cohort,
    run_validation_pipeline,
    validate_anomalies,
    validate_cross_partition_consistency,
    validate_entity_and_time_series,
    validate_file_level,
    validate_general_integrity,
    validate_hourly_data,
    validate_schema,
)

BASE_SCHEMA_DICT = {
    "city_id": "int64",
    "store_id": "int64",
    "management_group_id": "int64",
    "first_category_id": "int64",
    "second_category_id": "int64",
    "third_category_id": "int64",
    "product_id": "int64",
    "dt": "string",
    "sale_amount": "double",
    "hours_sale": "list<item: double>",
    "stock_hour6_22_cnt": "int32",
    "hours_stock_status": "list<item: int64>",
    "discount": "double",
    "holiday_flag": "int32",
    "activity_flag": "int32",
    "precpt": "double",
    "avg_temperature": "double",
    "avg_humidity": "double",
    "avg_wind_level": "double",
}


def get_test_arrow_schema() -> pa.Schema:
    """Return standard PyArrow schema matching dataset specification."""
    return pa.schema(
        [
            pa.field("city_id", pa.int64()),
            pa.field("store_id", pa.int64()),
            pa.field("management_group_id", pa.int64()),
            pa.field("first_category_id", pa.int64()),
            pa.field("second_category_id", pa.int64()),
            pa.field("third_category_id", pa.int64()),
            pa.field("product_id", pa.int64()),
            pa.field("dt", pa.string()),
            pa.field("sale_amount", pa.float64()),
            pa.field("hours_sale", pa.list_(pa.float64())),
            pa.field("stock_hour6_22_cnt", pa.int32()),
            pa.field("hours_stock_status", pa.list_(pa.int64())),
            pa.field("discount", pa.float64()),
            pa.field("holiday_flag", pa.int32()),
            pa.field("activity_flag", pa.int32()),
            pa.field("precpt", pa.float64()),
            pa.field("avg_temperature", pa.float64()),
            pa.field("avg_humidity", pa.float64()),
            pa.field("avg_wind_level", pa.float64()),
        ]
    )


def make_synthetic_records(
    num_series: int = 2,
    num_days: int = 3,
    start_date: str = "2024-01-01",
    discount_val: float = 0.95,
    sale_val: float = 24.0,
    hourly_sale_val: float = 1.0,
    stockout_pattern: list[int] | None = None,
) -> dict[str, list[Any]]:
    """Generate columnar dictionary of valid synthetic records."""
    if stockout_pattern is None:
        stockout_pattern = [0] * 24

    daytime_cnt = sum(stockout_pattern[6:22])
    hours_sale = [hourly_sale_val] * 24

    records: dict[str, list[Any]] = {
        "city_id": [],
        "store_id": [],
        "management_group_id": [],
        "first_category_id": [],
        "second_category_id": [],
        "third_category_id": [],
        "product_id": [],
        "dt": [],
        "sale_amount": [],
        "hours_sale": [],
        "stock_hour6_22_cnt": [],
        "hours_stock_status": [],
        "discount": [],
        "holiday_flag": [],
        "activity_flag": [],
        "precpt": [],
        "avg_temperature": [],
        "avg_humidity": [],
        "avg_wind_level": [],
    }

    base_dt = datetime.date.fromisoformat(start_date)

    for s_idx in range(num_series):
        store_id = 100 + s_idx
        product_id = 200 + s_idx
        for d_idx in range(num_days):
            current_dt = (base_dt + datetime.timedelta(days=d_idx)).isoformat()
            records["city_id"].append(1)
            records["store_id"].append(store_id)
            records["management_group_id"].append(1)
            records["first_category_id"].append(1)
            records["second_category_id"].append(1)
            records["third_category_id"].append(1)
            records["product_id"].append(product_id)
            records["dt"].append(current_dt)
            records["sale_amount"].append(sale_val)
            records["hours_sale"].append(list(hours_sale))
            records["stock_hour6_22_cnt"].append(daytime_cnt)
            records["hours_stock_status"].append(list(stockout_pattern))
            records["discount"].append(discount_val)
            records["holiday_flag"].append(0)
            records["activity_flag"].append(0)
            records["precpt"].append(0.0)
            records["avg_temperature"].append(20.0)
            records["avg_humidity"].append(50.0)
            records["avg_wind_level"].append(1.0)

    return records


def write_parquet(
    path: Path,
    data: dict[str, list[Any]],
    schema: pa.Schema | None = None,
) -> Path:
    """Write dictionary data to Parquet using PyArrow."""
    if schema is not None:
        table = pa.Table.from_pydict(data, schema=schema)
    else:
        # Check if columns match standard 19 columns
        if set(data.keys()) == set(BASE_SCHEMA_DICT.keys()):
            table = pa.Table.from_pydict(data, schema=get_test_arrow_schema())
        else:
            table = pa.Table.from_pydict(data)
    pq.write_table(table, path)
    return path


@pytest.fixture
def duckdb_con():
    """DuckDB connection fixture with low memory footprint."""
    con = create_duckdb_connection(threads=1, max_memory="128MB")
    yield con
    con.close()


# 1. Valid synthetic dataset
def test_valid_synthetic_dataset(tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection):
    """Test that a clean, valid synthetic dataset passes all standard checks."""
    data = make_synthetic_records(num_series=2, num_days=3)
    p = write_parquet(tmp_path / "valid.parquet", data)

    meta = read_parquet_metadata(p)
    assert meta.num_rows == 6
    assert meta.num_columns == 19

    checks = validate_schema(p, BASE_SCHEMA_DICT)
    assert all(c.status == ValidationStatus.PASS.value for c in checks)

    integrity_checks = validate_general_integrity(duckdb_con, p)
    assert all(c.status == ValidationStatus.PASS.value for c in integrity_checks)

    hourly_checks, diag = validate_hourly_data(p, batch_size=5)
    assert all(c.status == ValidationStatus.PASS.value for c in hourly_checks)
    assert diag["stockout_diagnostics"]["total_rows"] == 6


# 2. Missing file
def test_missing_file(tmp_path: Path):
    """Test handling of missing file."""
    p = tmp_path / "does_not_exist.parquet"
    assert not check_file_exists(p)
    checks, meta = validate_file_level(p)
    assert meta is None
    assert any(c.status == ValidationStatus.FAIL.value for c in checks)


# 3. Incorrect schema: missing column
def test_schema_missing_column(tmp_path: Path):
    """Test detection of missing columns in schema."""
    data = make_synthetic_records()
    del data["discount"]
    p = write_parquet(tmp_path / "missing_col.parquet", data)

    checks = validate_schema(p, BASE_SCHEMA_DICT)
    col_check = next(c for c in checks if c.name == "schema_missing_columns")
    assert col_check.status == ValidationStatus.FAIL.value
    assert "discount" in col_check.actual


# 4. Incorrect schema: unexpected column
def test_schema_unexpected_column(tmp_path: Path):
    """Test detection of unexpected extra columns in schema."""
    data = make_synthetic_records()
    data["extra_column"] = [1.0] * len(data["dt"])
    p = write_parquet(tmp_path / "extra_col.parquet", data)

    checks = validate_schema(p, BASE_SCHEMA_DICT)
    col_check = next(c for c in checks if c.name == "schema_unexpected_columns")
    assert col_check.status == ValidationStatus.FAIL.value
    assert "extra_column" in col_check.actual


# 5. Incorrect schema: wrong data type
def test_schema_wrong_data_type(tmp_path: Path):
    """Test detection of mismatched column data types."""
    data = make_synthetic_records()
    data["discount"] = ["0.95"] * len(data["dt"])

    # Build schema where discount is string
    fields = [f for f in get_test_arrow_schema() if f.name != "discount"]
    fields.append(pa.field("discount", pa.string()))
    custom_schema = pa.schema(fields)

    p = write_parquet(tmp_path / "wrong_type.parquet", data, schema=custom_schema)

    checks = validate_schema(p, BASE_SCHEMA_DICT)
    type_check = next(c for c in checks if c.name == "schema_data_types")
    assert type_check.status == ValidationStatus.FAIL.value
    assert "discount" in type_check.explanation


# 6. Missing values / nulls detected
def test_null_values_detected(tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection):
    """Test detection of null / missing values."""
    data = make_synthetic_records()
    data["discount"][0] = None  # type: ignore
    p = write_parquet(tmp_path / "nulls.parquet", data)

    checks = validate_general_integrity(duckdb_con, p)
    null_check = next(c for c in checks if c.name == "missing_values_scalar")
    assert null_check.status == ValidationStatus.FAIL.value
    assert null_check.actual > 0


# 7. Duplicate natural keys
def test_duplicate_natural_keys(tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection):
    """Test detection of duplicate natural keys (store_id, product_id, dt)."""
    data = make_synthetic_records(num_series=1, num_days=2)
    # Duplicate first record
    for k in data:
        data[k].append(data[k][0])
    p = write_parquet(tmp_path / "duplicates.parquet", data)

    checks = validate_general_integrity(duckdb_con, p)
    dup_check = next(c for c in checks if c.name == "natural_key_duplicates")
    assert dup_check.status == ValidationStatus.FAIL.value
    assert dup_check.actual == 1


# 8. Invalid date formats
def test_invalid_date_format(tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection):
    """Test detection of invalid date string formats."""
    data = make_synthetic_records()
    data["dt"][0] = "2024/01/01"  # Slashes instead of dashes
    p = write_parquet(tmp_path / "invalid_date.parquet", data)

    checks = validate_general_integrity(duckdb_con, p)
    date_check = next(c for c in checks if c.name == "date_format_check")
    assert date_check.status == ValidationStatus.FAIL.value
    assert date_check.actual == 1


# 9. Incorrect date boundaries
def test_incorrect_date_boundaries(
    tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection
):
    """Test validation of expected date boundaries."""
    data = make_synthetic_records(num_series=1, num_days=3, start_date="2024-05-01")
    p = write_parquet(tmp_path / "dates.parquet", data)

    expected_size = {"start_date": "2024-01-01", "end_date": "2024-01-03", "dates": 3}
    checks = validate_entity_and_time_series(
        duckdb_con, p, expected_size=expected_size, expected_entities={}
    )
    bound_check = next(c for c in checks if c.name == "date_boundaries")
    assert bound_check.status == ValidationStatus.FAIL.value


# 10. Incomplete panel: missing days for a series
def test_incomplete_panel_missing_days(
    tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection
):
    """Test detection of series with missing days."""
    data = make_synthetic_records(num_series=2, num_days=3)
    # Drop one row for series 1
    for k in data:
        data[k].pop(0)
    p = write_parquet(tmp_path / "missing_days.parquet", data)

    expected_size = {"dates": 3, "series": 2}
    checks = validate_entity_and_time_series(
        duckdb_con, p, expected_size=expected_size, expected_entities={}
    )
    panel_check = next(c for c in checks if c.name == "series_observation_balance")
    assert panel_check.status == ValidationStatus.FAIL.value


# 11. Incomplete panel: missing series on a day
def test_incomplete_panel_missing_series_on_day(
    tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection
):
    """Test detection of daily record count discrepancies."""
    data = make_synthetic_records(num_series=2, num_days=3)
    # Drop one row
    for k in data:
        data[k].pop(0)
    p = write_parquet(tmp_path / "missing_series.parquet", data)

    expected_size = {"dates": 3, "series": 2}
    checks = validate_entity_and_time_series(
        duckdb_con, p, expected_size=expected_size, expected_entities={}
    )
    daily_check = next(c for c in checks if c.name == "daily_record_balance")
    assert daily_check.status == ValidationStatus.FAIL.value


# 12. Invalid hourly array lengths (!= 24)
def test_invalid_hourly_array_lengths(tmp_path: Path):
    """Test failure when hourly array length does not equal 24."""
    data = make_synthetic_records()
    data["hours_sale"][0] = [1.0] * 23  # Length 23 instead of 24
    p = write_parquet(tmp_path / "bad_array_len.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=5)
    len_check = next(c for c in checks if c.name == "hourly_array_lengths")
    assert len_check.status == ValidationStatus.FAIL.value
    assert len_check.counts["invalid_length_rows"] == 1


# 13. Non-binary stockout flags
def test_non_binary_stockout_flags(tmp_path: Path):
    """Test failure when hourly stockout array contains values other than 0 or 1."""
    data = make_synthetic_records()
    data["hours_stock_status"][0][5] = 2  # Invalid non-binary flag
    p = write_parquet(tmp_path / "non_binary_stockout.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=5)
    binary_check = next(c for c in checks if c.name == "hourly_stock_status_binary")
    assert binary_check.status == ValidationStatus.FAIL.value


# 14. Invalid daytime stockout count calculation
def test_mismatched_daytime_stockout_count(tmp_path: Path):
    """Test failure when stock_hour6_22_cnt disagrees with hourly stockout status."""
    data = make_synthetic_records()
    data["stock_hour6_22_cnt"][0] = 99  # Does not equal sum of hours 6-21
    p = write_parquet(tmp_path / "mismatched_cnt.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=5)
    cnt_check = next(c for c in checks if c.name == "stockout_count_reconciliation")
    assert cnt_check.status == ValidationStatus.FAIL.value


# 15. Daily and hourly sales mismatch (> tolerance)
def test_daily_hourly_sales_mismatch(tmp_path: Path):
    """Test failure when sale_amount disagrees with sum(hours_sale)."""
    data = make_synthetic_records(
        sale_val=100.0, hourly_sale_val=1.0
    )  # sum is 24, diff is 76
    p = write_parquet(tmp_path / "sales_mismatch.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=5, tolerance=1e-5)
    reconcile_check = next(
        c for c in checks if c.name == "daily_hourly_sales_reconciliation"
    )
    assert reconcile_check.status == ValidationStatus.FAIL.value


# 16. Negative sales values
def test_negative_sales_values(tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection):
    """Test detection and failure on negative sales amounts."""
    data = make_synthetic_records()
    data["sale_amount"][0] = -5.0
    p = write_parquet(tmp_path / "neg_sales.parquet", data)

    checks = validate_general_integrity(duckdb_con, p)
    sales_check = next(c for c in checks if c.name == "negative_sales")
    assert sales_check.status == ValidationStatus.FAIL.value


# 17. Non-finite values (NaN / Inf)
def test_non_finite_values(tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection):
    """Test detection and failure on non-finite values (NaN / Inf)."""
    data = make_synthetic_records()
    data["avg_temperature"][0] = float("nan")
    p = write_parquet(tmp_path / "nan_values.parquet", data)

    checks = validate_general_integrity(duckdb_con, p)
    finite_check = next(c for c in checks if c.name == "non_finite_values")
    assert finite_check.status == ValidationStatus.FAIL.value


# 18. Discount > 1.0 produces WARNING (not FAIL)
def test_discount_greater_than_one_warning(
    tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection
):
    """Test that discount > 1.0 generates a WARNING and is not rejected as FAIL."""
    data = make_synthetic_records(discount_val=1.088)
    p = write_parquet(tmp_path / "discount_high.parquet", data)

    checks = validate_anomalies(duckdb_con, p)
    disc_upper_check = next(c for c in checks if c.name == "discount_ceiling_anomaly")
    assert disc_upper_check.status == ValidationStatus.WARNING.value
    assert disc_upper_check.counts["discount_gt_1_count"] > 0


# 19. Discount < 0.0 produces FAIL
def test_discount_negative_fail(tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection):
    """Test that discount < 0.0 is a hard failure (FAIL)."""
    data = make_synthetic_records(discount_val=-0.1)
    p = write_parquet(tmp_path / "discount_neg.parquet", data)

    checks = validate_anomalies(duckdb_con, p)
    disc_lower_check = next(c for c in checks if c.name == "discount_non_negative")
    assert disc_lower_check.status == ValidationStatus.FAIL.value


# Evaluation cohort breakdown test
def test_evaluation_cohort_breakdown(
    tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection
):
    """Test evaluation cohort categorization into daytime no-stockout vs affected."""
    # Create 4 rows: 3 with daytime stockouts = 0, 1 with daytime stockouts = 5
    data = make_synthetic_records(num_series=2, num_days=2)
    data["stock_hour6_22_cnt"] = [0, 0, 0, 5]
    p = write_parquet(tmp_path / "cohort.parquet", data)

    cohort = calculate_evaluation_cohort(duckdb_con, p)
    assert cohort["total_rows"] == 4
    assert cohort["daytime_no_stockout_rows"] == 3
    assert cohort["pct_daytime_no_stockout"] == 75.0
    assert cohort["daytime_with_stockout_rows"] == 1
    assert cohort["pct_daytime_with_stockout"] == 25.0


# Cross-partition consistency test
def test_cross_partition_consistency(
    tmp_path: Path, duckdb_con: duckdb.DuckDBPyConnection
):
    """Test cross-partition checks on disjoint series or overlapping dates."""
    # Clean train: 2 series, dates 2024-01-01 to 2024-01-02
    train_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-01"
    )
    train_p = write_parquet(tmp_path / "train.parquet", train_data)

    # Clean eval: same 2 series, dates 2024-01-03 to 2024-01-04
    eval_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-03"
    )
    eval_p = write_parquet(tmp_path / "eval.parquet", eval_data)

    res = validate_cross_partition_consistency(duckdb_con, train_p, eval_p)
    assert res.status == ValidationStatus.PASS.value

    # Mismatched eval series (different store_id)
    eval_bad_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-03"
    )
    eval_bad_data["store_id"] = [999] * len(eval_bad_data["store_id"])
    eval_bad_p = write_parquet(tmp_path / "eval_bad.parquet", eval_bad_data)

    res_bad = validate_cross_partition_consistency(duckdb_con, train_p, eval_bad_p)
    assert res_bad.status == ValidationStatus.FAIL.value


# SHA-256 Checksum mismatch test
def test_sha256_checksum_mismatch(tmp_path: Path):
    """Test that an incorrect expected SHA-256 produces a FAIL check."""
    data = make_synthetic_records()
    p = write_parquet(tmp_path / "chk.parquet", data)

    bad_hash = "0" * 64
    checks, meta = validate_file_level(p, expected_sha256=bad_hash)
    chk_check = next(c for c in checks if c.name == "sha256_checksum")
    assert chk_check.status == ValidationStatus.FAIL.value


# =========================================================================
# Task 1: Hourly Non-Finite & Array Integrity Unit Tests
# =========================================================================


def test_nan_in_hours_sale_fails(tmp_path: Path):
    """Test that NaN in hours_sale results in FAIL and is reported."""
    data = make_synthetic_records()
    data["hours_sale"][0][5] = float("nan")
    p = write_parquet(tmp_path / "nan.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=5)
    val_check = next(c for c in checks if c.name == "hourly_sales_validity")
    rec_check = next(c for c in checks if c.name == "daily_hourly_sales_reconciliation")
    assert val_check.status == ValidationStatus.FAIL.value
    assert val_check.counts["non_finite_sales_rows"] >= 1
    assert rec_check.status == ValidationStatus.FAIL.value


def test_pos_inf_in_hours_sale_fails(tmp_path: Path):
    """Test that +Inf in hours_sale results in FAIL and is reported."""
    data = make_synthetic_records()
    data["hours_sale"][0][5] = float("inf")
    p = write_parquet(tmp_path / "pos_inf.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=5)
    val_check = next(c for c in checks if c.name == "hourly_sales_validity")
    assert val_check.status == ValidationStatus.FAIL.value
    assert val_check.counts["non_finite_sales_rows"] >= 1


def test_neg_inf_in_hours_sale_fails(tmp_path: Path):
    """Test that -Inf in hours_sale results in FAIL and is reported."""
    data = make_synthetic_records()
    data["hours_sale"][0][5] = float("-inf")
    p = write_parquet(tmp_path / "neg_inf.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=5)
    val_check = next(c for c in checks if c.name == "hourly_sales_validity")
    assert val_check.status == ValidationStatus.FAIL.value
    assert val_check.counts["non_finite_sales_rows"] >= 1


def test_child_null_in_hours_sale_fails(tmp_path: Path):
    """Test that null element within hours_sale results in FAIL."""
    data = make_synthetic_records()
    data["hours_sale"][0][3] = None
    p = write_parquet(tmp_path / "child_null_sale.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=5)
    val_check = next(c for c in checks if c.name == "hourly_sales_validity")
    rec_check = next(c for c in checks if c.name == "daily_hourly_sales_reconciliation")
    assert val_check.status == ValidationStatus.FAIL.value
    assert val_check.counts["null_child_sales_rows"] >= 1
    assert rec_check.status == ValidationStatus.FAIL.value


def test_child_null_in_hours_stock_status_fails(tmp_path: Path):
    """Test that null element within hours_stock_status results in FAIL."""
    data = make_synthetic_records()
    data["hours_stock_status"][0][7] = None
    p = write_parquet(tmp_path / "child_null_stock.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=5)
    st_check = next(c for c in checks if c.name == "hourly_stock_status_binary")
    cnt_check = next(c for c in checks if c.name == "stockout_count_reconciliation")
    assert st_check.status == ValidationStatus.FAIL.value
    assert st_check.counts["null_child_stock_rows"] >= 1
    assert cnt_check.status == ValidationStatus.FAIL.value


def test_parent_null_in_hourly_arrays_fails(tmp_path: Path):
    """Test that null parent array in hourly columns results in FAIL."""
    data = make_synthetic_records()
    data["hours_sale"][0] = None
    p = write_parquet(tmp_path / "parent_null.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=5)
    len_check = next(c for c in checks if c.name == "hourly_array_lengths")
    assert len_check.status == ValidationStatus.FAIL.value
    assert len_check.counts["null_parent_rows"] >= 1


def test_finite_sale_amount_with_non_finite_hourly_fails_reconciliation(
    tmp_path: Path,
):
    """Test reconciliation fails when sale_amount is finite but hourly has NaN."""
    data = make_synthetic_records(sale_val=24.0)
    data["hours_sale"][0][0] = float("nan")
    p = write_parquet(tmp_path / "finite_sale_nan_hourly.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=5)
    rec_check = next(c for c in checks if c.name == "daily_hourly_sales_reconciliation")
    assert rec_check.status == ValidationStatus.FAIL.value
    assert rec_check.counts["mismatch_count"] >= 1


def test_batch_with_mixed_valid_and_invalid_rows(tmp_path: Path):
    """Test batch processing isolates invalid rows without dropping valid ones."""
    data = make_synthetic_records(num_series=3, num_days=2)
    # Row 0: non-finite sales
    data["hours_sale"][0][0] = float("nan")
    # Row 1: negative sales
    data["hours_sale"][1][0] = -1.0
    # Rows 2-5: completely valid
    p = write_parquet(tmp_path / "mixed_batch.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=10)
    val_check = next(c for c in checks if c.name == "hourly_sales_validity")
    assert val_check.status == ValidationStatus.FAIL.value
    assert val_check.counts["non_finite_sales_rows"] == 1
    assert val_check.counts["negative_sales_rows"] == 1


def test_fallback_path_with_malformed_array_lengths(tmp_path: Path):
    """Test fallback path triggered by malformed array length reports failure."""
    data = make_synthetic_records(num_series=2, num_days=2)
    # Row 0: length 10 instead of 24 triggers fallback
    data["hours_sale"][0] = [1.0] * 10
    p = write_parquet(tmp_path / "fallback_len.parquet", data)

    checks, _ = validate_hourly_data(p, batch_size=10)
    len_check = next(c for c in checks if c.name == "hourly_array_lengths")
    rec_check = next(c for c in checks if c.name == "daily_hourly_sales_reconciliation")
    assert len_check.status == ValidationStatus.FAIL.value
    assert len_check.counts["invalid_length_rows"] == 1
    assert rec_check.status == ValidationStatus.FAIL.value


# =========================================================================
# Task 2: Safe Pipeline Failure Integration Tests
# =========================================================================


def make_valid_pipeline_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    """Create valid synthetic datasets and return pipeline config and paths."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = tmp_path / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    art_dir = tmp_path / "artifacts" / "metadata"
    art_dir.mkdir(parents=True, exist_ok=True)

    train_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-01"
    )
    eval_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-03"
    )

    train_path = write_parquet(data_dir / "train.parquet", train_data)
    eval_path = write_parquet(data_dir / "eval.parquet", eval_data)

    config = {
        "raw_train_path": str(train_path),
        "raw_eval_path": str(eval_path),
        "expected_schema": BASE_SCHEMA_DICT,
        "dataset_size": {
            "train": {"rows": 4, "columns": 19, "dates": 2, "series": 2},
            "eval": {"rows": 4, "columns": 19, "dates": 2, "series": 2},
        },
        "expected_entities": {
            "stores": 2,
            "products": 2,
            "store_product_series": 2,
        },
        "hourly_settings": {
            "sales_reconciliation_tolerance": 1e-5,
            "daytime_stockout_start": 6,
            "daytime_stockout_end": 22,
        },
        "processing": {
            "batch_size": 10,
            "duckdb_threads": 1,
            "duckdb_memory_limit": "128MB",
        },
    }
    return config, rep_dir, art_dir / "dataset_manifest.json"


def _verify_deliverables_on_failure(
    summary: dict[str, Any], rep_dir: Path, manifest_path: Path
) -> None:
    """Verify that all deliverables are generated even on validation failure."""
    assert summary["overall_status"] == "FAIL"
    val_json_path = rep_dir / "tables" / "data_validation.json"
    profile_path = rep_dir / "data_profile.md"
    assert val_json_path.is_file()
    assert profile_path.is_file()
    assert manifest_path.is_file()

    # Verify JSON is valid and strictly compliant (no NaN/Inf)
    val_data = json.loads(val_json_path.read_text(encoding="utf-8"))
    assert val_data["overall_status"] == "FAIL"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["dataset_name"] == "FreshRetailNet-50K"


def test_pipeline_missing_train_file(tmp_path: Path):
    """Test pipeline handles missing training Parquet file safely."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    Path(config["raw_train_path"]).unlink()

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    _verify_deliverables_on_failure(summary, rep_dir, manifest_p)


def test_pipeline_missing_eval_file(tmp_path: Path):
    """Test pipeline handles missing evaluation Parquet file safely."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    Path(config["raw_eval_path"]).unlink()

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    _verify_deliverables_on_failure(summary, rep_dir, manifest_p)


def test_pipeline_corrupt_parquet_file(tmp_path: Path):
    """Test pipeline handles corrupted Parquet file safely."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    Path(config["raw_train_path"]).write_bytes(b"CORRUPTED_NOT_A_VALID_PARQUET_FILE")

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    _verify_deliverables_on_failure(summary, rep_dir, manifest_p)


def test_pipeline_missing_schema_column(tmp_path: Path):
    """Test pipeline handles missing schema column safely."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    train_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-01"
    )
    del train_data["discount"]
    write_parquet(Path(config["raw_train_path"]), train_data)

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    _verify_deliverables_on_failure(summary, rep_dir, manifest_p)


def test_pipeline_unexpected_schema_column(tmp_path: Path):
    """Test pipeline handles unexpected extra column safely."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    train_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-01"
    )
    train_data["unexpected_col"] = [1.0] * len(train_data["city_id"])
    write_parquet(Path(config["raw_train_path"]), train_data)

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    _verify_deliverables_on_failure(summary, rep_dir, manifest_p)


def test_pipeline_wrong_column_type(tmp_path: Path):
    """Test pipeline handles column with wrong logical data type safely."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    train_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-01"
    )
    # store_id as string instead of int64
    train_data["store_id"] = [str(x) for x in train_data["store_id"]]
    custom_schema = get_test_arrow_schema()
    idx = custom_schema.get_field_index("store_id")
    custom_schema = custom_schema.set(idx, pa.field("store_id", pa.string()))
    write_parquet(Path(config["raw_train_path"]), train_data, schema=custom_schema)

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    _verify_deliverables_on_failure(summary, rep_dir, manifest_p)


def test_pipeline_invalid_date_strings(tmp_path: Path):
    """Test pipeline handles unparseable date strings safely."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    train_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-01"
    )
    train_data["dt"][0] = "INVALID_DATE_FORMAT"
    write_parquet(Path(config["raw_train_path"]), train_data)

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    _verify_deliverables_on_failure(summary, rep_dir, manifest_p)


def test_pipeline_malformed_hourly_arrays(tmp_path: Path):
    """Test pipeline handles malformed hourly arrays safely."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    train_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-01"
    )
    train_data["hours_sale"][0] = [1.0] * 12
    write_parquet(Path(config["raw_train_path"]), train_data)

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    _verify_deliverables_on_failure(summary, rep_dir, manifest_p)


# =========================================================================
# Task 2: Hourly Parent-Null Separation Unit Tests
# =========================================================================


def test_hourly_parent_null_sales_only(tmp_path: Path):
    """Test parent null separation when hours_sale parent is null only."""
    p = tmp_path / "sales_parent_null.parquet"
    data = make_synthetic_records(num_series=1, num_days=2)
    data["hours_sale"][0] = None
    write_parquet(p, data)

    checks, _ = validate_hourly_data(p)
    chk_map = {c.name: c for c in checks}

    sales_chk = chk_map["hourly_sales_validity"]
    assert sales_chk.status == ValidationStatus.FAIL.value
    assert sales_chk.counts["null_parent_sales_rows"] == 1

    stock_chk = chk_map["hourly_stock_status_binary"]
    assert stock_chk.status == ValidationStatus.PASS.value
    assert stock_chk.counts["null_parent_stock_rows"] == 0


def test_hourly_parent_null_stock_only(tmp_path: Path):
    """Test parent null separation when hours_stock_status parent is null only."""
    p = tmp_path / "stock_parent_null.parquet"
    data = make_synthetic_records(num_series=1, num_days=2)
    data["hours_stock_status"][0] = None
    write_parquet(p, data)

    checks, _ = validate_hourly_data(p)
    chk_map = {c.name: c for c in checks}

    sales_chk = chk_map["hourly_sales_validity"]
    assert sales_chk.status == ValidationStatus.PASS.value
    assert sales_chk.counts["null_parent_sales_rows"] == 0

    stock_chk = chk_map["hourly_stock_status_binary"]
    assert stock_chk.status == ValidationStatus.FAIL.value
    assert stock_chk.counts["null_parent_stock_rows"] == 1


def test_hourly_parent_null_both_same_row(tmp_path: Path):
    """Test parent null separation when both arrays are null on the same row."""
    p = tmp_path / "both_parents_null_same_row.parquet"
    data = make_synthetic_records(num_series=1, num_days=2)
    data["hours_sale"][0] = None
    data["hours_stock_status"][0] = None
    write_parquet(p, data)

    checks, _ = validate_hourly_data(p)
    chk_map = {c.name: c for c in checks}

    sales_chk = chk_map["hourly_sales_validity"]
    stock_chk = chk_map["hourly_stock_status_binary"]
    lens_chk = chk_map["hourly_array_lengths"]

    assert sales_chk.status == ValidationStatus.FAIL.value
    assert sales_chk.counts["null_parent_sales_rows"] == 1
    assert stock_chk.status == ValidationStatus.FAIL.value
    assert stock_chk.counts["null_parent_stock_rows"] == 1
    assert lens_chk.counts["null_parent_rows"] == 1


def test_hourly_parent_null_different_rows(tmp_path: Path):
    """Test parent null separation with nulls in different columns."""
    p = tmp_path / "different_rows_null.parquet"
    data = make_synthetic_records(num_series=1, num_days=2)
    data["hours_sale"][0] = None
    data["hours_stock_status"][1] = None
    write_parquet(p, data)

    checks, _ = validate_hourly_data(p)
    chk_map = {c.name: c for c in checks}

    sales_chk = chk_map["hourly_sales_validity"]
    stock_chk = chk_map["hourly_stock_status_binary"]
    lens_chk = chk_map["hourly_array_lengths"]

    assert sales_chk.status == ValidationStatus.FAIL.value
    assert sales_chk.counts["null_parent_sales_rows"] == 1
    assert stock_chk.status == ValidationStatus.FAIL.value
    assert stock_chk.counts["null_parent_stock_rows"] == 1
    assert lens_chk.counts["null_parent_rows"] == 2


# =========================================================================
# Task 6: Comprehensive Integration Tests (A, B, C, D, E)
# =========================================================================


def test_pipeline_integration_complete_success(tmp_path: Path):
    """Test complete successful pipeline produces all deliverables meeting standards."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    assert summary["overall_status"] == "PASS"

    val_json_path = rep_dir / "tables" / "data_validation.json"
    profile_path = rep_dir / "data_profile.md"
    dict_path = get_project_root() / "docs" / "data_dictionary.md"

    assert val_json_path.is_file()
    assert profile_path.is_file()
    assert manifest_p.is_file()
    assert dict_path.is_file()

    # Verify data_profile.md contains no PASS, FAIL, or execution timestamps
    profile_text = profile_path.read_text(encoding="utf-8")
    assert not re.search(r"\b(PASS|FAIL|WARNING)\b", profile_text)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", profile_text)

    # Verify dataset_manifest.json contains no inspection_date
    manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert "inspection_date" not in manifest_data


def test_pipeline_integration_wrong_logical_types(tmp_path: Path):
    """Test wrong logical types in sale_amount, dt, discount, and hours_sale."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    config["dataset_size"]["train"]["start_date"] = "2024-01-01"
    config["dataset_size"]["train"]["end_date"] = "2024-01-02"
    train_path = Path(config["raw_train_path"])

    train_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-01"
    )
    n = len(train_data["city_id"])
    train_data["sale_amount"] = ["24.0"] * n
    train_data["dt"] = [20240101] * n
    train_data["discount"] = ["0.95"] * n
    train_data["hours_sale"] = [["1.0"] * 24] * n

    custom_schema = get_test_arrow_schema()
    custom_schema = custom_schema.set(
        custom_schema.get_field_index("sale_amount"),
        pa.field("sale_amount", pa.string()),
    )
    custom_schema = custom_schema.set(
        custom_schema.get_field_index("dt"), pa.field("dt", pa.int64())
    )
    custom_schema = custom_schema.set(
        custom_schema.get_field_index("discount"),
        pa.field("discount", pa.string()),
    )
    custom_schema = custom_schema.set(
        custom_schema.get_field_index("hours_sale"),
        pa.field("hours_sale", pa.list_(pa.string())),
    )

    write_parquet(train_path, train_data, schema=custom_schema)

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    assert summary["overall_status"] == "FAIL"

    train_checks = {c["name"]: c for c in summary["train"]["checks"]}
    assert train_checks["negative_sales"]["status"] == "FAIL"
    assert "non-numeric" in train_checks["negative_sales"]["explanation"].lower()

    assert train_checks["date_boundaries"]["status"] == "FAIL"
    assert "non-string" in train_checks["date_boundaries"]["explanation"].lower()

    assert train_checks["discount_non_negative"]["status"] == "FAIL"
    assert "non-numeric" in train_checks["discount_non_negative"]["explanation"].lower()

    assert train_checks["hourly_sales_validity"]["status"] == "FAIL"
    assert "non-numeric" in train_checks["hourly_sales_validity"]["explanation"].lower()

    assert train_checks["file_existence"]["status"] == "PASS"
    assert train_checks["row_count"]["status"] == "PASS"
    assert train_checks["column_count"]["status"] == "PASS"
    assert train_checks["missing_values_scalar"]["status"] == "PASS"
    assert train_checks["hourly_stock_status_binary"]["status"] == "PASS"


def test_pipeline_integration_unaffected_train_missing_eval_valid(
    tmp_path: Path,
):
    """When train is missing, eval partition must validate and produce results."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    Path(config["raw_train_path"]).unlink()

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    assert summary["overall_status"] == "FAIL"
    assert summary["train"]["status"] == "FAIL"
    assert summary["eval"]["status"] == "PASS"
    assert summary["cross_partition"]["status"] == "FAIL"


def test_pipeline_integration_unaffected_eval_missing_train_valid(
    tmp_path: Path,
):
    """When eval is missing, train partition must validate and produce results."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    Path(config["raw_eval_path"]).unlink()

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    assert summary["overall_status"] == "FAIL"
    assert summary["train"]["status"] == "PASS"
    assert summary["eval"]["status"] == "FAIL"
    assert summary["cross_partition"]["status"] == "FAIL"


def test_cli_exit_codes(tmp_path: Path):
    """Test CLI returns exit code 0 for healthy and 1 for failing dataset."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(yaml.dump(config), encoding="utf-8")
    cli_script = get_project_root() / "scripts" / "inspect_data.py"

    # Healthy dataset
    ret_healthy = subprocess.run(
        [
            sys.executable,
            str(cli_script),
            "--config",
            str(cfg_file),
            "--reports-dir",
            str(rep_dir),
            "--manifest-path",
            str(manifest_p),
        ],
        capture_output=True,
    )
    assert ret_healthy.returncode == 0

    # Failing dataset (corrupt train)
    Path(config["raw_train_path"]).write_bytes(b"NOT_A_PARQUET")
    ret_failing = subprocess.run(
        [
            sys.executable,
            str(cli_script),
            "--config",
            str(cfg_file),
            "--reports-dir",
            str(rep_dir),
            "--manifest-path",
            str(manifest_p),
        ],
        capture_output=True,
    )
    assert ret_failing.returncode == 1


def test_report_correctness(tmp_path: Path):
    """Test data_profile.md contains no badges and metrics match JSON."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )

    profile_path = rep_dir / "data_profile.md"
    val_json_path = rep_dir / "tables" / "data_validation.json"

    profile_text = profile_path.read_text(encoding="utf-8")
    assert not re.search(r"\b(PASS|FAIL|WARNING)\b", profile_text)
    assert not re.search(r"\d+\s+(PASS|FAIL)", profile_text)

    val_json = json.loads(val_json_path.read_text(encoding="utf-8"))
    assert val_json["overall_status"] == summary["overall_status"]
    train_rows = val_json["train"]["metadata"]["num_rows"]
    assert f"{train_rows:,}" in profile_text


def test_pipeline_integration_non_zero_scalar_nulls_and_duplicate_keys(
    tmp_path: Path,
):
    """Test non-zero scalar null count and duplicate keys are mapped accurately."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    train_path = Path(config["raw_train_path"])

    train_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-01"
    )
    # Duplicate natural key group by copying the second record
    for k in train_data:
        train_data[k].append(train_data[k][1])

    # Add scalar null in precpt
    train_data["precpt"][0] = None

    write_parquet(train_path, train_data)

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    assert summary["overall_status"] == "FAIL"

    val_json_path = rep_dir / "tables" / "data_validation.json"
    val_json = json.loads(val_json_path.read_text(encoding="utf-8"))
    train_checks = {c["name"]: c for c in val_json["train"]["checks"]}

    assert train_checks["missing_values_scalar"]["status"] == "FAIL"
    assert train_checks["missing_values_scalar"]["counts"]["null_count"] == 1

    assert train_checks["natural_key_duplicates"]["status"] == "FAIL"
    assert train_checks["natural_key_duplicates"]["counts"]["duplicate_groups"] == 1

    profile_path = rep_dir / "data_profile.md"
    profile_text = profile_path.read_text(encoding="utf-8")

    assert (
        "Scalar missing values detected: 1 in training, 0 in evaluation."
        in profile_text
    )
    assert (
        "Natural key duplicates detected: 1 in training, 0 in evaluation."
        in profile_text
    )
    assert (
        "No missing scalar values exist across all columns in either partition."
        not in profile_text
    )
    assert "Zero duplicate records exist on the primary natural key" not in profile_text


def test_pipeline_integration_invalid_key_and_date_types(tmp_path: Path):
    """Test that invalid required key or date type fails safely without crashing."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    train_path = Path(config["raw_train_path"])

    train_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-01"
    )
    n = len(train_data["city_id"])
    train_data["store_id"] = [[1, 2]] * n
    train_data["dt"] = [1.5] * n

    custom_schema = get_test_arrow_schema()
    custom_schema = custom_schema.set(
        custom_schema.get_field_index("store_id"),
        pa.field("store_id", pa.list_(pa.int64())),
    )
    custom_schema = custom_schema.set(
        custom_schema.get_field_index("dt"),
        pa.field("dt", pa.float64()),
    )
    write_parquet(train_path, train_data, schema=custom_schema)

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    assert summary["overall_status"] == "FAIL"

    train_checks = {c["name"]: c for c in summary["train"]["checks"]}
    assert train_checks["unique_stores"]["status"] == "FAIL"
    assert "invalid type" in train_checks["unique_stores"]["explanation"].lower()
    assert train_checks["unique_series"]["status"] == "FAIL"
    assert train_checks["series_observation_balance"]["status"] == "FAIL"
    assert train_checks["daily_record_balance"]["status"] == "FAIL"

    profile_text = (rep_dir / "data_profile.md").read_text(encoding="utf-8")
    assert not re.search(r"\b(PASS|FAIL|WARNING)\b", profile_text)


def test_pipeline_integration_incompatible_cross_partition_key_types(
    tmp_path: Path,
):
    """Test that incompatible cross-partition identifier types fail safely."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    eval_path = Path(config["raw_eval_path"])

    eval_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-03"
    )
    eval_data["store_id"] = [str(s) for s in eval_data["store_id"]]

    custom_schema = get_test_arrow_schema()
    custom_schema = custom_schema.set(
        custom_schema.get_field_index("store_id"),
        pa.field("store_id", pa.string()),
    )
    write_parquet(eval_path, eval_data, schema=custom_schema)

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    assert summary["overall_status"] == "FAIL"

    cross_checks = {c["name"]: c for c in summary["cross_partition"]["checks"]}
    assert cross_checks["cross_partition_series_consistency"]["status"] == "FAIL"
    assert (
        "incompatible"
        in cross_checks["cross_partition_series_consistency"]["explanation"].lower()
    )


def test_pipeline_integration_invalid_hourly_sales_diagnostics_unavailable(
    tmp_path: Path,
):
    """Test invalid hourly sales input marks diagnostics as unavailable, not 0.0."""
    config, rep_dir, manifest_p = make_valid_pipeline_config(tmp_path)
    train_path = Path(config["raw_train_path"])

    train_data = make_synthetic_records(
        num_series=2, num_days=2, start_date="2024-01-01"
    )
    n = len(train_data["city_id"])
    train_data["hours_sale"] = [["1.0"] * 24] * n

    custom_schema = get_test_arrow_schema()
    custom_schema = custom_schema.set(
        custom_schema.get_field_index("hours_sale"),
        pa.field("hours_sale", pa.list_(pa.string())),
    )
    write_parquet(train_path, train_data, schema=custom_schema)

    summary = run_validation_pipeline(
        config=config, reports_dir=rep_dir, manifest_path=manifest_p
    )
    assert summary["overall_status"] == "FAIL"

    train_diag = summary["train"]["diagnostics"]["stockout_diagnostics"]
    assert train_diag["cooccurrence_available"] is False
    assert train_diag["pct_positive_sale_stockout_hours"] is None
    assert train_diag["positive_sale_stockout_hours"] is None
    assert train_diag["sales_volume_during_stockout"] is None

    profile_text = (rep_dir / "data_profile.md").read_text(encoding="utf-8")
    assert (
        "- Positive Sales During Stockout Flags: Diagnostic unavailable."
        in profile_text
    )
    assert "0.00% of stockout hours" not in profile_text

    manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert not any("0.00%" in lim for lim in manifest_data["known_limitations"])
