"""Unit tests for Aarohan data ingestion, validation, and reporting pipeline.

Tests use synthetic Parquet fixtures constructed via PyArrow in temporary directories (tmp_path).
Covers all integrity, schema, continuity, hourly sequences, anomalies, and cohort checks.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aarohan.data.dataset import ValidationStatus
from aarohan.data.load import (
    check_file_exists,
    create_duckdb_connection,
    read_parquet_metadata,
)
from aarohan.data.validate import (
    calculate_evaluation_cohort,
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
    """Test cross-partition consistency checks on disjoint series or overlapping dates."""
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

    checks, meta = validate_file_level(
        p,
        expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
    )
    chk_check = next(c for c in checks if c.name == "sha256_checksum")
    assert chk_check.status == ValidationStatus.FAIL.value
