"""Focused unit tests for exploratory data analysis functions.

Tests calendar calculations, zero-sales diagnostics, availability fractions,
demand segment boundaries, and zero-denominator CV edge cases using synthetic
fixtures only.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aarohan.analysis.diagnostics import (
    calculate_hourly_concordance,
    calculate_stockout_severity_summary,
    calculate_zero_sales_diagnostics,
)
from aarohan.analysis.profiling import (
    calculate_series_profiles,
    define_demand_segments,
    get_weekday_info,
)


def test_get_weekday_info() -> None:
    """Verify correct calendar day mapping, names, and weekend detection."""
    # 2024-03-28 is Thursday
    thurs = get_weekday_info("2024-03-28")
    assert thurs["weekday_num"] == 3
    assert thurs["weekday_name"] == "Thursday"
    assert thurs["is_weekend"] is False

    # 2024-03-30 is Saturday
    sat = get_weekday_info("2024-03-30")
    assert sat["weekday_num"] == 5
    assert sat["weekday_name"] == "Saturday"
    assert sat["is_weekend"] is True

    # 2024-03-31 is Sunday
    sun = get_weekday_info("2024-03-31")
    assert sun["weekday_num"] == 6
    assert sun["weekday_name"] == "Sunday"
    assert sun["is_weekend"] is True

    # 2024-04-01 is Monday
    mon = get_weekday_info("2024-04-01")
    assert mon["weekday_num"] == 0
    assert mon["weekday_name"] == "Monday"
    assert mon["is_weekend"] is False


def test_zero_sales_frequency_and_decomposition() -> None:
    """Verify zero-sales frequency calculation and stockout decomposition."""
    records = [
        # In-stock positive sale
        {"sale_amount": 1.5, "stock_hour6_22_cnt": 0},
        # In-stock zero sale
        {"sale_amount": 0.0, "stock_hour6_22_cnt": 0},
        # Partial stockout positive sale
        {"sale_amount": 0.8, "stock_hour6_22_cnt": 4},
        # Partial stockout zero sale
        {"sale_amount": 0.0, "stock_hour6_22_cnt": 6},
        # Full daytime stockout zero sale
        {"sale_amount": 0.0, "stock_hour6_22_cnt": 16},
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        pq_path = Path(tmp_dir) / "test_zeros.parquet"
        pd.DataFrame(records).to_parquet(pq_path)

        con = duckdb.connect(database=":memory:")
        try:
            diag = calculate_zero_sales_diagnostics(con, pq_path.as_posix())
            assert diag["total_observations"] == 5
            assert diag["total_zero_sales"] == 3
            assert diag["total_zero_sales_pct"] == 60.0
            assert diag["in_stock_zero_sales"] == 1
            assert diag["in_stock_zero_sales_pct_of_zeros"] == round(1 / 3 * 100, 2)
            assert diag["stockout_zero_sales"] == 2
            assert diag["fully_daytime_stockout_zeros"] == 1
        finally:
            con.close()


def test_daytime_and_24h_stockout_availability_fractions() -> None:
    """Verify daytime and 24-hour stockout availability fractions and categories."""
    # Daytime availability: 1 - stock_hour6_22_cnt / 16
    # 24h availability: 1 - sum(hours_stock_status) / 24
    row_free = {
        "sale_amount": 2.0,
        "stock_hour6_22_cnt": 0,
        "hours_stock_status": [0] * 24,
    }
    row_partial = {
        "sale_amount": 1.0,
        "stock_hour6_22_cnt": 4,  # daytime avail = 12/16 = 0.75
        "hours_stock_status": [1] * 4 + [0] * 20,  # 24h avail = 20/24 = 0.8333
    }
    row_full_day = {
        "sale_amount": 0.0,
        "stock_hour6_22_cnt": 16,  # daytime avail = 0/16 = 0.0
        "hours_stock_status": [0] * 4 + [1] * 16 + [0] * 4,  # 24h avail = 8/24
    }
    row_full_24h = {
        "sale_amount": 0.0,
        "stock_hour6_22_cnt": 16,
        "hours_stock_status": [1] * 24,  # 24h avail = 0/24 = 0.0
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        pq_path = Path(tmp_dir) / "test_avail.parquet"
        schema = pa.schema(
            [
                ("sale_amount", pa.float64()),
                ("stock_hour6_22_cnt", pa.int32()),
                ("hours_stock_status", pa.list_(pa.int64())),
            ]
        )
        table = pa.Table.from_pylist(
            [row_free, row_partial, row_full_day, row_full_24h], schema=schema
        )
        pq.write_table(table, pq_path)

        con = duckdb.connect(database=":memory:")
        try:
            summary = calculate_stockout_severity_summary(con, pq_path.as_posix())
            # Daytime stockout-free
            free_row = summary[
                summary["stockout_category"] == "Daytime Stockout-Free"
            ].iloc[0]
            assert free_row["observation_count"] == 1
            assert free_row["mean_daytime_stockout_hours"] == 0.0

            # Fully daytime stockout
            full_row = summary[
                summary["stockout_category"] == "Fully Daytime Stockout"
            ].iloc[0]
            assert full_row["observation_count"] == 2
            assert full_row["mean_daytime_stockout_hours"] == 16.0

            # Fully 24h stockout
            row_24 = summary[
                summary["stockout_category"] == "Fully 24-Hour Stockout"
            ].iloc[0]
            assert row_24["observation_count"] == 1
            assert row_24["mean_24h_stockout_hours"] == 24.0
        finally:
            con.close()


def test_demand_segment_boundaries_and_monotonicity() -> None:
    """Verify segment assignment, monotonic boundaries, and count consistency."""
    series_data = [
        {
            "store_id": 1,
            "product_id": i,
            "mean_demand": demand,
            "zero_sales_pct": zeros,
            "mean_daytime_stockout_pct": stk,
        }
        for i, (demand, zeros, stk) in enumerate(
            [
                (0.1, 5.0, 10.0),
                (0.3, 10.0, 12.0),
                (0.6, 15.0, 18.0),
                (0.9, 22.0, 20.0),
                (1.5, 30.0, 25.0),
                (3.0, 0.0, 35.0),
            ]
        )
    ]
    df = pd.DataFrame(series_data)

    enriched_df, thresholds_df = define_demand_segments(
        df,
        demand_quantiles=(1 / 3, 2 / 3),
        intermittency_threshold=20.0,
        stockout_quantiles=(1 / 3, 2 / 3),
    )

    # All series assigned
    assert len(enriched_df) == 6
    assert set(enriched_df["demand_segment"].unique()).issubset(
        {"Low", "Medium", "High"}
    )
    assert set(enriched_df["intermittency_segment"].unique()).issubset(
        {"Regular", "Intermittent"}
    )
    assert set(enriched_df["stockout_segment"].unique()).issubset(
        {"Low", "Medium", "High"}
    )

    # Threshold checks
    d_rows = thresholds_df[thresholds_df["segment_dimension"] == "Demand Volume"]
    assert len(d_rows) == 3
    assert int(d_rows["series_count"].sum()) == 6

    # Monotonicity of demand bounds
    bounds = d_rows["upper_bound"].tolist()
    assert bounds[0] <= bounds[1] <= bounds[2]

    # Intermittent count check (series with zeros >= 20.0%: 22.0 and 30.0 -> exactly 2)
    inter_rows = thresholds_df[thresholds_df["segment_dimension"] == "Intermittency"]
    inter_cnt = int(
        inter_rows[inter_rows["segment_name"] == "Intermittent"]["series_count"].iloc[0]
    )
    assert inter_cnt == 2


def test_cv_zero_denominator_handling() -> None:
    """Verify coefficient of variation handling for zero mean demand series."""
    # Series with all 0.0 demand
    records = [
        {"store_id": 1, "product_id": 100, "sale_amount": 0.0, "stock_hour6_22_cnt": 0},
        {"store_id": 1, "product_id": 100, "sale_amount": 0.0, "stock_hour6_22_cnt": 0},
        {"store_id": 1, "product_id": 200, "sale_amount": 1.0, "stock_hour6_22_cnt": 0},
        {"store_id": 1, "product_id": 200, "sale_amount": 3.0, "stock_hour6_22_cnt": 0},
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        pq_path = Path(tmp_dir) / "test_cv.parquet"
        pd.DataFrame(records).to_parquet(pq_path)

        con = duckdb.connect(database=":memory:")
        try:
            df = calculate_series_profiles(con, pq_path.as_posix())
            # Product 100 has mean = 0.0, cv should be 0.0 safely
            p100 = df[df["product_id"] == 100].iloc[0]
            assert p100["mean_demand"] == 0.0
            assert p100["cv"] == 0.0
            assert not np.isnan(p100["cv"])
            assert not np.isinf(p100["cv"])

            # Product 200 has mean = 2.0, std = sqrt(2) ≈ 1.4142, cv ≈ 0.7071
            p200 = df[df["product_id"] == 200].iloc[0]
            assert p200["mean_demand"] == 2.0
            assert p200["cv"] > 0.0
            assert not np.isnan(p200["cv"])
        finally:
            con.close()


def test_hourly_concordance_synthetic() -> None:
    """Verify hourly concordance logic on synthetic hourly array data."""
    # Row 1: in stock, positive sales
    r1 = {"hours_sale": [1.0] * 24, "hours_stock_status": [0] * 24}
    # Row 2: stockout, positive sales in 2 hours, zero sales in 22 hours
    r2 = {
        "hours_sale": [0.5, 0.5] + [0.0] * 22,
        "hours_stock_status": [1] * 24,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        pq_path = Path(tmp_dir) / "test_concordance.parquet"
        schema = pa.schema(
            [
                ("hours_sale", pa.list_(pa.float64())),
                ("hours_stock_status", pa.list_(pa.int64())),
            ]
        )
        table = pa.Table.from_pylist([r1, r2], schema=schema)
        pq.write_table(table, pq_path)

        concordance = calculate_hourly_concordance(pq_path, batch_size=2)
        assert concordance["total_hourly_slots"] == 48
        assert concordance["total_stockout_hours"] == 24
        assert concordance["total_instock_hours"] == 24
        assert concordance["positive_sales_during_stockout_hours"] == 2
        assert concordance["zero_sales_during_stockout_hours"] == 22
        assert concordance["positive_sales_during_stockout_pct"] == round(
            2 / 24 * 100, 2
        )
        assert concordance["positive_sales_during_instock_hours"] == 24
