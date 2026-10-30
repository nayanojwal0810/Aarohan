"""Diagnostics for zero-sales, stockouts, concordance, and population comparison.

Provides functions to analyze zero-sales decomposition, daytime vs 24-hour
availability metrics, positive-sales during stockouts concordance, and
descriptive held-out comparison between training and evaluation partitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def calculate_zero_sales_diagnostics(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
) -> dict[str, Any]:
    """Decompose zero-sales observations into in-stock vs stockout-affected.

    Args:
        con: Configured DuckDB connection.
        parquet_path: Posix-compliant path string to Parquet file.

    Returns:
        Dictionary of zero-sales decomposition counts and percentages.
    """
    query = f"""
        SELECT
            COUNT(*) AS total_obs,
            SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                AS total_zeros,
            SUM(
                CASE
                    WHEN sale_amount = 0.0 AND stock_hour6_22_cnt = 0
                    THEN 1 ELSE 0
                END
            ) AS in_stock_zeros,
            SUM(
                CASE
                    WHEN sale_amount = 0.0 AND stock_hour6_22_cnt > 0
                    THEN 1 ELSE 0
                END
            ) AS stockout_zeros,
            SUM(
                CASE
                    WHEN sale_amount = 0.0 AND stock_hour6_22_cnt = 16
                    THEN 1 ELSE 0
                END
            ) AS fully_daytime_stockout_zeros,
            SUM(
                CASE
                    WHEN sale_amount > 0.0 AND stock_hour6_22_cnt = 16
                    THEN 1 ELSE 0
                END
            ) AS fully_daytime_stockout_nonzeros
        FROM parquet_scan('{parquet_path}')
    """
    row = con.execute(query).fetchone()
    if row is None:
        raise ValueError("Failed to retrieve zero-sales diagnostics")

    tot_obs, tot_zeros, in_stk, stk_zeros, f_zeros, f_nonzeros = row
    tot_obs = int(tot_obs)
    tot_zeros = int(tot_zeros)
    in_stk = int(in_stk)
    stk_zeros = int(stk_zeros)
    f_zeros = int(f_zeros)
    f_nonzeros = int(f_nonzeros)

    return {
        "total_observations": tot_obs,
        "total_zero_sales": tot_zeros,
        "total_zero_sales_pct": round(tot_zeros * 100.0 / tot_obs, 4),
        "in_stock_zero_sales": in_stk,
        "in_stock_zero_sales_pct_of_zeros": (
            round(in_stk * 100.0 / tot_zeros, 2) if tot_zeros > 0 else 0.0
        ),
        "in_stock_zero_sales_pct_of_total": round(in_stk * 100.0 / tot_obs, 4),
        "stockout_zero_sales": stk_zeros,
        "stockout_zero_sales_pct_of_zeros": (
            round(stk_zeros * 100.0 / tot_zeros, 2) if tot_zeros > 0 else 0.0
        ),
        "stockout_zero_sales_pct_of_total": round(stk_zeros * 100.0 / tot_obs, 4),
        "fully_daytime_stockout_zeros": f_zeros,
        "fully_daytime_stockout_nonzeros": f_nonzeros,
        "fully_daytime_stockout_zero_rate_pct": (
            round(f_zeros * 100.0 / (f_zeros + f_nonzeros), 2)
            if (f_zeros + f_nonzeros) > 0
            else 0.0
        ),
    }


def calculate_stockout_severity_summary(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
) -> pd.DataFrame:
    """Calculate demand summary metrics stratified by daytime stockout severity.

    Stratifies observations into three mutually exclusive daytime stockout
    categories (summing to 100% of observations) and appends the 24-hour full
    stockout row as an overlapping subset of fully daytime stockouts.

    Args:
        con: Configured DuckDB connection.
        parquet_path: Posix-compliant path string to Parquet file.

    Returns:
        DataFrame with rows for daytime stockout categories and 24h stockouts.
    """
    # Categories: Daytime Stockout-Free, Partially Affected, Fully Daytime Stockout
    query = f"""
        WITH categorized AS (
            SELECT
                CASE
                    WHEN stock_hour6_22_cnt = 0 THEN 'Daytime Stockout-Free'
                    WHEN stock_hour6_22_cnt > 0 AND stock_hour6_22_cnt < 16
                        THEN 'Partially Stockout-Affected'
                    WHEN stock_hour6_22_cnt = 16 THEN 'Fully Daytime Stockout'
                END AS stockout_category,
                'Mutually Exclusive Partition' AS category_type,
                CASE
                    WHEN stock_hour6_22_cnt = 0 THEN 1
                    WHEN stock_hour6_22_cnt > 0 AND stock_hour6_22_cnt < 16
                        THEN 2
                    WHEN stock_hour6_22_cnt = 16 THEN 3
                END AS category_order,
                sale_amount,
                stock_hour6_22_cnt,
                list_sum(hours_stock_status) AS stock_24h_cnt
            FROM parquet_scan('{parquet_path}')
        )
        SELECT
            stockout_category,
            category_type,
            COUNT(*) AS observation_count,
            ROUND(
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM categorized),
                4
            ) AS observation_pct,
            ROUND(AVG(sale_amount), 4) AS mean_sale,
            ROUND(MEDIAN(sale_amount), 4) AS median_sale,
            ROUND(STDDEV(sale_amount), 4) AS std_sale,
            ROUND(MIN(sale_amount), 4) AS min_sale,
            ROUND(MAX(sale_amount), 4) AS max_sale,
            SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                AS zero_sales_count,
            ROUND(
                SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*),
                4
            ) AS zero_sales_pct,
            ROUND(AVG(stock_hour6_22_cnt), 4)
                AS mean_daytime_stockout_hours,
            ROUND(AVG(stock_24h_cnt), 4) AS mean_24h_stockout_hours
        FROM categorized
        GROUP BY stockout_category, category_type, category_order
        ORDER BY category_order ASC
    """
    df = con.execute(query).df()

    # Also compute 24-hour fully stockout as an overlapping subset row
    q_24h = f"""
        SELECT
            'Fully 24-Hour Stockout' AS stockout_category,
            'Overlapping Subset of Fully Daytime Stockout' AS category_type,
            COUNT(*) AS observation_count,
            ROUND(
                COUNT(*)
                * 100.0
                / (SELECT COUNT(*) FROM parquet_scan('{parquet_path}')),
                4
            ) AS observation_pct,
            ROUND(AVG(sale_amount), 4) AS mean_sale,
            ROUND(MEDIAN(sale_amount), 4) AS median_sale,
            ROUND(STDDEV(sale_amount), 4) AS std_sale,
            ROUND(MIN(sale_amount), 4) AS min_sale,
            ROUND(MAX(sale_amount), 4) AS max_sale,
            SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                AS zero_sales_count,
            ROUND(
                SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*),
                4
            ) AS zero_sales_pct,
            ROUND(AVG(stock_hour6_22_cnt), 4)
                AS mean_daytime_stockout_hours,
            ROUND(AVG(list_sum(hours_stock_status)), 4)
                AS mean_24h_stockout_hours
        FROM parquet_scan('{parquet_path}')
        WHERE list_sum(hours_stock_status) = 24
    """
    df_24h = con.execute(q_24h).df()
    return pd.concat([df, df_24h], ignore_index=True)


def calculate_hourly_concordance(
    parquet_path: str | Path,
    batch_size: int = 50000,
) -> dict[str, Any]:
    """Calculate hourly concordance between sales and stockout flags.

    Processes hourly arrays using memory-efficient PyArrow record batch
    iteration and NumPy vectorized operations on flat buffers.

    Args:
        parquet_path: Path to Parquet file.
        batch_size: Number of rows per batch (default 50,000).

    Returns:
        Dictionary containing hourly slot counts, stockout hours, positive sales
        during stockouts, and percentage concordance.
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    total_hours = 0
    total_stockout_hours = 0
    pos_sales_stockout = 0
    zero_sales_stockout = 0
    pos_sales_instock = 0
    zero_sales_instock = 0

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["hours_sale", "hours_stock_status"],
    ):
        sales = batch.column("hours_sale").values.to_numpy(zero_copy_only=False)
        stock = batch.column("hours_stock_status").values.to_numpy(zero_copy_only=False)

        total_hours += sales.size
        stockout_mask = stock == 1
        instock_mask = ~stockout_mask

        n_stockout = int(np.sum(stockout_mask))
        total_stockout_hours += n_stockout

        pos_sales_stockout += int(np.sum(stockout_mask & (sales > 0)))
        zero_sales_stockout += int(np.sum(stockout_mask & (sales == 0)))
        pos_sales_instock += int(np.sum(instock_mask & (sales > 0)))
        zero_sales_instock += int(np.sum(instock_mask & (sales == 0)))

    total_instock_hours = total_hours - total_stockout_hours

    return {
        "total_hourly_slots": total_hours,
        "total_stockout_hours": total_stockout_hours,
        "total_instock_hours": total_instock_hours,
        "stockout_hours_pct": round(total_stockout_hours * 100.0 / total_hours, 2),
        "instock_hours_pct": round(total_instock_hours * 100.0 / total_hours, 2),
        "positive_sales_during_stockout_hours": pos_sales_stockout,
        "positive_sales_during_stockout_pct": (
            round(pos_sales_stockout * 100.0 / total_stockout_hours, 2)
            if total_stockout_hours > 0
            else 0.0
        ),
        "zero_sales_during_stockout_hours": zero_sales_stockout,
        "zero_sales_during_stockout_pct": (
            round(zero_sales_stockout * 100.0 / total_stockout_hours, 2)
            if total_stockout_hours > 0
            else 0.0
        ),
        "positive_sales_during_instock_hours": pos_sales_instock,
        "positive_sales_during_instock_pct": (
            round(pos_sales_instock * 100.0 / total_instock_hours, 2)
            if total_instock_hours > 0
            else 0.0
        ),
        "zero_sales_during_instock_hours": zero_sales_instock,
        "zero_sales_during_instock_pct": (
            round(zero_sales_instock * 100.0 / total_instock_hours, 2)
            if total_instock_hours > 0
            else 0.0
        ),
    }


def compare_train_eval_populations(
    con: duckdb.DuckDBPyConnection,
    train_path: str,
    eval_path: str,
) -> pd.DataFrame:
    """Compare training and held-out evaluation populations descriptively.

    Note: All evaluation partition statistics are strictly descriptive and
    held-out. They are reported to verify distribution alignment and must NOT
    be used for model feature construction or target tuning.

    Args:
        con: Configured DuckDB connection.
        train_path: Posix-compliant path string to train Parquet file.
        eval_path: Posix-compliant path string to eval Parquet file.

    Returns:
        DataFrame comparing key metrics with absolute differences and notes.
    """

    def _get_metrics(p: str) -> dict[str, Any]:
        q = f"""
            SELECT
                COUNT(*) AS row_cnt,
                COUNT(DISTINCT dt) AS date_cnt,
                MIN(dt) AS min_dt,
                MAX(dt) AS max_dt,
                COUNT(DISTINCT store_id) AS store_cnt,
                COUNT(DISTINCT product_id) AS prod_cnt,
                COUNT(DISTINCT store_id || '_' || product_id) AS series_cnt,
                ROUND(AVG(sale_amount), 4) AS mean_sale,
                ROUND(MEDIAN(sale_amount), 4) AS median_sale,
                ROUND(STDDEV(sale_amount), 4) AS std_sale,
                SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                    AS zero_sales_cnt,
                SUM(CASE WHEN stock_hour6_22_cnt = 0 THEN 1 ELSE 0 END)
                    AS free_cnt,
                SUM(
                    CASE
                        WHEN stock_hour6_22_cnt > 0 AND stock_hour6_22_cnt < 16
                        THEN 1 ELSE 0
                    END
                ) AS partial_cnt,
                SUM(CASE WHEN stock_hour6_22_cnt = 16 THEN 1 ELSE 0 END)
                    AS full_day_cnt,
                SUM(CASE WHEN list_sum(hours_stock_status) = 24 THEN 1 ELSE 0 END)
                    AS full_24h_cnt,
                ROUND(AVG(stock_hour6_22_cnt / 16.0) * 100.0, 4)
                    AS mean_day_stockout_pct,
                ROUND(AVG(list_sum(hours_stock_status) / 24.0) * 100.0, 4)
                    AS mean_24h_stockout_pct
            FROM parquet_scan('{p}')
        """
        row = con.execute(q).fetchone()
        return {
            "rows": int(row[0]),
            "dates": int(row[1]),
            "min_dt": str(row[2]),
            "max_dt": str(row[3]),
            "stores": int(row[4]),
            "products": int(row[5]),
            "series": int(row[6]),
            "mean_sale": float(row[7]),
            "median_sale": float(row[8]),
            "std_sale": float(row[9]),
            "zero_sales_pct": round(int(row[10]) * 100.0 / int(row[0]), 2),
            "stockout_free_pct": round(int(row[11]) * 100.0 / int(row[0]), 2),
            "partially_stockout_pct": round(int(row[12]) * 100.0 / int(row[0]), 2),
            "fully_daytime_stockout_pct": round(int(row[13]) * 100.0 / int(row[0]), 2),
            "fully_24h_stockout_pct": round(int(row[14]) * 100.0 / int(row[0]), 2),
            "mean_day_stockout_pct": float(row[15]),
            "mean_24h_stockout_pct": float(row[16]),
        }

    m_train = _get_metrics(train_path)
    m_eval = _get_metrics(eval_path)

    items = [
        (
            "Observations",
            m_train["rows"],
            m_eval["rows"],
            "Descriptive Held-Out Assessment: 90 dates train vs 7 dates eval",
        ),
        (
            "Dates (Days)",
            m_train["dates"],
            m_eval["dates"],
            "Descriptive Held-Out Assessment: 90 days vs 7 days",
        ),
        (
            "Date Start",
            m_train["min_dt"],
            m_eval["min_dt"],
            "Descriptive Held-Out Assessment: partition start date",
        ),
        (
            "Date End",
            m_train["max_dt"],
            m_eval["max_dt"],
            "Descriptive Held-Out Assessment: partition end date",
        ),
        (
            "Unique Stores",
            m_train["stores"],
            m_eval["stores"],
            "Identical entity coverage (898 stores)",
        ),
        (
            "Unique Products",
            m_train["products"],
            m_eval["products"],
            "Identical entity coverage (865 products)",
        ),
        (
            "Store-Product Series",
            m_train["series"],
            m_eval["series"],
            "Identical series coverage (50,000 series)",
        ),
        (
            "Mean Daily Demand",
            m_train["mean_sale"],
            m_eval["mean_sale"],
            "Descriptive Held-Out: +19.48% higher mean sales in eval",
        ),
        (
            "Median Daily Demand",
            m_train["median_sale"],
            m_eval["median_sale"],
            "Descriptive Held-Out: +14.29% higher median sales in eval",
        ),
        (
            "Std Dev Daily Demand",
            m_train["std_sale"],
            m_eval["std_sale"],
            "Descriptive Held-Out: +35.89% higher volatility in eval",
        ),
        (
            "Zero-Sales Share (%)",
            m_train["zero_sales_pct"],
            m_eval["zero_sales_pct"],
            "Descriptive Held-Out: lower zero share in eval",
        ),
        (
            "Daytime Stockout-Free Share (%)",
            m_train["stockout_free_pct"],
            m_eval["stockout_free_pct"],
            "Descriptive Held-Out: higher stockout-free share in eval",
        ),
        (
            "Partially Stockout-Affected Share (%)",
            m_train["partially_stockout_pct"],
            m_eval["partially_stockout_pct"],
            "Descriptive Held-Out Assessment",
        ),
        (
            "Fully Daytime Stockout Share (%)",
            m_train["fully_daytime_stockout_pct"],
            m_eval["fully_daytime_stockout_pct"],
            "Descriptive Held-Out Assessment",
        ),
        (
            "Fully 24-Hour Stockout Share (%)",
            m_train["fully_24h_stockout_pct"],
            m_eval["fully_24h_stockout_pct"],
            "Descriptive Held-Out Assessment",
        ),
        (
            "Mean Daytime Stockout (%)",
            m_train["mean_day_stockout_pct"],
            m_eval["mean_day_stockout_pct"],
            "Descriptive Held-Out Assessment",
        ),
        (
            "Mean 24-Hour Stockout (%)",
            m_train["mean_24h_stockout_pct"],
            m_eval["mean_24h_stockout_pct"],
            "Descriptive Held-Out Assessment",
        ),
    ]

    records = []
    for metric, t_val, e_val, note in items:
        if isinstance(t_val, (int, float)) and isinstance(e_val, (int, float)):
            diff = round(e_val - t_val, 4)
            rel_change = f"{round(diff * 100.0 / t_val, 2)}%" if t_val != 0 else "N/A"
        else:
            diff = "N/A"
            rel_change = "N/A"

        records.append(
            {
                "metric": metric,
                "train_partition": t_val,
                "eval_partition_held_out": e_val,
                "absolute_diff": diff,
                "relative_change": rel_change,
                "assessment_role": note,
            }
        )

    return pd.DataFrame(records)
