"""Demand profiling and descriptive series analysis for Aarohan.

Provides reusable analytical calculations for overall demand distribution,
series-level demand metrics, analytical segmentation, entity aggregations,
product-level summaries, and temporal/calendar demand patterns.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import duckdb
import numpy as np
import pandas as pd


def get_weekday_info(iso_date: str) -> dict[str, Any]:
    """Parse an ISO 8601 date string and return calendar metadata.

    Args:
        iso_date: Date string formatted as 'YYYY-MM-DD'.

    Returns:
        Dictionary containing weekday_num (0=Monday..6=Sunday),
        weekday_name ('Monday'..'Sunday'), and is_weekend boolean.
    """
    dt = date.fromisoformat(iso_date)
    dow = dt.weekday()
    names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    return {
        "iso_date": iso_date,
        "weekday_num": dow,
        "weekday_name": names[dow],
        "is_weekend": dow in (5, 6),
    }


def calculate_overall_demand_summary(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
) -> dict[str, Any]:
    """Calculate overall observation-level demand statistics across the partition.

    Computes distributional metrics directly at the row level across all
    observations, rather than averaging daily medians or standard deviations.

    Args:
        con: Configured DuckDB connection.
        parquet_path: Posix-compliant path string to Parquet file.

    Returns:
        Dictionary of observation-level distribution statistics and quantiles.
    """
    query = f"""
        SELECT
            COUNT(*) AS total_observations,
            ROUND(SUM(sale_amount), 4) AS total_demand,
            ROUND(AVG(sale_amount), 4) AS mean_demand,
            ROUND(MEDIAN(sale_amount), 4) AS median_demand,
            ROUND(STDDEV(sale_amount), 4) AS std_demand,
            ROUND(MIN(sale_amount), 4) AS min_demand,
            ROUND(MAX(sale_amount), 4) AS max_demand,
            ROUND(QUANTILE_CONT(sale_amount, 0.25), 4) AS p25_demand,
            ROUND(QUANTILE_CONT(sale_amount, 0.50), 4) AS p50_demand,
            ROUND(QUANTILE_CONT(sale_amount, 0.75), 4) AS p75_demand,
            ROUND(QUANTILE_CONT(sale_amount, 0.90), 4) AS p90_demand,
            ROUND(QUANTILE_CONT(sale_amount, 0.95), 4) AS p95_demand,
            ROUND(QUANTILE_CONT(sale_amount, 0.99), 4) AS p99_demand,
            SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                AS zero_sales_count,
            ROUND(
                SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*),
                4
            ) AS zero_sales_pct
        FROM parquet_scan('{parquet_path}')
    """
    row = con.execute(query).fetchone()
    if row is None:
        raise ValueError("Failed to retrieve overall demand summary")

    return {
        "total_observations": int(row[0]),
        "total_demand": float(row[1]),
        "mean_demand": float(row[2]),
        "median_demand": float(row[3]),
        "std_demand": float(row[4]),
        "min_demand": float(row[5]),
        "max_demand": float(row[6]),
        "p25_demand": float(row[7]),
        "p50_demand": float(row[8]),
        "p75_demand": float(row[9]),
        "p90_demand": float(row[10]),
        "p95_demand": float(row[11]),
        "p99_demand": float(row[12]),
        "zero_sales_count": int(row[13]),
        "zero_sales_pct": float(row[14]),
    }


def calculate_daily_demand_summary(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
) -> pd.DataFrame:
    """Calculate daily aggregate and distributional demand statistics.

    Args:
        con: Configured DuckDB connection.
        parquet_path: Posix-compliant path string to Parquet file.

    Returns:
        DataFrame sorted by date with daily aggregations and quantiles.
    """
    query = f"""
        SELECT
            dt,
            isodow(CAST(dt AS DATE)) - 1 AS weekday_num,
            strftime(CAST(dt AS DATE), '%A') AS weekday_name,
            CASE WHEN isodow(CAST(dt AS DATE)) IN (6, 7) THEN 1 ELSE 0 END
                AS is_weekend,
            COUNT(*) AS observation_count,
            ROUND(SUM(sale_amount), 4) AS total_demand,
            ROUND(AVG(sale_amount), 4) AS mean_demand,
            ROUND(MEDIAN(sale_amount), 4) AS median_demand,
            ROUND(STDDEV(sale_amount), 4) AS std_demand,
            ROUND(MIN(sale_amount), 4) AS min_demand,
            ROUND(MAX(sale_amount), 4) AS max_demand,
            ROUND(QUANTILE_CONT(sale_amount, 0.25), 4) AS p25_demand,
            ROUND(QUANTILE_CONT(sale_amount, 0.75), 4) AS p75_demand,
            ROUND(QUANTILE_CONT(sale_amount, 0.90), 4) AS p90_demand,
            ROUND(QUANTILE_CONT(sale_amount, 0.95), 4) AS p95_demand,
            ROUND(QUANTILE_CONT(sale_amount, 0.99), 4) AS p99_demand,
            SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                AS zero_sales_count,
            ROUND(
                SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*),
                4
            ) AS zero_sales_pct,
            SUM(CASE WHEN stock_hour6_22_cnt = 0 THEN 1 ELSE 0 END)
                AS daytime_stockout_free_count,
            SUM(
                CASE
                    WHEN stock_hour6_22_cnt > 0 AND stock_hour6_22_cnt < 16
                    THEN 1 ELSE 0
                END
            ) AS partially_stockout_count,
            SUM(CASE WHEN stock_hour6_22_cnt = 16 THEN 1 ELSE 0 END)
                AS fully_daytime_stockout_count,
            ROUND(AVG(stock_hour6_22_cnt / 16.0) * 100.0, 4)
                AS mean_daytime_stockout_pct
        FROM parquet_scan('{parquet_path}')
        GROUP BY dt
        ORDER BY dt ASC
    """
    return con.execute(query).df()


def calculate_series_profiles(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
) -> pd.DataFrame:
    """Calculate store-product series summary statistics across observations.

    Args:
        con: Configured DuckDB connection.
        parquet_path: Posix-compliant path string to Parquet file.

    Returns:
        DataFrame with 1 row per series containing demand, volatility,
        intermittency, and stockout metrics.
    """
    query = f"""
        SELECT
            store_id,
            product_id,
            COUNT(*) AS day_count,
            ROUND(AVG(sale_amount), 4) AS mean_demand,
            ROUND(MEDIAN(sale_amount), 4) AS median_demand,
            ROUND(STDDEV(sale_amount), 4) AS std_demand,
            ROUND(SUM(sale_amount), 4) AS total_demand,
            SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                AS zero_sales_count,
            ROUND(
                SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*),
                4
            ) AS zero_sales_pct,
            SUM(CASE WHEN sale_amount > 0.0 THEN 1 ELSE 0 END)
                AS nonzero_count,
            ROUND(AVG(stock_hour6_22_cnt), 4)
                AS mean_daytime_stockout_hours,
            ROUND(AVG(stock_hour6_22_cnt / 16.0) * 100.0, 4)
                AS mean_daytime_stockout_pct
        FROM parquet_scan('{parquet_path}')
        GROUP BY store_id, product_id
        ORDER BY store_id ASC, product_id ASC
    """
    df = con.execute(query).df()

    # Coefficient of Variation: std / mean.
    # When mean == 0, CV is mathematically undefined (NaN), not 0.0.
    cv_series = np.where(
        (df["mean_demand"] > 0) & (df["std_demand"].notna()),
        df["std_demand"] / df["mean_demand"],
        np.nan,
    )
    df["cv"] = np.round(cv_series, 4)
    return df


def calculate_series_distribution_summary(
    series_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate summary distribution metrics across store-product series.

    Computes distribution moments (count, mean, std, min, quantiles, max)
    for demand, dispersion, CV, and stockout exposure across series.
    Undefined CV values (NaN) are explicitly excluded from CV moments.

    Args:
        series_df: DataFrame of series profiles.

    Returns:
        DataFrame summarizing series-level distributional metrics.
    """
    cols = [
        "mean_demand",
        "median_demand",
        "std_demand",
        "cv",
        "zero_sales_pct",
        "mean_daytime_stockout_pct",
        "day_count",
    ]
    records = []
    for col in cols:
        if col not in series_df.columns:
            continue
        valid_s = series_df[col].dropna()
        records.append(
            {
                "metric": col,
                "valid_series_count": int(len(valid_s)),
                "mean": round(float(valid_s.mean()), 4),
                "std": round(float(valid_s.std()), 4),
                "min": round(float(valid_s.min()), 4),
                "p10": round(float(valid_s.quantile(0.10)), 4),
                "p25": round(float(valid_s.quantile(0.25)), 4),
                "median": round(float(valid_s.median()), 4),
                "p75": round(float(valid_s.quantile(0.75)), 4),
                "p90": round(float(valid_s.quantile(0.90)), 4),
                "p99": round(float(valid_s.quantile(0.99)), 4),
                "max": round(float(valid_s.max()), 4),
            }
        )
    return pd.DataFrame(records)


def calculate_product_summary(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
) -> pd.DataFrame:
    """Calculate product-level demand and stockout metrics across all products.

    Args:
        con: Configured DuckDB connection.
        parquet_path: Posix-compliant path string to Parquet file.

    Returns:
        DataFrame sorted by total demand descending with product statistics.
    """
    query = f"""
        SELECT
            product_id,
            COUNT(DISTINCT store_id) AS store_count,
            COUNT(*) AS observation_count,
            ROUND(SUM(sale_amount), 4) AS total_demand,
            ROUND(AVG(sale_amount), 4) AS mean_demand,
            ROUND(MEDIAN(sale_amount), 4) AS median_demand,
            ROUND(STDDEV(sale_amount), 4) AS std_demand,
            ROUND(
                SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*),
                4
            ) AS zero_sales_pct,
            ROUND(AVG(stock_hour6_22_cnt / 16.0) * 100.0, 4)
                AS mean_daytime_stockout_pct
        FROM parquet_scan('{parquet_path}')
        GROUP BY product_id
        ORDER BY total_demand DESC
    """
    return con.execute(query).df()


def define_demand_segments(
    series_df: pd.DataFrame,
    demand_quantiles: tuple[float, float] = (1 / 3, 2 / 3),
    intermittency_threshold: float = 20.0,
    stockout_quantiles: tuple[float, float] = (1 / 3, 2 / 3),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Define empirical segments for series demand, intermittency, and stockouts.

    Args:
        series_df: Series-level profiles DataFrame.
        demand_quantiles: Quantile cutoffs for low/medium/high demand.
        intermittency_threshold: Zero-sales percentage cutoff for intermittency.
        stockout_quantiles: Quantile cutoffs for stockout exposure tiers.

    Returns:
        Tuple of:
          - Enriched series_df with segment assignment columns.
          - Summary thresholds DataFrame documenting boundaries and distributions.
    """
    df = series_df.copy()

    # Demand terciles
    d_q1 = float(df["mean_demand"].quantile(demand_quantiles[0]))
    d_q2 = float(df["mean_demand"].quantile(demand_quantiles[1]))

    demand_conditions = [
        df["mean_demand"] <= d_q1,
        (df["mean_demand"] > d_q1) & (df["mean_demand"] <= d_q2),
        df["mean_demand"] > d_q2,
    ]
    demand_labels = ["Low", "Medium", "High"]
    df["demand_segment"] = np.select(demand_conditions, demand_labels, default="Medium")

    # Intermittency definition (descriptive threshold based on zero-sales %)
    df["intermittency_segment"] = np.where(
        df["zero_sales_pct"] >= intermittency_threshold,
        "Intermittent",
        "Regular",
    )

    # Stockout exposure terciles
    s_q1 = float(df["mean_daytime_stockout_pct"].quantile(stockout_quantiles[0]))
    s_q2 = float(df["mean_daytime_stockout_pct"].quantile(stockout_quantiles[1]))

    stockout_conditions = [
        df["mean_daytime_stockout_pct"] <= s_q1,
        (df["mean_daytime_stockout_pct"] > s_q1)
        & (df["mean_daytime_stockout_pct"] <= s_q2),
        df["mean_daytime_stockout_pct"] > s_q2,
    ]
    stockout_labels = ["Low", "Medium", "High"]
    df["stockout_segment"] = np.select(
        stockout_conditions, stockout_labels, default="Medium"
    )

    # Compile threshold documentation
    total_series = len(df)
    records = []

    # Demand tiers
    for label, lower, upper in [
        ("Low", 0.0, d_q1),
        ("Medium", d_q1, d_q2),
        ("High", d_q2, float(df["mean_demand"].max())),
    ]:
        sub = df[df["demand_segment"] == label]
        records.append(
            {
                "segment_dimension": "Demand Volume",
                "segment_name": label,
                "metric": "mean_demand",
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "series_count": len(sub),
                "series_pct": round(len(sub) * 100.0 / total_series, 2),
                "mean_demand": round(float(sub["mean_demand"].mean()), 4),
                "mean_daytime_stockout_pct": round(
                    float(sub["mean_daytime_stockout_pct"].mean()), 2
                ),
            }
        )

    # Intermittency tiers
    for label, lower, upper in [
        ("Regular", 0.0, intermittency_threshold),
        ("Intermittent", intermittency_threshold, 100.0),
    ]:
        sub = df[df["intermittency_segment"] == label]
        records.append(
            {
                "segment_dimension": "Intermittency",
                "segment_name": label,
                "metric": "zero_sales_pct",
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "series_count": len(sub),
                "series_pct": round(len(sub) * 100.0 / total_series, 2),
                "mean_demand": round(float(sub["mean_demand"].mean()), 4),
                "mean_daytime_stockout_pct": round(
                    float(sub["mean_daytime_stockout_pct"].mean()), 2
                ),
            }
        )

    # Stockout exposure tiers
    for label, lower, upper in [
        ("Low", 0.0, s_q1),
        ("Medium", s_q1, s_q2),
        ("High", s_q2, float(df["mean_daytime_stockout_pct"].max())),
    ]:
        sub = df[df["stockout_segment"] == label]
        records.append(
            {
                "segment_dimension": "Stockout Exposure",
                "segment_name": label,
                "metric": "mean_daytime_stockout_pct",
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "series_count": len(sub),
                "series_pct": round(len(sub) * 100.0 / total_series, 2),
                "mean_demand": round(float(sub["mean_demand"].mean()), 4),
                "mean_daytime_stockout_pct": round(
                    float(sub["mean_daytime_stockout_pct"].mean()), 2
                ),
            }
        )

    thresholds_df = pd.DataFrame(records)
    return df, thresholds_df


def calculate_entity_aggregations(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
    entity_col: str,
) -> pd.DataFrame:
    """Calculate demand and stockout metrics grouped by a specified entity column.

    Args:
        con: Configured DuckDB connection.
        parquet_path: Posix-compliant path string to Parquet file.
        entity_col: Column name to aggregate by (e.g. 'first_category_id', 'store_id').

    Returns:
        Aggregated DataFrame sorted by total demand descending.
    """
    valid_cols = {
        "store_id",
        "product_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "management_group_id",
        "city_id",
    }
    if entity_col not in valid_cols:
        raise ValueError(
            f"Unsupported entity column '{entity_col}'. Must be one of {valid_cols}"
        )

    query = f"""
        SELECT
            {entity_col},
            COUNT(DISTINCT store_id || '_' || product_id) AS series_count,
            COUNT(*) AS observation_count,
            ROUND(SUM(sale_amount), 4) AS total_demand,
            ROUND(AVG(sale_amount), 4) AS mean_demand,
            ROUND(MEDIAN(sale_amount), 4) AS median_demand,
            ROUND(STDDEV(sale_amount), 4) AS std_demand,
            ROUND(
                SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*),
                4
            ) AS zero_sales_pct,
            ROUND(AVG(stock_hour6_22_cnt / 16.0) * 100.0, 4)
                AS mean_daytime_stockout_pct
        FROM parquet_scan('{parquet_path}')
        GROUP BY {entity_col}
        ORDER BY total_demand DESC
    """
    return con.execute(query).df()


def calculate_weekday_patterns(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
) -> pd.DataFrame:
    """Calculate weekday demand and stockout metrics across Monday to Sunday.

    Args:
        con: Configured DuckDB connection.
        parquet_path: Posix-compliant path string to Parquet file.

    Returns:
        DataFrame with 7 rows sorted from Monday (0) to Sunday (6).
    """
    query = f"""
        SELECT
            isodow(CAST(dt AS DATE)) - 1 AS weekday_num,
            strftime(CAST(dt AS DATE), '%A') AS weekday_name,
            CASE WHEN isodow(CAST(dt AS DATE)) IN (6, 7) THEN 1 ELSE 0 END
                AS is_weekend,
            COUNT(DISTINCT dt) AS date_count,
            COUNT(*) AS observation_count,
            ROUND(SUM(sale_amount), 4) AS total_demand,
            ROUND(AVG(sale_amount), 4) AS mean_demand,
            ROUND(MEDIAN(sale_amount), 4) AS median_demand,
            ROUND(STDDEV(sale_amount), 4) AS std_demand,
            ROUND(
                SUM(CASE WHEN sale_amount = 0.0 THEN 1 ELSE 0 END)
                * 100.0 / COUNT(*),
                4
            ) AS zero_sales_pct,
            ROUND(AVG(stock_hour6_22_cnt / 16.0) * 100.0, 4)
                AS mean_daytime_stockout_pct
        FROM parquet_scan('{parquet_path}')
        GROUP BY 1, 2, 3
        ORDER BY weekday_num ASC
    """
    return con.execute(query).df()


# Alias for calendar patterns
calculate_calendar_patterns = calculate_weekday_patterns
