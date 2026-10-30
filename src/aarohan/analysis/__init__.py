"""Exploratory data analysis and diagnostics package for Aarohan.

Provides reusable profiling, diagnostics, and visualization tools for
analyzing demand patterns, series heterogeneity, stockouts, and data
distributions across FreshRetailNet-50K.
"""

from __future__ import annotations

from aarohan.analysis.diagnostics import (
    calculate_hourly_concordance,
    calculate_stockout_severity_summary,
    calculate_zero_sales_diagnostics,
    compare_train_eval_populations,
)
from aarohan.analysis.profiling import (
    calculate_calendar_patterns,
    calculate_daily_demand_summary,
    calculate_entity_aggregations,
    calculate_series_profiles,
    calculate_weekday_patterns,
    define_demand_segments,
    get_weekday_info,
)
from aarohan.analysis.visualize import (
    generate_category_heterogeneity_chart,
    generate_daily_demand_trend_chart,
    generate_demand_segments_profile_chart,
    generate_sales_by_stockout_category_chart,
    generate_series_distribution_chart,
    generate_stockout_exposure_distribution_chart,
    generate_weekday_pattern_chart,
    generate_zero_sales_distribution_chart,
)

__all__ = [
    "calculate_calendar_patterns",
    "calculate_daily_demand_summary",
    "calculate_entity_aggregations",
    "calculate_hourly_concordance",
    "calculate_series_profiles",
    "calculate_stockout_severity_summary",
    "calculate_weekday_patterns",
    "calculate_zero_sales_diagnostics",
    "compare_train_eval_populations",
    "define_demand_segments",
    "generate_category_heterogeneity_chart",
    "generate_daily_demand_trend_chart",
    "generate_demand_segments_profile_chart",
    "generate_sales_by_stockout_category_chart",
    "generate_series_distribution_chart",
    "generate_stockout_exposure_distribution_chart",
    "generate_weekday_pattern_chart",
    "generate_zero_sales_distribution_chart",
    "get_weekday_info",
]
