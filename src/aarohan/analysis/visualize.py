"""Vector SVG visualization generator for Aarohan exploratory data analysis.

Generates self-contained, publication-grade SVG figures without requiring
external plotting dependencies, maintaining strict compliance with the
approved repository requirements.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


def _esc(text: Any) -> str:
    """Escape text for safe inclusion in SVG XML."""
    return html.escape(str(text))


def _svg_start(width: int, height: int) -> str:
    """Return root SVG opening tag and style block."""
    font_stack = (
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Roboto, Helvetica, Arial, sans-serif"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n'
        f"  <style>\n"
        f"    text {{ font-family: {font_stack}; }}\n"
        f"    .title {{ font-size: 16px; font-weight: 600; fill: #0f172a; }}\n"
        f"    .subtitle {{ font-size: 12px; fill: #64748b; }}\n"
        f"  </style>\n"
        f'  <rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>\n'
    )


def _svg_axes(
    margin_left: float,
    margin_top: float,
    plot_width: float,
    plot_height: float,
    width: float,
    include_right: bool = False,
) -> str:
    """Return SVG left, bottom, and optionally right axis lines."""
    y_bottom = margin_top + plot_height
    x_right = width - (width - margin_left - plot_width)
    lines = [
        f'  <line x1="{margin_left:.1f}" y1="{margin_top:.1f}" '
        f'x2="{margin_left:.1f}" y2="{y_bottom:.1f}" '
        f'stroke="#94a3b8" stroke-width="1.5"/>',
        f'  <line x1="{margin_left:.1f}" y1="{y_bottom:.1f}" '
        f'x2="{x_right:.1f}" y2="{y_bottom:.1f}" '
        f'stroke="#94a3b8" stroke-width="1.5"/>',
    ]
    if include_right:
        lines.append(
            f'  <line x1="{x_right:.1f}" y1="{margin_top:.1f}" '
            f'x2="{x_right:.1f}" y2="{y_bottom:.1f}" '
            f'stroke="#94a3b8" stroke-width="1.5"/>'
        )
    return "\n".join(lines) + "\n"


def generate_daily_demand_trend_chart(
    daily_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Daily Aggregate Normalized Demand Trend (FreshRetailNet-50K)",
) -> None:
    """Generate SVG line chart of daily aggregate demand across observed dates."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = daily_df.sort_values("dt").reset_index(drop=True)
    n_points = len(df)
    if n_points == 0:
        return

    width = 900
    height = 500
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 80

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    vals = df["total_demand"].to_numpy()
    min_val = 0.0
    max_val = float(np.max(vals)) * 1.15

    def get_x(idx: int) -> float:
        return margin_left + (idx / max(1, n_points - 1)) * plot_width

    def get_y(v: float) -> float:
        return (
            margin_top
            + plot_height
            - ((v - min_val) / (max_val - min_val)) * plot_height
        )

    # Shading for weekends
    weekend_rects = []
    for i, row in df.iterrows():
        if row.get("is_weekend", 0) == 1:
            w = plot_width / n_points
            x = get_x(i) - w / 2
            weekend_rects.append(
                f'  <rect x="{x:.1f}" y="{margin_top:.1f}" width="{w:.1f}" '
                f'height="{plot_height:.1f}" fill="#f1f5f9" opacity="0.6"/>'
            )

    # Y ticks and grid
    y_ticks = np.linspace(min_val, max_val, 6)
    grid_lines = []
    y_labels = []
    for yt in y_ticks:
        y_pos = get_y(yt)
        grid_lines.append(
            f'  <line x1="{margin_left:.1f}" y1="{y_pos:.1f}" '
            f'x2="{width - margin_right:.1f}" y2="{y_pos:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        y_labels.append(
            f'  <text x="{margin_left - 12:.1f}" y="{y_pos + 4:.1f}" '
            f'font-size="11" fill="#64748b" text-anchor="end">'
            f"{yt / 1000:.1f}k</text>"
        )

    # X ticks
    x_labels = []
    step = max(1, n_points // 8)
    for i in range(0, n_points, step):
        dt_str = df.loc[i, "dt"]
        x_pos = get_x(i)
        y_tick = margin_top + plot_height
        x_labels.append(
            f'  <line x1="{x_pos:.1f}" y1="{y_tick:.1f}" '
            f'x2="{x_pos:.1f}" y2="{y_tick + 5:.1f}" '
            f'stroke="#94a3b8" stroke-width="1"/>\n'
            f'  <text x="{x_pos:.1f}" y="{y_tick + 22:.1f}" font-size="10" '
            f'fill="#64748b" text-anchor="middle" '
            f'transform="rotate(30, {x_pos:.1f}, {y_tick + 22:.1f})">'
            f"{dt_str}</text>"
        )

    points = [f"{get_x(i):.1f},{get_y(v):.1f}" for i, v in enumerate(vals)]
    polyline = " ".join(points)

    sub = (
        "Daily aggregated sum of normalized demand index "
        "(Shaded areas indicate Saturdays and Sundays)"
    )
    y_mid = margin_top + plot_height / 2
    x_mid = margin_left + plot_width / 2

    parts = [
        _svg_start(width, height),
        f'  <text x="{margin_left}" y="32" class="title">{_esc(title)}</text>\n',
        f'  <text x="{margin_left}" y="50" class="subtitle">{_esc(sub)}</text>\n',
        "\n".join(weekend_rects) + "\n",
        "\n".join(grid_lines) + "\n",
        _svg_axes(margin_left, margin_top, plot_width, plot_height, width),
        "\n".join(y_labels) + "\n",
        f'  <text x="{margin_left - 50}" y="{y_mid:.1f}" font-size="12" '
        f'fill="#334155" text-anchor="middle" '
        f'transform="rotate(-90, {margin_left - 50}, {y_mid:.1f})">'
        f"Total Daily Demand Index</text>\n",
        "\n".join(x_labels) + "\n",
        f'  <text x="{x_mid:.1f}" y="{height - 15}" font-size="12" '
        f'fill="#334155" text-anchor="middle">Observation Date (YYYY-MM-DD)</text>\n',
        f'  <polyline fill="none" stroke="#2563eb" stroke-width="2.5" '
        f'stroke-linecap="round" stroke-linejoin="round" points="{polyline}"/>\n',
        "</svg>",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def generate_series_distribution_chart(
    series_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate SVG histogram of series-level mean daily demand."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vals = series_df["mean_demand"].to_numpy()
    vals_clipped = np.clip(vals, 0.0, 5.0)

    bins = np.linspace(0.0, 5.0, 26)
    counts, _ = np.histogram(vals_clipped, bins=bins)

    width = 900
    height = 500
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 70

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_count = float(np.max(counts)) * 1.15

    d_rows = thresholds_df[thresholds_df["segment_dimension"] == "Demand Volume"]
    q1 = float(d_rows[d_rows["segment_name"] == "Low"]["upper_bound"].iloc[0])
    q2 = float(d_rows[d_rows["segment_name"] == "Medium"]["upper_bound"].iloc[0])

    bars = []
    n_bins = len(counts)
    bin_w = plot_width / n_bins

    for i in range(n_bins):
        c = counts[i]
        b_mid = (bins[i] + bins[i + 1]) / 2.0
        bar_h = (c / max_count) * plot_height
        x = margin_left + i * bin_w + 2
        y = margin_top + plot_height - bar_h

        color = "#3b82f6" if b_mid <= q1 else ("#10b981" if b_mid <= q2 else "#f59e0b")
        bars.append(
            f'  <rect x="{x:.1f}" y="{y:.1f}" width="{max(1.0, bin_w - 4):.1f}" '
            f'height="{bar_h:.1f}" fill="{color}" rx="2"/>'
        )

    def val_to_x(v: float) -> float:
        return margin_left + (v / 5.0) * plot_width

    x_q1 = val_to_x(q1)
    x_q2 = val_to_x(q2)
    y_bot = margin_top + plot_height

    thresh_lines = [
        f'  <line x1="{x_q1:.1f}" y1="{margin_top}" x2="{x_q1:.1f}" y2="{y_bot:.1f}" '
        f'stroke="#ef4444" stroke-width="2" stroke-dasharray="4,4"/>',
        f'  <text x="{x_q1 + 5:.1f}" y="{margin_top + 25}" font-size="11" '
        f'fill="#ef4444" font-weight="600">Q1 ({q1:.3f})</text>',
        f'  <line x1="{x_q2:.1f}" y1="{margin_top}" x2="{x_q2:.1f}" y2="{y_bot:.1f}" '
        f'stroke="#8b5cf6" stroke-width="2" stroke-dasharray="4,4"/>',
        f'  <text x="{x_q2 + 5:.1f}" y="{margin_top + 45}" font-size="11" '
        f'fill="#8b5cf6" font-weight="600">Q2 ({q2:.3f})</text>',
    ]

    grid_lines = []
    y_labels = []
    for yt in np.linspace(0, max_count, 6):
        y_pos = margin_top + plot_height - (yt / max_count) * plot_height
        grid_lines.append(
            f'  <line x1="{margin_left:.1f}" y1="{y_pos:.1f}" '
            f'x2="{width - margin_right:.1f}" y2="{y_pos:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        y_labels.append(
            f'  <text x="{margin_left - 12:.1f}" y="{y_pos + 4:.1f}" font-size="11" '
            f'fill="#64748b" text-anchor="end">{int(yt):,}</text>'
        )

    x_labels = []
    for xt in np.linspace(0.0, 5.0, 11):
        x_pos = val_to_x(xt)
        x_labels.append(
            f'  <line x1="{x_pos:.1f}" y1="{y_bot:.1f}" x2="{x_pos:.1f}" '
            f'y2="{y_bot + 5:.1f}" stroke="#94a3b8" stroke-width="1"/>\n'
            f'  <text x="{x_pos:.1f}" y="{y_bot + 20:.1f}" font-size="11" '
            f'fill="#64748b" text-anchor="middle">{xt:.1f}</text>'
        )

    y_mid = margin_top + plot_height / 2
    x_mid = margin_left + plot_width / 2
    sub = (
        "Histogram across 50,000 store-product series "
        "(Values clipped at 5.0 for display; dashed lines denote terciles)"
    )

    legend = (
        f'  <g transform="translate({width - 250}, 20)">\n'
        f'    <rect x="0" y="0" width="12" height="12" fill="#3b82f6" rx="2"/>\n'
        f'    <text x="18" y="10" font-size="11" fill="#334155">'
        f"Low Demand (≤{q1:.2f})</text>\n"
        f'    <rect x="0" y="18" width="12" height="12" fill="#10b981" rx="2"/>\n'
        f'    <text x="18" y="28" font-size="11" fill="#334155">'
        f"Medium ({q1:.2f}–{q2:.2f})</text>\n"
        f'    <rect x="0" y="36" width="12" height="12" fill="#f59e0b" rx="2"/>\n'
        f'    <text x="18" y="46" font-size="11" fill="#334155">'
        f"High Demand (&gt;{q2:.2f})</text>\n"
        f"  </g>\n"
    )

    parts = [
        _svg_start(width, height),
        f'  <text x="{margin_left}" y="32" class="title">'
        f"Distribution of Series-Level Mean Daily Demand</text>\n",
        f'  <text x="{margin_left}" y="50" class="subtitle">{_esc(sub)}</text>\n',
        "\n".join(grid_lines) + "\n",
        "\n".join(bars) + "\n",
        "\n".join(thresh_lines) + "\n",
        _svg_axes(margin_left, margin_top, plot_width, plot_height, width),
        "\n".join(y_labels) + "\n",
        f'  <text x="{margin_left - 55}" y="{y_mid:.1f}" font-size="12" '
        f'fill="#334155" text-anchor="middle" '
        f'transform="rotate(-90, {margin_left - 55}, {y_mid:.1f})">'
        f"Number of Series</text>\n",
        "\n".join(x_labels) + "\n",
        f'  <text x="{x_mid:.1f}" y="{height - 18}" font-size="12" '
        f'fill="#334155" text-anchor="middle">'
        f"Mean Daily Normalized Demand Index</text>\n",
        legend,
        "</svg>",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def generate_zero_sales_distribution_chart(
    series_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate SVG histogram of zero-sales day percentage across series."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vals = series_df["zero_sales_pct"].to_numpy()
    bins = [0.0, 0.001, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]
    labels = [
        "0%",
        "(0%, 5%]",
        "(5%, 10%]",
        "(10%, 20%]",
        "(20%, 30%]",
        "(30%, 50%]",
        "&gt; 50%",
    ]

    counts = [int(np.sum(vals == 0.0))]
    for i in range(1, len(bins) - 1):
        counts.append(int(np.sum((vals > bins[i]) & (vals <= bins[i + 1]))))

    width = 900
    height = 500
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 70

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_count = float(np.max(counts)) * 1.15
    n_bars = len(counts)
    bar_w = plot_width / n_bars

    bars = []
    x_labels = []

    for i in range(n_bars):
        c = counts[i]
        bar_h = (c / max_count) * plot_height
        x = margin_left + i * bar_w + 10
        y = margin_top + plot_height - bar_h

        color = "#e11d48" if i >= 4 else "#3b82f6"
        bars.append(
            f'  <rect x="{x:.1f}" y="{y:.1f}" width="{max(1.0, bar_w - 20):.1f}" '
            f'height="{bar_h:.1f}" fill="{color}" rx="3"/>\n'
            f'  <text x="{x + (bar_w - 20) / 2:.1f}" y="{y - 6:.1f}" font-size="11" '
            f'fill="#334155" font-weight="500" text-anchor="middle">{c:,}</text>'
        )

        mid_x = margin_left + i * bar_w + bar_w / 2
        y_bot = margin_top + plot_height
        x_labels.append(
            f'  <text x="{mid_x:.1f}" y="{y_bot + 22:.1f}" font-size="11" '
            f'fill="#64748b" text-anchor="middle">{labels[i]}</text>'
        )

    grid_lines = []
    y_labels = []
    for yt in np.linspace(0, max_count, 6):
        y_pos = margin_top + plot_height - (yt / max_count) * plot_height
        grid_lines.append(
            f'  <line x1="{margin_left:.1f}" y1="{y_pos:.1f}" '
            f'x2="{width - margin_right:.1f}" y2="{y_pos:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        y_labels.append(
            f'  <text x="{margin_left - 12:.1f}" y="{y_pos + 4:.1f}" font-size="11" '
            f'fill="#64748b" text-anchor="end">{int(yt):,}</text>'
        )

    y_mid = margin_top + plot_height / 2
    x_mid = margin_left + plot_width / 2
    sub = (
        "Percentage of zero-sales observation days across 50,000 series "
        "(Demonstrates fast-moving retail continuity)"
    )

    parts = [
        _svg_start(width, height),
        f'  <text x="{margin_left}" y="32" class="title">'
        f"Distribution of Zero-Sales Frequency across Series</text>\n",
        f'  <text x="{margin_left}" y="50" class="subtitle">{_esc(sub)}</text>\n',
        "\n".join(grid_lines) + "\n",
        "\n".join(bars) + "\n",
        _svg_axes(margin_left, margin_top, plot_width, plot_height, width),
        "\n".join(y_labels) + "\n",
        f'  <text x="{margin_left - 55}" y="{y_mid:.1f}" font-size="12" '
        f'fill="#334155" text-anchor="middle" '
        f'transform="rotate(-90, {margin_left - 55}, {y_mid:.1f})">'
        f"Number of Store-Product Series</text>\n",
        "\n".join(x_labels) + "\n",
        f'  <text x="{x_mid:.1f}" y="{height - 18}" font-size="12" '
        f'fill="#334155" text-anchor="middle">'
        f"Historical Zero-Sales Days Share (%)</text>\n",
        "</svg>",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def generate_weekday_pattern_chart(
    weekday_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate SVG bar chart of demand patterns across weekdays Monday to Sunday."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = weekday_df.sort_values("weekday_num").reset_index(drop=True)
    width = 900
    height = 500
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 70

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_val = float(df["mean_demand"].max()) * 1.25
    n_bars = len(df)
    bar_w = plot_width / n_bars

    bars = []
    x_labels = []

    for i, row in df.iterrows():
        mean_d = float(row["mean_demand"])
        med_d = float(row["median_demand"])
        day_name = str(row["weekday_name"])
        is_wk = int(row["is_weekend"])

        bar_h = (mean_d / max_val) * plot_height
        x = margin_left + i * bar_w + 14
        y = margin_top + plot_height - bar_h
        color = "#f59e0b" if is_wk == 1 else "#3b82f6"
        med_y = margin_top + plot_height - (med_d / max_val) * plot_height
        mid_bar = x + (bar_w - 28) / 2

        bars.append(
            f'  <rect x="{x:.1f}" y="{y:.1f}" width="{max(1.0, bar_w - 28):.1f}" '
            f'height="{bar_h:.1f}" fill="{color}" rx="3"/>\n'
            f'  <text x="{mid_bar:.1f}" y="{y - 8:.1f}" font-size="11" '
            f'fill="#0f172a" font-weight="600" text-anchor="middle">\n'
            f"    {mean_d:.3f}\n"
            f"  </text>\n"
            f'  <circle cx="{mid_bar:.1f}" cy="{med_y:.1f}" r="4" fill="#ffffff" '
            f'stroke="#0f172a" stroke-width="2"/>'
        )

        mid_x = margin_left + i * bar_w + bar_w / 2
        y_bot = margin_top + plot_height
        weight = 600 if is_wk == 1 else 400
        x_labels.append(
            f'  <text x="{mid_x:.1f}" y="{y_bot + 22:.1f}" font-size="12" '
            f'fill="#334155" font-weight="{weight}" text-anchor="middle">'
            f"{day_name}</text>"
        )

    grid_lines = []
    y_labels = []
    for yt in np.linspace(0, max_val, 6):
        y_pos = margin_top + plot_height - (yt / max_val) * plot_height
        grid_lines.append(
            f'  <line x1="{margin_left:.1f}" y1="{y_pos:.1f}" '
            f'x2="{width - margin_right:.1f}" y2="{y_pos:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        y_labels.append(
            f'  <text x="{margin_left - 12:.1f}" y="{y_pos + 4:.1f}" font-size="11" '
            f'fill="#64748b" text-anchor="end">{yt:.2f}</text>'
        )

    y_mid = margin_top + plot_height / 2
    x_mid = margin_left + plot_width / 2
    sub = (
        "Mean demand (Bars) and Median demand (White Circles) "
        "across 4.5M training observations"
    )

    legend = (
        f'  <g transform="translate({width - 240}, 20)">\n'
        f'    <rect x="0" y="0" width="12" height="12" fill="#3b82f6" rx="2"/>\n'
        f'    <text x="18" y="10" font-size="11" fill="#334155">Weekday</text>\n'
        f'    <rect x="0" y="18" width="12" height="12" fill="#f59e0b" rx="2"/>\n'
        f'    <text x="18" y="28" font-size="11" fill="#334155">'
        f"Weekend (Sat/Sun)</text>\n"
        f'    <circle cx="6" cy="42" r="4" fill="#ffffff" '
        f'stroke="#0f172a" stroke-width="2"/>\n'
        f'    <text x="18" y="46" font-size="11" fill="#334155">'
        f"Median Demand Point</text>\n"
        f"  </g>\n"
    )

    parts = [
        _svg_start(width, height),
        f'  <text x="{margin_left}" y="32" class="title">'
        f"Observed Demand by Day of Week</text>\n",
        f'  <text x="{margin_left}" y="50" class="subtitle">{_esc(sub)}</text>\n',
        "\n".join(grid_lines) + "\n",
        "\n".join(bars) + "\n",
        _svg_axes(margin_left, margin_top, plot_width, plot_height, width),
        "\n".join(y_labels) + "\n",
        f'  <text x="{margin_left - 50}" y="{y_mid:.1f}" font-size="12" '
        f'fill="#334155" text-anchor="middle" '
        f'transform="rotate(-90, {margin_left - 50}, {y_mid:.1f})">'
        f"Mean Daily Normalized Demand</text>\n",
        "\n".join(x_labels) + "\n",
        f'  <text x="{x_mid:.1f}" y="{height - 18}" font-size="12" '
        f'fill="#334155" text-anchor="middle">Calendar Day of Week</text>\n',
        legend,
        "</svg>",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def generate_stockout_exposure_distribution_chart(
    con: duckdb.DuckDBPyConnection,
    train_path: str,
    output_path: str | Path,
) -> None:
    """Generate SVG histogram of daytime stockout hours (0 to 16 hours)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    query = f"""
        SELECT
            stock_hour6_22_cnt AS hours,
            COUNT(*) AS cnt,
            ROUND(COUNT(*) * 100.0 / 4500000.0, 2) AS pct
        FROM parquet_scan('{train_path}')
        GROUP BY 1
        ORDER BY 1 ASC
    """
    df = con.execute(query).df()

    width = 900
    height = 500
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 70

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_cnt = float(df["cnt"].max()) * 1.15
    n_bars = 17
    bar_w = plot_width / n_bars

    bars = []
    x_labels = []

    for i in range(17):
        sub_df = df[df["hours"] == i]
        c = int(sub_df["cnt"].iloc[0]) if len(sub_df) > 0 else 0
        pct = float(sub_df["pct"].iloc[0]) if len(sub_df) > 0 else 0.0

        bar_h = (c / max_cnt) * plot_height
        x = margin_left + i * bar_w + 3
        y = margin_top + plot_height - bar_h

        color = "#10b981" if i == 0 else ("#ef4444" if i == 16 else "#6366f1")
        mid_bar = x + (bar_w - 6) / 2
        bars.append(
            f'  <rect x="{x:.1f}" y="{y:.1f}" width="{max(1.0, bar_w - 6):.1f}" '
            f'height="{bar_h:.1f}" fill="{color}" rx="2"/>\n'
            f'  <text x="{mid_bar:.1f}" y="{y - 6:.1f}" font-size="9" '
            f'fill="#334155" text-anchor="middle">{pct:.1f}%</text>'
        )

        mid_x = margin_left + i * bar_w + bar_w / 2
        y_bot = margin_top + plot_height
        x_labels.append(
            f'  <text x="{mid_x:.1f}" y="{y_bot + 18:.1f}" font-size="10" '
            f'fill="#64748b" text-anchor="middle">{i}h</text>'
        )

    grid_lines = []
    y_labels = []
    for yt in np.linspace(0, max_cnt, 6):
        y_pos = margin_top + plot_height - (yt / max_cnt) * plot_height
        grid_lines.append(
            f'  <line x1="{margin_left:.1f}" y1="{y_pos:.1f}" '
            f'x2="{width - margin_right:.1f}" y2="{y_pos:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        y_labels.append(
            f'  <text x="{margin_left - 12:.1f}" y="{y_pos + 4:.1f}" font-size="11" '
            f'fill="#64748b" text-anchor="end">{yt / 1000000:.2f}M</text>'
        )

    y_mid = margin_top + plot_height / 2
    x_mid = margin_left + plot_width / 2
    sub = (
        "Observation count by daytime stockout hours in training partition "
        "(55.7% at 0h, 40.2% partial 1–15h, 4.0% at 16h)"
    )

    legend = (
        f'  <g transform="translate({width - 250}, 20)">\n'
        f'    <rect x="0" y="0" width="12" height="12" fill="#10b981" rx="2"/>\n'
        f'    <text x="18" y="10" font-size="11" fill="#334155">'
        f"Stockout-Free (0h, 55.7%)</text>\n"
        f'    <rect x="0" y="18" width="12" height="12" fill="#6366f1" rx="2"/>\n'
        f'    <text x="18" y="28" font-size="11" fill="#334155">'
        f"Partially Affected (1–15h)</text>\n"
        f'    <rect x="0" y="36" width="12" height="12" fill="#ef4444" rx="2"/>\n'
        f'    <text x="18" y="46" font-size="11" fill="#334155">'
        f"Fully Stockout (16h, 4.0%)</text>\n"
        f"  </g>\n"
    )

    parts = [
        _svg_start(width, height),
        f'  <text x="{margin_left}" y="32" class="title">'
        f"Distribution of Daytime Stockout Hours (06:00 to 21:00)</text>\n",
        f'  <text x="{margin_left}" y="50" class="subtitle">{_esc(sub)}</text>\n',
        "\n".join(grid_lines) + "\n",
        "\n".join(bars) + "\n",
        _svg_axes(margin_left, margin_top, plot_width, plot_height, width),
        "\n".join(y_labels) + "\n",
        f'  <text x="{margin_left - 55}" y="{y_mid:.1f}" font-size="12" '
        f'fill="#334155" text-anchor="middle" '
        f'transform="rotate(-90, {margin_left - 55}, {y_mid:.1f})">'
        f"Observation Count (Millions)</text>\n",
        "\n".join(x_labels) + "\n",
        f'  <text x="{x_mid:.1f}" y="{height - 18}" font-size="12" '
        f'fill="#334155" text-anchor="middle">'
        f"Daytime Stockout Duration (Hours in 06:00–21:00 Window)</text>\n",
        legend,
        "</svg>",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def generate_sales_by_stockout_category_chart(
    severity_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate SVG comparison chart of observed sales across stockout categories."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = severity_df[
        severity_df["stockout_category"].isin(
            [
                "Daytime Stockout-Free",
                "Partially Stockout-Affected",
                "Fully Daytime Stockout",
            ]
        )
    ].reset_index(drop=True)

    width = 900
    height = 500
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 70

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_val = float(df["mean_sale"].max()) * 1.35
    n_cats = len(df)
    cat_w = plot_width / n_cats

    bars = []
    x_labels = []

    for i, row in df.iterrows():
        cat = str(row["stockout_category"])
        mean_s = float(row["mean_sale"])
        med_s = float(row["median_sale"])
        zero_pct = float(row["zero_sales_pct"])
        obs_cnt = int(row["observation_count"])
        obs_pct = float(row["observation_pct"])

        mean_h = (mean_s / max_val) * plot_height
        bar_x = margin_left + i * cat_w + 35
        bar_y = margin_top + plot_height - mean_h

        med_h = (med_s / max_val) * plot_height
        med_x = bar_x + 55
        med_y = margin_top + plot_height - med_h

        bars.append(
            f'  <rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="45" '
            f'height="{mean_h:.1f}" fill="#2563eb" rx="3"/>\n'
            f'  <text x="{bar_x + 22.5:.1f}" y="{bar_y - 8:.1f}" font-size="12" '
            f'fill="#0f172a" font-weight="600" text-anchor="middle">'
            f"{mean_s:.3f}</text>\n"
            f'  <rect x="{med_x:.1f}" y="{med_y:.1f}" width="45" '
            f'height="{med_h:.1f}" fill="#93c5fd" rx="3"/>\n'
            f'  <text x="{med_x + 22.5:.1f}" y="{med_y - 8:.1f}" font-size="12" '
            f'fill="#0f172a" font-weight="600" text-anchor="middle">'
            f"{med_s:.3f}</text>\n"
            f'  <text x="{bar_x + 50:.1f}" y="{margin_top + 30}" font-size="11" '
            f'fill="#64748b" text-anchor="middle">'
            f"Share: {obs_pct:.1f}% ({obs_cnt:,})</text>\n"
            f'  <text x="{bar_x + 50:.1f}" y="{margin_top + 48}" font-size="11" '
            f'fill="#e11d48" font-weight="500" text-anchor="middle">'
            f"Zero Sales: {zero_pct:.2f}%</text>"
        )

        mid_x = margin_left + i * cat_w + cat_w / 2
        y_bot = margin_top + plot_height
        x_labels.append(
            f'  <text x="{mid_x:.1f}" y="{y_bot + 22:.1f}" font-size="12" '
            f'font-weight="500" fill="#334155" text-anchor="middle">{cat}</text>'
        )

    grid_lines = []
    y_labels = []
    for yt in np.linspace(0, max_val, 6):
        y_pos = margin_top + plot_height - (yt / max_val) * plot_height
        grid_lines.append(
            f'  <line x1="{margin_left:.1f}" y1="{y_pos:.1f}" '
            f'x2="{width - margin_right:.1f}" y2="{y_pos:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        y_labels.append(
            f'  <text x="{margin_left - 12:.1f}" y="{y_pos + 4:.1f}" font-size="11" '
            f'fill="#64748b" text-anchor="end">{yt:.2f}</text>'
        )

    y_mid = margin_top + plot_height / 2
    x_mid = margin_left + plot_width / 2
    sub = (
        "Comparison of Mean and Median Sales with Zero-Sales Percentages "
        "across Stockout Categories"
    )

    legend = (
        f'  <g transform="translate({width - 200}, 20)">\n'
        f'    <rect x="0" y="0" width="12" height="12" fill="#2563eb" rx="2"/>\n'
        f'    <text x="18" y="10" font-size="11" fill="#334155">'
        f"Mean Sale Amount</text>\n"
        f'    <rect x="0" y="18" width="12" height="12" fill="#93c5fd" rx="2"/>\n'
        f'    <text x="18" y="28" font-size="11" fill="#334155">'
        f"Median Sale Amount</text>\n"
        f"  </g>\n"
    )

    parts = [
        _svg_start(width, height),
        f'  <text x="{margin_left}" y="32" class="title">'
        f"Observed Normalized Sales by Daytime Stockout Category</text>\n",
        f'  <text x="{margin_left}" y="50" class="subtitle">{_esc(sub)}</text>\n',
        "\n".join(grid_lines) + "\n",
        "\n".join(bars) + "\n",
        _svg_axes(margin_left, margin_top, plot_width, plot_height, width),
        "\n".join(y_labels) + "\n",
        f'  <text x="{margin_left - 50}" y="{y_mid:.1f}" font-size="12" '
        f'fill="#334155" text-anchor="middle" '
        f'transform="rotate(-90, {margin_left - 50}, {y_mid:.1f})">'
        f"Observed Demand Index</text>\n",
        "\n".join(x_labels) + "\n",
        f'  <text x="{x_mid:.1f}" y="{height - 18}" font-size="12" '
        f'fill="#334155" text-anchor="middle">Stockout Severity Category</text>\n',
        legend,
        "</svg>",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def generate_demand_segments_profile_chart(
    thresholds_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate SVG summary chart of demand tiers vs stockout exposure."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = thresholds_df[
        thresholds_df["segment_dimension"] == "Demand Volume"
    ].reset_index(drop=True)

    width = 900
    height = 500
    margin_left = 80
    margin_right = 80
    margin_top = 70
    margin_bottom = 70

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_demand = float(df["mean_demand"].max()) * 1.35
    max_stockout = 30.0

    n_tiers = len(df)
    tier_w = plot_width / n_tiers

    bars = []
    x_labels = []

    for i, row in df.iterrows():
        tier = str(row["segment_name"])
        mean_d = float(row["mean_demand"])
        stk_pct = float(row["mean_daytime_stockout_pct"])
        s_cnt = int(row["series_count"])

        d_h = (mean_d / max_demand) * plot_height
        bar_x = margin_left + i * tier_w + 40
        bar_y = margin_top + plot_height - d_h

        s_h = (stk_pct / max_stockout) * plot_height
        stk_x = bar_x + 60
        stk_y = margin_top + plot_height - s_h

        bars.append(
            f'  <rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="50" '
            f'height="{d_h:.1f}" fill="#3b82f6" rx="3"/>\n'
            f'  <text x="{bar_x + 25:.1f}" y="{bar_y - 8:.1f}" font-size="12" '
            f'fill="#0f172a" font-weight="600" text-anchor="middle">'
            f"{mean_d:.3f}</text>\n"
            f'  <rect x="{stk_x:.1f}" y="{stk_y:.1f}" width="50" '
            f'height="{s_h:.1f}" fill="#f59e0b" rx="3"/>\n'
            f'  <text x="{stk_x + 25:.1f}" y="{stk_y - 8:.1f}" font-size="12" '
            f'fill="#0f172a" font-weight="600" text-anchor="middle">'
            f"{stk_pct:.1f}%</text>\n"
            f'  <text x="{bar_x + 55:.1f}" y="{margin_top + 30}" font-size="11" '
            f'fill="#64748b" text-anchor="middle">{s_cnt:,} series</text>'
        )

        mid_x = margin_left + i * tier_w + tier_w / 2
        y_bot = margin_top + plot_height
        x_labels.append(
            f'  <text x="{mid_x:.1f}" y="{y_bot + 22:.1f}" font-size="12" '
            f'font-weight="500" fill="#334155" text-anchor="middle">'
            f"{tier} Demand</text>"
        )

    grid_lines = []
    y_labels_left = []
    for yt in np.linspace(0, max_demand, 6):
        y_pos = margin_top + plot_height - (yt / max_demand) * plot_height
        grid_lines.append(
            f'  <line x1="{margin_left:.1f}" y1="{y_pos:.1f}" '
            f'x2="{width - margin_right:.1f}" y2="{y_pos:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        y_labels_left.append(
            f'  <text x="{margin_left - 12:.1f}" y="{y_pos + 4:.1f}" font-size="11" '
            f'fill="#3b82f6" text-anchor="end">{yt:.2f}</text>'
        )

    y_labels_right = []
    for st in np.linspace(0, max_stockout, 6):
        y_pos = margin_top + plot_height - (st / max_stockout) * plot_height
        x_right_lbl = width - margin_right + 12
        y_labels_right.append(
            f'  <text x="{x_right_lbl:.1f}" y="{y_pos + 4:.1f}" font-size="11" '
            f'fill="#f59e0b" text-anchor="start">{st:.0f}%</text>'
        )

    y_mid = margin_top + plot_height / 2
    x_mid = margin_left + plot_width / 2
    sub = (
        "Tercile-based segmentation (Low, Medium, High) showing demand index "
        "(Blue) and stockout rate (Amber)"
    )

    legend = (
        f'  <g transform="translate({width - 240}, 20)">\n'
        f'    <rect x="0" y="0" width="12" height="12" fill="#3b82f6" rx="2"/>\n'
        f'    <text x="18" y="10" font-size="11" fill="#334155">'
        f"Mean Demand Index</text>\n"
        f'    <rect x="0" y="18" width="12" height="12" fill="#f59e0b" rx="2"/>\n'
        f'    <text x="18" y="28" font-size="11" fill="#334155">'
        f"Mean Daytime Stockout (%)</text>\n"
        f"  </g>\n"
    )

    parts = [
        _svg_start(width, height),
        f'  <text x="{margin_left}" y="32" class="title">'
        f"Demand and Stockout Exposure across Demand Segments</text>\n",
        f'  <text x="{margin_left}" y="50" class="subtitle">{_esc(sub)}</text>\n',
        "\n".join(grid_lines) + "\n",
        "\n".join(bars) + "\n",
        _svg_axes(
            margin_left, margin_top, plot_width, plot_height, width, include_right=True
        ),
        "\n".join(y_labels_left) + "\n",
        f'  <text x="{margin_left - 50}" y="{y_mid:.1f}" font-size="12" '
        f'fill="#3b82f6" text-anchor="middle" '
        f'transform="rotate(-90, {margin_left - 50}, {y_mid:.1f})">'
        f"Mean Demand Index</text>\n",
        "\n".join(y_labels_right) + "\n",
        f'  <text x="{width - margin_right + 50}" y="{y_mid:.1f}" font-size="12" '
        f'fill="#f59e0b" text-anchor="middle" '
        f'transform="rotate(90, {width - margin_right + 50}, {y_mid:.1f})">'
        f"Mean Daytime Stockout (%)</text>\n",
        "\n".join(x_labels) + "\n",
        f'  <text x="{x_mid:.1f}" y="{height - 18}" font-size="12" '
        f'fill="#334155" text-anchor="middle">Demand Segment</text>\n',
        legend,
        "</svg>",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def generate_category_heterogeneity_chart(
    category_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate SVG chart of demand heterogeneity across top product categories."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = (
        category_df.sort_values("total_demand", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    width = 900
    height = 500
    margin_left = 140
    margin_right = 60
    margin_top = 70
    margin_bottom = 70

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_demand = float(df["total_demand"].max()) * 1.15
    n_cats = len(df)
    bar_h = plot_height / n_cats

    bars = []
    y_labels = []

    for i, row in df.iterrows():
        cat_id = int(row["first_category_id"])
        tot_d = float(row["total_demand"])
        mean_d = float(row["mean_demand"])
        stk_pct = float(row["mean_daytime_stockout_pct"])

        bar_w = (tot_d / max_demand) * plot_width
        y = margin_top + i * bar_h + 4
        h = max(1.0, bar_h - 8)

        lbl = f"{tot_d / 1000:.1f}k (Avg {mean_d:.2f}, Stkout {stk_pct:.1f}%)"
        bars.append(
            f'  <rect x="{margin_left}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{h:.1f}" fill="#0284c7" rx="2"/>\n'
            f'  <text x="{margin_left + bar_w + 8:.1f}" y="{y + h / 2 + 4:.1f}" '
            f'font-size="11" fill="#0f172a">{lbl}</text>'
        )

        y_labels.append(
            f'  <text x="{margin_left - 12}" y="{y + h / 2 + 4:.1f}" '
            f'font-size="11" fill="#334155" text-anchor="end">'
            f"Category {cat_id}</text>"
        )

    grid_lines = []
    x_labels = []
    for xt in np.linspace(0, max_demand, 6):
        x_pos = margin_left + (xt / max_demand) * plot_width
        grid_lines.append(
            f'  <line x1="{x_pos:.1f}" y1="{margin_top:.1f}" '
            f'x2="{x_pos:.1f}" y2="{margin_top + plot_height:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        x_labels.append(
            f'  <text x="{x_pos:.1f}" y="{margin_top + plot_height + 20:.1f}" '
            f'font-size="11" fill="#64748b" text-anchor="middle">'
            f"{xt / 1000:.0f}k</text>"
        )

    x_mid = margin_left + plot_width / 2
    sub = (
        "Ranked by Total Observed Demand Index "
        "(with series mean demand and stockout rate annotated)"
    )

    parts = [
        _svg_start(width, height),
        f'  <text x="{margin_left}" y="32" class="title">'
        f"Demand Heterogeneity: Top 10 Product Categories</text>\n",
        f'  <text x="{margin_left}" y="50" class="subtitle">{_esc(sub)}</text>\n',
        "\n".join(grid_lines) + "\n",
        "\n".join(bars) + "\n",
        _svg_axes(margin_left, margin_top, plot_width, plot_height, width),
        "\n".join(y_labels) + "\n",
        "\n".join(x_labels) + "\n",
        f'  <text x="{x_mid:.1f}" y="{height - 18}" font-size="12" '
        f'fill="#334155" text-anchor="middle">'
        f"Total Observed Normalized Demand Index</text>\n",
        "</svg>",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
