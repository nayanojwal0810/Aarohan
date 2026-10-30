# Aarohan — Stage 2: Exploratory Data Analysis Findings

## 1. Executive Summary & Scope

This report presents empirical findings from Stage 2 Exploratory Data
Analysis on the **FreshRetailNet-50K** dataset for Project Aarohan
(*Intelligent Demand Planning for Perishable Retail*).

The analysis was performed across all **4,850,000 observations**
(4,500,000 training observations across 90 dates from `2024-03-28` to
`2024-06-25`, and 350,000 held-out evaluation observations across 7
dates from `2024-06-26` to `2024-07-02`). All 50,000 store-product
series across 898 stores and 865 products are represented.

The entire analytical pipeline was executed within strict memory-efficient
constraints (DuckDB memory ceiling: 512 MB, 2 execution threads, PyArrow
chunk streaming), completing in **29.45 seconds** with a
peak resident set size of **247.3 MB**.

> [!IMPORTANT]
> **Data Semantics Notice**:
> 1. `sale_amount` and `hours_sale` represent globally normalized demand
> index values, not physical unit counts.
> 2. `stock_hour6_22_cnt` counts daytime stockout hours between 06:00 and
> 21:00 inclusive (16 daytime hours total). Daytime availability fraction
> is defined as `1 - stock_hour6_22_cnt / 16`.
> 3. `hours_stock_status` records 24-hour binary status (1 = stockout,
> 0 = in stock). A value of zero in `stock_hour6_22_cnt` does not guarantee
> 24-hour stockout-free availability.
> 4. All evaluation partition statistics reported herein are **strictly
> descriptive held-out analyses** used exclusively to verify population
> alignment. They are never used to tune features or construct models.

---

## 2. Overall Demand Distribution

| Metric | Training Population Value | Description / Interpretation |
| :--- | :--- | :--- |
| **Total Observations** | 4,500,000 | 50,000 series × 90 calendar dates |
| **Total Normalized Demand** | 4,493,660.787 | Sum of sales index across training |
| **Mean Daily Demand Index** | 0.9986 | Mean per store-product-day |
| **Median Daily Demand Index** | 0.6800 | 50th percentile |
| **Standard Deviation** | 1.3934 | Standard deviation of daily demand |
| **Minimum Demand** | 0.0000 | Zero observed sales |
| **Maximum Demand** | 44.9000 | Peak observed normalized sales index |
| **25th Percentile (P25)** | 0.3200 | Lower quartile |
| **75th Percentile (P75)** | 1.2500 | Upper quartile |
| **90th Percentile (P90)** | 2.1500 | 90th percentile of daily demand |
| **95th Percentile (P95)** | 2.9700 | 95th percentile of daily demand |
| **99th Percentile (P99)** | 5.5600 | Long-tail heavy demand threshold |
| **Zero-Sales Observations** | 200,760 (4.46%) | Total zero sales days |

### Distribution Characteristics
The observed daily demand exhibits a right-skewed distribution. The mean
(0.9986) exceeds the median (0.6800) due to high-volume products during peak
demand periods. The top 1% of transactions (P99 = 5.5600) extends up to
44.9000, illustrating pronounced demand concentration.

---

## 3. Store-Product Series Heterogeneity & Demand Segmentation

Across the 50,000 series, mean daily demand varies substantially:
- Minimum series mean demand: **0.0000**
- Median series mean demand: **0.6578**
- Maximum series mean demand: **25.2156**

### Empirical Demand Segments
Using training-population terciles (33.33% and 66.67% quantiles), series
are segmented into three operational volume tiers:

| Demand Tier | Lower Bound | Upper Bound | Series Count | Series Share | Mean Demand Index | Mean Daytime Stockout |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low Demand** | 0.0000 | 0.5367 | 16,673 | 33.35% | 0.4608 | 19.69% |
| **Medium Demand** | 0.5367 | 0.8713 | 16,662 | 33.32% | 0.6726 | 19.71% |
| **High Demand** | 0.8713 | 19.0744 | 16,665 | 33.33% | 1.8625 | 20.23% |

### Intermittency Analysis
FreshRetailNet-50K exhibits strong demand continuity, reflecting fresh
grocery retail characteristics rather than slow-moving spare parts:
- **28.81% of series** have **0.0% zero-sales days** throughout the 90 days.
- **74.27% of series** have **≤ 5.0% zero-sales days**.
- **88.21% of series** have **≤ 10.0% zero-sales days**.
- Using a threshold of `zero_sales_pct >= 20.0%`, only **2,573 series
  (5.15%)** are classified as intermittent.
- Only **82 series (0.16%)** experience ≥ 50.0% zero-sales days.

---

## 4. Availability and Stockout Patterns

Daytime availability (06:00 to 21:00 inclusive) is a primary operational
determinant of observed retail demand:

| Stockout Severity Category | Criteria | Observations | Share (%) | Mean Sale Index | Median Sale Index | Zero-Sales Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Daytime Stockout-Free** | `stock_hour6_22_cnt == 0` | 2,507,994 | 55.73% | 0.9745 | 0.6800 | 1.87% |
| **Partially Stockout-Affected** | `1 <= stock_hour6_22_cnt < 16` | 1,810,215 | 40.23% | 1.1276 | 0.7000 | 0.32% |
| **Fully Daytime Stockout** | `stock_hour6_22_cnt == 16` | 181,791 | 4.04% | 0.0460 | 0.0000 | 81.44% |
| **Fully 24-Hour Stockout** | `sum(hours_stock_status) == 24` | 170,717 | 3.79% | 0.0369 | 0.0000 | 85.24% |

---

## 5. Zero-Sales Decomposition & Causality Analysis

A critical analytical finding is that zero sales in FreshRetailNet-50K are
predominantly driven by **stockouts**, not customer demand absence:
- **Total Zero-Sales Observations**: 200,760 (4.46%).
- **Stockout-Affected Zero Sales**: 153,816 (**76.62% of all zero sales**). When a zero occurs, there
  is an over 3-to-1 probability that the item was out of stock during the day.
- **In-Stock Zero Sales (True Zero Demand)**: 46,944 (**23.38% of all zero sales**, representing 1.04% of total).
- On fully daytime-stockout days, zero-sales rate is **81.44%**.

---

## 6. Observed Sales vs Stockout Endogeneity

Notice a key counter-intuitive diagnostic:
- Mean sales on **Partially Stockout-Affected days** (1.1276)
  is **higher** than on **Daytime Stockout-Free days** (0.9745).

**Explanation**: Fast-moving, high-demand items deplete inventory rapidly
during high-traffic periods, triggering partial stockouts before store close.
Conversely, slower-moving items remain in stock throughout the full 16 hours.
This reflects **stockout endogeneity and right-censoring**: observed sales
under partial stockout are truncated measures of an even larger demand.

### Hourly Concordance Diagnostics
Vectorized analysis across all **108,000,000 hourly observations** reveals:
- Total stockout hours: **26,876,404** (24.89% of all hourly slots).
- Zero sales during stockout hours: **26,103,978** (97.13% concordance).
- Positive sales during stockout-flagged hours: **772,426** (2.87%).
- As established in project methodology, positive sales during stockout
  represent intra-hour transactions before stockout onset or inventory latency.
  It is a descriptive diagnostic and does not indicate causality.

---

## 7. Calendar and Temporal Dynamics

| Day of Week | Observation Count | Date Count | Mean Demand Index | Median Demand Index | Mean Daytime Stockout (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Monday** | 650,000 | 13 | 0.9426 | 0.6000 | 20.15% |
| **Tuesday** | 650,000 | 13 | 0.9620 | 0.6600 | 20.81% |
| **Wednesday** | 600,000 | 12 | 0.9069 | 0.6000 | 19.53% |
| **Thursday** | 650,000 | 13 | 0.8660 | 0.6000 | 18.71% |
| **Friday** | 650,000 | 13 | 0.9268 | 0.6000 | 17.55% |
| **Saturday** | 650,000 | 13 | 1.1781 | 0.8000 | 19.92% |
| **Sunday** | 650,000 | 13 | 1.2007 | 0.8000 | 22.44% |

Demand jumps sharply on weekends:
- Mean demand on Sunday (**1.2007**) is **38.6% higher**
  than Thursday (**0.8660**).
- Median demand shifts from **0.60** on weekdays to **0.80** on weekends.
- Sunday daytime stockouts peak at **22.44%**,
  confirming severe supply constraints during weekend peak traffic.

---

## 8. Training vs Held-Out Evaluation Population Comparison

All evaluation figures represent descriptive held-out comparisons:

| Metric | Training Partition (90 Days) | Evaluation Partition (7 Days, Held-Out) | Relative Change |
| :--- | :--- | :--- | :--- |
| **Observations** | 4,500,000 | 350,000 | -92.22% |
| **Unique Stores** | 898 | 898 | 0.00% (Full match) |
| **Unique Products** | 865 | 865 | 0.00% (Full match) |
| **Store-Product Series** | 50,000 | 50,000 | 0.00% (Full match) |
| **Mean Daily Demand Index** | 0.9986 | 0.9829 | -1.57% |
| **Median Daily Demand Index** | 0.6800 | 0.6700 | -1.47% |
| **Zero-Sales Share (%)** | 4.46% | 4.16% | -6.73% |
| **Daytime Stockout-Free Share (%)** | 55.73% | 59.02% | +5.90% |
| **Partially Stockout Share (%)** | 40.23% | 37.40% | -7.03% |
| **Fully Daytime Stockout Share (%)** | 4.04% | 3.58% | -11.39% |
| **Fully 24-Hour Stockout Share (%)** | 3.79% | 3.37% | -11.08% |

The evaluation population demonstrates remarkable structural and
distributional stability relative to the training period, with exactly
identical series, store, and product footprints.

---

## 9. Generated Figures and Summary Tables

### Generated Summary Tables (`reports/tables/`)
1. `eda_daily_demand_summary.csv`: Daily aggregate observations, totals,
   means, medians, percentiles, and stockout counts.
2. `eda_series_segment_thresholds.csv`: Documentation of boundaries,
   series shares, and means for demand, intermittency, and stockout tiers.
3. `eda_category_summary.csv`: Aggregated volume, mean sales, and
   stockout rates across 32 first-level product categories.
4. `eda_store_summary.csv`: Store-level volume, series count, and
   stockout exposure across 898 stores.
5. `eda_stockout_severity_summary.csv`: Complete metrics stratified across
   daytime and 24-hour stockout severity tiers.
6. `eda_train_eval_comparison.csv`: Descriptive comparison between training
   and held-out evaluation populations.

### Generated Vector Figures (`reports/figures/`)
1. `fig01_daily_aggregate_demand.svg`: Daily aggregate demand trend.
2. `fig02_series_mean_demand_distribution.svg`: Histogram of series mean demand.
3. `fig03_zero_sales_distribution.svg`: Histogram of series zero-sales share.
4. `fig04_weekday_demand_pattern.svg`: Weekday demand pattern.
5. `fig05_stockout_exposure_distribution.svg`: Daytime stockout hours distribution.
6. `fig06_sales_by_stockout_category.svg`: Observed sales across stockout tiers.
7. `fig07_demand_segments_stockout_profile.svg`: Demand and stockout exposure.
8. `fig08_category_heterogeneity.svg`: Top 10 product categories volume & stockouts.

---

## 10. Implications for Historical Demand Estimation

The analytical findings provide empirical guideposts for Stage 3
(Historical Demand Estimation under Stockouts):

1. **Observed Sales Significantly Understate True Latent Demand on Stockout Days**:
   Because 40.23% of series-days experience partial daytime stockouts and
   4.04% suffer full daytime stockouts, naive training on observed sales
   would introduce systemic downward bias. Historical demand must be imputed
   or unconstrained on stockout-affected periods.

2. **Zero-Sales Cannot Be Treated as Zero Demand Without Stockout Qualification**:
   76.62% of zero-sales days occur when products are flagged as out of stock.
   Treating these as zero customer demand would distort intermittency models.
   In-stock zero sales (true zero demand) constitute only 1.04% of total
   observations.

3. **Partial Stockout Endogeneity Must Be Addressed**:
   Fast-moving items have higher observed sales on partially stockout days
   (1.1276) than on stockout-free days (0.9745), because higher sales velocity
   induces earlier stock exhaustion. Estimation algorithms must account for
   available selling hours (e.g. daytime availability fraction) rather than
   treating partially stockout days as unconstrained.

4. **Weekend Seasonality Is a Strong Demand Driver**:
   With weekend demand 25–38% above midweek levels and stockout rates peaking
   on Sunday (22.44%), demand estimation models must preserve calendar
   day-of-week interactions.

5. **Daytime vs 24-Hour Stockout Discrepancy Must Be Explicitly Handled**:
   While 55.73% of days are daytime stockout-free, only a subset are fully
   24-hour stockout-free. Daytime availability (`1 - stock_hour6_22_cnt / 16`)
   aligns with customer shopping hours and is the relevant operational exposure
   window for retail demand planning.

