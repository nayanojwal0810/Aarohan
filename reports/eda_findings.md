# Aarohan — Stage 2: Exploratory Data Analysis Findings

## 1. Executive Summary & Scope

This report presents empirical findings from Stage 2 Exploratory Data
Analysis on the **FreshRetailNet-50K** dataset for Project Aarohan
(*Intelligent Demand Planning for Perishable Retail*).

The analysis was performed across all **4,850,000 observations**
(4,500,000 training observations
across 90 dates from `2024-03-28` to `2024-06-25`, and 350,000
held-out evaluation observations across 7 dates from `2024-06-26` to
`2024-07-02`). All 50,000 store-product series across
898 stores and 865 products are
represented.

The analytical pipeline was executed within strict memory constraints
(DuckDB memory ceiling: 512 MB, 2 execution threads, PyArrow record
batch streaming), completing in **54.54 seconds** with a
peak working set of **616.7 MB** (Windows `peak_wset`)
and completion resident set size of **190.5 MB**.

> [!IMPORTANT]
> **Data Semantics and Operational Boundaries**:
> 1. `sale_amount` and `hours_sale` represent globally normalized
> demand index values, not physical unit counts.
> 2. `stock_hour6_22_cnt` counts daytime stockout hours between
> 06:00 and 21:00 inclusive (16 daytime hours total). Daytime
> availability fraction is defined as `1 - stock_hour6_22_cnt / 16`.
> 3. `hours_stock_status` records 24-hour binary status (1 = stockout,
> 0 = in stock). A value of zero in `stock_hour6_22_cnt` does not
> guarantee 24-hour stockout-free availability.
> 4. All evaluation partition statistics reported herein are strictly
> **descriptive held-out analyses** used exclusively to verify
> population alignment. They are never used to tune features or
> construct models.

---

## 2. Overall Demand Distribution

Distributional metrics are computed directly at the observation level
across all 4,500,000 training records rather than averaging daily
medians or standard deviations:

| Metric | Observation-Level Value | Description / Interpretation |
| :--- | :--- | :--- |
| **Total Observations** | 4,500,000 | Total rows across 50,000 series × 90 calendar dates |
| **Total Normalized Demand** | 4,493,660.7870 | Sum of normalized sales index in training partition |
| **Mean Daily Demand Index** | 0.9986 | Arithmetic mean across all 4.5M individual observations |
| **Median Daily Demand Index** | 0.7000 | 50th percentile across all 4.5M individual observations |
| **Standard Deviation** | 1.4067 | Observation-level standard deviation |
| **Minimum Demand** | 0.0000 | Zero observed sales |
| **Maximum Demand** | 44.9000 | Peak observed normalized sales index |
| **25th Percentile (P25)** | 0.4000 | Lower quartile of observation-level demand |
| **50th Percentile (P50)** | 0.7000 | Median of observation-level demand |
| **75th Percentile (P75)** | 1.1000 | Upper quartile of observation-level demand |
| **90th Percentile (P90)** | 2.0000 | 90th percentile of observation-level demand |
| **95th Percentile (P95)** | 2.9000 | 95th percentile of observation-level demand |
| **99th Percentile (P99)** | 5.8000 | 99th percentile of observation-level demand |
| **Zero-Sales Observations** | 200,760 (4.46%) | Observations with sale_amount == 0.0 |

### Distribution Characteristics
The observed daily demand exhibits a right-skewed distribution. The
overall mean (0.9986) substantially
exceeds the overall median (0.7000).
The upper 1% of transactions (P99 = 5.8000)
extends up to 44.9000, demonstrating heavy
right-tail volume concentration.

---

## 3. Store-Product Series Heterogeneity & Demand Segmentation

### Balanced Panel Structure
All 50,000 store-product series contain exactly 90 calendar
observations (minimum 90, maximum 90, 0 missing dates), constituting
a 100% complete balanced panel across the training window.

### Series-Level Distribution Moments (`eda_series_summary_metrics.csv`)
Across the 50,000 series, mean daily demand, volatility, CV, and
zero-sales frequency exhibit wide variation. Note: when mean demand is
zero, the Coefficient of Variation (CV = std / mean) is mathematically
undefined (`np.nan`) and excluded from CV summary calculations:

| Metric | Valid Series | Mean | Std Dev | Min | P25 | Median | P75 | P90 | P99 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **mean_demand** | 50,000 | 0.9986 | 1.1526 | 0.2220 | 0.4942 | 0.6578 | 1.0544 | 1.8222 | 4.9478 | 19.0744 |
| **median_demand** | 50,000 | 0.9410 | 1.2227 | 0.0000 | 0.5000 | 0.6000 | 1.0000 | 1.7000 | 4.7500 | 22.7500 |
| **std_demand** | 50,000 | 0.5303 | 0.6136 | 0.1018 | 0.2702 | 0.3684 | 0.5829 | 0.9417 | 2.5860 | 12.2109 |
| **cv** | 50,000 | 0.5665 | 0.2093 | 0.1643 | 0.4309 | 0.5141 | 0.6414 | 0.8297 | 1.3318 | 3.0078 |
| **zero_sales_pct** | 50,000 | 4.4613 | 7.0650 | 0.0000 | 0.0000 | 2.2222 | 5.5556 | 12.2222 | 34.4444 | 72.2222 |
| **mean_daytime_stockout_pct** | 50,000 | 19.8771 | 8.2140 | 3.8889 | 14.8611 | 17.6736 | 21.9444 | 28.8889 | 53.0563 | 97.0139 |
| **day_count** | 50,000 | 90.0000 | 0.0000 | 90.0000 | 90.0000 | 90.0000 | 90.0000 | 90.0000 | 90.0000 | 90.0000 |

### Empirical Demand Segments
Using training-population terciles (33.33% and 66.67% quantiles),
series are segmented into three operational volume tiers:

| Demand Tier | Lower Bound | Upper Bound | Series Count | Series Share | Mean Demand Index | Mean Daytime Stockout |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low Demand** | 0.0000 | 0.5367 | 16,673 | 33.35% | 0.4608 | 19.69% |
| **Medium Demand** | 0.5367 | 0.8713 | 16,662 | 33.32% | 0.6726 | 19.71% |
| **High Demand** | 0.8713 | 19.0744 | 16,665 | 33.33% | 1.8625 | 20.23% |

### Intermittency Analysis
FreshRetailNet-50K exhibits high demand continuity, reflecting
fast-moving fresh grocery retail rather than slow-moving spare parts:
- **28.81% of series** have **0.0% zero-sales days**
  throughout the 90-day training period.
- **74.27% of series** have **≤ 5.0% zero-sales days**.
- **88.21% of series** have **≤ 10.0% zero-sales days**.
- Using a threshold of `zero_sales_pct >= 20.0%`, **2,573 series (5.15%)** are classified as intermittent.
- Only **82 series (0.16%)** experience ≥ 50.0% zero-sales days.

---

## 4. Product-Level Demand Distribution & Volume Concentration

Across all **865 distinct products**
(`eda_product_summary.csv`):
- Product store presence ranges from **1** to **783 stores** (median: **10 stores**).
- Product mean daily demand ranges from **0.3644** to **15.5551** (median: **0.5215**).
- Top 10% of products (86 items) account for **72.30%** of total aggregate normalized demand.
- The 32 first-level product categories (`eda_category_summary.csv`)
  exhibit wide heterogeneity, with category mean daily demand ranging
  from
  **0.4150** to **1.7951**.

---

## 5. Availability and Stockout Patterns

Daytime availability (06:00 to 21:00 inclusive, 16 hours) is a primary
operational determinant of observed retail demand. The three daytime
categories form a **mutually exclusive and exhaustive partition**
summing to 100% of observations, while the 24-hour stockout category
is an **overlapping subset of fully daytime stockouts**:

| Stockout Severity Category | Category Type | Observations | Share (%) | Mean Sale Index | Median Sale Index | Zero-Sales Share (%) | Mean Daytime Stockout | Mean 24h Stockout |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Daytime Stockout-Free** | Mutually Exclusive Partition | 2,507,994 | 55.73% | 0.9745 | 0.6800 | 1.87% | 0.00h | 1.26h |
| **Partially Stockout-Affected** | Mutually Exclusive Partition | 1,810,215 | 40.23% | 1.1276 | 0.7000 | 0.32% | 6.30h | 10.71h |
| **Fully Daytime Stockout** | Mutually Exclusive Partition | 181,791 | 4.04% | 0.0460 | 0.0000 | 81.44% | 16.00h | 23.86h |
| **Fully 24-Hour Stockout** | Overlapping Subset of Fully Daytime Stockout | 170,717 | 3.79% | 0.0369 | 0.0000 | 85.24% | 16.00h | 24.00h |

### Daytime vs 24-Hour Stockout Relationship
- **Daytime Stockout-Free** (2,507,994 rows, 55.73%), **Partially Affected** (1,810,215 rows, 40.23%), and **Fully Daytime Stockout** (181,791 rows, 4.04%) sum to exactly **4,500,000 rows (100.00%)**.
- **Fully 24-Hour Stockout** (170,717 rows, 3.79%) is a strict **93.91% subset** of the 181,791 fully daytime stockout observations. Zero observations have 24-hour stockout with daytime stockout < 16 hours.

---

## 6. Zero-Sales Decomposition & Availability Diagnostics

Empirical zero-sales decomposition across all 4.5M training observations:
- **Total Zero-Sales Observations**: **200,760** (4.46% of all observations).
- **Stockout-Associated Zero Sales**: **153,816** (**76.62% of all zero sales**, representing 3.42% of all observations). When zero sales occur, daytime stockout flags co-occur in over 3 out of 4 cases.
- **In-Stock Zero Sales**: **46,944** (**23.38% of zeros**, representing **1.04% of all observations**). These represent observed zero sales occurring during daytime stockout-free hours.
- On fully daytime-stockout days, zero-sales co-occurrence is **81.44%** (148,045 days).

> [!NOTE]
> **Diagnostic Interpretation Note**:
> - Zero sales co-occurring with stockout flags reflect empirical
>   association, not definitive causal proof of lost demand.
> - In-stock zero sales indicate zero recorded sales during available
>   hours; they are descriptive observations rather than proven zero
>   customer demand.

---

## 7. Observed Sales Patterns & Hourly Concordance Diagnostics

### Partial Stockout Association
Mean observed sales on **Partially Stockout-Affected days** (**1.1276**) are higher than on **Daytime Stockout-Free days** (**0.9745**).

This empirical pattern reflects an observed association/correlation:
fast-moving, high-demand items tend to sell at higher velocities and
exhaust stock before the end of trading hours, whereas slow-moving items
remain in stock throughout the full 16 hours. Observed sales under
partial
stockout are truncated measures of true unconstrained demand.

### Hourly Concordance Diagnostics
Vectorized PyArrow streaming analysis across all **108,000,000 hourly
observation slots** reveals:
- Total stockout hours: **26,876,404** (24.89% of all hourly slots).
- Total in-stock hours: **81,123,596** (75.11% of all hourly slots).
- Zero sales during stockout hours: **26,103,978** (97.13% concordance).
- Positive sales during stockout-flagged hours: **772,426** (2.87% of stockout hours).
- Positive sales during stockout hours represent an empirical descriptive
  diagnostic (such as intra-hour transactions occurring prior to stockout
  onset within the hourly reporting bucket). Observed sales and inferred
  demand are kept strictly distinct.

---

## 8. Calendar and Temporal Dynamics

Demand patterns stratified across days of the week
(`eda_weekday_patterns`):

| Day of Week | Weekend | Observations | Dates | Mean Demand Index | Median Demand Index | Mean Daytime Stockout (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Monday** | No | 650,000 | 13 | 0.9426 | 0.6000 | 20.15% |
| **Tuesday** | No | 650,000 | 13 | 0.9620 | 0.6600 | 20.81% |
| **Wednesday** | No | 600,000 | 12 | 0.9069 | 0.6000 | 19.53% |
| **Thursday** | No | 650,000 | 13 | 0.8660 | 0.6000 | 18.71% |
| **Friday** | No | 650,000 | 13 | 0.9268 | 0.6000 | 17.55% |
| **Saturday** | Yes | 650,000 | 13 | 1.1781 | 0.8000 | 19.92% |
| **Sunday** | Yes | 650,000 | 13 | 1.2007 | 0.8000 | 22.44% |

Demand increases substantially on weekends: Sunday mean demand (**1.2007**) is **38.6% higher** than Thursday midweek demand (**0.8660**). Daytime stockouts also peak on Sunday (**22.44%**), indicating increased inventory pressure during high-traffic weekend periods.

---

## 9. Data Missingness and Outlier Diagnostics

Diagnostics derived from data validation pipeline (`reports/tables/data_validation.json`):

- **Validation File Status**: Available.
- **Scalar Missingness**: **0 missing / null values** across 19 columns
  and 4,500,000 records (100% complete).
- **Non-Finite Values**: **0 NaN / Inf values** across numeric fields.
- **Hourly Array Integrity**: 24 elements per array across records
  with zero null parent rows or child elements.
- **Reconciliation Checks**: 0 mismatches between daily `sale_amount`
  and sum of `hours_sale` (max diff: 7.11e-15); exactly 0 mismatches
  between
  `stock_hour6_22_cnt` and daytime status flags.
- **Discount Ceiling Anomaly**: Found **34 records in train with discount > 1.0**
  (maximum 1.088). Raw values are preserved as authentic empirical data
  without
  synthetic clipping.
- **Demand Outliers**: Extreme right-tail demand reaches maximum **44.9000** (P99 is 5.8000), reflecting high-volume promotional spikes rather than corrupt data.

---

## 10. Training vs Held-Out Evaluation Population Comparison

All evaluation partition statistics are strictly descriptive and
held out.
Generated dynamically from `eda_train_eval_comparison.csv`:

| Metric | Training Partition (90 Days) | Evaluation Partition (7 Days, Held-Out) | Absolute Diff | Relative Change | Assessment Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Observations** | 4500000 | 350000 | -4150000 | -92.22% | Descriptive Held-Out Assessment: 90 dates train vs 7 dates eval |
| **Dates (Days)** | 90 | 7 | -83 | -92.22% | Descriptive Held-Out Assessment: 90 days vs 7 days |
| **Date Start** | 2024-03-28 | 2024-06-26 | N/A | N/A | Descriptive Held-Out Assessment: partition start date |
| **Date End** | 2024-06-25 | 2024-07-02 | N/A | N/A | Descriptive Held-Out Assessment: partition end date |
| **Unique Stores** | 898 | 898 | 0 | 0.0% | Identical entity coverage (898 stores) |
| **Unique Products** | 865 | 865 | 0 | 0.0% | Identical entity coverage (865 products) |
| **Store-Product Series** | 50000 | 50000 | 0 | 0.0% | Identical series coverage (50,000 series) |
| **Mean Daily Demand** | 0.9986 | 1.1931 | 0.1945 | 19.48% | Descriptive Held-Out: +19.48% higher mean sales in eval |
| **Median Daily Demand** | 0.7 | 0.8 | 0.1 | 14.29% | Descriptive Held-Out: +14.29% higher median sales in eval |
| **Std Dev Daily Demand** | 1.4067 | 1.9116 | 0.5049 | 35.89% | Descriptive Held-Out: +35.89% higher volatility in eval |
| **Zero-Sales Share (%)** | 4.46 | 4.16 | -0.3 | -6.73% | Descriptive Held-Out: lower zero share in eval |
| **Daytime Stockout-Free Share (%)** | 55.73 | 59.02 | 3.29 | 5.9% | Descriptive Held-Out: higher stockout-free share in eval |
| **Partially Stockout-Affected Share (%)** | 40.23 | 37.4 | -2.83 | -7.03% | Descriptive Held-Out Assessment |
| **Fully Daytime Stockout Share (%)** | 4.04 | 3.58 | -0.46 | -11.39% | Descriptive Held-Out Assessment |
| **Fully 24-Hour Stockout Share (%)** | 3.79 | 3.37 | -0.42 | -11.08% | Descriptive Held-Out Assessment |
| **Mean Daytime Stockout (%)** | 19.8771 | 18.1077 | -1.7694 | -8.9% | Descriptive Held-Out Assessment |
| **Mean 24-Hour Stockout (%)** | 24.8856 | 22.7746 | -2.111 | -8.48% | Descriptive Held-Out Assessment |

### Key Population Differences
1. **Exact Entity Footprint Alignment**: The evaluation partition has
   identical coverage of all 50,000 series, 898 stores, and 865 products (100% match).
2. **Higher Evaluation Demand Volume**: Mean daily demand in the evaluation period is **1.1931**, which is **+19.48% higher** than the training mean (**0.9986**). Std dev is similarly +35.89% higher.
3. **Coverage vs Distribution Distinction**: Identical entity footprints do NOT imply
   identical demand distributions. The evaluation window (late June / early July) exhibits
   higher average demand and volatility than the overall 90-day training average.

---

## 11. Segments Difficult to Forecast

EDA identifies several operational segments that present heightened forecasting difficulty:
1. **Intermittent Series (2,573 series, 5.15%)**:
   Series with `zero_sales_pct >= 20.0%` require specialized intermittent handling (e.g. hurdle models or zero-inflated formulations).
2. **High Stockout-Exposure Series (16,519 series)**: Series with mean daytime stockout >= 33.33% experience severe censoring that distorts observed sales distributions.
3. **High Volatility Series (CV > 1.0)**: Series with wide demand dispersion require
   quantile loss functions or probabilistic prediction intervals.
4. **Tail Products**: Products with low store penetration and sparse transaction volume.

---

## 12. Modelling Decisions Supported by EDA

The empirical findings directly support key architectural and modeling decisions for Stage 3 & 4:
1. **Unconstrain Demand on Stockout Periods**: Because 40.23% of rows experience
   partial daytime stockouts and 4.04% full daytime stockouts, training models directly on
   raw observed sales would introduce systemic downward bias. Historical demand must be imputed
   or adjusted during stockout intervals.
2. **Separate In-Stock Zeros from Censored Zeros**: In-stock zero sales represent authentic
   zero customer purchases, whereas stockout-associated zeros reflect
   lack of availability.
   These must be treated distinctly in demand estimation.
3. **Daytime Availability Exposure Feature**: Operational demand is
   governed by the 16-hour
   daytime window (06:00–21:00). Daytime availability fraction (`1 - stock_hour6_22_cnt / 16`)
   should be used as an exposure feature.
4. **Explicit Calendar and Day-of-Week Seasonality**: Given significant weekend demand lift
   (Sunday mean demand is +38.6% higher than midweek), models must explicitly capture day-of-week interactions.
5. **Hierarchical Pooling & Entity Embeddings**: Heterogeneity across 50,000 series and
   32 categories necessitates hierarchical pooling (store-level, product-level, category-level)
   to prevent overfitting on sparse series.
6. **Robust Target Transformations**: Right-skewed demand (P99 = 5.80 vs max 44.90) and promotional
   discount anomalies (> 1.0) motivate log1p target transformations or robust loss formulations
   (e.g. Huber / Pinball loss).

---

## 13. Summary of Generated Tables and Figures

### Summary Tables (`reports/tables/`)
1. `eda_overall_demand_summary.csv`: Observation-level moments across all 4.5M rows.
2. `eda_daily_demand_summary.csv`: Daily aggregate totals, means, and percentiles.
3. `eda_series_summary_metrics.csv`: Distribution moments across all 50,000 series.
4. `eda_series_segment_thresholds.csv`: Boundaries and counts for volume, intermittency, and stockouts.
5. `eda_product_summary.csv`: Product-level demand, store count, and stockout metrics.
6. `eda_category_summary.csv`: Aggregated volume across 32 first-level product categories.
7. `eda_store_summary.csv`: Store-level volume and stockout exposure
   across 898 stores.
8. `eda_stockout_severity_summary.csv`: Metrics stratified across
   stockout severity tiers.
9. `eda_train_eval_comparison.csv`: Descriptive comparison between train and held-out evaluation.

### Vector Figures (`reports/figures/`)
1. `fig01_daily_aggregate_demand.svg`: Daily aggregate demand trend
   across 90 days.
2. `fig02_series_mean_demand_distribution.svg`: Histogram of series mean demand.
3. `fig03_zero_sales_distribution.svg`: Histogram of series zero-sales share.
4. `fig04_weekday_demand_pattern.svg`: Weekday demand pattern with
   mean and median.
5. `fig05_stockout_exposure_distribution.svg`: Daytime stockout hours distribution.
6. `fig06_sales_by_stockout_category.svg`: Observed sales across stockout tiers.
7. `fig07_demand_segments_stockout_profile.svg`: Demand volume and
   stockout exposure.
8. `fig08_category_heterogeneity.svg`: Product category volume and stockouts.

