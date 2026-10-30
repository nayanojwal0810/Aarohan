# Aarohan: Dataset Profile and Validation Report

## 1. Dataset Overview

Aarohan utilizes the FreshRetailNet-50K retail demand dataset published by Dingdong Inc. under CC BY 4.0. The dataset contains daily panel observations across store-product series in perishable grocery retail, annotated with 24-hour stockout status flags and hourly sales distributions.

---

## 2. Dataset Coverage

### Partition File Specifications

| Partition | File Path | File Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| Training | `data/raw/train.parquet` | 101.51 MB (106,436,287 B) | `6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d` |
| Evaluation | `data/raw/eval.parquet` | 8.05 MB (8,440,124 B) | `1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e` |

### Dataset Dimensions

| Dimension | Training Partition | Evaluation Partition | Combined Total |
| :--- | :--- | :--- | :--- |
| Row Count | 4,500,000 | 350,000 | 4,850,000 |
| Column Count | 19 | 19 | 19 |
| Row Groups | 4,500 | 350 | 4,850 |
| Unique Dates | 90 days | 7 days | 97 continuous days |
| Unique Series | 50,000 | 50,000 | 50,000 series |
| Unique Stores | 898 | 898 | 898 stores |
| Unique SKUs | 865 | 865 | 865 products |

---

## 3. Data Structure

All 19 required columns conform to expected logical Arrow types:
- Entity Identifiers: `city_id` (int64), `store_id` (int64), `management_group_id` (int64), `first_category_id` (int64), `second_category_id` (int64), `third_category_id` (int64), `product_id` (int64)
- Temporal and Sales: `dt` (string), `sale_amount` (double), `hours_sale` (list<double>)
- Stockout Indicators: `stock_hour6_22_cnt` (int32), `hours_stock_status` (list<int64>)
- Covariates: `discount` (double), `holiday_flag` (int32), `activity_flag` (int32), `precpt` (double), `avg_temperature` (double), `avg_humidity` (double), `avg_wind_level` (double)

---

## 4. Data Completeness and Integrity

No missing scalar values exist across all columns in either partition.
All floating-point numerical values are finite across both partitions (zero NaN or Infinity occurrences).
No negative sales values exist in either partition.
Zero duplicate records exist on the primary natural key `(store_id, product_id, dt)` across all observations in both partitions.
Holiday and marketing activity indicators strictly conform to binary values {0, 1}.

---

## 5. Temporal Coverage

- Training Window: `2024-03-28` to `2024-06-25` (90 calendar days).
- Evaluation Window: `2024-06-26` to `2024-07-02` (7 calendar days).
- Temporal Partition Ordering: The training partition strictly precedes the evaluation partition without temporal overlap (`2024-06-25` < `2024-06-26`).
- Series Continuity and Balance: Every store-product series contains an unbroken daily observation record without missing calendar dates (90 days in training, 7 days in evaluation). Daily record counts are exactly balanced at 50,000 observations per day.

---

## 6. Hourly Sales and Stock Availability

- Nested Array Structure: Every record in both partitions contains exactly 24 hourly observations for both sales and stock status with non-null parent arrays.
- Hourly Sales Elements: Hourly sales arrays contain exclusively finite, non-negative numerical values with zero missing child elements across all 24 hours.
- Hourly Stockout Indicators: Hourly stockout flags are strictly binary integers in {0, 1} across all hours in both partitions.
- Sales Reconciliation: Daily recorded sales reconcile with the sum of hourly sales within floating-point tolerance (1e-5) across all records in both partitions (maximum absolute difference: 7.11e-15).
- Daytime Stockout Count Reconciliation: Daytime stockout hour counts (`stock_hour6_22_cnt`) reconcile exactly with the sum of hourly stockout indicators across hours 06:00 through 21:00 inclusive (`hours_stock_status[6:22]`) across all records in both partitions.

### Stockout Availability Diagnostics

- Daytime Stockout-Free Series-Days (`stock_hour6_22_cnt == 0`): Training 2,507,994 rows (55.73%), Evaluation 206,562 rows (59.02%).
- Full Daytime Stockout Series-Days (`stock_hour6_22_cnt == 16`): Training 181,791 rows (4.04%), Evaluation 12,541 rows (3.58%).
- Full 24-Hour Stockout Series-Days: Training 170,717 rows (3.79%), Evaluation 11,809 rows (3.37%).
- Total Stockout Hours: Training 26,876,404 hours, Evaluation 1,913,065 hours.
- Positive Sales During Stockout Flags: Training 2.87% of stockout hours (772,426 hours, volume: 112,931.35 normalized demand index values); Evaluation 3.10% of stockout hours (59,300 hours, volume: 9,080.19 normalized demand index values). Positive sales co-occurring with stockout flags are an observed diagnostic. Their within-hour timing and cause are not established by this validation.

---

## 7. Observed Data Anomalies

- Non-Negative Discount Rates: Discount rates are non-negative across all records in both partitions.
- Training Partition Discount Ceiling: 34 records feature discount values exceeding 1.0 (maximum observed discount: 1.088). discount > 1.0 is an observed anomaly relative to the conventional discount range with an unknown cause. Raw values are preserved without clipping or correction.
- Evaluation Partition Discount Ceiling: Discount ceiling check unavailable.

---

## 8. Evaluation Cohort

The project's selected primary evaluation cohort is defined by `stock_hour6_22_cnt == 0` (series-days free of daytime stockouts):
- Total Evaluation Records: 350,000
- Daytime Stockout-Free Cohort: 206,562 records (59.02%)
- Daytime Stockout-Affected Cohort: 143,438 records (40.98%)

---

## 9. Data Limitations and Modeling Considerations

1. Demand Index Scale: Recorded sales values represent globally normalized demand index figures rather than raw physical unit quantities. Original physical unit sales cannot be reconstructed from the public dataset. Demand forecasting and evaluation operate in this normalized space. Observed sales reflect censored transaction volume during periods of product availability; stockout status flags indicate product unavailability; and customer demand remains unobserved and latent when stockouts occur.
2. Daytime Operating Scope: The primary evaluation cohort (`stock_hour6_22_cnt == 0`) captures stockout absence across daytime operating hours (06:00 through 21:00 inclusive); daytime availability does not imply complete stockout absence during overnight fulfillment and restocking windows (22:00 through 06:00).
3. Discount Anomalies: 34 training records feature discount values exceeding 1.0 (reaching 1.088), which are observed anomalies with an unknown cause. Raw observed values are preserved without clipping, truncation, or synthetic adjustment.
4. Sales and Stockout Co-occurrence: Positive sales co-occur with stockout flags in approximately 2.87% of training stockout hours and 3.10% of evaluation stockout hours. Their within-hour timing and cause are not established by this validation.
