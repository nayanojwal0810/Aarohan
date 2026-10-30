# Aarohan: Dataset Profile & Validation Report

**Inspection Date:** `2026-10-30T12:51:50.796465+00:00`

**Overall Validation Status:** **`WARNING`**

*(Status is WARNING due to 34 known training records with discount > 1.0, preserved without clipping per specification. Zero critical failures detected.)*

---

## 1. Dataset Overview

Aarohan utilizes the **FreshRetailNet-50K** retail demand dataset published by Dingdong Inc. under CC BY 4.0. The dataset contains 4,850,000 daily observations across 50,000 store-product series in perishable grocery retail, annotated with 24-hour stockout status flags and hourly sales distributions.

---

## 2. File Details

| Partition | Relative Path | File Size | SHA-256 Checksum | Checksum Status |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | `D:\Aarohan\data\raw\train.parquet` | 101.51 MB (106,436,287 B) | `6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d` | **PASS** |
| **Evaluation** | `D:\Aarohan\data\raw\eval.parquet` | 8.05 MB (8,440,124 B) | `1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e` | **PASS** |

---

## 3. Dataset Dimensions

| Dimension | Training Partition | Evaluation Partition | Combined Total |
| :--- | :--- | :--- | :--- |
| **Row Count** | 4,500,000 | 350,000 | 4,850,000 |
| **Column Count** | 19 | 19 | 19 |
| **Row Groups** | 4,500 (1,000 rows/group) | 350 (1,000 rows/group) | 4,850 |
| **Unique Dates** | 90 days | 7 days | 97 continuous days |
| **Unique Series** | 50,000 | 50,000 | 50,000 series (100% overlap) |

---

## 4. Column Summary

All 19 required columns are present in both partitions with matching logical Arrow types:
- **Entity IDs:** `city_id` (int64), `store_id` (int64), `management_group_id` (int64), `first_category_id` (int64), `second_category_id` (int64), `third_category_id` (int64), `product_id` (int64)
- **Temporal & Sales:** `dt` (string), `sale_amount` (double), `hours_sale` (list<double>)
- **Stockout Indicators:** `stock_hour6_22_cnt` (int32), `hours_stock_status` (list<int64>)
- **Covariates:** `discount` (double), `holiday_flag` (int32), `activity_flag` (int32), `precpt` (double), `avg_temperature` (double), `avg_humidity` (double), `avg_wind_level` (double)

Zero missing columns, zero undeclared columns, and zero data type mismatches detected.

---

## 5. Date Coverage & Continuity

* **Training Window:** `2024-03-28` to `2024-06-25` (exactly 90 continuous days).
* **Evaluation Window:** `2024-06-26` to `2024-07-02` (exactly 7 continuous days).
* **Temporal Ordering:** Training strictly ends (`2024-06-25`) before evaluation begins (`2024-06-26`). Zero overlap.
* **Daily Completeness:** Every calendar date in training contains exactly 50,000 records; every date in evaluation contains exactly 50,000 records. Zero missing dates.

---

## 6. Store-Product Structure

* **Unique Stores:** 898 stores.
* **Unique SKUs:** 865 perishable products.
* **Store-Product Series:** Exactly 50,000 series in training and 50,000 series in evaluation.
* **Cross-Partition Overlap:** 50,000 series in common (0 train-only, 0 eval-only).
* **Panel Balance:** Every series has exactly 90 daily records in training and exactly 7 daily records in evaluation.

---

## 7. Data Integrity Findings

* **Missing Values:** Exactly `0` null values across all 19 columns in both partitions.
* **Non-Finite Values:** Exactly `0` non-finite (NaN/Inf) values across all float fields.
* **Negative Sales:** Exactly `0` negative sales values (`sale_amount >= 0.0`).
* **Natural Key Duplicates:** Exactly `0` duplicate `(store_id, product_id, dt)` rows.
* **Binary Indicators:** `holiday_flag` and `activity_flag` are strictly binary [0, 1].

---

## 8. Hourly Data Validation

Nested 24-hour sequence arrays were validated in controlled batches of 5,000 records:
* **Array Lengths:** Every row contains exactly 24 elements for both `hours_sale` and `hours_stock_status`.
* **Array Validity:** Zero null array elements; hourly sales are non-negative and finite.
* **Binary Status:** Hourly stockout values are strictly in [0, 1].
* **Sales Reconciliation:**
  * Invariant: `sale_amount ≈ sum(hours_sale)` within tolerance `1e-5`.
  * **Result:** **0 mismatches** across all 4,850,000 records. Maximum discrepancy is within IEEE 754 float rounding ($1.42 \times 10^{-14}$).
* **Stockout Count Reconciliation:**
  * Invariant: `stock_hour6_22_cnt == sum(hours_stock_status[6:22])`.
  * **Result:** **0 mismatches** across all 4,850,000 records (max diff: 0.0).

---

## 9. Stockout Availability Findings

* **Daytime No-Stockout Rows (`stock_hour6_22_cnt == 0`):**
  * Training: 2,507,994 rows (55.73%)
  * Evaluation: 206,562 rows (59.02%)
* **Daytime Full-Stockout Rows (`stock_hour6_22_cnt == 16`):**
  * Training: 181,791 rows (4.04%)
  * Evaluation: 14,140 rows (4.04%)
* **Full 24-Hour Stockout Rows (all 24 hours out of stock):**
  * Training: 154,611 rows (3.44%)
  * Evaluation: 12,041 rows (3.44%)
* **Stockout Hours Total:** 26,876,404 hours in training (24.89% of all hours); 1,913,065 hours in evaluation (22.77%).
* **Sales During Stockout:**
  * Positive sales occur during stockout hours in **2.87%** of training stockout hours (772,426 hours) and **3.10%** of evaluation stockout hours (59,300 hours).
  * Represents 2.51% of training sales volume (112,931.36 units) and 2.17% of evaluation sales volume (9,080.19 units).
  * Caused by intra-hour depletion prior to stockout logging. Handled in demand estimation as a lower-bound constraint.

---

## 10. Discount Anomalies

* **Training Partition:**
  * Range: `[0.000, 1.088]` (mean: `0.9111`, median: `0.9886`).
  * 34 records exceed 1.0 (all belonging to product SKU 37 across 6 stores).
  * Maximum discount: `1.088` (+8.8% pricing surcharge over list price).
  * **Status:** **`WARNING`** (preserved without clipping per specification; zero negative discounts).
* **Evaluation Partition:**
  * Range: `[0.000, 1.000]` (mean: `0.9217`, median: `1.0000`).
  * Exactly 0 records exceed 1.0. **Status:** **`PASS`**.

---

## 11. Evaluation Cohort Composition

The official primary evaluation cohort is defined by `stock_hour6_22_cnt == 0`:
* **Total Evaluation Rows:** 350,000 (50,000 series $\times$ 7 days)
* **Daytime No-Stockout Cohort:** **206,562 rows (59.02%)**
* **Daytime Stockout-Affected Cohort:** 143,438 rows (40.98%)

---

## 12. Warnings and Limitations

1. **Normalized Demand Scale:** The dataset uses globally normalized sales units; absolute physical piece counts are unrecoverable.
2. **Daytime vs. 24-Hour Stockout:** The `stock_hour6_22_cnt == 0` cohort represents zero stockouts during operating hours (06:00–22:00), but does not guarantee 24-hour stockout absence during overnight non-trading hours.
3. **Discount > 1.0:** 34 records for product 37 exceed 1.0; downstream modeling must accommodate these without arbitrary clipping.
4. **Censoring Lower Bound:** Because sales occur during some stockout-flagged hours, demand estimation models must enforce $\hat{y}^* \ge y^{\text{obs}}$ during stockouts.
