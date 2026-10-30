# Aarohan: Dataset Profile & Validation Report

**Inspection Date:** `2026-10-30T13:35:47.748273+00:00`
**Overall Validation Status:** **`WARNING`**
*(Validation completed with WARNING in checks: discount_ceiling_anomaly. Documented anomalies preserved without artificial clipping. Zero critical failures detected.)*

---

## 1. Dataset Overview

Aarohan utilizes the **FreshRetailNet-50K** retail demand dataset published by Dingdong Inc. under CC BY 4.0. The dataset contains daily observations across store-product series in perishable grocery retail, annotated with 24-hour stockout status flags and hourly sales distributions.

---

## 2. File Details

| Partition | Relative Path | File Size | SHA-256 Checksum | Checksum Status |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | `data/raw/train.parquet` | 101.51 MB (106,436,287 B) | `6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d` | **PASS** |
| **Evaluation** | `data/raw/eval.parquet` | 8.05 MB (8,440,124 B) | `1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e` | **PASS** |

---

## 3. Dataset Dimensions

| Dimension | Training Partition | Evaluation Partition | Combined Total |
| :--- | :--- | :--- | :--- |
| **Row Count** | 4,500,000 | 350,000 | 4,850,000 |
| **Column Count** | 19 | 19 | 19 |
| **Row Groups** | 4,500 | 350 | 4,850 |
| **Unique Dates** | 90 days | 7 days | 97 continuous days |
| **Unique Series** | 50,000 | 50,000 | 50,000 series |

---

## 4. Column Summary

All 19 required columns were verified against expected logical Arrow types:
- **Entity IDs:** `city_id` (int64), `store_id` (int64), `management_group_id` (int64), `first_category_id` (int64), `second_category_id` (int64), `third_category_id` (int64), `product_id` (int64)
- **Temporal & Sales:** `dt` (string), `sale_amount` (double), `hours_sale` (list<double>)
- **Stockout Indicators:** `stock_hour6_22_cnt` (int32), `hours_stock_status` (list<int64>)
- **Covariates:** `discount` (double), `holiday_flag` (int32), `activity_flag` (int32), `precpt` (double), `avg_temperature` (double), `avg_humidity` (double), `avg_wind_level` (double)

---

## 5. Date Coverage & Continuity

* **Training Window:** `2024-03-28` to `2024-06-25` (90 days).
* **Evaluation Window:** `2024-06-26` to `2024-07-02` (7 days).
* **Temporal Ordering:** Training strictly ends (`2024-06-25`) before evaluation begins (`2024-06-26`). Zero overlap.

---

## 6. Store-Product Structure

* **Unique Stores:** 898 stores in training, 898 in evaluation.
* **Unique SKUs:** 865 products in training, 865 in evaluation.
* **Store-Product Series:** 50,000 in training, 50,000 in evaluation.

---

## 7. Data Integrity Findings

* **Missing Values:** 0 scalar nulls in training; 0 in evaluation.
* **Non-Finite Values:** 0 in training; 0 in evaluation.
* **Negative Sales:** 0 in training; 0 in evaluation.
* **Natural Key Duplicates:** 0 in training; 0 in evaluation.

---

## 8. Hourly Data Validation

Nested 24-hour sequence arrays were validated in controlled batches:
* **Array Lengths:** All 24 elements with non-null parents (train); All 24 elements with non-null parents (eval).
* **Element Validity:** Negative: 0, Non-finite: 0, Child nulls: 0, Parent nulls: 0 (train); Negative: 0, Non-finite: 0, Child nulls: 0, Parent nulls: 0 (eval).
* **Sales Reconciliation:** Invariant `sale_amount ≈ sum(hours_sale)` within tolerance `1e-5`.
  * **Result:** **0 mismatches** across all records. Maximum discrepancy is within float rounding (7.11e-15).
* **Stockout Count Reconciliation:** Invariant `stock_hour6_22_cnt == sum(hours_stock_status[6:22])`.
  * **Result:** **0 mismatches** across all records (max diff: 0.0).

---

## 9. Stockout Availability Diagnostics

* **Daytime No-Stockout Rows (`stock_hour6_22_cnt == 0`):
  * Training: 2,507,994 rows (55.73%)
  * Evaluation: 206,562 rows (59.02%)
* **Daytime Full-Stockout Rows (`stock_hour6_22_cnt == 16`):
  * Training: 181,791 rows (4.04%)
  * Evaluation: 12,541 rows (3.58%)
* **Full 24-Hour Stockout Rows (all 24 hours out of stock):
  * Training: 170,717 rows (3.79%)
  * Evaluation: 11,809 rows (3.37%)
* **Stockout Hours Total:** 26,876,404 hours in training; 1,913,065 hours in evaluation.
* **Positive Sales During Stockout Flags:**
  * Training: Positive sales co-occur with stockout flags in **2.874%** of stockout hours (772,426 hours, volume: 112,931.35 normalized units).
  * Evaluation: Positive sales co-occur with stockout flags in **3.0997%** of stockout hours (59,300 hours, volume: 9,080.19 normalized units).
  * *Note:* Positive sales co-occurring with stockout flags are an observed diagnostic. Their within-hour timing and cause are not established by this validation.

---

## 10. Discount Anomalies

* **Training Partition:** 34 records with discount > 1.0 (max: 1.088).
  * Status: **WARNING** (34 records > 1.0; discount > 1.0 is an observed anomaly relative to the conventional discount range. Its cause is not established by this validation. Raw values are preserved without clipping or correction).
* **Evaluation Partition:** Max discount: 1.0.
  * Status: **PASS** (Zero records exceeding 1.0).

---

## 11. Evaluation Cohort Composition

The project's selected primary evaluation cohort is defined by `stock_hour6_22_cnt == 0`:
* **Total Evaluation Rows:** 350,000
* **Daytime No-Stockout Cohort:** **206,562 rows (59.02%)**
* **Daytime Stockout-Affected Cohort:** 143,438 rows (40.98%)

---

## 12. Warnings and Limitations

1. **Normalized Demand Scale:** The dataset uses globally normalized sales index values. Raw physical unit counts cannot be reconstructed.
2. **Daytime vs. 24-Hour Stockout:** The `stock_hour6_22_cnt == 0` cohort represents daytime operating stockout freedom (06:00–22:00); it does not guarantee 24-hour stockout absence.
3. **Discount > 1.0:** 34 records in training exhibit discount > 1.0 (max 1.088); these raw observed values are preserved without clipping or correction.
4. **Positive Sales Co-occurring with Stockout Flags:** Positive sales co-occur with stockout flags in ~2.874% of training stockout hours and ~3.0997% of evaluation stockout hours. Their within-hour timing and cause are not established by this validation.
