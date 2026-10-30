# Aarohan: Data Dictionary

This document details all 19 columns in the FreshRetailNet-50K dataset used by the Aarohan demand planning system.

---

## Column Specifications

| Column Name | Data Type | Status | Business Meaning | Interpretation Notes |
| :--- | :--- | :--- | :--- | :--- |
| `city_id` | `int64` | Observed | City entity identifier | Categorical ID denoting metropolitan delivery area (18 unique cities). |
| `store_id` | `int64` | Observed | Retail store/warehouse identifier | Frontline fulfilment station ID (898 unique stores). |
| `management_group_id` | `int64` | Observed | Operational cluster identifier | Administrative store management cluster (7 unique groups). |
| `first_category_id` | `int64` | Observed | L1 Product hierarchy category | Broad merchandise department (32 categories, e.g. Vegetables, Meat). |
| `second_category_id` | `int64` | Observed | L2 Product hierarchy category | Sub-department category (84 categories). |
| `third_category_id` | `int64` | Observed | L3 Product hierarchy category | Granular product sub-category (233 categories). |
| `product_id` | `int64` | Observed | SKU item identifier | Stock keeping unit identifier (865 unique perishable items). |
| `dt` | `string` | Observed | Calendar date | Date of transaction in `YYYY-MM-DD` ISO-8601 format. |
| `sale_amount` | `double` | Observed | Daily sales volume | Daily recorded sales quantity after global normalization by Dingdong Inc. Values are non-negative normalized demand index values, not physical unit counts. Reconciles with `sum(hours_sale)` within float precision ($10^{-5}$). |
| `hours_sale` | `list<double>` | Observed | 24-hour sales distribution | Array of exactly 24 non-negative floating-point values representing sales for each hour from 00:00 to 23:00. |
| `stock_hour6_22_cnt` | `int32` | Derived | Daytime stockout hour count | Count of daytime hours (06:00 through 21:00 inclusive) with recorded stockout. Integer range $[0, 16]$. Equals $\sum_{h=6}^{21} \text{hours\_stock\_status}_h$. |
| `hours_stock_status` | `list<int64>` | Observed | 24-hour stockout flags | Array of exactly 24 binary integers. `1` indicates out-of-stock during that hour; `0` indicates item was in stock. |
| `discount` | `double` | Observed | Daily discount rate | Promotional pricing factor (selling price / list price). 1.0 indicates full retail price; 0.9 represents a 10% discount. 34 records in training exceed 1.0 (up to 1.088), which is an observed anomaly relative to the conventional discount range preserved without clipping. |
| `holiday_flag` | `int32` | Derived | Statutory holiday indicator | Binary flag (`1` for statutory holidays, `0` for regular days). |
| `activity_flag` | `int32` | Derived | Marketing campaign indicator | Binary flag (`1` if a promotional campaign was active, `0` otherwise). |
| `precpt` | `double` | Observed | Daily precipitation | Realized rainfall measurement in millimeters (mm). |
| `avg_temperature` | `double` | Observed | Daily average temperature | Ambient outdoor temperature in degrees Celsius (°C). |
| `avg_humidity` | `double` | Observed | Daily average relative humidity | Relative humidity percentage ($[0, 100]$%). |
| `avg_wind_level` | `double` | Observed | Daily average wind level | Meteorological wind intensity index. |

---

## Critical Modeling Notes

1. **Normalized Scale:** `sale_amount` and `hours_sale` are globally normalized demand index values. Raw physical item counts cannot be reconstructed. Predictions represent normalized demand index values.
2. **Stockout Semantics:** `hours_stock_status = 1` indicates unavailable stock. Positive sales co-occur with stockout flags in ~2.87% of stockout hours; their within-hour timing and cause are not established by this validation.
3. **Evaluation Filter:** `stock_hour6_22_cnt == 0` designates the project's selected primary evaluation cohort free of daytime stockouts (206,562 rows / 59.02% of evaluation partition). It must not be interpreted as guaranteeing zero stockouts across overnight non-trading hours (22:00–06:00).
