# Aarohan: Demand Planning Decision-Support Prototype for Perishable Retail

Aarohan is an end-to-end demand-planning decision-support prototype designed for perishable grocery retail. It investigates how to estimate customer demand that may be obscured by stockouts, generate 7-day store-product forecasts, and translate predictions into illustrative replenishment recommendations.

Aarohan is an end-to-end demand-planning decision-support prototype. It is not connected to a live retailer's automated ordering or enterprise resource planning (ERP) system.

---

## 1. Problem Context

In fresh and perishable grocery retail, recorded sales often fail to reflect actual customer demand. When an inventory stockout occurs, sales drop to zero regardless of how many customers intended to purchase the item.

Forecasting models trained directly on raw sales data risk learning historical inventory availability constraints rather than genuine consumer demand patterns. In grocery operations, this creates an adverse feedback loop: unadjusted sales lead to lower forecasts, lower forecasts cause under-replenishment, and under-replenishment causes recurring stockouts.

Aarohan explores methods to address this problem by accounting for stockout censoring, estimating demand hidden during unavailable intervals, producing multi-day store-product forecasts, and providing transparent replenishment guidance.

---

## 2. Project Objectives

The project investigates a connected five-stage workflow:

1. **Data Ingestion and Integrity Checks:** Load and inspect historical transaction panels, verifying schema consistency, daily continuity, and stockout indicator fidelity.
2. **Availability and Stockout Profiling:** Analyze the frequency, duration, and sales impact of partial-day and full-day stockout events across store-product hierarchies.
3. **Latent Demand Estimation:** Estimate customer demand during stockout-affected periods using censored demand imputation methods.
4. **Multi-Horizon Demand Forecasting:** Generate 7-day store-product forecasts using global gradient-boosted trees (LightGBM) trained on leakage-safe temporal features.
5. **Illustrative Replenishment Guidance:** Evaluate prediction uncertainty and convert demand estimates into illustrative safety-stock and replenishment recommendations.

---

## 3. Dataset and Attribution

Aarohan uses the **FreshRetailNet-50K** dataset published by Dingdong Inc.

* **Dataset Source:** [FreshRetailNet-50K on Hugging Face](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
* **Data Provider:** Dingdong Inc.
* **Dataset License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
* **Dataset Scope:** 4,850,000 daily observations across 50,000 store-product series, 898 stores, 865 perishable SKUs, and hourly stockout annotations.
* **Scale and Units:** As documented by Dingdong Inc., sales quantities (`sale_amount` and `hours_sale`) are provided after global normalization by a proprietary scaling coefficient to protect commercial sales volumes. The original physical integer unit counts cannot be reconstructed from the dataset; all forecast outputs represent normalized demand index values.

---

## 4. Dataset Setup

The raw Parquet files are excluded from Git tracking via `.gitignore` because of their size. To run the project locally, download the two official files and place them in the `data/raw/` directory:

```text
data/
└── raw/
    ├── train.parquet
    └── eval.parquet
```

* **Training file (~101.5 MB):**
  [Download train.parquet](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/main/data/train.parquet)
* **Evaluation file (~8.05 MB):**
  [Download eval.parquet](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/main/data/eval.parquet)

---

## 5. Technology Stack

Aarohan is implemented in Python 3.12 using standard open-source libraries:

* **Language:** Python 3.12 (64-bit)
* **Data Processing:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [PyArrow](https://arrow.apache.org/pyarrow/)
* **Columnar SQL Engine:** [DuckDB](https://duckdb.org/)
* **Scientific Computing & ML:** [SciPy](https://scipy.org/), [Scikit-learn](https://scikit-learn.org/)
* **Forecasting Engine:** [LightGBM](https://lightgbm.readthedocs.io/)
* **Configuration & Logging:** [PyYAML](https://pyyaml.org/), Python Standard Library `logging`
* **System Telemetry:** [psutil](https://github.com/giampaolo/psutil)
* **Testing & Code Quality:** [pytest](https://docs.pytest.org/), [Ruff](https://docs.astral.sh/ruff/)

---

## 6. Repository Layout

```text
Aarohan/
├── README.md                   # Project documentation and setup guide
├── LICENSE                     # MIT License for codebase
├── pyproject.toml              # Build specification and package configuration
├── requirements.txt            # Tested direct dependency versions
├── .gitignore                  # Git exclusion rules
│
├── configs/                    # YAML configuration files
│   ├── data.yaml               # Dataset paths, entity keys, date bounds
│   ├── features.yaml           # Lag parameters and feature controls
│   └── training.yaml           # Model settings and temporal validation folds
│
├── data/
│   ├── raw/                    # Original raw Parquet files (untracked)
│   ├── interim/                # Transformed intermediate tables (untracked)
│   └── processed/              # Feature matrices for modeling (untracked)
│
├── src/
│   └── aarohan/                # Core Python package
│       ├── __init__.py         # Package root and version
│       ├── config.py           # Configuration loader and path resolver
│       └── logging_config.py   # Reusable structured logging setup
│
├── notebooks/                  # Exploratory notebooks
├── scripts/                    # Command-line execution scripts
├── tests/                      # Automated unit tests
├── reports/                    # Generated project reports
│   ├── figures/                # Output plots and diagrams
│   └── tables/                 # Benchmark tables and metrics
├── artifacts/                  # Serialized outputs and experiment records (untracked)
└── docs/                       # Technical notes and architecture documentation
```

---

## 7. Environment Setup

### Prerequisites
* Windows 11 / Linux / macOS
* Python 3.12 (64-bit)
* Git

### Installation (Windows PowerShell)

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/nayanojwal0810/Aarohan.git
   cd Aarohan
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies and project:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install --no-deps -e .
   ```

4. **Run verification tests:**
   ```powershell
   pytest -q
   ruff check .
   ruff format --check .
   ```

---

## 8. Project Status

Aarohan is under active development. Project scaffolding, dependency management, configuration validation, logging safeguards, and initial test suites are operational.

---

## 9. License

* Code in this repository is licensed under the [MIT License](LICENSE).
* The underlying benchmark dataset (FreshRetailNet-50K) is subject to its own separate [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
