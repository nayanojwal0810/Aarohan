# Aarohan: Intelligent Demand Planning for Perishable Retail

Aarohan is an end-to-end demand forecasting and inventory replenishment system designed specifically for perishable retail environments.

---

## 1. Business Problem

In retail supply chains—particularly for fresh and perishable products—observed sales records do not always reflect true consumer demand. When an item runs out of stock, recorded sales drop to zero regardless of how many customers intend to purchase it.

Standard time-series forecasting models trained directly on raw sales data risk learning supply constraints rather than underlying consumer demand. In grocery retail, this causes a compounding feedback loop: under-forecasting leads to insufficient replenishment, which causes recurring stockouts, further depressing recorded sales.

Aarohan addresses this challenge by explicitly accounting for stockout censoring, recovering latent consumer demand, and translating uncertainty-aware multi-day forecasts into operational replenishment orders.

---

## 2. Project Objectives

Aarohan executes a connected five-stage operational workflow:

1. **Data Ingestion & Integrity Audit:** Ingest high-volume historical transaction panels, validating schema consistency, temporal continuity, and stockout indicator fidelity.
2. **Availability & Stockout Profiling:** Quantify the frequency, duration, and sales impact of partial-day and full-day stockout events across store-product hierarchies.
3. **Latent Demand Estimation:** Model and reconstruct customer demand obscured by stockouts using principled censored demand imputation.
4. **Multi-Horizon Demand Forecasting:** Generate robust 7-day store-product demand forecasts using global gradient-boosted trees (LightGBM) trained on leakage-safe temporal features.
5. **Operational Replenishment Guidance:** Quantify prediction uncertainty and transform demand distributions into transparent safety-stock and replenishment recommendations.

---

## 3. Dataset & Attribution

Aarohan is built upon the **FreshRetailNet-50K** dataset, a large-scale, real-world retail demand benchmark released by Dingdong Inc.

* **Dataset Source:** [FreshRetailNet-50K on Hugging Face](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
* **Citation & Paper:** *FreshRetailNet-50K: A Large-Scale Real-World Dataset for Stockout-Annotated Demand Forecasting in Fresh Retail* ([arXiv:2505.16319](https://arxiv.org/abs/2505.16319))
* **Dataset License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
* **Data Scale:** 4,850,000 daily observations across 50,000 store-product series, 898 stores, 865 perishable SKUs, and hourly stockout annotations.
* **Important Note on Scale:** As documented by the dataset authors, sales quantities (`sale_amount` and `hours_sale`) are provided after global normalization by a proprietary scaling coefficient to protect sensitive commercial metrics. Original integer physical unit counts cannot be reconstructed; all forecasts represent normalized demand index units.

---

## 4. Technology Stack

Aarohan is implemented in Python 3.12 with a focused, production-grade dependency stack:

* **Language:** Python 3.12 (64-bit)
* **Data Manipulation & Analysis:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [PyArrow](https://arrow.apache.org/pyarrow/)
* **Out-of-Core Data Querying:** [DuckDB](https://duckdb.org/)
* **Scientific Computing & ML:** [SciPy](https://scipy.org/), [Scikit-learn](https://scikit-learn.org/)
* **Forecasting Engine:** [LightGBM](https://lightgbm.readthedocs.io/)
* **Configuration & Logging:** [PyYAML](https://pyyaml.org/), Python Standard Library `logging`
* **System Telemetry:** [psutil](https://github.com/giampaolo/psutil)
* **Testing & Quality Assurance:** [pytest](https://docs.pytest.org/), [Ruff](https://docs.astral.sh/ruff/)

---

## 5. Repository Structure

```text
Aarohan/
├── README.md                   # Project overview and documentation
├── LICENSE                     # MIT License for codebase
├── pyproject.toml              # Build configuration and dependency declarations
├── requirements.txt            # Locked tested runtime and dev dependencies
├── .gitignore                  # Git exclusion rules
│
├── configs/                    # Declarative configuration files
│   ├── data.yaml               # Dataset paths, entity keys, date bounds
│   ├── features.yaml           # Lag and rolling feature parameters
│   └── training.yaml           # Model hyperparameters and validation folds
│
├── data/
│   ├── raw/                    # Original read-only dataset (untracked)
│   ├── interim/                # Intermediate transformed datasets (untracked)
│   └── processed/              # Feature matrices ready for modeling (untracked)
│
├── src/
│   └── aarohan/                # Core Python package
│       ├── __init__.py         # Package root and version definition
│       ├── config.py           # Configuration loader and path resolver
│       └── logging_config.py   # Centralized structured logging setup
│
├── notebooks/                  # Interactive exploratory notebooks
├── scripts/                    # Command-line execution scripts
├── tests/                      # Automated unit and integration tests
├── reports/                    # Generated project evaluation reports
│   ├── figures/                # Visualizations and diagnostic plots
│   └── tables/                 # Benchmark metrics and census summaries
├── artifacts/                  # Serialized models and experiment metadata (untracked)
└── docs/                       # Architecture diagrams and technical notes
```

---

## 6. Environment Setup

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

3. **Install the package in editable mode with development tools:**
   ```powershell
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

4. **Verify installation and run tests:**
   ```powershell
   pytest -q
   ruff check .
   ruff format --check .
   ```

---

## 7. Project Status

Aarohan is currently under active development. Core foundation, configuration management, logging infrastructure, and test harnesses are established. Subsequent pipeline stages are being implemented incrementally.

---

## 8. License

* The software source code in this repository is licensed under the [MIT License](LICENSE).
* The underlying benchmark dataset (FreshRetailNet-50K) is subject to its own separate [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
