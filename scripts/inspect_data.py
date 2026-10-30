"""CLI script to execute Aarohan data ingestion and validation pipeline.

Inspects raw dataset Parquet files (train.parquet, eval.parquet), verifies schema,
integrity, continuity, hourly sequences, anomalies, and cohort structure.
Generates machine-readable tables and human-readable reports.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from aarohan.config import load_config
from aarohan.data.validate import run_validation_pipeline
from aarohan.logging_config import setup_logging


def get_process_memory_stats(process: psutil.Process) -> dict[str, Any]:
    """Return process memory statistics honestly labeled by platform."""
    mem_info = process.memory_info()
    end_rss_mb = mem_info.rss / (1024 * 1024)
    peak_wset = getattr(mem_info, "peak_wset", None)
    if peak_wset is not None:
        peak_mb = peak_wset / (1024 * 1024)
        peak_label = "Peak Process Working Set (RSS)"
    else:
        peak_mb = end_rss_mb
        peak_label = "Process RSS (end-of-run)"
    return {
        "end_rss_mb": end_rss_mb,
        "peak_mb": peak_mb,
        "peak_label": peak_label,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aarohan Data Ingestion and Validation Pipeline CLI"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to configs/data.yaml)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Custom directory to store validation reports (default: reports)",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help=(
            "Custom path to store dataset manifest "
            "(default: artifacts/metadata/dataset_manifest.json)"
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to write log output file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main CLI execution routine."""
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)
    logger = logging.getLogger("aarohan.scripts.inspect_data")

    logger.info("Starting Aarohan data inspection and validation")
    start_time = time.perf_counter()
    process = psutil.Process()

    config = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            logger.error("Configuration file not found: %s", config_path)
            return 1
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = load_config("data")

    try:
        summary = run_validation_pipeline(
            config=config,
            reports_dir=args.reports_dir,
            manifest_path=args.manifest_path,
        )
    except Exception as exc:
        logger.exception("Validation pipeline encountered an unhandled error: %s", exc)
        return 1

    elapsed = time.perf_counter() - start_time
    mem_stats = get_process_memory_stats(process)

    # Console summary output
    overall_status = summary.get("overall_status", "UNKNOWN")
    train = summary.get("train", {})
    eval_part = summary.get("eval", {})
    cross = summary.get("cross_partition", {})
    cohort = summary.get("evaluation_cohort", {})

    print("\n" + "=" * 70)
    print("           AAROHAN DATA VALIDATION EXECUTION SUMMARY           ")
    print("=" * 70)
    print(f"Overall Status:            {overall_status}")
    print(f"Execution Duration:        {elapsed:.2f} seconds")
    peak_hdr = f"{mem_stats['peak_label']}:"
    print(f"{peak_hdr:<32}{mem_stats['peak_mb']:.2f} MB")
    print(f"{'Process RSS (end-of-run):':<32}{mem_stats['end_rss_mb']:.2f} MB")
    print("-" * 70)

    # Partition breakdowns
    for name, part in [
        ("Training Partition", train),
        ("Evaluation Partition", eval_part),
    ]:
        meta = part.get("metadata", {})
        status = part.get("status", "N/A")
        print(f"[{name}] -> Status: {status}")
        if meta:
            print(f"  File:                    {meta.get('path', 'N/A')}")
            rows_cols = f"{meta.get('num_rows', 0):,} x {meta.get('num_columns', 0)}"
            print(f"  Rows x Columns:          {rows_cols}")
            print(f"  Unique Series:           {meta.get('num_series', 0):,}")
            dt_range = (
                f"{meta.get('min_dt', 'N/A')} to {meta.get('max_dt', 'N/A')} "
                f"({meta.get('num_days', 0)} days)"
            )
            print(f"  Date Range:              {dt_range}")
            sha_val = meta.get("sha256", "N/A")
            sha_short = f"{sha_val[:16]}..." if len(sha_val) > 16 else sha_val
            sha_check = next(
                (
                    c
                    for c in part.get("checks", [])
                    if c.get("name") == "sha256_checksum"
                ),
                None,
            )
            sha_match = (
                " (match)"
                if sha_check and sha_check.get("status") == "PASS"
                else (
                    " (mismatch)"
                    if sha_check and sha_check.get("status") == "FAIL"
                    else ""
                )
            )
            print(f"  SHA-256:                 {sha_short}{sha_match}")
        checks = part.get("checks", [])
        passed = sum(1 for c in checks if c["status"] == "PASS")
        warnings = sum(1 for c in checks if c["status"] == "WARNING")
        failed = sum(1 for c in checks if c["status"] == "FAIL")
        print(
            f"  Checks:                  {passed} PASS, "
            f"{warnings} WARNING, {failed} FAIL"
        )
        print("-" * 70)

    # Cross-partition
    cross_checks = cross.get("checks", [])
    cross_passed = sum(1 for c in cross_checks if c["status"] == "PASS")
    cross_failed = sum(1 for c in cross_checks if c["status"] == "FAIL")
    series_check = next(
        (
            c
            for c in cross_checks
            if c.get("name") == "cross_partition_series_consistency"
        ),
        {},
    )
    temp_check = next(
        (c for c in cross_checks if c.get("name") == "temporal_partition_separation"),
        {},
    )
    print(f"[Cross-Partition Integrity] -> Status: {cross.get('status', 'N/A')}")
    if series_check:
        print(f"  Series Overlap:          {series_check.get('actual', 'N/A')}")
    if temp_check:
        print(f"  Temporal Boundary:       {temp_check.get('actual', 'N/A')}")
    print(f"  Checks:                  {cross_passed} PASS, {cross_failed} FAIL")
    print("-" * 70)

    # Cohort breakdown
    print("[Primary Evaluation Cohort (stock_hour6_22_cnt == 0)]")
    print(f"  Total Evaluation Rows:   {cohort.get('total_rows', 0):,}")
    no_st = (
        f"{cohort.get('daytime_no_stockout_rows', 0):,} "
        f"({cohort.get('pct_daytime_no_stockout', 0)}%)"
    )
    with_st = (
        f"{cohort.get('daytime_with_stockout_rows', 0):,} "
        f"({cohort.get('pct_daytime_with_stockout', 0)}%)"
    )
    print(f"  Daytime No-Stockout:     {no_st}")
    print(f"  Daytime Stockout:        {with_st}")
    print("-" * 70)

    # Deliverables generated
    print("Deliverables Generated:")
    print("  - reports/tables/data_validation.json")
    print("  - reports/data_profile.md")
    print("  - docs/data_dictionary.md")
    print("  - artifacts/metadata/dataset_manifest.json")
    print("=" * 70)

    if overall_status == "FAIL":
        logger.error("Data validation pipeline failed with status FAIL.")
        return 1

    if overall_status == "WARNING":
        logger.warning(
            "Data validation pipeline completed with status WARNING "
            "(documented anomalies preserved)."
        )
    else:
        logger.info("Data validation pipeline completed successfully with status PASS.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
