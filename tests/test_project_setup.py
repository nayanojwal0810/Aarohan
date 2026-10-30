"""Unit tests verifying project setup, package imports, configuration, and logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

import aarohan
from aarohan.config import get_project_root, load_config, load_yaml
from aarohan.logging_config import get_logger, setup_logging


@pytest.fixture(autouse=True)
def isolate_logging_environment():
    """Isolate logging state for every test, preserving pytest and external handlers.

    Saves initial root logger level and external handlers before each test.
    During teardown, safely removes and closes any Aarohan-managed handlers
    created during test execution, and restores the original root level.
    """
    root_logger = logging.getLogger()
    initial_level = root_logger.level
    initial_handlers = [
        h for h in root_logger.handlers if not getattr(h, "_aarohan_managed", False)
    ]

    yield root_logger

    # Teardown: close and remove only Aarohan-managed handlers
    for handler in list(root_logger.handlers):
        if getattr(handler, "_aarohan_managed", False):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    # Ensure pre-existing external handlers remain attached
    for handler in initial_handlers:
        if handler not in root_logger.handlers:
            root_logger.addHandler(handler)

    # Restore initial root level
    root_logger.setLevel(initial_level)


def test_package_metadata():
    """Verify package name and version definition."""
    assert aarohan.__version__ == "0.1.0"


def test_project_root_resolution():
    """Verify project root path exists and contains configs directory."""
    root = get_project_root()
    assert root.exists()
    assert (root / "configs").is_dir()
    assert (root / "src" / "aarohan").is_dir()


def test_load_valid_configs():
    """Verify that all default configurations load successfully with expected keys."""
    data_cfg = load_config("data")
    assert "raw_train_path" in data_cfg
    assert "raw_eval_path" in data_cfg
    assert "date_boundaries" in data_cfg
    assert data_cfg["expected_rows"]["train"] == 4500000

    features_cfg = load_config("features")
    assert features_cfg["forecast_horizon_days"] == 7
    assert 28 in features_cfg["lags"]
    assert 28 in features_cfg["rolling_mean_windows"]
    assert "day_of_week" in features_cfg["known_calendar_features"]
    assert "activity_flag" in features_cfg["conditional_future_covariates"]
    assert "discount" in features_cfg["conditional_future_covariates"]
    assert "weather" in features_cfg["conditional_future_covariates"]

    training_cfg = load_config("training")
    assert training_cfg["random_seed"] == 42
    assert training_cfg["model_type"] == "LightGBM"
    assert training_cfg["reproducibility"]["n_jobs"] == 2
    assert len(training_cfg["validation_folds"]) == 4


def test_missing_config_raises_file_not_found(tmp_path: Path):
    """Verify loading non-existent configuration raises FileNotFoundError."""
    non_existent = tmp_path / "non_existent_config.yaml"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_yaml(non_existent)
    assert "not found" in str(exc_info.value).lower()


def test_invalid_yaml_raises_error(tmp_path: Path):
    """Verify that malformed YAML content raises a yaml.YAMLError."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [unclosed list", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_yaml(bad_yaml)


def test_empty_config_raises_value_error(tmp_path: Path):
    """Verify that an empty YAML file raises ValueError."""
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_yaml(empty_yaml)
    assert "empty" in str(exc_info.value).lower()


def test_setup_logging_preserves_external_handlers():
    """Verify that existing external root handlers remain untouched by setup_logging."""
    root_logger = logging.getLogger()
    external_handler = logging.NullHandler()
    root_logger.addHandler(external_handler)

    try:
        setup_logging(level="INFO")
        assert external_handler in root_logger.handlers
    finally:
        root_logger.removeHandler(external_handler)
        external_handler.close()


def test_setup_logging_no_duplicates_on_repeated_calls():
    """Verify that calling setup_logging repeatedly does not duplicate handlers."""
    root_logger = logging.getLogger()
    setup_logging(level="INFO")
    count_first = sum(
        1 for h in root_logger.handlers if getattr(h, "_aarohan_managed", False)
    )
    assert count_first == 1

    setup_logging(level="DEBUG")
    count_second = sum(
        1 for h in root_logger.handlers if getattr(h, "_aarohan_managed", False)
    )
    assert count_second == 1


def test_setup_logging_invalid_level_raises_value_error():
    """Verify that an invalid logging level string raises a clear ValueError."""
    with pytest.raises(ValueError, match="Invalid logging level"):
        setup_logging(level="SUPER_VERBOSE")


def test_setup_logging_file_output(tmp_path: Path):
    """Verify that optional file logging writes structured log messages."""
    log_file = tmp_path / "test_run.log"
    setup_logging(level="INFO", log_file=log_file)

    test_logger = get_logger("aarohan.test_file_output")
    test_msg = "Verification message for file logging test"
    test_logger.info(test_msg)

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert test_msg in content
    assert "aarohan.test_file_output" in content
    assert "INFO" in content


def test_logging_isolation_leaves_no_residual_aarohan_handlers():
    """Verify that setup_logging cleanup leaves no dangling Aarohan handlers."""
    root_logger = logging.getLogger()
    original_level = root_logger.level

    setup_logging(level="DEBUG")
    assert root_logger.level == logging.DEBUG
    aarohan_handlers = [
        h for h in root_logger.handlers if getattr(h, "_aarohan_managed", False)
    ]
    assert len(aarohan_handlers) >= 1

    # Simulate teardown cleanup
    for h in aarohan_handlers:
        root_logger.removeHandler(h)
        h.close()
    root_logger.setLevel(original_level)

    assert not any(getattr(h, "_aarohan_managed", False) for h in root_logger.handlers)
    assert root_logger.level == original_level


def test_readme_dataset_attribution_and_no_paper_citation():
    """Verify README retains attribution and license without paper references."""
    readme_path = get_project_root() / "README.md"
    assert readme_path.is_file()
    content = readme_path.read_text(encoding="utf-8")

    # Required dataset attribution
    assert "FreshRetailNet-50K" in content
    assert "Dingdong Inc" in content
    assert "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K" in content
    assert "CC BY 4.0" in content

    # Excluded research paper citation and arXiv references
    assert "arxiv.org" not in content.lower()
    assert "2505.16319" not in content
    assert "research paper" not in content.lower()
    paper_title = "latent demand recovery and forecasting in fresh retail"
    assert paper_title not in content.lower()
