"""Unit tests verifying project setup, package imports, configuration, and logging."""

from pathlib import Path

import pytest
import yaml

import aarohan
from aarohan.config import get_project_root, load_config, load_yaml
from aarohan.logging_config import get_logger, setup_logging


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

    training_cfg = load_config("training")
    assert training_cfg["random_seed"] == 42
    assert training_cfg["model_type"] == "LightGBM"
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


def test_logging_setup():
    """Verify logger setup, formatting, and handler instantiation."""
    logger = setup_logging(level="DEBUG")
    assert logger.level == 10  # logging.DEBUG

    module_logger = get_logger("aarohan.test")
    assert module_logger.name == "aarohan.test"
    # Logger can emit messages without error
    module_logger.info("Test message for logging verification")
