"""Configuration management for Aarohan.

Provides centralized, safe loading of project configuration files with path
resolution relative to the project root and strict validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Assumes this file is located at <project_root>/src/aarohan/config.py.
    """
    return Path(__file__).resolve().parent.parent.parent


def resolve_project_path(
    relative_path: str | Path, base_dir: Path | None = None
) -> Path:
    """Resolve a relative path against the project root or specified base directory.

    Args:
        relative_path: Path relative to project root.
        base_dir: Optional base directory. Defaults to project root.

    Returns:
        Absolute Path object.
    """
    base = base_dir if base_dir is not None else get_project_root()
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Safely load and parse a YAML file.

    Args:
        file_path: Path to YAML file. Can be relative to project root or absolute.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If the file is empty or cannot be parsed as a valid mapping.
        yaml.YAMLError: If the YAML syntax is malformed.
    """
    resolved_path = resolve_project_path(file_path)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {resolved_path} "
            f"(resolved from '{file_path}')"
        )

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Specified configuration path is not a regular file: {resolved_path}"
        )

    with open(resolved_path, encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(
                f"Failed to parse YAML configuration at {resolved_path}: {exc}"
            ) from exc

    if data is None:
        raise ValueError(f"Configuration file is empty: {resolved_path}")

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration file must contain a top-level mapping/dictionary, "
            f"got {type(data).__name__} from {resolved_path}"
        )

    return data


def load_config(config_name: str) -> dict[str, Any]:
    """Load a named configuration file from the project's configs/ directory.

    Args:
        config_name: Name of config file, with or without .yaml extension
            (e.g., 'data' or 'data.yaml').

    Returns:
        Parsed configuration dictionary.
    """
    if not config_name.endswith((".yaml", ".yml")):
        config_name = f"{config_name}.yaml"

    config_path = get_project_root() / "configs" / config_name
    return load_yaml(config_path)
