"""Centralized logging configuration for Aarohan.

Provides standard, structured log formatting with configurable log levels,
module-level loggers, and optional file logging while safely preserving external
handlers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Standardized format: Timestamp | Level | Logger/Module | Message
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

VALID_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.CRITICAL,
    "NOTSET": logging.NOTSET,
}


def _validate_log_level(level: int | str) -> int:
    """Validate and normalize a log level input.

    Args:
        level: Integer or string representation of a logging level.

    Returns:
        Standard logging level integer.

    Raises:
        ValueError: If string does not match a valid standard logging level.
    """
    if isinstance(level, str):
        upper_level = level.strip().upper()
        if upper_level not in VALID_LOG_LEVELS:
            valid_names = ", ".join(sorted(set(VALID_LOG_LEVELS.keys())))
            raise ValueError(
                f"Invalid logging level '{level}'. Must be one of: {valid_names}"
            )
        return VALID_LOG_LEVELS[upper_level]
    if isinstance(level, int):
        return level
    raise ValueError(
        f"Logging level must be a string or integer, got {type(level).__name__}"
    )


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Configure the root logger with Aarohan handlers and formatting.

    Preserves pre-existing handlers configured by pytest, third-party libraries,
    or the consuming application. Replaces only handlers previously registered by
    Aarohan to prevent handler duplication on repeated calls.

    Args:
        level: Minimum logging level (e.g., logging.INFO, "INFO", "DEBUG").
        log_file: Optional file path to write log output.
        log_format: Format string for log messages.
        date_format: Format string for timestamps.

    Returns:
        The configured root logger.

    Raises:
        ValueError: If an invalid logging level string is provided.
    """
    numeric_level = _validate_log_level(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove only handlers previously created and tagged by Aarohan
    handlers_to_remove = [
        h for h in root_logger.handlers if getattr(h, "_aarohan_managed", False)
    ]
    for h in handlers_to_remove:
        root_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    console_handler._aarohan_managed = True
    root_logger.addHandler(console_handler)

    # File Handler (optional)
    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        file_handler._aarohan_managed = True
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger.

    Args:
        name: Name of the module or stage (typically __name__).

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)
