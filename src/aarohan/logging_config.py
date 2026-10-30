"""Centralized logging configuration for Aarohan.

Provides standard, structured log formatting with configurable log levels,
module-level loggers, and optional file logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Standardized format: Timestamp | Level | Logger/Module | Message
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Configure the root logger with standard handlers and formatting.

    Args:
        level: Minimum logging level (e.g., logging.INFO or "INFO").
        log_file: Optional file path to write log output.
        log_format: Format string for log messages.
        date_format: Format string for timestamps.

    Returns:
        The configured root logger.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logging is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (optional)
    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
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
