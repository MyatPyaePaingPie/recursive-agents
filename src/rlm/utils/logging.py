"""Logging configuration for the RLM system.

This module provides structured logging using loguru with support for
request tracking, JSON output, and log rotation.

Example:
    >>> from rlm.utils.logging import setup_logging, get_logger
    >>> setup_logging(level="DEBUG")
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing query", query="test", depth=0)
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Remove default handler
logger.remove()

# Track if logging has been configured
_logging_configured = False


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    json_format: bool = False,
    rotation: str = "100 MB",
    retention: str = "10 days",
) -> None:
    """Configure logging for the RLM system.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        json_format: If True, output logs as JSON
        rotation: When to rotate log files (e.g., "100 MB", "1 day")
        retention: How long to keep old log files

    Example:
        >>> setup_logging(level="DEBUG", log_file="logs/rlm.log")
    """
    global _logging_configured

    # Remove any existing handlers
    logger.remove()

    # Console format
    if json_format:
        console_format = "{message}"
    else:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
        serialize=json_format,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="gz",
            serialize=json_format,
        )

    _logging_configured = True
    logger.debug(f"Logging configured with level={level}")


def get_logger(name: str | None = None) -> Any:
    """Get a logger instance with optional context binding.

    Args:
        name: Optional name for the logger (typically __name__)

    Returns:
        A loguru logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    global _logging_configured

    # Auto-configure if not done
    if not _logging_configured:
        setup_logging()

    if name:
        return logger.bind(name=name)
    return logger


class LogContext:
    """Context manager for adding context to log messages.

    This allows adding request-specific information to all log
    messages within a context block.

    Example:
        >>> with LogContext(request_id="abc123", depth=0):
        ...     logger.info("Processing query")
        # Output includes request_id and depth
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with context key-value pairs."""
        self.context = kwargs
        self._token: Any = None

    def __enter__(self) -> "LogContext":
        """Enter the context and bind values."""
        self._token = logger.contextualize(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context."""
        pass
