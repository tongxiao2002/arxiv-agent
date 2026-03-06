"""Logging configuration for Arxiv-Agent."""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Any


def setup_logging(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    console: bool = True,
    file: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 7,
) -> None:
    """
    Configure structured logging with rotation.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Enable console logging
        file: Enable file logging
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Convert log level string to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create log directory if it doesn't exist
    if file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if file:
        log_file = log_path / "arxiv-agent.log"

        # Use RotatingFileHandler for size-based rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

        # Also set up a TimedRotatingFileHandler for daily rotation
        daily_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_path / "arxiv-agent-daily.log",
            when="midnight",
            interval=1,
            backupCount=30,  # Keep 30 days of logs
            encoding="utf-8",
        )
        daily_handler.setLevel(level)
        daily_handler.setFormatter(detailed_formatter)
        daily_handler.suffix = "%Y-%m-%d"
        root_logger.addHandler(daily_handler)

        logging.info(f"Logging configured. Files will be written to {log_path}")

    # Capture warnings from warnings module
    logging.captureWarnings(True)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Example of structured logging with context
class StructuredLogger:
    """Helper for structured logging with context."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def info(self, message: str, **context: Any) -> None:
        """Log info message with structured context."""
        if context:
            extra = " ".join(f"{k}={v}" for k, v in context.items())
            self.logger.info(f"{message} [{extra}]")
        else:
            self.logger.info(message)

    def error(self, message: str, **context: Any) -> None:
        """Log error message with structured context."""
        if context:
            extra = " ".join(f"{k}={v}" for k, v in context.items())
            self.logger.error(f"{message} [{extra}]")
        else:
            self.logger.error(message)

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message with structured context."""
        if context:
            extra = " ".join(f"{k}={v}" for k, v in context.items())
            self.logger.warning(f"{message} [{extra}]")
        else:
            self.logger.warning(message)

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message with structured context."""
        if context:
            extra = " ".join(f"{k}={v}" for k, v in context.items())
            self.logger.debug(f"{message} [{extra}]")
        else:
            self.logger.debug(message)
