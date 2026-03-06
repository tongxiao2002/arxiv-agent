"""Tests for logging configuration."""

import logging
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from arxiv_agent.utils.logging import StructuredLogger, get_logger, setup_logging


def test_setup_logging_console_only():
    """Test logging setup with only console output."""
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    setup_logging(console=True, file=False)

    # Check that console handler was added
    console_handlers = [
        h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(console_handlers) > 0
    # Check that no file handlers were added
    file_handlers = [
        h
        for h in root_logger.handlers
        if isinstance(h, logging.FileHandler)
        or isinstance(h, logging.handlers.RotatingFileHandler)
        or isinstance(h, logging.handlers.TimedRotatingFileHandler)
    ]
    assert len(file_handlers) == 0


def test_setup_logging_file_creation(temp_dir):
    """Test that log files are created in specified directory."""
    log_dir = temp_dir / "logs"

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    setup_logging(log_dir=str(log_dir), console=False, file=True)

    # Log a test message
    test_message = "Test file logging"
    logging.info(test_message)

    # Check that log directory exists
    assert log_dir.exists()
    assert log_dir.is_dir()

    # Check that log files were created
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) > 0

    # Check that our message appears in at least one log file
    message_found = False
    for log_file in log_files:
        if log_file.exists():
            content = log_file.read_text()
            if test_message in content:
                message_found = True
                break

    assert message_found, f"Test message '{test_message}' not found in any log file"


def test_setup_logging_both_console_and_file(temp_dir):
    """Test logging setup with both console and file output."""
    log_dir = temp_dir / "logs"

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    setup_logging(log_dir=str(log_dir), console=True, file=True)

    # Check that both console and file handlers were added
    console_handlers = [
        h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    file_handlers = [
        h
        for h in root_logger.handlers
        if isinstance(h, logging.FileHandler)
        or isinstance(h, logging.handlers.RotatingFileHandler)
        or isinstance(h, logging.handlers.TimedRotatingFileHandler)
    ]
    assert len(console_handlers) > 0
    assert len(file_handlers) > 0

    # Log a test message
    test_message = "Test both console and file logging"
    logging.info(test_message)

    # Check file output
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) > 0

    message_found = False
    for log_file in log_files:
        if log_file.exists():
            content = log_file.read_text()
            if test_message in content:
                message_found = True
                break

    assert message_found, f"Test message not found in log files"


def test_get_logger():
    """Test get_logger function returns proper logger."""
    logger_name = "test.module"
    logger = get_logger(logger_name)

    assert isinstance(logger, logging.Logger)
    assert logger.name == logger_name


def test_structured_logger_basic():
    """Test StructuredLogger basic functionality."""
    import unittest.mock as mock

    logger = mock.Mock(spec=logging.Logger)
    structured = StructuredLogger(logger)

    # Test with context
    structured.info("Test message", user="test", action="login")
    logger.info.assert_called_once_with("Test message [user=test action=login]")

    logger.reset_mock()

    # Test without context
    structured.info("Test message")
    logger.info.assert_called_once_with("Test message")


def test_structured_logger_different_levels():
    """Test StructuredLogger with different log levels."""
    import unittest.mock as mock

    logger = mock.Mock(spec=logging.Logger)
    structured = StructuredLogger(logger)

    # Test each level
    structured.debug("Debug message", data="test")
    logger.debug.assert_called_once_with("Debug message [data=test]")

    logger.reset_mock()
    structured.info("Info message", data="test")
    logger.info.assert_called_once_with("Info message [data=test]")

    logger.reset_mock()
    structured.warning("Warning message", data="test")
    logger.warning.assert_called_once_with("Warning message [data=test]")

    logger.reset_mock()
    structured.error("Error message", data="test")
    logger.error.assert_called_once_with("Error message [data=test]")


def test_logging_levels(temp_dir):
    """Test that different log levels work correctly."""
    log_dir = temp_dir / "logs"

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    setup_logging(log_dir=str(log_dir), log_level="DEBUG", console=False, file=True)

    # Log messages at different levels
    logging.debug("Debug message")
    logging.info("Info message")
    logging.warning("Warning message")
    logging.error("Error message")

    # Check that all messages appear in log file
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) > 0

    content = log_files[0].read_text()
    assert "Debug message" in content
    assert "Info message" in content
    assert "Warning message" in content
    assert "Error message" in content


def test_logging_rotation(temp_dir):
    """Test that log rotation works (basic test)."""
    log_dir = temp_dir / "logs"

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Setup logging with small maxBytes to trigger rotation quickly
    setup_logging(
        log_dir=str(log_dir),
        console=False,
        file=True,
        max_bytes=100,  # Very small to potentially trigger rotation
        backup_count=2,
    )

    # Write enough data to potentially trigger rotation
    for i in range(10):
        logging.info(f"Test message {i}: " + "x" * 20)

    # Check that we have log files
    log_files = list(log_dir.glob("*.log*"))
    assert len(log_files) > 0

    # At minimum we should have the main log file
    main_log = log_dir / "arxiv-agent.log"
    assert main_log.exists()


def test_logging_directory_creation(temp_dir):
    """Test that log directory is created if it doesn't exist."""
    log_dir = temp_dir / "nonexistent" / "logs" / "deep"

    # Ensure directory doesn't exist
    if log_dir.exists():
        shutil.rmtree(log_dir)

    assert not log_dir.exists()

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    setup_logging(log_dir=str(log_dir), console=False, file=True)

    # Directory should now exist
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_logging_format():
    """Test that log messages have expected format."""
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    setup_logging(console=True, file=False)

    # Capture formatted output
    import io
    import sys

    stream = io.StringIO()
    console_handler = logging.StreamHandler(stream)
    console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(console_handler)

    logging.info("Formatted message")

    output = stream.getvalue()
    assert " - " in output  # Should have separators
    assert "INFO" in output
    assert "Formatted message" in output


def test_setup_logging_multiple_calls(temp_dir):
    """Test that calling setup_logging multiple times doesn't duplicate handlers."""
    log_dir = temp_dir / "logs"

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # First call
    setup_logging(log_dir=str(log_dir), console=True, file=True)
    handler_count_first = len(root_logger.handlers)

    # Second call
    setup_logging(log_dir=str(log_dir), console=True, file=True)
    handler_count_second = len(root_logger.handlers)

    # Should have same number of handlers (old ones removed)
    assert handler_count_first == handler_count_second


def test_cleanup():
    """Clean up any global logging state after tests."""
    # Remove all handlers from root logger to avoid affecting other tests
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add a null handler to prevent "No handlers found" warnings
    root_logger.addHandler(logging.NullHandler())
