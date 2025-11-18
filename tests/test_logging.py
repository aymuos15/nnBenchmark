"""
Tests for src.logging module.
Tests logger setup, helpers, and system info logging.
"""

from __future__ import annotations

import os
import time
from io import StringIO

from loguru import logger

from src.logging.helpers import log_and_print, log_header, log_separator
from src.logging.setup import (
    setup_logger,
    setup_test_logger,
    setup_train_logger,
    setup_verbose_logger,
)
from src.logging.system import (
    log_system_info,
)


def _wait_for_log_message(log_path: str, message: str, timeout: float = 5.0) -> bool:
    """Poll for a message in a log file with timeout.

    Args:
        log_path: Path to the log file
        message: Message to search for
        timeout: Maximum time to wait in seconds

    Returns:
        True if message found, False if timeout exceeded
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                content = f.read()
            if message in content:
                return True
        time.sleep(0.01)
    return False


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_setup_logger_creates_file(self, temp_dir: str) -> None:
        """Test that setup_logger creates a log file."""
        log_name = "test_logger"
        log = setup_logger(temp_dir, log_name=log_name)
        log.info("Test message")

        # Give async logger time to flush
        time.sleep(0.2)

        # Verify log file exists in logs/ subdirectory
        log_path = os.path.join(temp_dir, "logs", f"{log_name}.log")
        assert os.path.exists(log_path)

    def test_setup_logger_writes_message(self, temp_dir: str) -> None:
        """Test that logger writes messages to file."""
        log_name = "test_write"
        log = setup_logger(temp_dir, log_name=log_name)

        test_message = "Test log message"
        log.info(test_message)

        # Poll for message in log file (more reliable than fixed sleep)
        log_path = os.path.join(temp_dir, "logs", f"{log_name}.log")
        message_found = _wait_for_log_message(log_path, test_message)

        assert message_found, f"Message '{test_message}' not found in log file"

    def test_setup_logger_always_appends(self, temp_dir: str) -> None:
        """Test that setup_logger always appends to existing log file."""
        log_name = "test_append"

        # Create initial log
        log1 = setup_logger(temp_dir, log_name=log_name)
        log1.info("Initial message")

        log_path = os.path.join(temp_dir, "logs", f"{log_name}.log")
        assert _wait_for_log_message(log_path, "Initial message")

        # Create new logger - should append, not overwrite
        logger.remove()  # Clear handlers before setup
        log2 = setup_logger(temp_dir, log_name=log_name)
        log2.info("New message")

        assert _wait_for_log_message(log_path, "New message")

        with open(log_path, "r") as f:
            content = f.read()

        # Should have both messages since logger always appends
        assert "New message" in content
        assert "Initial message" in content

    def test_setup_logger_appends_to_existing_file(self, temp_dir: str) -> None:
        """Test that setup_logger appends to existing log file."""
        log_name = "test_append"

        # Create initial log
        log1 = setup_logger(temp_dir, log_name=log_name)
        log1.info("Initial message")

        log_path = os.path.join(temp_dir, "logs", f"{log_name}.log")
        assert _wait_for_log_message(log_path, "Initial message")

        # Clear and create new logger (setup_logger always appends)
        logger.remove()
        log2 = setup_logger(temp_dir, log_name=log_name)
        log2.info("Appended message")

        assert _wait_for_log_message(log_path, "Appended message")

        with open(log_path, "r") as f:
            content = f.read()

        # Should have both messages since logger appends by default
        assert "Initial message" in content
        assert "Appended message" in content


class TestSetupTrainLogger:
    """Tests for setup_train_logger function."""

    def test_setup_train_logger_creates_train_log(self, temp_dir: str) -> None:
        """Test that setup_train_logger creates train.log file."""
        log = setup_train_logger(temp_dir)

        log.info("Training started")

        log_path = os.path.join(temp_dir, "logs", "train.log")
        assert _wait_for_log_message(log_path, "Training started")

    def test_setup_train_logger_appends_to_existing(self, temp_dir: str) -> None:
        """Test setup_train_logger appends to existing log file."""
        # First logger
        log1 = setup_train_logger(temp_dir)
        log1.info("First run")

        log_path = os.path.join(temp_dir, "logs", "train.log")
        assert _wait_for_log_message(log_path, "First run")

        # Second logger (setup_logger always appends by default)
        logger.remove()
        log2 = setup_train_logger(temp_dir)
        log2.info("Second run")

        assert _wait_for_log_message(log_path, "Second run")

        with open(log_path, "r") as f:
            content = f.read()

        # Both messages should be present (logger appends by default)
        assert "First run" in content
        assert "Second run" in content


class TestSetupTestLogger:
    """Tests for setup_test_logger function."""

    def test_setup_test_logger_creates_test_log(self, temp_dir: str) -> None:
        """Test that setup_test_logger creates test.log file."""
        log = setup_test_logger(temp_dir)

        log.info("Testing started")

        log_path = os.path.join(temp_dir, "logs", "test.log")
        assert _wait_for_log_message(log_path, "Testing started")


class TestSetupVerboseLogger:
    """Tests for setup_verbose_logger function."""

    def test_setup_verbose_logger_default(self) -> None:
        """Test setup_verbose_logger with default level."""
        import sys

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            setup_verbose_logger(level="DEBUG")
            logger.debug("Debug message")
            logger.info("Info message")

            output = sys.stderr.getvalue()
            # Both messages should be captured
            assert "Debug message" in output
            assert "Info message" in output
        finally:
            sys.stderr = old_stderr
            logger.remove()

    def test_setup_verbose_logger_warning_level(self) -> None:
        """Test setup_verbose_logger with WARNING level."""
        import sys

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            setup_verbose_logger(level="WARNING")
            logger.info("Info message")
            logger.warning("Warning message")

            output = sys.stderr.getvalue()
            # Only warning and above should be captured
            assert "Warning message" in output
            assert "Info message" not in output
        finally:
            sys.stderr = old_stderr
            logger.remove()

    def test_setup_verbose_logger_custom_format(self) -> None:
        """Test setup_verbose_logger with custom format string."""
        import sys

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            custom_format = "{level} - {message}"
            setup_verbose_logger(level="INFO", format_string=custom_format)
            logger.info("Test message")

            output = sys.stderr.getvalue()
            assert "Test message" in output
        finally:
            sys.stderr = old_stderr
            logger.remove()


class TestLogAndPrint:
    """Tests for log_and_print helper function."""

    def test_log_and_print_info(self, temp_dir: str, capsys) -> None:
        """Test log_and_print with INFO level."""
        log = setup_logger(temp_dir, log_name="test")

        log_and_print(log, "Test message", level="INFO")

        # Check console output
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_log_and_print_warning(self, temp_dir: str, capsys) -> None:
        """Test log_and_print with WARNING level."""
        log = setup_logger(temp_dir, log_name="test2")

        log_and_print(log, "Warning message", level="WARNING")

        captured = capsys.readouterr()
        assert "Warning message" in captured.out

    def test_log_and_print_error(self, temp_dir: str, capsys) -> None:
        """Test log_and_print with ERROR level."""
        log = setup_logger(temp_dir, log_name="test3")

        log_and_print(log, "Error message", level="ERROR")

        captured = capsys.readouterr()
        assert "Error message" in captured.out

    def test_log_and_print_invalid_level_defaults_to_info(
        self, temp_dir: str, capsys
    ) -> None:
        """Test log_and_print with invalid level defaults to INFO."""
        log = setup_logger(temp_dir, log_name="test4")

        log_and_print(log, "Default message", level="INVALID")

        captured = capsys.readouterr()
        assert "Default message" in captured.out


class TestLogSeparator:
    """Tests for log_separator helper function."""

    def test_log_separator_default(self, temp_dir: str, capsys) -> None:
        """Test log_separator with default parameters."""
        log = setup_logger(temp_dir, log_name="sep")

        log_separator(log)

        captured = capsys.readouterr()
        assert "=" in captured.out

    def test_log_separator_custom_char(self, temp_dir: str, capsys) -> None:
        """Test log_separator with custom character."""
        log = setup_logger(temp_dir, log_name="sep2")

        log_separator(log, char="-", width=10)

        captured = capsys.readouterr()
        assert "----------" in captured.out

    def test_log_separator_no_print(self, temp_dir: str, capsys) -> None:
        """Test log_separator with print_too=False."""
        log = setup_logger(temp_dir, log_name="sep3")

        log_separator(log, print_too=False)

        captured = capsys.readouterr()
        # Should not print to console
        assert captured.out == ""


class TestLogHeader:
    """Tests for log_header helper function."""

    def test_log_header_default(self, temp_dir: str, capsys) -> None:
        """Test log_header with default parameters."""
        log = setup_logger(temp_dir, log_name="hdr")

        log_header(log, "Test Header")

        captured = capsys.readouterr()
        assert "Test Header" in captured.out
        assert "=" in captured.out

    def test_log_header_custom_char(self, temp_dir: str, capsys) -> None:
        """Test log_header with custom character."""
        log = setup_logger(temp_dir, log_name="hdr2")

        log_header(log, "Custom Header", char="-")

        captured = capsys.readouterr()
        assert "Custom Header" in captured.out
        assert "-" in captured.out

    def test_log_header_format(self, temp_dir: str, capsys) -> None:
        """Test log_header has proper sandwich format."""
        log = setup_logger(temp_dir, log_name="hdr3")

        log_header(log, "Header Message")

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        # Should have separator, message, separator
        assert len(lines) >= 3
        assert lines[0].startswith("=")
        assert "Header Message" in lines[1]
        assert lines[2].startswith("=")


class TestLogSystemInfo:
    """Tests for log_system_info function."""

    def test_log_system_info_cpu(self, temp_dir: str) -> None:
        """Test log_system_info logs system information on CPU."""
        import torch

        log = setup_logger(temp_dir, log_name="sysinfo")
        device = torch.device("cpu")

        log_system_info(log, device)

        # Give a moment for async logging to complete
        import time

        time.sleep(0.1)

        log_path = os.path.join(temp_dir, "logs", "sysinfo.log")
        assert _wait_for_log_message(log_path, "SYSTEM INFO")

        with open(log_path, "r") as f:
            content = f.read()

        # Should log system info
        assert "SYSTEM INFO" in content
        assert "Python version" in content
        assert "PyTorch version" in content
        assert "Platform" in content
