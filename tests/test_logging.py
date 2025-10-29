"""
Tests for src.logging module.
Tests logger setup, helpers, and system info logging.
"""

from __future__ import annotations

import os
import time
from io import StringIO

import pytest
from loguru import logger

from src.logging.helpers import log_and_print, log_header, log_separator
from src.logging.setup import (
    setup_logger,
    setup_test_logger,
    setup_train_logger,
    setup_verbose_logger,
)
from src.logging.system import (
    get_gpu_memory_string,
    log_gpu_memory,
    log_system_info,
)


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_setup_logger_creates_file(self, temp_dir: str) -> None:
        """Test that setup_logger creates a log file."""
        log_name = "test_logger"
        log = setup_logger(temp_dir, log_name=log_name)
        log.info("Test message")

        # Give async logger time to flush
        time.sleep(0.2)

        # Verify log file exists
        log_path = os.path.join(temp_dir, f"{log_name}.log")
        assert os.path.exists(log_path)

    def test_setup_logger_writes_message(self, temp_dir: str) -> None:
        """Test that logger writes messages to file."""
        log_name = "test_write"
        log = setup_logger(temp_dir, log_name=log_name)

        test_message = "Test log message"
        log.info(test_message)

        # Give async logger time to flush
        time.sleep(0.2)

        # Verify message is in file
        log_path = os.path.join(temp_dir, f"{log_name}.log")
        with open(log_path, "r") as f:
            content = f.read()

        assert test_message in content

    def test_setup_logger_resume_false_deletes_file(self, temp_dir: str) -> None:
        """Test that resume=False deletes existing log file before setup."""
        log_name = "test_overwrite"

        # Create initial log
        log1 = setup_logger(temp_dir, log_name=log_name)
        log1.info("Initial message")

        time.sleep(0.2)

        log_path = os.path.join(temp_dir, f"{log_name}.log")
        assert os.path.exists(log_path)

        # Create new logger with resume=False (default)
        logger.remove()  # Clear handlers before setup
        log2 = setup_logger(temp_dir, log_name=log_name)
        log2.info("New message")

        time.sleep(0.2)

        with open(log_path, "r") as f:
            content = f.read()

        # Should only have new message, not initial
        assert "New message" in content
        assert "Initial message" not in content

    def test_setup_logger_resume_true_keeps_file(self, temp_dir: str) -> None:
        """Test that resume=True keeps existing log file."""
        log_name = "test_append"

        # Create initial log
        log1 = setup_logger(temp_dir, log_name=log_name)
        log1.info("Initial message")

        time.sleep(0.2)

        log_path = os.path.join(temp_dir, f"{log_name}.log")
        assert os.path.exists(log_path)

        # Clear and create new logger with resume=True
        logger.remove()
        log2 = setup_logger(temp_dir, log_name=log_name, resume=True)
        log2.info("Appended message")

        time.sleep(0.2)

        with open(log_path, "r") as f:
            content = f.read()

        # Should have both messages since we didn't delete the file
        assert "Initial message" in content
        assert "Appended message" in content


class TestSetupTrainLogger:
    """Tests for setup_train_logger function."""

    def test_setup_train_logger_creates_train_log(self, temp_dir: str) -> None:
        """Test that setup_train_logger creates train.log file."""
        log = setup_train_logger(temp_dir)

        log.info("Training started")

        time.sleep(0.2)

        log_path = os.path.join(temp_dir, "train.log")
        assert os.path.exists(log_path)

    def test_setup_train_logger_with_resume(self, temp_dir: str) -> None:
        """Test setup_train_logger with resume parameter."""
        # First logger
        log1 = setup_train_logger(temp_dir)
        log1.info("First run")

        time.sleep(0.2)

        # Second logger resuming
        logger.remove()
        log2 = setup_train_logger(temp_dir, resume=True)
        log2.info("Resumed run")

        time.sleep(0.2)

        log_path = os.path.join(temp_dir, "train.log")
        with open(log_path, "r") as f:
            content = f.read()

        assert "First run" in content
        assert "Resumed run" in content


class TestSetupTestLogger:
    """Tests for setup_test_logger function."""

    def test_setup_test_logger_creates_test_log(self, temp_dir: str) -> None:
        """Test that setup_test_logger creates test.log file."""
        log = setup_test_logger(temp_dir)

        log.info("Testing started")

        time.sleep(0.2)

        log_path = os.path.join(temp_dir, "test.log")
        assert os.path.exists(log_path)


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


class TestGetGpuMemoryString:
    """Tests for get_gpu_memory_string function."""

    def test_get_gpu_memory_string_cpu(self) -> None:
        """Test get_gpu_memory_string on CPU device."""
        import torch

        device = torch.device("cpu")
        result = get_gpu_memory_string(device)

        # Should return empty string for CPU
        assert result == ""

    def test_get_gpu_memory_string_cuda_if_available(self) -> None:
        """Test get_gpu_memory_string on CUDA if available."""
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
            result = get_gpu_memory_string(device)

            # Should contain GPU memory info
            assert "GPU" in result
            assert "MB" in result
        else:
            pytest.skip("CUDA not available")


class TestLogGpuMemory:
    """Tests for log_gpu_memory function."""

    def test_log_gpu_memory_cpu_noop(self, temp_dir: str) -> None:
        """Test that log_gpu_memory is no-op on CPU."""
        import torch

        log = setup_logger(temp_dir, log_name="gpu")
        device = torch.device("cpu")

        # Should not raise any errors
        log_gpu_memory(log, "Test context", device)

        time.sleep(0.2)

        log_path = os.path.join(temp_dir, "gpu.log")
        with open(log_path, "r") as f:
            content = f.read()

        # Should not log anything for CPU
        assert "GPU Memory" not in content

    def test_log_gpu_memory_cuda_if_available(self, temp_dir: str) -> None:
        """Test log_gpu_memory on CUDA if available."""
        import torch

        if torch.cuda.is_available():
            log = setup_logger(temp_dir, log_name="gpu2")
            device = torch.device("cuda")

            log_gpu_memory(log, "Test context", device)

            time.sleep(0.2)

            log_path = os.path.join(temp_dir, "gpu2.log")
            with open(log_path, "r") as f:
                content = f.read()

            # Should log GPU memory info
            assert "GPU Memory" in content
            assert "Test context" in content
        else:
            pytest.skip("CUDA not available")


class TestLogSystemInfo:
    """Tests for log_system_info function."""

    def test_log_system_info_cpu(self, temp_dir: str) -> None:
        """Test log_system_info logs system information on CPU."""
        import torch

        log = setup_logger(temp_dir, log_name="sysinfo")
        device = torch.device("cpu")

        log_system_info(log, device)

        time.sleep(0.2)

        log_path = os.path.join(temp_dir, "sysinfo.log")
        with open(log_path, "r") as f:
            content = f.read()

        # Should log system info
        assert "SYSTEM INFO" in content
        assert "Python version" in content
        assert "PyTorch version" in content
        assert "Platform" in content

    def test_log_system_info_cuda_if_available(self, temp_dir: str) -> None:
        """Test log_system_info includes GPU info on CUDA."""
        import torch

        if torch.cuda.is_available():
            log = setup_logger(temp_dir, log_name="sysinfo2")
            device = torch.device("cuda")

            log_system_info(log, device)

            time.sleep(0.2)

            log_path = os.path.join(temp_dir, "sysinfo2.log")
            with open(log_path, "r") as f:
                content = f.read()

            # Should log GPU info
            assert "CUDA version" in content
            assert "GPU" in content
        else:
            pytest.skip("CUDA not available")
