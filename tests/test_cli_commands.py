"""
Tests for CLI command entry points.
Ensures all nnBench.* commands are properly registered and functional.
"""

import subprocess
import sys


class TestCLICommands:
    """Test that all CLI commands are properly installed and working."""

    def test_nnbench_train_command_exists(self) -> None:
        """Test that nnBench.train command is installed."""
        result = subprocess.run(
            ["nnBench.train", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "train" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_nnbench_test_command_exists(self) -> None:
        """Test that nnBench.test command is installed."""
        result = subprocess.run(
            ["nnBench.test", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "test" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_nnbench_plot_command_exists(self) -> None:
        """Test that nnBench.plot command is installed."""
        result = subprocess.run(
            ["nnBench.plot", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "plot" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_nnbench_plan_command_exists(self) -> None:
        """Test that nnBench.plan command is installed."""
        result = subprocess.run(
            ["nnBench.plan", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "plan" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_python_module_plotting_works(self) -> None:
        """Test that python -m src.plotting.cli works."""
        result = subprocess.run(
            [sys.executable, "-m", "src.plotting.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "plot" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_python_module_planning_works(self) -> None:
        """Test that python -m src.planning.cli works."""
        result = subprocess.run(
            [sys.executable, "-m", "src.planning.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "plan" in result.stdout.lower() or "usage" in result.stdout.lower()
