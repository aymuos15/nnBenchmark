"""
Tests for src.engines.common module.
Tests result formatting utilities including test results and metric history recording.
"""

from __future__ import annotations

from typing import Any

from src.engines.common import print_results


class TestPrintResults:
    """Tests for print_results function."""

    def test_print_basic_results(self, capsys: Any) -> None:
        """Test printing basic test results."""
        results = {
            "mean": 0.85,
            "std": 0.05,
            "min": 0.75,
            "max": 0.95,
        }

        print_results(results, "Dice", context="TEST")

        captured = capsys.readouterr()
        assert "Dice TEST RESULTS" in captured.out
        assert "Mean Dice Score: 0.8500 ± 0.0500" in captured.out
        assert "Min Dice Score: 0.7500" in captured.out
        assert "Max Dice Score: 0.9500" in captured.out

    def test_print_results_with_per_class(self, capsys: Any) -> None:
        """Test printing test results with per-class metrics."""
        results = {
            "mean": 0.85,
            "std": 0.05,
            "min": 0.75,
            "max": 0.95,
            "per_class": {
                "Class_A": {"mean": 0.90, "std": 0.03},
                "Class_B": {"mean": 0.80, "std": 0.07},
            },
        }

        print_results(results, "IoU", context="TEST")

        captured = capsys.readouterr()
        assert "IoU TEST RESULTS" in captured.out
        assert "Mean IoU Score: 0.8500 ± 0.0500" in captured.out
        assert "Class_A: 0.9000 ± 0.0300" in captured.out
        assert "Class_B: 0.8000 ± 0.0700" in captured.out

    def test_print_results_formatting(self, capsys: Any) -> None:
        """Test that results are formatted with 4 decimal places."""
        results = {
            "mean": 0.123456,
            "std": 0.023456,
            "min": 0.023456,
            "max": 0.923456,
        }

        print_results(results, "Accuracy", context="TEST")

        captured = capsys.readouterr()
        # Should format to 4 decimal places
        assert "0.1235" in captured.out
        assert "0.0235" in captured.out

    def test_print_results_separator_lines(self, capsys: Any) -> None:
        """Test that results have separator lines."""
        results = {
            "mean": 0.85,
            "std": 0.05,
            "min": 0.75,
            "max": 0.95,
        }

        print_results(results, "Dice", context="TEST")

        captured = capsys.readouterr()
        # Should have separator lines
        assert "=" * 50 in captured.out
