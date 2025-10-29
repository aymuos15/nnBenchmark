"""
Tests for src.config.validation module.
Tests configuration validation utilities including required field checks and metric validation.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.config.validation import (
    validate_metrics_config,
    validate_required_field,
    validate_sliding_window_config,
)


class TestValidateRequiredField:
    """Tests for validate_required_field function."""

    def test_valid_nested_field(self, sample_config: dict[str, Any]) -> None:
        """Test validation passes for existing nested field."""
        # Should not raise any exception
        validate_required_field(sample_config, ["dataset", "fold"], "fold")
        validate_required_field(sample_config, ["model", "type"], "model type")
        validate_required_field(sample_config, ["training", "epochs"], "epochs")

    def test_missing_nested_field(self, sample_config: dict[str, Any]) -> None:
        """Test validation raises ValueError for missing nested field."""
        with pytest.raises(ValueError) as excinfo:
            validate_required_field(sample_config, ["dataset", "missing_field"], "missing_field")

        assert "'missing_field' parameter is required" in str(excinfo.value)

    def test_missing_section(self) -> None:
        """Test validation raises ValueError for missing section."""
        config = {"dataset": {"name": "test"}}

        with pytest.raises(ValueError) as excinfo:
            validate_required_field(config, ["model", "type"], "model type")

        assert "Missing section 'model'" in str(excinfo.value)

    def test_deeply_nested_field(self) -> None:
        """Test validation with deeply nested path."""
        config = {
            "level1": {
                "level2": {
                    "level3": {
                        "field": "value"
                    }
                }
            }
        }

        # Should not raise
        validate_required_field(config, ["level1", "level2", "level3", "field"], "deep field")

    def test_missing_deeply_nested_section(self) -> None:
        """Test validation with missing deeply nested section."""
        config = {
            "level1": {
                "level2": {}
            }
        }

        with pytest.raises(ValueError) as excinfo:
            validate_required_field(config, ["level1", "level2", "level3", "field"], "field")

        assert "Missing section" in str(excinfo.value)
        assert "level1 -> level2 -> level3" in str(excinfo.value)

    def test_with_example_message(self, sample_config: dict[str, Any]) -> None:
        """Test validation error message includes example when provided."""
        with pytest.raises(ValueError) as excinfo:
            validate_required_field(
                sample_config,
                ["dataset", "missing_param"],
                "missing_param",
                "missing_param: value"
            )

        assert "missing_param: value" in str(excinfo.value)


class TestValidateMetricsConfig:
    """Tests for validate_metrics_config function."""

    def test_valid_metrics_config(self, sample_config: dict[str, Any]) -> None:
        """Test validation passes for valid metrics configuration."""
        # Mock metric dictionary
        metric_dict = {"Dice": "mock_metric"}

        checkpoint_metric, plot_metrics = validate_metrics_config(sample_config, metric_dict)

        assert checkpoint_metric == "Dice"
        assert plot_metrics == ["Dice"]

    def test_multiple_plot_metrics(self, sample_config: dict[str, Any]) -> None:
        """Test validation with multiple plot metrics."""
        # Modify config to have multiple metrics
        sample_config["training"]["plot_metrics"] = ["Dice", "IoU"]

        metric_dict = {"Dice": "mock_metric_1", "IoU": "mock_metric_2"}

        checkpoint_metric, plot_metrics = validate_metrics_config(sample_config, metric_dict)

        assert checkpoint_metric == "Dice"
        assert plot_metrics == ["Dice", "IoU"]

    def test_missing_checkpoint_metric_field(self, sample_config: dict[str, Any]) -> None:
        """Test validation raises ValueError when checkpoint_metric field is missing."""
        del sample_config["training"]["checkpoint_metric"]
        metric_dict = {"Dice": "mock_metric"}

        with pytest.raises(ValueError) as excinfo:
            validate_metrics_config(sample_config, metric_dict)

        assert "Missing required field 'checkpoint_metric'" in str(excinfo.value)

    def test_missing_plot_metrics_field(self, sample_config: dict[str, Any]) -> None:
        """Test validation raises ValueError when plot_metrics field is missing."""
        del sample_config["training"]["plot_metrics"]
        metric_dict = {"Dice": "mock_metric"}

        with pytest.raises(ValueError) as excinfo:
            validate_metrics_config(sample_config, metric_dict)

        assert "Missing required field 'plot_metrics'" in str(excinfo.value)

    def test_checkpoint_metric_not_in_defined_metrics(self, sample_config: dict[str, Any]) -> None:
        """Test validation raises ValueError when checkpoint_metric not in defined metrics."""
        sample_config["training"]["checkpoint_metric"] = "InvalidMetric"
        metric_dict = {"Dice": "mock_metric"}

        with pytest.raises(ValueError) as excinfo:
            validate_metrics_config(sample_config, metric_dict)

        assert "checkpoint_metric 'InvalidMetric' not found" in str(excinfo.value)
        assert "Available metrics" in str(excinfo.value)

    def test_plot_metric_not_in_defined_metrics(self, sample_config: dict[str, Any]) -> None:
        """Test validation raises ValueError when plot_metric not in defined metrics."""
        sample_config["training"]["plot_metrics"] = ["Dice", "InvalidMetric"]
        metric_dict = {"Dice": "mock_metric"}

        with pytest.raises(ValueError) as excinfo:
            validate_metrics_config(sample_config, metric_dict)

        assert "plot_metric 'InvalidMetric' not found" in str(excinfo.value)
        assert "Available metrics" in str(excinfo.value)


class TestValidateSlidingWindowConfig:
    """Tests for validate_sliding_window_config function."""

    def test_disabled_sliding_window_passes(self) -> None:
        """Test that disabled sliding window passes validation."""
        config = {"testing": {"sliding_window": {"enabled": False}}}
        validate_sliding_window_config(config)  # Should not raise

    def test_missing_sliding_window_section_passes(self) -> None:
        """Test that missing sliding_window section passes validation."""
        config = {"testing": {}}
        validate_sliding_window_config(config)  # Should not raise

    def test_empty_sliding_window_passes(self) -> None:
        """Test that empty sliding_window dict passes validation."""
        config = {"testing": {"sliding_window": {}}}
        validate_sliding_window_config(config)  # Should not raise

    def test_valid_mode_gaussian(self) -> None:
        """Test that 'gaussian' mode is accepted."""
        config = {
            "testing": {
                "sliding_window": {"enabled": True, "mode": "gaussian"}
            }
        }
        validate_sliding_window_config(config)

    def test_valid_mode_constant(self) -> None:
        """Test that 'constant' mode is accepted."""
        config = {
            "testing": {
                "sliding_window": {"enabled": True, "mode": "constant"}
            }
        }
        validate_sliding_window_config(config)

    def test_invalid_mode_median_rejected(self) -> None:
        """Test that 'median' mode is rejected (not supported)."""
        config = {
            "testing": {
                "sliding_window": {"enabled": True, "mode": "median"}
            }
        }
        with pytest.raises(ValueError) as excinfo:
            validate_sliding_window_config(config)
        assert "mode" in str(excinfo.value).lower()
        assert "gaussian" in str(excinfo.value)
        assert "constant" in str(excinfo.value)

    def test_invalid_mode_other(self) -> None:
        """Test that other invalid modes are rejected."""
        for mode in ["average", "invalid", "linear"]:
            config = {
                "testing": {
                    "sliding_window": {"enabled": True, "mode": mode}
                }
            }
            with pytest.raises(ValueError) as excinfo:
                validate_sliding_window_config(config)
            assert "mode" in str(excinfo.value).lower()

    def test_overlap_valid_range(self) -> None:
        """Test that valid overlap values are accepted."""
        for overlap in [0.0, 0.25, 0.5, 0.75, 0.99]:
            config = {
                "testing": {
                    "sliding_window": {"enabled": True, "overlap": overlap}
                }
            }
            validate_sliding_window_config(config)

    def test_overlap_too_high(self) -> None:
        """Test that overlap >= 1.0 is rejected."""
        for overlap in [1.0, 1.5, 2.0]:
            config = {
                "testing": {
                    "sliding_window": {"enabled": True, "overlap": overlap}
                }
            }
            with pytest.raises(ValueError) as excinfo:
                validate_sliding_window_config(config)
            assert "overlap" in str(excinfo.value).lower()

    def test_overlap_negative(self) -> None:
        """Test that negative overlap is rejected."""
        config = {
            "testing": {
                "sliding_window": {"enabled": True, "overlap": -0.5}
            }
        }
        with pytest.raises(ValueError) as excinfo:
            validate_sliding_window_config(config)
        assert "overlap" in str(excinfo.value).lower()

    def test_sw_batch_size_positive(self) -> None:
        """Test that positive sw_batch_size values are accepted."""
        for size in [1, 2, 4, 8, 16]:
            config = {
                "testing": {
                    "sliding_window": {"enabled": True, "sw_batch_size": size}
                }
            }
            validate_sliding_window_config(config)

    def test_sw_batch_size_zero(self) -> None:
        """Test that sw_batch_size of 0 is rejected."""
        config = {
            "testing": {
                "sliding_window": {"enabled": True, "sw_batch_size": 0}
            }
        }
        with pytest.raises(ValueError) as excinfo:
            validate_sliding_window_config(config)
        assert "positive" in str(excinfo.value).lower()

    def test_sw_batch_size_negative(self) -> None:
        """Test that negative sw_batch_size is rejected."""
        config = {
            "testing": {
                "sliding_window": {"enabled": True, "sw_batch_size": -4}
            }
        }
        with pytest.raises(ValueError) as excinfo:
            validate_sliding_window_config(config)
        assert "positive" in str(excinfo.value).lower()

    def test_padding_mode_valid_options(self) -> None:
        """Test that all valid padding modes are accepted."""
        for mode in ["constant", "edge", "reflect", "wrap"]:
            config = {
                "testing": {
                    "sliding_window": {
                        "enabled": True,
                        "padding_mode": mode
                    }
                }
            }
            validate_sliding_window_config(config)

    def test_padding_mode_invalid(self) -> None:
        """Test that invalid padding modes are rejected."""
        for mode in ["invalid", "mirror", "replicate"]:
            config = {
                "testing": {
                    "sliding_window": {
                        "enabled": True,
                        "padding_mode": mode
                    }
                }
            }
            with pytest.raises(ValueError) as excinfo:
                validate_sliding_window_config(config)
            assert "padding_mode" in str(excinfo.value).lower()
