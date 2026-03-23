"""
Tests for src.config.validation module.
Tests configuration validation utilities including required field checks and metric validation.
"""


from typing import Any

import pytest

from src.config.validation import (
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
            validate_required_field(
                sample_config, ["dataset", "missing_field"], "missing_field"
            )

        assert "'missing_field' parameter is required" in str(excinfo.value)

    def test_missing_section(self) -> None:
        """Test validation raises ValueError for missing section."""
        config = {"dataset": {"name": "test"}}

        with pytest.raises(ValueError) as excinfo:
            validate_required_field(config, ["model", "type"], "model type")

        assert "Missing section 'model'" in str(excinfo.value)

    def test_deeply_nested_field(self) -> None:
        """Test validation with deeply nested path."""
        config = {"level1": {"level2": {"level3": {"field": "value"}}}}

        # Should not raise
        validate_required_field(
            config, ["level1", "level2", "level3", "field"], "deep field"
        )

    def test_missing_deeply_nested_section(self) -> None:
        """Test validation with missing deeply nested section."""
        config = {"level1": {"level2": {}}}

        with pytest.raises(ValueError) as excinfo:
            validate_required_field(
                config, ["level1", "level2", "level3", "field"], "field"
            )

        assert "Missing section" in str(excinfo.value)
        assert "level1 -> level2 -> level3" in str(excinfo.value)

    def test_with_example_message(self, sample_config: dict[str, Any]) -> None:
        """Test validation error message includes example when provided."""
        with pytest.raises(ValueError) as excinfo:
            validate_required_field(
                sample_config,
                ["dataset", "missing_param"],
                "missing_param",
                "missing_param: value",
            )

        assert "missing_param: value" in str(excinfo.value)


class TestValidateSlidingWindowConfig:
    """Tests for validate_sliding_window_config function."""

    def test_disabled_sliding_window_passes(self) -> None:
        """Test that disabled sliding window passes validation."""
        config = {"inference": {"sliding_window": {"enabled": False}}}
        validate_sliding_window_config(config)  # Should not raise

    def test_missing_sliding_window_section_passes(self) -> None:
        """Test that missing sliding_window section passes validation."""
        config = {"inference": {}}
        validate_sliding_window_config(config)  # Should not raise

    def test_empty_sliding_window_passes(self) -> None:
        """Test that empty sliding_window dict passes validation."""
        config = {"inference": {"sliding_window": {}}}
        validate_sliding_window_config(config)  # Should not raise

    @pytest.mark.parametrize(
        "mode,should_pass",
        [
            pytest.param("gaussian", True, id="valid_gaussian"),
            pytest.param("constant", True, id="valid_constant"),
            pytest.param("median", False, id="invalid_median"),
            pytest.param("average", False, id="invalid_average"),
            pytest.param("invalid", False, id="invalid_string"),
            pytest.param("linear", False, id="invalid_linear"),
        ],
    )
    def test_sliding_window_mode_validation(self, mode: str, should_pass: bool) -> None:
        """Test sliding_window mode validation with various modes.

        Parameters:
        - mode: Mode to test
        - should_pass: Whether validation should succeed
        """
        config = {"inference": {"sliding_window": {"enabled": True, "mode": mode}}}
        if should_pass:
            validate_sliding_window_config(config)
        else:
            with pytest.raises(ValueError) as excinfo:
                validate_sliding_window_config(config)
            assert "mode" in str(excinfo.value).lower()

    def test_overlap_valid_range(self) -> None:
        """Test that valid overlap values are accepted."""
        for overlap in [0.0, 0.25, 0.5, 0.75, 0.99]:
            config = {
                "inference": {"sliding_window": {"enabled": True, "overlap": overlap}}
            }
            validate_sliding_window_config(config)

    def test_overlap_too_high(self) -> None:
        """Test that overlap >= 1.0 is rejected."""
        for overlap in [1.0, 1.5, 2.0]:
            config = {
                "inference": {"sliding_window": {"enabled": True, "overlap": overlap}}
            }
            with pytest.raises(ValueError) as excinfo:
                validate_sliding_window_config(config)
            assert "overlap" in str(excinfo.value).lower()

    def test_overlap_negative(self) -> None:
        """Test that negative overlap is rejected."""
        config = {"inference": {"sliding_window": {"enabled": True, "overlap": -0.5}}}
        with pytest.raises(ValueError) as excinfo:
            validate_sliding_window_config(config)
        assert "overlap" in str(excinfo.value).lower()

    @pytest.mark.parametrize(
        "batch_size,should_pass",
        [
            pytest.param(1, True, id="positive_1"),
            pytest.param(2, True, id="positive_2"),
            pytest.param(8, True, id="positive_8"),
            pytest.param(16, True, id="positive_16"),
            pytest.param(0, False, id="zero"),
            pytest.param(-4, False, id="negative"),
        ],
    )
    def test_sw_batch_size_validation(self, batch_size: int, should_pass: bool) -> None:
        """Test sliding_window batch size validation.

        Parameters:
        - batch_size: Batch size to test
        - should_pass: Whether validation should succeed
        """
        config = {
            "inference": {
                "sliding_window": {"enabled": True, "sw_batch_size": batch_size}
            }
        }
        if should_pass:
            validate_sliding_window_config(config)
        else:
            with pytest.raises(ValueError) as excinfo:
                validate_sliding_window_config(config)
            assert "positive" in str(excinfo.value).lower()

    def test_padding_mode_valid_options(self) -> None:
        """Test that all valid padding modes are accepted."""
        for mode in ["constant", "edge", "reflect", "wrap"]:
            config = {
                "inference": {"sliding_window": {"enabled": True, "padding_mode": mode}}
            }
            validate_sliding_window_config(config)

    def test_padding_mode_invalid(self) -> None:
        """Test that invalid padding modes are rejected."""
        for mode in ["invalid", "mirror", "replicate"]:
            config = {
                "inference": {"sliding_window": {"enabled": True, "padding_mode": mode}}
            }
            with pytest.raises(ValueError) as excinfo:
                validate_sliding_window_config(config)
            assert "padding_mode" in str(excinfo.value).lower()


