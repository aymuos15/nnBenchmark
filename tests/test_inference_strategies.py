"""
Unit tests for inference strategy classes.

Tests both FullVolumeInferer and SlidingWindowInferer strategies,
including configuration, parameter validation, and inference behavior.

Note: The infer() method returns a union type for flexibility with different model outputs,
but test models in this file all return torch.Tensor. Type narrowing via isinstance checks
or explicit casting is used where needed.
"""  # noqa: PYI

from __future__ import annotations

from typing import cast

import pytest
import torch
import torch.nn as nn

from src.engines.inference.strategy import (
    FullVolumeInferer,
    SlidingWindowInferer,
    create_inferer,
)


class SimpleMockModel(nn.Module):
    """Simple mock model for testing.

    Returns output with same spatial dimensions as input but with specified number of channels.
    """

    num_channels: int

    def __init__(self, num_channels: int = 2) -> None:
        super().__init__()
        self.num_channels: int = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tensor with same spatial shape as input but specified num_channels.

        Args:
            x: Input tensor of shape [B, C, H, W, D] or [B, C, H, W]

        Returns:
            Output tensor of shape [B, num_channels, H, W, D] or [B, num_channels, H, W]
        """
        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]
        return torch.randn(batch_size, self.num_channels, *spatial_shape)


class TestFullVolumeInferer:
    """Test cases for FullVolumeInferer."""

    def test_initialization(self) -> None:
        """Test that FullVolumeInferer initializes correctly."""
        inferer = FullVolumeInferer()
        assert inferer is not None

    def test_infer_basic(self) -> None:
        """Test basic inference with full-volume inferer."""
        device = torch.device("cpu")
        model = SimpleMockModel(num_channels=2)
        inferer = FullVolumeInferer()

        # Create input tensor
        inputs = torch.randn(1, 1, 32, 32, 32, device=device)

        # Perform inference
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        # Check output shape
        assert outputs.shape == (1, 2, 32, 32, 32)

    def test_infer_batch(self) -> None:
        """Test inference with batch size > 1."""
        device = torch.device("cpu")
        model = SimpleMockModel(num_channels=3)
        inferer = FullVolumeInferer()

        # Create batch input
        inputs = torch.randn(4, 1, 48, 48, 48, device=device)

        # Perform inference
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        # Check output shape
        assert outputs.shape == (4, 3, 48, 48, 48)

    def test_infer_output_type(self) -> None:
        """Test that inference returns torch.Tensor."""
        device = torch.device("cpu")
        model = SimpleMockModel()
        inferer = FullVolumeInferer()
        inputs = torch.randn(1, 1, 64, 64, 64, device=device)

        outputs = inferer.infer(model, inputs, device)

        assert isinstance(outputs, torch.Tensor)

    def test_infer_with_different_devices(self) -> None:
        """Test inference on CPU device (skip CUDA due to environment constraints)."""
        device = torch.device("cpu")
        model = SimpleMockModel()
        inferer = FullVolumeInferer()
        inputs = torch.randn(1, 1, 32, 32, 32, device=device)

        outputs = inferer.infer(model, inputs, device)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.device.type == "cpu"
        assert outputs.shape == (1, 2, 32, 32, 32)


class TestSlidingWindowInferer:
    """Test cases for SlidingWindowInferer."""

    def test_initialization_default_params(self) -> None:
        """Test initialization with default parameters."""
        roi_size = (32, 48, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size)

        assert inferer.roi_size == roi_size
        assert inferer.sw_batch_size == 4
        assert inferer.overlap == 0.5
        assert inferer.mode == "gaussian"
        assert inferer.padding_mode == "constant"

    def test_initialization_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        roi_size = (32, 48, 32)
        inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=2,
            overlap=0.75,
            mode="constant",
            padding_mode="edge",
        )

        assert inferer.roi_size == roi_size
        assert inferer.sw_batch_size == 2
        assert inferer.overlap == 0.75
        assert inferer.mode == "constant"
        assert inferer.padding_mode == "edge"

    def test_roi_size_list_conversion(self) -> None:
        """Test that roi_size list is converted to tuple."""
        roi_size_list = [32, 48, 32]
        inferer = SlidingWindowInferer(roi_size=roi_size_list)

        assert isinstance(inferer.roi_size, tuple)
        assert inferer.roi_size == (32, 48, 32)

    def test_infer_basic(self) -> None:
        """Test basic sliding window inference."""
        device = torch.device("cpu")
        model = SimpleMockModel(num_channels=2)
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size)

        # Create input tensor larger than roi_size
        inputs = torch.randn(1, 1, 64, 64, 64, device=device)

        # Perform inference
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        # Check output shape matches input spatial dims
        assert outputs.shape == (1, 2, 64, 64, 64)

    def test_infer_output_shape_matches_input(self) -> None:
        """Test that output spatial shape matches input shape."""
        device = torch.device("cpu")
        model = SimpleMockModel(num_channels=3)
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size)

        # Test with various input sizes
        for spatial_size in [48, 64, 80]:
            inputs = torch.randn(
                1, 1, spatial_size, spatial_size, spatial_size, device=device
            )
            outputs = inferer.infer(model, inputs, device, use_amp=False)
            assert isinstance(outputs, torch.Tensor)

            # Output spatial dims should match input spatial dims
            expected_shape = (1, 3, spatial_size, spatial_size, spatial_size)
            assert (
                outputs.shape == expected_shape
            ), f"Expected {expected_shape}, got {outputs.shape}"

    def test_infer_creates_monai_inferer(self) -> None:
        """Test that MONAI inferer is created correctly."""
        roi_size = (32, 48, 32)
        inferer = SlidingWindowInferer(
            roi_size=roi_size, sw_batch_size=2, overlap=0.5, mode="gaussian"
        )

        # Check that inferer attribute exists and is MONAI inferer
        assert hasattr(inferer, "inferer")
        assert inferer.inferer is not None

    def test_infer_batch_processing(self) -> None:
        """Test inference with batch size > 1."""
        device = torch.device("cpu")
        model = SimpleMockModel(num_channels=2)
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=2)

        # Create batch input
        inputs = torch.randn(2, 1, 64, 64, 64, device=device)

        # Perform inference
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        # Check output shape
        assert outputs.shape == (2, 2, 64, 64, 64)

    def test_infer_different_overlap_values(self) -> None:
        """Test inference with different overlap values."""
        device = torch.device("cpu")
        model = SimpleMockModel(num_channels=2)
        roi_size = (32, 32, 32)

        for overlap_val in [0.25, 0.5, 0.75]:
            inferer = SlidingWindowInferer(roi_size=roi_size, overlap=overlap_val)
            inputs = torch.randn(1, 1, 64, 64, 64, device=device)
            outputs = inferer.infer(model, inputs, device, use_amp=False)
            assert isinstance(outputs, torch.Tensor)

            assert outputs.shape == (1, 2, 64, 64, 64)

    def test_infer_different_modes(self) -> None:
        """Test inference with different blending modes."""
        device = torch.device("cpu")
        model = SimpleMockModel(num_channels=2)
        roi_size = (32, 32, 32)

        for mode_val in ["gaussian", "constant"]:
            inferer = SlidingWindowInferer(roi_size=roi_size, mode=mode_val)
            inputs = torch.randn(1, 1, 64, 64, 64, device=device)
            outputs = inferer.infer(model, inputs, device, use_amp=False)
            assert isinstance(outputs, torch.Tensor)

            assert outputs.shape == (1, 2, 64, 64, 64)

    def test_infer_with_different_devices(self) -> None:
        """Test sliding window inference on CPU device (skip CUDA due to environment constraints)."""
        device = torch.device("cpu")
        model = SimpleMockModel(num_channels=2)
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)
        outputs = inferer.infer(model, inputs, device)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.device.type == "cpu"
        assert outputs.shape == (1, 2, 64, 64, 64)


class TestCreateInferer:
    """Test cases for create_inferer factory function."""

    def test_create_full_volume_inferer_default(self) -> None:
        """Test creating full-volume inferer with default config."""
        config = {"inference": {}}

        inferer = create_inferer(config)

        assert isinstance(inferer, FullVolumeInferer)

    def test_create_full_volume_inferer_disabled(self) -> None:
        """Test creating full-volume inferer when sliding window is disabled."""
        config = {"inference": {"sliding_window": {"enabled": False}}}

        inferer = create_inferer(config)

        assert isinstance(inferer, FullVolumeInferer)

    def test_create_sliding_window_inferer_enabled(self) -> None:
        """Test creating sliding window inferer when enabled."""
        config = {
            "dataset": {"spatial_size": [32, 48, 32]},
            "inference": {"sliding_window": {"enabled": True}},
        }

        inferer = create_inferer(config)

        assert isinstance(inferer, SlidingWindowInferer)
        assert inferer.roi_size == (32, 48, 32)

    def test_create_sliding_window_inferer_with_explicit_roi_size(self) -> None:
        """Test creating sliding window inferer with explicit roi_size."""
        config = {
            "inference": {
                "sliding_window": {
                    "enabled": True,
                    "roi_size": [64, 64, 64],
                }
            }
        }

        inferer = create_inferer(config)

        assert isinstance(inferer, SlidingWindowInferer)
        assert inferer.roi_size == (64, 64, 64)

    def test_create_sliding_window_inferer_custom_params(self) -> None:
        """Test creating sliding window inferer with custom parameters."""
        config = {
            "dataset": {"spatial_size": [32, 48, 32]},
            "inference": {
                "sliding_window": {
                    "enabled": True,
                    "sw_batch_size": 2,
                    "overlap": 0.75,
                    "mode": "constant",
                    "padding_mode": "edge",
                }
            },
        }

        inferer = create_inferer(config)

        assert isinstance(inferer, SlidingWindowInferer)
        assert inferer.sw_batch_size == 2
        assert inferer.overlap == 0.75
        assert inferer.mode == "constant"
        assert inferer.padding_mode == "edge"

    def test_create_sliding_window_inferer_missing_roi_size(self) -> None:
        """Test error when roi_size is not provided and cannot be inferred."""
        config = {
            "inference": {
                "sliding_window": {
                    "enabled": True,
                    "roi_size": None,
                }
            }
        }

        with pytest.raises(ValueError, match="roi_size"):
            create_inferer(config)

    def test_create_sliding_window_inferer_invalid_roi_size_type(self) -> None:
        """Test error when roi_size has invalid type."""
        config = {
            "inference": {
                "sliding_window": {
                    "enabled": True,
                    "roi_size": "invalid",
                }
            }
        }

        with pytest.raises(ValueError, match="roi_size"):
            create_inferer(config)

    def test_create_with_empty_config(self) -> None:
        """Test creating inferer with minimal/empty config."""
        config = {}

        inferer = create_inferer(config)

        assert isinstance(inferer, FullVolumeInferer)


class _FixedOutputModel(nn.Module):
    """Mock model that returns fixed output for testing edge cases."""

    value: float

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value: float = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tensor filled with fixed value."""
        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]
        return torch.full((batch_size, 2, *spatial_shape), self.value)


class TestSlidingWindowVolumeVariations:
    """Comprehensive edge case tests for sliding window inference.

    Tests sliding window inferer with:
    - Variable volume sizes (small, medium, large)
    - Anisotropic volumes
    - Edge cases (volumes smaller than roi_size, exact size matches)
    - Output quality verification
    """

    def test_small_volume_inference(self) -> None:
        """Test inference on volume smaller than roi_size."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        inputs = torch.randn(1, 1, 16, 16, 16, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.shape == (1, 2, 16, 16, 16)

    def test_medium_volume_inference(self) -> None:
        """Test inference on volume approximately roi_size."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        inputs = torch.randn(1, 1, 32, 32, 32, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.shape == (1, 2, 32, 32, 32)

    def test_large_volume_inference(self) -> None:
        """Test inference on volume much larger than roi_size."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        inputs = torch.randn(1, 1, 128, 128, 128, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.shape == (1, 2, 128, 128, 128)

    def test_anisotropic_volume_inference(self) -> None:
        """Test inference on anisotropic volume (different aspect ratios)."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (32, 48, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        inputs = torch.randn(1, 1, 64, 96, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.shape == (1, 2, 64, 96, 64)

    def test_highly_anisotropic_volume(self) -> None:
        """Test inference on highly anisotropic volume (e.g., 2D-like)."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (64, 64, 16)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.25)

        inputs = torch.randn(1, 1, 128, 128, 32, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.shape == (1, 2, 128, 128, 32)

    def test_volume_smaller_than_roi_size(self) -> None:
        """Test inference when volume is much smaller than roi_size."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size)

        inputs = torch.randn(1, 1, 8, 8, 8, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.shape == (1, 2, 8, 8, 8)

    def test_volume_exactly_roi_size(self) -> None:
        """Test inference when volume exactly matches roi_size."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size)

        inputs = torch.randn(1, 1, 32, 32, 32, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.shape == (1, 2, 32, 32, 32)

    def test_non_divisible_overlap(self) -> None:
        """Test inference with overlap that doesn't divide evenly."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.33)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.shape == (1, 2, 64, 64, 64)

    def test_asymmetric_roi_size(self) -> None:
        """Test inference with asymmetric roi_size."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (16, 32, 64)
        inferer = SlidingWindowInferer(roi_size=roi_size)

        inputs = torch.randn(1, 1, 32, 64, 128, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.shape == (1, 2, 32, 64, 128)

    def test_very_high_overlap(self) -> None:
        """Test inference with very high overlap (0.9)."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.9)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.shape == (1, 2, 64, 64, 64)

    def test_very_low_overlap(self) -> None:
        """Test inference with very low overlap (0.1)."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.1)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.shape == (1, 2, 64, 64, 64)

    def test_output_shape_consistency(self) -> None:
        """Test that output shape always matches input spatial shape."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        test_shapes = [
            (1, 1, 20, 20, 20),
            (1, 1, 32, 32, 32),
            (1, 1, 64, 64, 64),
            (1, 1, 100, 100, 100),
            (2, 1, 48, 48, 48),
        ]

        for shape in test_shapes:
            inputs = torch.randn(shape, device=device)
            outputs = inferer.infer(model, inputs, device, use_amp=False)

            assert outputs.shape[0] == shape[0]  # type: ignore[union-attr]
            assert outputs.shape[1] == 2  # type: ignore[union-attr]
            assert outputs.shape[2:] == shape[2:]  # type: ignore[union-attr]

    def test_output_values_range(self) -> None:
        """Test that output values are reasonable (not NaN or inf)."""
        device = torch.device("cpu")
        model = _FixedOutputModel(value=0.5)
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5, mode="gaussian")

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        outputs_tensor = cast(torch.Tensor, outputs)

        assert not torch.isnan(outputs_tensor).any(), "Output contains NaN values"  # type: ignore[arg-type]
        assert not torch.isinf(outputs_tensor).any(), "Output contains inf values"  # type: ignore[arg-type]
        assert outputs_tensor.min() >= -1.0, "Output values too low"
        assert outputs_tensor.max() <= 2.0, "Output values too high"

    def test_overlap_consistency(self) -> None:
        """Test that increased overlap produces similar results."""
        device = torch.device("cpu")
        model = _FixedOutputModel(value=1.0)
        roi_size = (32, 32, 32)
        inputs = torch.randn(1, 1, 64, 64, 64, device=device)

        outputs_low_overlap = SlidingWindowInferer(
            roi_size=roi_size, overlap=0.25, mode="gaussian"
        ).infer(model, inputs, device)

        outputs_high_overlap = SlidingWindowInferer(
            roi_size=roi_size, overlap=0.75, mode="gaussian"
        ).infer(model, inputs, device)

        assert outputs_low_overlap.shape == outputs_high_overlap.shape  # type: ignore[union-attr]

        diff = torch.abs(outputs_low_overlap - outputs_high_overlap).max()  # type: ignore[operator]
        assert (
            diff < 1.0
        ), "High overlap overlap significantly different from low overlap"

    def test_blending_mode_consistency(self) -> None:
        """Test that different blending modes produce reasonable outputs."""
        device = torch.device("cpu")
        model = _FixedOutputModel(value=1.0)
        roi_size = (32, 32, 32)
        inputs = torch.randn(1, 1, 64, 64, 64, device=device)

        outputs_gaussian = SlidingWindowInferer(
            roi_size=roi_size, overlap=0.5, mode="gaussian"
        ).infer(model, inputs, device)

        outputs_constant = SlidingWindowInferer(
            roi_size=roi_size, overlap=0.5, mode="constant"
        ).infer(model, inputs, device)

        assert outputs_gaussian.shape == outputs_constant.shape  # type: ignore[union-attr]

        mean_diff = torch.abs(outputs_gaussian - outputs_constant).mean()  # type: ignore[operator]
        assert mean_diff < 1.0, "Blending modes produce vastly different results"

    def test_large_batch_size(self) -> None:
        """Test inference with large batch size."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=2)

        inputs = torch.randn(8, 1, 48, 48, 48, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)
        assert isinstance(outputs, torch.Tensor)

        assert outputs.shape == (8, 2, 48, 48, 48)

    def test_sliding_window_deterministic(self) -> None:
        """Test that sliding window produces consistent results."""
        device = torch.device("cpu")
        model = _FixedOutputModel(value=0.5)
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)

        outputs1 = inferer.infer(model, inputs, device, use_amp=False)
        outputs2 = inferer.infer(model, inputs, device, use_amp=False)

        assert torch.allclose(outputs1, outputs2), (  # type: ignore[arg-type]
            "Sliding window inference not deterministic"
        )
