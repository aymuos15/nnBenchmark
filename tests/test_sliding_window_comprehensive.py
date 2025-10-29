"""
Comprehensive integration and edge case tests for sliding window inference.

Tests sliding window inferer with:
- Variable volume sizes (small, medium, large)
- Anisotropic volumes
- Edge cases (volumes smaller than roi_size, exact size matches)
- Output quality verification
"""

import torch
import torch.nn as nn

from src.inference.strategy import FullVolumeInferer, SlidingWindowInferer


class FixedOutputModel(nn.Module):
    """Mock model that returns fixed output for testing."""

    value: float

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = value  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tensor filled with fixed value."""
        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]
        return torch.full((batch_size, 2, *spatial_shape), self.value)


class TestSlidingWindowVariableVolumeSizes:
    """Test sliding window inference with different volume sizes."""

    def test_small_volume_inference(self) -> None:
        """Test inference on volume smaller than roi_size."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        # Volume much smaller than roi_size
        inputs = torch.randn(1, 1, 16, 16, 16, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        # Output shape should match input spatial dims
        assert outputs.shape == (1, 2, 16, 16, 16)

    def test_medium_volume_inference(self) -> None:
        """Test inference on volume approximately roi_size."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        # Volume similar to roi_size
        inputs = torch.randn(1, 1, 32, 32, 32, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 32, 32, 32)

    def test_large_volume_inference(self) -> None:
        """Test inference on volume much larger than roi_size."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        # Volume much larger than roi_size (4x each dimension)
        inputs = torch.randn(1, 1, 128, 128, 128, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 128, 128, 128)

    def test_anisotropic_volume_inference(self) -> None:
        """Test inference on anisotropic volume (different aspect ratios)."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 48, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        # Anisotropic volume
        inputs = torch.randn(1, 1, 64, 96, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 64, 96, 64)

    def test_highly_anisotropic_volume(self) -> None:
        """Test inference on highly anisotropic volume (e.g., 2D-like)."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (64, 64, 16)  # Much larger in X, Y than Z
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.25)

        # Anisotropic input
        inputs = torch.randn(1, 1, 128, 128, 32, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 128, 128, 32)


class TestSlidingWindowOutputQuality:
    """Test output quality and correctness of sliding window inference."""

    def test_output_shape_consistency(self) -> None:
        """Test that output shape always matches input spatial shape."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        # Test various input shapes
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

            # Output spatial dims should match input
            assert outputs.shape[0] == shape[0]  # batch
            assert outputs.shape[1] == 2  # num classes
            assert outputs.shape[2:] == shape[2:]  # spatial dims

    def test_output_values_range(self) -> None:
        """Test that output values are reasonable (not NaN or inf)."""
        device = torch.device("cpu")
        model = FixedOutputModel(value=0.5)
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5, mode="gaussian")

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        # Check for NaN and inf
        assert not torch.isnan(outputs).any(), "Output contains NaN values"
        assert not torch.isinf(outputs).any(), "Output contains inf values"

        # Check value range (blending with input may produce values around 0.5)
        assert outputs.min() >= -1.0, "Output values too low"
        assert outputs.max() <= 2.0, "Output values too high"

    def test_overlap_consistency(self) -> None:
        """Test that increased overlap produces similar results."""
        device = torch.device("cpu")
        model = FixedOutputModel(value=1.0)
        roi_size = (32, 32, 32)
        inputs = torch.randn(1, 1, 64, 64, 64, device=device)

        # Test with different overlaps
        outputs_low_overlap = SlidingWindowInferer(
            roi_size=roi_size, overlap=0.25, mode="gaussian"
        ).infer(model, inputs, device)

        outputs_high_overlap = SlidingWindowInferer(
            roi_size=roi_size, overlap=0.75, mode="gaussian"
        ).infer(model, inputs, device)

        # Outputs should have same shape
        assert outputs_low_overlap.shape == outputs_high_overlap.shape

        # With fixed output model, values should be similar
        # (exact values depend on blending, but should be close)
        diff = torch.abs(outputs_low_overlap - outputs_high_overlap).max()
        assert diff < 1.0, (
            "High overlap overlap significantly different from low overlap"
        )

    def test_blending_mode_consistency(self) -> None:
        """Test that different blending modes produce reasonable outputs."""
        device = torch.device("cpu")
        model = FixedOutputModel(value=1.0)
        roi_size = (32, 32, 32)
        inputs = torch.randn(1, 1, 64, 64, 64, device=device)

        # Test with different modes
        outputs_gaussian = SlidingWindowInferer(
            roi_size=roi_size, overlap=0.5, mode="gaussian"
        ).infer(model, inputs, device)

        outputs_constant = SlidingWindowInferer(
            roi_size=roi_size, overlap=0.5, mode="constant"
        ).infer(model, inputs, device)

        # Both should have same shape
        assert outputs_gaussian.shape == outputs_constant.shape

        # Values should be reasonable (not completely different)
        mean_diff = torch.abs(outputs_gaussian - outputs_constant).mean()
        assert mean_diff < 1.0, "Blending modes produce vastly different results"


class TestSlidingWindowEdgeCases:
    """Test edge cases for sliding window inference."""

    def test_volume_smaller_than_roi_size(self) -> None:
        """Test inference when volume is much smaller than roi_size."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size)

        # Very small volume (should still work)
        inputs = torch.randn(1, 1, 8, 8, 8, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 8, 8, 8)

    def test_volume_exactly_roi_size(self) -> None:
        """Test inference when volume exactly matches roi_size."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size)

        # Volume exactly matches roi_size
        inputs = torch.randn(1, 1, 32, 32, 32, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 32, 32, 32)

    def test_single_patch_volume(self) -> None:
        """Test inference on volume that requires exactly one patch."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        # Slightly larger than roi_size (should use 2 patches minimum)
        inputs = torch.randn(1, 1, 40, 40, 40, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 40, 40, 40)

    def test_batch_size_one(self) -> None:
        """Test inference with batch size 1."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 64, 64, 64)

    def test_large_batch_size(self) -> None:
        """Test inference with large batch size."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=2)

        # Large batch (relative to patch processing)
        inputs = torch.randn(8, 1, 48, 48, 48, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (8, 2, 48, 48, 48)

    def test_non_divisible_overlap(self) -> None:
        """Test inference with overlap that doesn't divide evenly."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.33)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 64, 64, 64)

    def test_asymmetric_roi_size(self) -> None:
        """Test inference with asymmetric roi_size."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (16, 32, 64)
        inferer = SlidingWindowInferer(roi_size=roi_size)

        inputs = torch.randn(1, 1, 32, 64, 128, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 32, 64, 128)

    def test_very_high_overlap(self) -> None:
        """Test inference with very high overlap (0.9)."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.9)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 64, 64, 64)

    def test_very_low_overlap(self) -> None:
        """Test inference with very low overlap (0.1)."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.1)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 64, 64, 64)


class TestSlidingWindowVsFullVolume:
    """Test consistency between sliding window and full-volume inference."""

    def test_consistent_output_shape(self) -> None:
        """Test that sliding window produces same shape as full-volume."""
        device = torch.device("cpu")
        model = FixedOutputModel()
        roi_size = (32, 32, 32)

        full_volume_inferer = FullVolumeInferer()
        sliding_window_inferer = SlidingWindowInferer(roi_size=roi_size)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)

        outputs_full = full_volume_inferer.infer(model, inputs, device)
        outputs_sw = sliding_window_inferer.infer(model, inputs, device)

        # Shapes should match
        assert outputs_full.shape == outputs_sw.shape == (1, 2, 64, 64, 64)

    def test_sliding_window_deterministic(self) -> None:
        """Test that sliding window produces consistent results."""
        device = torch.device("cpu")
        model = FixedOutputModel(value=0.5)
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)

        # Run inference twice
        outputs1 = inferer.infer(model, inputs, device, use_amp=False)
        outputs2 = inferer.infer(model, inputs, device, use_amp=False)

        # Results should be identical
        assert torch.allclose(outputs1, outputs2), (
            "Sliding window inference not deterministic"
        )
