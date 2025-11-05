"""
Unit tests for KiU-Net model implementation.

Tests the configurable KiU-Net architecture including:
- Forward pass for 2D and 3D variants
- Deep supervision functionality
- Parameter configurability
- Model registry integration
- Config validation
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.config.validation import validate_model_config
from src.factory.models.kiunet import KiUNet, KiUNet2D, KiUNet3D
from src.factory.models.registry import model_registry


class TestKiUNetArchitecture:
    """Tests for core KiU-Net architecture."""

    @pytest.mark.parametrize(
        "spatial_dims,input_shape",
        [
            (2, (2, 1, 64, 64)),  # 2D: batch=2, channels=1, H=64, W=64
            (3, (2, 1, 32, 32, 32)),  # 3D: batch=2, channels=1, H=32, W=32, D=32
        ],
    )
    def test_forward_pass_basic(
        self, spatial_dims: int, input_shape: tuple[int, ...]
    ) -> None:
        """Test basic forward pass produces correct output shape."""
        model = KiUNet(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=2,
            features=[16, 32, 64],
            deep_supervision=False,
        )

        x = torch.randn(*input_shape)
        output = model(x)

        # Output should match input spatial size with out_channels
        expected_shape = (input_shape[0], 2) + input_shape[2:]
        assert output.shape == expected_shape

    @pytest.mark.parametrize(
        "spatial_dims,input_shape",
        [
            (2, (2, 1, 64, 64)),
            (3, (2, 1, 32, 32, 32)),
        ],
    )
    def test_forward_pass_with_deep_supervision(
        self, spatial_dims: int, input_shape: tuple[int, ...]
    ) -> None:
        """Test forward pass with deep supervision returns multiple outputs."""
        model = KiUNet(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=2,
            features=[16, 32, 64],
            deep_supervision=True,
            deep_supr_num=2,
        )

        x = torch.randn(*input_shape)
        output = model(x)

        # Should return tuple of (main_output, [aux_outputs])
        assert isinstance(output, tuple)
        assert len(output) == 2
        main_output, aux_outputs = output

        # Main output shape
        expected_shape = (input_shape[0], 2) + input_shape[2:]
        assert main_output.shape == expected_shape

        # Auxiliary outputs
        assert isinstance(aux_outputs, list)
        assert len(aux_outputs) == 2  # deep_supr_num=2
        for aux_out in aux_outputs:
            assert aux_out.shape == expected_shape  # Upsampled to input size

    def test_different_feature_levels(self) -> None:
        """Test KiU-Net with different numbers of encoder levels."""
        # 2-level encoder
        model_2 = KiUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            features=[32, 64],
        )
        x = torch.randn(1, 1, 64, 64)
        output_2 = model_2(x)
        assert output_2.shape == (1, 2, 64, 64)

        # 4-level encoder
        model_4 = KiUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            features=[16, 32, 64, 128],
        )
        output_4 = model_4(x)
        assert output_4.shape == (1, 2, 64, 64)

    @pytest.mark.parametrize(
        "norm_name",
        ["batch", "instance", "group"],
    )
    def test_different_normalizations(self, norm_name: str) -> None:
        """Test KiU-Net with different normalization types."""
        model = KiUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            features=[16, 32],
            norm_name=norm_name,
        )

        x = torch.randn(2, 1, 64, 64)
        output = model(x)
        assert output.shape == (2, 2, 64, 64)

    @pytest.mark.parametrize(
        "act_name",
        ["relu", "leakyrelu", "prelu"],
    )
    def test_different_activations(self, act_name: str) -> None:
        """Test KiU-Net with different activation functions."""
        model = KiUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            features=[16, 32],
            act_name=act_name,
        )

        x = torch.randn(2, 1, 64, 64)
        output = model(x)
        assert output.shape == (2, 2, 64, 64)

    def test_multi_class_segmentation(self) -> None:
        """Test KiU-Net with multiple output classes."""
        model = KiUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=5,  # 5-class segmentation
            features=[16, 32, 64],
        )

        x = torch.randn(2, 1, 64, 64)
        output = model(x)
        assert output.shape == (2, 5, 64, 64)

    def test_multi_channel_input(self) -> None:
        """Test KiU-Net with multi-channel input (e.g., RGB or multi-modal)."""
        model = KiUNet(
            spatial_dims=2,
            in_channels=3,  # RGB or 3-modality input
            out_channels=2,
            features=[16, 32, 64],
        )

        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        assert output.shape == (2, 2, 64, 64)


class TestKiUNet2D:
    """Tests for KiUNet2D convenience function."""

    def test_kiunet2d_instantiation(self) -> None:
        """Test that KiUNet2D creates a 2D model."""
        model = KiUNet2D(
            in_channels=1,
            out_channels=2,
            features=[16, 32, 64],
        )

        assert model.spatial_dims == 2
        assert model.in_channels == 1
        assert model.out_channels == 2

    def test_kiunet2d_forward(self) -> None:
        """Test forward pass through KiUNet2D."""
        model = KiUNet2D(
            in_channels=1,
            out_channels=2,
            features=[16, 32, 64],
        )

        x = torch.randn(2, 1, 128, 128)
        output = model(x)
        assert output.shape == (2, 2, 128, 128)


class TestKiUNet3D:
    """Tests for KiUNet3D convenience function."""

    def test_kiunet3d_instantiation(self) -> None:
        """Test that KiUNet3D creates a 3D model."""
        model = KiUNet3D(
            in_channels=1,
            out_channels=2,
            features=[16, 32, 64],
        )

        assert model.spatial_dims == 3
        assert model.in_channels == 1
        assert model.out_channels == 2

    def test_kiunet3d_forward(self) -> None:
        """Test forward pass through KiUNet3D."""
        model = KiUNet3D(
            in_channels=1,
            out_channels=2,
            features=[16, 32, 64],
        )

        x = torch.randn(2, 1, 32, 32, 32)
        output = model(x)
        assert output.shape == (2, 2, 32, 32, 32)


class TestModelRegistry:
    """Tests for KiU-Net integration with model registry."""

    def test_kiunet2d_registered(self) -> None:
        """Test that KiUNet2D is registered in model registry."""
        assert "KiUNet2D" in model_registry._registry

    def test_kiunet3d_registered(self) -> None:
        """Test that KiUNet3D is registered in model registry."""
        assert "KiUNet3D" in model_registry._registry

    def test_build_kiunet2d_from_config(self) -> None:
        """Test building KiUNet2D from config."""
        config = {
            "type": "KiUNet2D",
            "in_channels": 1,
            "out_channels": 2,
            "features": [16, 32, 64],
            "norm_name": "instance",
            "act_name": "relu",
            "deep_supervision": False,
        }

        device = torch.device("cpu")
        model = model_registry.build(config, device)

        assert isinstance(model, nn.Module)
        assert model.spatial_dims == 2

        # Test forward pass
        x = torch.randn(1, 1, 64, 64)
        output = model(x)
        assert output.shape == (1, 2, 64, 64)

    def test_build_kiunet3d_from_config(self) -> None:
        """Test building KiUNet3D from config."""
        config = {
            "type": "KiUNet3D",
            "in_channels": 1,
            "out_channels": 2,
            "features": [16, 32, 64],
            "norm_name": "instance",
            "act_name": "relu",
            "deep_supervision": False,
        }

        device = torch.device("cpu")
        model = model_registry.build(config, device)

        assert isinstance(model, nn.Module)
        assert model.spatial_dims == 3

        # Test forward pass
        x = torch.randn(1, 1, 32, 32, 32)
        output = model(x)
        assert output.shape == (1, 2, 32, 32, 32)

    def test_build_kiunet_with_nested_config(self) -> None:
        """Test building KiU-Net with nested config format."""
        config = {
            "type": "KiUNet2D",
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 2,
            "deep_supervision": True,
            "ds_weights": [1.0, 0.5],
            "deep_supr_num": 1,
            "KiUNet2D": {
                "features": [32, 64, 128],
                "norm_name": "batch",
                "act_name": "leakyrelu",
            },
        }

        device = torch.device("cpu")
        model = model_registry.build(config, device)

        # Test forward pass with deep supervision
        x = torch.randn(1, 1, 64, 64)
        output = model(x)
        assert isinstance(output, tuple)
        main_output, aux_outputs = output
        assert main_output.shape == (1, 2, 64, 64)
        assert len(aux_outputs) == 1


class TestConfigValidation:
    """Tests for KiU-Net config validation."""

    def test_valid_kiunet2d_config(self) -> None:
        """Test that valid KiUNet2D config passes validation."""
        config = {
            "model": {
                "type": "KiUNet2D",
                "spatial_dims": 2,
                "in_channels": 1,
                "out_channels": 2,
                "features": [16, 32, 64],
            }
        }
        # Should not raise
        validate_model_config(config)

    def test_valid_kiunet3d_config(self) -> None:
        """Test that valid KiUNet3D config passes validation."""
        config = {
            "model": {
                "type": "KiUNet3D",
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 2,
                "features": [16, 32, 64],
            }
        }
        # Should not raise
        validate_model_config(config)

    def test_missing_features_raises_error(self) -> None:
        """Test that missing 'features' parameter raises validation error."""
        config = {
            "model": {
                "type": "KiUNet2D",
                "spatial_dims": 2,
                "in_channels": 1,
                "out_channels": 2,
                # Missing 'features'
            }
        }
        with pytest.raises(ValueError, match="Missing required KiUNet parameter"):
            validate_model_config(config)

    def test_invalid_features_type_raises_error(self) -> None:
        """Test that invalid 'features' type raises validation error."""
        config = {
            "model": {
                "type": "KiUNet2D",
                "spatial_dims": 2,
                "in_channels": 1,
                "out_channels": 2,
                "features": 64,  # Should be list/tuple
            }
        }
        with pytest.raises(ValueError, match="must be a list or tuple"):
            validate_model_config(config)

    def test_insufficient_feature_levels_raises_error(self) -> None:
        """Test that too few feature levels raises validation error."""
        config = {
            "model": {
                "type": "KiUNet2D",
                "spatial_dims": 2,
                "in_channels": 1,
                "out_channels": 2,
                "features": [16],  # Need at least 2 levels
            }
        }
        with pytest.raises(ValueError, match="at least 2 levels"):
            validate_model_config(config)

    def test_nested_config_validation(self) -> None:
        """Test validation of nested KiU-Net config."""
        config = {
            "model": {
                "type": "KiUNet2D",
                "spatial_dims": 2,
                "in_channels": 1,
                "out_channels": 2,
                "deep_supervision": True,
                "ds_weights": [1.0, 0.5],
                "KiUNet2D": {
                    "features": [32, 64, 128],
                    "norm_name": "instance",
                },
            }
        }
        # Should not raise
        validate_model_config(config)


class TestGradientFlow:
    """Tests for gradient flow through KiU-Net."""

    def test_gradients_flow_through_model(self) -> None:
        """Test that gradients flow correctly through the model."""
        model = KiUNet2D(
            in_channels=1,
            out_channels=2,
            features=[16, 32, 64],
        )

        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        output = model(x)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_gradients_with_deep_supervision(self) -> None:
        """Test gradient flow with deep supervision enabled."""
        model = KiUNet2D(
            in_channels=1,
            out_channels=2,
            features=[16, 32, 64],
            deep_supervision=True,
            deep_supr_num=2,
        )

        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        main_output, aux_outputs = model(x)

        # Compute loss from all outputs
        loss = main_output.sum()
        for aux in aux_outputs:
            loss += 0.5 * aux.sum()

        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
