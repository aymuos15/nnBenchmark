"""
Tests for src.utils.viz module.
Tests visualization utilities for saving validation visualizations.
"""

from __future__ import annotations

import os

import torch

from src.plotting.validation import save_validation_visualizations


class TestSaveValidationVisualizations:
    """Tests for save_validation_visualizations function."""

    def test_save_2d_visualization(self, temp_dir: str) -> None:
        """Test saving 2D validation visualization."""
        batch_size = 2
        channels = 1
        height = 64
        width = 64

        # Create dummy tensors
        images = torch.randn(batch_size, channels, height, width)
        labels = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        predictions = torch.randint(0, 2, (batch_size, 1, height, width)).float()

        save_validation_visualizations(
            images, labels, predictions, temp_dir, epoch=1, spatial_dims=2
        )

        # Check that visualization directory was created
        vis_dir = os.path.join(temp_dir, "visualizations")
        assert os.path.exists(vis_dir)

        # Check that image file was created
        viz_file = os.path.join(vis_dir, "validation_epoch_001.png")
        assert os.path.exists(viz_file)
        assert os.path.getsize(viz_file) > 0

    def test_save_3d_visualization(self, temp_dir: str) -> None:
        """Test saving 3D validation visualization."""
        batch_size = 1
        channels = 1
        depth = 32
        height = 64
        width = 64

        # Create dummy tensors
        images = torch.randn(batch_size, channels, depth, height, width)
        labels = torch.randint(0, 2, (batch_size, 1, depth, height, width)).float()
        predictions = torch.randint(0, 2, (batch_size, 1, depth, height, width)).float()

        save_validation_visualizations(
            images, labels, predictions, temp_dir, epoch=5, spatial_dims=3
        )

        # Check that visualization was created
        vis_dir = os.path.join(temp_dir, "visualizations")
        assert os.path.exists(vis_dir)

        viz_file = os.path.join(vis_dir, "validation_epoch_005.png")
        assert os.path.exists(viz_file)

    def test_visualization_with_rgb_images(self, temp_dir: str) -> None:
        """Test visualization with RGB input images."""
        batch_size = 1
        # RGB image: 3 channels
        images = torch.randn(batch_size, 3, 64, 64)
        labels = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()
        predictions = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()

        save_validation_visualizations(
            images, labels, predictions, temp_dir, epoch=1, spatial_dims=2
        )

        vis_dir = os.path.join(temp_dir, "visualizations")
        viz_file = os.path.join(vis_dir, "validation_epoch_001.png")
        assert os.path.exists(viz_file)

    def test_visualization_tensor_on_gpu(self, temp_dir: str) -> None:
        """Test visualization with tensors on GPU if available."""
        batch_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        images = torch.randn(batch_size, 1, 32, 32, device=device)
        labels = torch.randint(0, 2, (batch_size, 1, 32, 32), device=device).float()
        predictions = torch.randint(
            0, 2, (batch_size, 1, 32, 32), device=device
        ).float()

        # Should handle GPU tensors correctly
        save_validation_visualizations(
            images, labels, predictions, temp_dir, epoch=1, spatial_dims=2
        )

        vis_dir = os.path.join(temp_dir, "visualizations")
        viz_file = os.path.join(vis_dir, "validation_epoch_001.png")
        assert os.path.exists(viz_file)

    def test_visualization_multiclass_predictions(self, temp_dir: str) -> None:
        """Test visualization with multi-class predictions."""
        batch_size = 1
        num_classes = 3
        # Multi-class predictions as class indices
        images = torch.randn(batch_size, 1, 32, 32)
        labels = torch.randint(0, num_classes, (batch_size, 1, 32, 32)).float()
        predictions = torch.randint(0, num_classes, (batch_size, 1, 32, 32)).float()

        save_validation_visualizations(
            images, labels, predictions, temp_dir, epoch=1, spatial_dims=2
        )

        vis_dir = os.path.join(temp_dir, "visualizations")
        viz_file = os.path.join(vis_dir, "validation_epoch_001.png")
        assert os.path.exists(viz_file)

    def test_visualization_3d_takes_middle_slice(self, temp_dir: str) -> None:
        """Test that 3D visualization takes middle slice along depth."""
        batch_size = 1
        depth = 16
        height = 32
        width = 32

        images = torch.randn(batch_size, 1, depth, height, width)
        labels = torch.randint(0, 2, (batch_size, 1, depth, height, width)).float()
        predictions = torch.randint(0, 2, (batch_size, 1, depth, height, width)).float()

        # Should not raise error when processing 3D data
        save_validation_visualizations(
            images, labels, predictions, temp_dir, epoch=1, spatial_dims=3
        )

        vis_file = os.path.join(temp_dir, "visualizations", "validation_epoch_001.png")
        assert os.path.exists(vis_file)
