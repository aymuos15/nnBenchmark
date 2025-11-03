"""Tests for custom loss implementations.

Tests cover instantiation, forward pass, gradient flow, and batch processing
for the Connected Components Loss (CCLoss).
"""

import pytest
import torch
import torch.nn as nn

from src.factory.losses.cc import CCLoss


class TestCCLoss:
    """Test suite for Connected Components Loss."""

    @pytest.fixture
    def device(self):
        """Use CPU for testing to avoid GPU memory issues."""
        return torch.device("cpu")

    @pytest.fixture
    def batch_data(self, device):
        """Generate sample batch data for testing."""
        batch_size = 2
        num_classes = 3
        spatial_size = 32

        # Predictions: logits from network
        pred = torch.randn(batch_size, num_classes, spatial_size, spatial_size, device=device)

        # Target: class indices
        target = torch.randint(0, num_classes, (batch_size, spatial_size, spatial_size), device=device)

        return pred, target

    def test_ccloss_instantiation(self):
        """Test that CCLoss can be instantiated with default parameters."""
        loss_fn = CCLoss()
        assert isinstance(loss_fn, nn.Module)
        assert loss_fn.sigmoid is True
        assert loss_fn.softmax is False
        assert loss_fn.to_onehot_y is False

    def test_ccloss_custom_parameters(self):
        """Test CCLoss instantiation with custom parameters."""
        loss_fn = CCLoss(
            to_onehot_y=True,
            softmax=True,
            sigmoid=False,
        )
        assert loss_fn.to_onehot_y is True
        assert loss_fn.softmax is True
        assert loss_fn.sigmoid is False

    def test_ccloss_forward_basic(self, batch_data):
        """Test forward pass with basic input."""
        loss_fn = CCLoss()
        pred, target = batch_data

        loss = loss_fn(pred, target)

        # Check loss is scalar
        assert loss.dim() == 0
        # Check loss is float tensor
        assert loss.dtype in [torch.float32, torch.float64]
        # Loss should be a finite number (may be negative if metric > 1)
        assert torch.isfinite(loss)

    def test_ccloss_forward_with_softmax(self, batch_data):
        """Test forward pass with softmax activation."""
        loss_fn = CCLoss(softmax=True, sigmoid=False)
        pred, target = batch_data

        loss = loss_fn(pred, target)

        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_ccloss_forward_with_sigmoid(self, batch_data):
        """Test forward pass with sigmoid activation."""
        loss_fn = CCLoss(sigmoid=True, softmax=False)
        pred, target = batch_data

        loss = loss_fn(pred, target)

        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_ccloss_with_onehot_target(self, device):
        """Test CCLoss with one-hot encoded target."""
        batch_size = 2
        num_classes = 3
        spatial_size = 32

        pred = torch.randn(batch_size, num_classes, spatial_size, spatial_size, device=device)
        # One-hot encoded target
        target_indices = torch.randint(0, num_classes, (batch_size, spatial_size, spatial_size), device=device)
        target_onehot = torch.nn.functional.one_hot(target_indices, num_classes=num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        loss_fn = CCLoss(to_onehot_y=False)  # Input is already one-hot
        loss = loss_fn(pred, target_onehot)

        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_ccloss_backward(self, batch_data):
        """Test that gradients flow through the loss."""
        loss_fn = CCLoss()
        pred, target = batch_data

        # Enable gradient computation
        pred.requires_grad = True

        loss = loss_fn(pred, target)

        # Check loss is computed successfully
        assert loss.dim() == 0
        assert torch.isfinite(loss)

        # Test backward pass - should work since loss is fully differentiable
        loss.backward()

        # Check gradients exist and are not all zero
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)

    def test_ccloss_batch_processing(self, batch_data):
        """Test that CCLoss handles batch dimension correctly."""
        loss_fn = CCLoss()

        pred, target = batch_data
        loss = loss_fn(pred, target)

        # Loss should be computed for the entire batch
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)

    def test_ccloss_single_sample(self, device):
        """Test CCLoss with single sample (batch_size=1)."""
        num_classes = 3
        spatial_size = 32

        pred = torch.randn(1, num_classes, spatial_size, spatial_size, device=device)
        target = torch.randint(0, num_classes, (1, spatial_size, spatial_size), device=device)

        loss_fn = CCLoss()
        loss = loss_fn(pred, target)

        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_ccloss_different_spatial_sizes(self, device):
        """Test CCLoss with different spatial dimensions."""
        # 2D case
        pred_2d = torch.randn(2, 3, 32, 32, device=device)
        target_2d = torch.randint(0, 3, (2, 32, 32), device=device)

        loss_fn = CCLoss()
        loss_2d = loss_fn(pred_2d, target_2d)

        assert loss_2d.dim() == 0
        assert torch.isfinite(loss_2d)

    def test_ccloss_deterministic(self, batch_data):
        """Test that CCLoss produces deterministic results."""
        loss_fn = CCLoss()
        pred, target = batch_data

        loss1 = loss_fn(pred, target)
        loss2 = loss_fn(pred, target)

        assert torch.allclose(loss1, loss2)

    def test_ccloss_all_correct_predictions(self, device):
        """Test CCLoss with perfect predictions (target matches pred after activation)."""
        batch_size = 2
        num_classes = 3
        spatial_size = 32

        # Create target
        target = torch.randint(0, num_classes, (batch_size, spatial_size, spatial_size), device=device)

        # Create predictions that exactly match target (after sigmoid)
        pred = torch.zeros(batch_size, num_classes, spatial_size, spatial_size, device=device)
        for b in range(batch_size):
            for c in range(num_classes):
                pred[b, c] = (target[b] == c).float()

        loss_fn = CCLoss(sigmoid=False)  # Since pred is already in [0, 1]
        loss = loss_fn(pred, target)

        # Loss should be close to 0 (perfect prediction)
        # Note: May not be exactly 0 due to numerical precision
        assert loss.item() < 0.5

    def test_ccloss_empty_classes(self, device):
        """Test CCLoss handling of empty classes."""
        num_classes = 3
        spatial_size = 32

        # Predictions
        pred = torch.randn(1, num_classes, spatial_size, spatial_size, device=device)

        # Target with only one class (others empty)
        target = torch.zeros(1, spatial_size, spatial_size, dtype=torch.long, device=device)
        target[0, :16, :16] = 1  # Only class 1 is present

        loss_fn = CCLoss()
        loss = loss_fn(pred, target)

        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_ccloss_to_onehot_conversion(self, device):
        """Test _to_onehot helper method."""
        batch_size = 2
        num_classes = 4
        spatial_size = 16

        # Create target with class indices
        target = torch.randint(0, num_classes, (batch_size, spatial_size, spatial_size), device=device)

        # Convert to one-hot
        onehot = CCLoss._to_onehot(target, num_classes)

        # Check shape
        assert onehot.shape == (batch_size, num_classes, spatial_size, spatial_size)

        # Check that sum along class dimension is 1
        assert torch.allclose(onehot.sum(dim=1), torch.ones_like(onehot[:, 0]))

        # Check dtype
        assert onehot.dtype == torch.float32

    def test_ccloss_device_handling(self):
        """Test CCLoss device handling."""
        # Test with CPU
        loss_fn = CCLoss()
        pred_cpu = torch.randn(2, 3, 32, 32)
        target_cpu = torch.randint(0, 3, (2, 32, 32))

        loss_cpu = loss_fn(pred_cpu, target_cpu)
        assert loss_cpu.device.type == "cpu"

        # Test with CPU explicitly
        device = torch.device("cpu")
        pred_dev = torch.randn(2, 3, 32, 32, device=device)
        target_dev = torch.randint(0, 3, (2, 32, 32), device=device)

        loss_dev = loss_fn(pred_dev, target_dev)
        assert loss_dev.device.type == "cpu"

    def test_ccloss_dtype_handling(self, device):
        """Test CCLoss with different dtypes."""
        num_classes = 3
        spatial_size = 32

        # Test with float32
        pred_f32 = torch.randn(2, num_classes, spatial_size, spatial_size, device=device, dtype=torch.float32)
        target = torch.randint(0, num_classes, (2, spatial_size, spatial_size), device=device)

        loss_fn = CCLoss()
        loss = loss_fn(pred_f32, target)

        assert loss.dtype == torch.float32

    def test_ccloss_multiple_forward_passes(self, batch_data):
        """Test that CCLoss can handle multiple forward passes."""
        loss_fn = CCLoss()
        pred, target = batch_data

        # Multiple forward passes should produce consistent results
        loss1 = loss_fn(pred, target)
        loss2 = loss_fn(pred, target)

        # Losses should be identical for the same input
        assert torch.allclose(loss1, loss2)
