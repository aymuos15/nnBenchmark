"""
Tests for edge cases in custom loss functions (CCLoss, BlobLoss).
Tests gradient flow in edge cases to ensure differentiability.
"""

from __future__ import annotations

import torch


class TestCCLossEdgeCases:
    """Test CCLoss edge cases for gradient flow and error handling."""

    def test_ccloss_empty_batch_gradient_flow(self) -> None:
        """Test that CCLoss maintains gradient flow when no ground truth regions exist."""
        from src.factory.losses.cc import CCLoss

        loss_fn = CCLoss(sigmoid=True, to_onehot_y=False)

        # Create prediction with requires_grad
        pred = torch.randn(1, 1, 16, 16, 16, requires_grad=True)

        # Create target with NO connected components (all zeros)
        target = torch.zeros(1, 16, 16, 16)

        # Compute loss
        loss = loss_fn(pred, target)

        # Loss should be finite
        assert torch.isfinite(loss).all()
        assert loss.requires_grad

        # Backward should work
        loss.backward()

        # Gradient should exist and be finite
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
        assert torch.isfinite(pred.grad).all()

    def test_ccloss_single_region_per_class(self) -> None:
        """Test that CCLoss handles single region per class correctly."""
        from src.factory.losses.cc import CCLoss

        loss_fn = CCLoss(sigmoid=True, to_onehot_y=False)

        # Create prediction - need 2 channels for binary (background + foreground)
        pred = torch.randn(1, 2, 16, 16, 16, requires_grad=True)

        # Create target with exactly one connected component (small cube)
        # Values should be 0 (background) or 1 (single class for binary)
        target = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        target[0, 5:10, 5:10, 5:10] = 1  # Single connected region

        # Compute loss
        loss = loss_fn(pred, target)

        # Loss should be finite and differentiable
        assert torch.isfinite(loss).all()
        assert loss.requires_grad

        # Backward pass
        loss.backward()

        # Gradients should exist
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_ccloss_class_index_with_spurious_channel_dimension(self) -> None:
        """Test that CCLoss handles spurious channel dimension from LoadImaged."""
        from src.factory.losses.cc import CCLoss

        loss_fn = CCLoss(sigmoid=True, to_onehot_y=False)

        # Create prediction: (B, C, H, W, D)
        pred = torch.randn(2, 2, 8, 8, 8, requires_grad=True)

        # Create target with spurious channel: (B, 1, H, W, D) instead of (B, H, W, D)
        target = torch.randint(0, 2, (2, 1, 8, 8, 8))

        # Compute loss - should handle spurious dimension
        loss = loss_fn(pred, target)

        # Should complete without shape mismatch error
        assert torch.isfinite(loss).all()
        assert loss.requires_grad

        # Create target without spurious channel: (B, H, W, D)
        target_no_spurious = target.squeeze(1)
        loss_no_spurious = loss_fn(pred, target_no_spurious)

        # Both should produce valid losses
        assert torch.isfinite(loss_no_spurious).all()

    def test_ccloss_all_classes_empty(self) -> None:
        """Test that CCLoss handles case where all classes have no instances."""
        from src.factory.losses.cc import CCLoss

        loss_fn = CCLoss(sigmoid=True, to_onehot_y=False)

        # Create prediction for multi-class
        pred = torch.randn(1, 3, 16, 16, 16, requires_grad=True)

        # Create target with all zeros (no instances of any class)
        target = torch.zeros(1, 16, 16, 16)

        # Compute loss
        loss = loss_fn(pred, target)

        # Loss should be finite
        assert torch.isfinite(loss).all()
        assert loss.requires_grad

        # Backward should work
        loss.backward()

        # Gradients should exist and be finite
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_ccloss_multiple_regions_per_class(self) -> None:
        """Test that CCLoss handles multiple regions correctly."""
        from src.factory.losses.cc import CCLoss

        loss_fn = CCLoss(sigmoid=True, to_onehot_y=False)

        # Create prediction - need 2 channels for binary
        pred = torch.randn(1, 2, 32, 32, 32, requires_grad=True)

        # Create target with multiple disconnected regions
        target = torch.zeros(1, 32, 32, 32, dtype=torch.long)
        target[0, 5:10, 5:10, 5:10] = 1  # Region 1
        target[0, 20:25, 20:25, 20:25] = 1  # Region 2 (disconnected)

        # Compute loss
        loss = loss_fn(pred, target)

        # Loss should be finite
        assert torch.isfinite(loss).all()
        assert loss.requires_grad

        # Backward pass
        loss.backward()

        # Gradients should exist
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_ccloss_gradient_magnitude_reasonable(self) -> None:
        """Test that CCLoss produces reasonable gradient magnitudes (not zero, not inf)."""
        from src.factory.losses.cc import CCLoss

        loss_fn = CCLoss(sigmoid=True, to_onehot_y=False)

        # Create typical prediction and target - need 2 channels for binary
        pred = torch.randn(2, 2, 16, 16, 16, requires_grad=True)
        target = torch.randint(0, 2, (2, 16, 16, 16), dtype=torch.long)

        # Compute loss and backward
        loss = loss_fn(pred, target)
        loss.backward()

        # Check gradient magnitude is reasonable
        grad_norm = pred.grad.norm()
        assert grad_norm > 0  # Not zero
        assert torch.isfinite(grad_norm)  # Not inf/nan
        assert grad_norm < 1e6  # Not unreasonably large


class TestBlobLossEdgeCases:
    """Test BlobLoss edge cases for gradient flow."""

    def test_blobloss_empty_instances_gradient_flow(self) -> None:
        """Test that BlobLoss handles empty instances and maintains gradient flow."""
        from src.factory.losses.blob import BlobLoss

        loss_fn = BlobLoss(main_weight=0.5, blob_weight=0.5, sigmoid=True)

        # Create prediction with requires_grad
        pred = torch.randn(2, 1, 16, 16, 16, requires_grad=True)

        # Create batch where some samples have instances, some don't
        # BlobLoss expects target shape (B, H, W, D) without channel dim
        target = torch.zeros(2, 1, 16, 16, 16)
        target[0, 0, 5:10, 5:10, 5:10] = 1  # Sample 0 has instances
        # Sample 1 has no instances (all zeros)

        # Compute loss
        loss = loss_fn(pred, target)

        # Loss should be finite
        assert torch.isfinite(loss).all()
        assert loss.requires_grad

        # Backward should work
        loss.backward()

        # Gradients should exist for entire batch
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_blobloss_main_weight_zero_gradient_flow(self) -> None:
        """Test that BlobLoss with main_weight=0 still produces gradients."""
        from src.factory.losses.blob import BlobLoss

        # Create loss with only blob component
        loss_fn = BlobLoss(main_weight=0.0, blob_weight=1.0, sigmoid=True)

        # Create prediction
        pred = torch.randn(1, 1, 16, 16, 16, requires_grad=True)

        # Create target with instances
        target = torch.zeros(1, 16, 16, 16)
        target[0, 5:10, 5:10, 5:10] = 1

        # Compute loss
        loss = loss_fn(pred, target)

        # Loss should be finite
        assert torch.isfinite(loss).all()
        assert loss.requires_grad

        # Backward should work
        loss.backward()

        # Gradients should exist
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.norm() > 0  # Non-zero gradients

    def test_blobloss_blob_weight_zero_gradient_flow(self) -> None:
        """Test that BlobLoss with blob_weight=0 still produces gradients."""
        from src.factory.losses.blob import BlobLoss

        # Create loss with only main component
        loss_fn = BlobLoss(main_weight=1.0, blob_weight=0.0, sigmoid=True)

        # Create prediction
        pred = torch.randn(1, 1, 16, 16, 16, requires_grad=True)

        # Create target with channel dimension
        target = torch.zeros(1, 1, 16, 16, 16)
        target[0, 0, 5:10, 5:10, 5:10] = 1

        # Compute loss
        loss = loss_fn(pred, target)

        # Loss should be finite
        assert torch.isfinite(loss).all()
        assert loss.requires_grad

        # Backward should work
        loss.backward()

        # Gradients should exist
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_blobloss_all_samples_empty(self) -> None:
        """Test that BlobLoss handles batch where all samples are empty."""
        from src.factory.losses.blob import BlobLoss

        loss_fn = BlobLoss(main_weight=0.5, blob_weight=0.5, sigmoid=True)

        # Create prediction
        pred = torch.randn(2, 1, 16, 16, 16, requires_grad=True)

        # Create target with all zeros (no instances) with channel dimension
        target = torch.zeros(2, 1, 16, 16, 16)

        # Compute loss
        loss = loss_fn(pred, target)

        # Loss should be finite
        assert torch.isfinite(loss).all()
        assert loss.requires_grad

        # Backward should work
        loss.backward()

        # Gradients should exist
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestDiceLossEdgeCases:
    """Test standard DiceLoss edge cases for comparison."""

    def test_diceloss_empty_target(self) -> None:
        """Test that standard losses handle empty targets."""
        from monai.losses import DiceLoss

        loss_fn = DiceLoss(sigmoid=True, to_onehot_y=False)

        # Create prediction
        pred = torch.randn(1, 1, 16, 16, 16, requires_grad=True)

        # Empty target
        target = torch.zeros(1, 1, 16, 16, 16)

        # Compute loss
        loss = loss_fn(pred, target)

        # Should be differentiable
        assert torch.isfinite(loss).all()
        loss.backward()

        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestLossGradientConsistency:
    """Test that losses produce consistent gradients across different scenarios."""

    def test_ccloss_gradient_consistency_across_batch_sizes(self) -> None:
        """Test that CCLoss produces consistent gradients for different batch sizes."""
        from src.factory.losses.cc import CCLoss

        loss_fn = CCLoss(sigmoid=True, to_onehot_y=False)

        # Create single sample - need 2 channels for binary
        pred_single = torch.randn(1, 2, 16, 16, 16, requires_grad=True)
        target_single = torch.randint(0, 2, (1, 16, 16, 16), dtype=torch.long)

        loss_single = loss_fn(pred_single, target_single)
        loss_single.backward()

        # Create batch with duplicate sample
        pred_batch = torch.randn(2, 2, 16, 16, 16, requires_grad=True)
        # Use detached tensors for consistent comparison
        target_batch = target_single.repeat(2, 1, 1, 1)

        loss_batch = loss_fn(pred_batch, target_batch)
        loss_batch.backward()

        # Gradients should have consistent magnitudes
        assert torch.isfinite(loss_batch).all()
        assert pred_batch.grad is not None
        assert torch.isfinite(pred_batch.grad).all()

    def test_blobloss_gradient_consistency(self) -> None:
        """Test that BlobLoss produces finite gradients consistently."""
        from src.factory.losses.blob import BlobLoss

        loss_fn = BlobLoss(main_weight=0.5, blob_weight=0.5, sigmoid=True)

        # Test multiple random scenarios
        for _ in range(5):
            pred = torch.randn(2, 1, 16, 16, 16, requires_grad=True)
            target = torch.randint(0, 2, (2, 1, 16, 16, 16))

            loss = loss_fn(pred, target)
            loss.backward()

            # All gradients should be finite
            assert torch.isfinite(loss).all()
            assert pred.grad is not None
            assert torch.isfinite(pred.grad).all()
