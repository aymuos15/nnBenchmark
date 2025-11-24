"""Tests for CCMetric FP/TP/FN classification and tracking.

Tests cover:
- Basic FP/TP/FN classification logic
- Instance size binning
- Per-sample tracking
- Edge cases (empty predictions, empty GT, no overlap)
"""

import pytest
import torch

from src.factory.metrics.cc import CCMetric


@pytest.mark.gpu_deps
class TestCCMetricFPTPFN:
    """Test suite for CCMetric FP/TP/FN functionality."""

    @pytest.fixture
    def device(self):
        """Use CPU for testing to avoid GPU memory issues."""
        return torch.device("cpu")

    @pytest.fixture
    def simple_metric(self):
        """Create a simple CCMetric instance for testing."""
        return CCMetric(
            include_background=False,
            reduction="mean_batch",
            num_classes=2,  # Background + 1 foreground class
            metric_type="dice",
        )

    def test_basic_tp_classification(self, simple_metric, device):
        """Test that GT instances with prediction overlap are classified as TP."""
        # Create simple scenario: 1 GT instance, prediction overlaps it
        batch_size = 1
        num_classes = 2
        h, w = 10, 10

        # GT: single instance at center (3x3 region)
        target = torch.zeros(batch_size, 1, h, w, device=device)
        target[0, 0, 4:7, 4:7] = 1  # 9 pixels

        # Prediction: overlaps with GT (softmax probabilities)
        pred = torch.zeros(batch_size, num_classes, h, w, device=device)
        pred[0, 0, :, :] = 0.1  # Background low probability
        pred[0, 1, 4:7, 4:7] = 0.9  # Foreground high probability where GT is

        # Reset and run metric
        simple_metric.reset_instance_scores()
        simple_metric(pred, target)

        # Get FP/TP/FN statistics
        stats = simple_metric.get_fp_tp_fn_statistics()

        # Should have 1 TP (GT was detected), 0 FN, 0 FP
        assert stats["all"]["TP"] == 1
        assert stats["all"]["FN"] == 0
        assert stats["all"]["FP"] == 0

    def test_basic_fn_classification(self, simple_metric, device):
        """Test that GT instances without prediction overlap are classified as FN."""
        batch_size = 1
        num_classes = 2
        h, w = 10, 10

        # GT: single instance at center
        target = torch.zeros(batch_size, 1, h, w, device=device)
        target[0, 0, 4:7, 4:7] = 1

        # Prediction: NO overlap with GT (predicts elsewhere)
        pred = torch.zeros(batch_size, num_classes, h, w, device=device)
        pred[0, 0, :, :] = 0.9  # Background high probability
        pred[0, 1, 0:2, 0:2] = 0.9  # Foreground prediction away from GT

        # Reset and run metric
        simple_metric.reset_instance_scores()
        simple_metric(pred, target)

        # Get FP/TP/FN statistics
        stats = simple_metric.get_fp_tp_fn_statistics()

        # Should have 0 TP, 1 FN (GT was missed), some FP
        assert stats["all"]["TP"] == 0
        assert stats["all"]["FN"] == 1
        assert stats["all"]["FP"] > 0  # The prediction elsewhere is FP

    def test_basic_fp_classification(self, simple_metric, device):
        """Test that predicted instances without GT overlap are classified as FP."""
        batch_size = 1
        num_classes = 2
        h, w = 10, 10

        # GT: single instance at center
        target = torch.zeros(batch_size, 1, h, w, device=device)
        target[0, 0, 4:7, 4:7] = 1

        # Prediction: Overlaps GT + extra predictions elsewhere
        pred = torch.zeros(batch_size, num_classes, h, w, device=device)
        pred[0, 0, :, :] = 0.1
        pred[0, 1, 4:7, 4:7] = 0.9  # Correct prediction (TP)
        pred[0, 1, 0:2, 0:2] = 0.9  # False positive
        pred[0, 1, 8:10, 8:10] = 0.9  # Another false positive

        # Reset and run metric
        simple_metric.reset_instance_scores()
        simple_metric(pred, target)

        # Get FP/TP/FN statistics
        stats = simple_metric.get_fp_tp_fn_statistics()

        # Should have 1 TP (GT detected), 0 FN, 2 FP (extra predictions)
        assert stats["all"]["TP"] == 1
        assert stats["all"]["FN"] == 0
        assert stats["all"]["FP"] == 2

    def test_instance_size_binning(self, simple_metric, device):
        """Test that instances are correctly binned by size."""
        batch_size = 1
        num_classes = 2
        h, w = 20, 20

        # GT: Three instances of different sizes
        target = torch.zeros(batch_size, 1, h, w, device=device)
        target[0, 0, 0, 0] = 1  # Tiny: 1 pixel
        target[0, 0, 5:8, 5:8] = 1  # Medium: 9 pixels
        target[0, 0, 10:15, 10:18] = 1  # Large: 40 pixels

        # Prediction: Overlaps all GT instances
        pred = torch.zeros(batch_size, num_classes, h, w, device=device)
        pred[0, 0, :, :] = 0.1
        pred[0, 1, 0, 0] = 0.9
        pred[0, 1, 5:8, 5:8] = 0.9
        pred[0, 1, 10:15, 10:18] = 0.9

        # Reset and run metric
        simple_metric.reset_instance_scores()
        simple_metric(pred, target)

        # Get FP/TP/FN statistics
        stats = simple_metric.get_fp_tp_fn_statistics()

        # Check binning: 1 in 0-2cc, 1 in 2-10cc, 1 in >10cc
        assert stats["0-2cc"]["TP"] == 1  # 1 pixel instance
        assert stats["2-10cc"]["TP"] == 1  # 9 pixel instance
        assert stats[">10cc"]["TP"] == 1  # 40 pixel instance
        assert stats["all"]["TP"] == 3

    def test_empty_predictions(self, simple_metric, device):
        """Test edge case: GT exists but predictions are all background."""
        batch_size = 1
        num_classes = 2
        h, w = 10, 10

        # GT: single instance
        target = torch.zeros(batch_size, 1, h, w, device=device)
        target[0, 0, 4:7, 4:7] = 1

        # Prediction: All background (no foreground predictions)
        pred = torch.zeros(batch_size, num_classes, h, w, device=device)
        pred[0, 0, :, :] = 0.9  # All background
        pred[0, 1, :, :] = 0.1  # No foreground

        # Reset and run metric
        simple_metric.reset_instance_scores()
        simple_metric(pred, target)

        # Get FP/TP/FN statistics
        stats = simple_metric.get_fp_tp_fn_statistics()

        # Should have 0 TP, 1 FN (GT missed), 0 FP (no predictions)
        assert stats["all"]["TP"] == 0
        assert stats["all"]["FN"] == 1
        assert stats["all"]["FP"] == 0

    def test_empty_gt(self, simple_metric, device):
        """Test edge case: Predictions exist but GT is empty."""
        batch_size = 1
        num_classes = 2
        h, w = 10, 10

        # GT: empty (all background)
        target = torch.zeros(batch_size, 1, h, w, device=device)

        # Prediction: Has foreground predictions
        pred = torch.zeros(batch_size, num_classes, h, w, device=device)
        pred[0, 0, :, :] = 0.1
        pred[0, 1, 4:7, 4:7] = 0.9  # False positive

        # Reset and run metric
        simple_metric.reset_instance_scores()
        simple_metric(pred, target)

        # Get FP/TP/FN statistics
        stats = simple_metric.get_fp_tp_fn_statistics()

        # Should have 0 TP, 0 FN (no GT), 1 FP (spurious prediction)
        assert stats["all"]["TP"] == 0
        assert stats["all"]["FN"] == 0
        assert stats["all"]["FP"] == 1

    def test_multiple_predictions_one_gt(self, simple_metric, device):
        """Test that multiple predictions overlapping one GT count as 1 TP."""
        batch_size = 1
        num_classes = 2
        h, w = 10, 10

        # GT: single large instance
        target = torch.zeros(batch_size, 1, h, w, device=device)
        target[0, 0, 3:8, 3:8] = 1  # 25 pixels

        # Prediction: Multiple small predictions overlapping the GT
        pred = torch.zeros(batch_size, num_classes, h, w, device=device)
        pred[0, 0, :, :] = 0.1
        pred[0, 1, 3:5, 3:5] = 0.9  # First prediction (overlaps GT)
        pred[0, 1, 6:8, 6:8] = 0.9  # Second prediction (also overlaps GT)

        # Reset and run metric
        simple_metric.reset_instance_scores()
        simple_metric(pred, target)

        # Get FP/TP/FN statistics
        stats = simple_metric.get_fp_tp_fn_statistics()

        # Should have 1 TP (the GT instance, detected by any prediction)
        # The multiple predictions are not counted separately as TP
        assert stats["all"]["TP"] == 1
        assert stats["all"]["FN"] == 0
        # No FP because both predictions overlap with GT
        assert stats["all"]["FP"] == 0

    def test_per_sample_tracking(self, simple_metric, device):
        """Test that per-sample FP/TP/FN statistics are correctly tracked."""
        batch_size = 2
        num_classes = 2
        h, w = 10, 10

        # Sample 1: 1 TP, 0 FN, 1 FP
        target_1 = torch.zeros(1, 1, h, w, device=device)
        target_1[0, 0, 4:7, 4:7] = 1
        pred_1 = torch.zeros(1, num_classes, h, w, device=device)
        pred_1[0, 0, :, :] = 0.1
        pred_1[0, 1, 4:7, 4:7] = 0.9  # TP
        pred_1[0, 1, 0:2, 0:2] = 0.9  # FP

        # Sample 2: 0 TP, 1 FN, 0 FP
        target_2 = torch.zeros(1, 1, h, w, device=device)
        target_2[0, 0, 4:7, 4:7] = 1
        pred_2 = torch.zeros(1, num_classes, h, w, device=device)
        pred_2[0, 0, :, :] = 0.9  # All background (FN)
        pred_2[0, 1, :, :] = 0.1

        # Concatenate into batch
        target = torch.cat([target_1, target_2], dim=0)
        pred = torch.cat([pred_1, pred_2], dim=0)

        # Reset and run metric
        simple_metric.reset_instance_scores()
        simple_metric(pred, target)

        # Get per-sample statistics
        per_sample_stats = simple_metric.get_per_sample_fp_tp_fn_statistics()

        # Check sample 1
        assert per_sample_stats[0]["all"]["TP"] == 1
        assert per_sample_stats[0]["all"]["FN"] == 0
        assert per_sample_stats[0]["all"]["FP"] == 1

        # Check sample 2
        assert per_sample_stats[1]["all"]["TP"] == 0
        assert per_sample_stats[1]["all"]["FN"] == 1
        assert per_sample_stats[1]["all"]["FP"] == 0

    def test_bin_boundaries(self, simple_metric, device):
        """Test that bin boundaries are correct (< 2, < 10, >= 10)."""
        batch_size = 1
        num_classes = 2
        h, w = 30, 30

        # GT: Instances at exact bin boundaries
        target = torch.zeros(batch_size, 1, h, w, device=device)
        target[0, 0, 0:1, 0:1] = 1  # 1 pixel (should be in 0-2cc)
        target[0, 0, 5:6, 5:6] = 1  # 1 pixel (should be in 0-2cc)
        target[0, 0, 10:11, 10:11] = 1  # 1 pixel (should be in 0-2cc)
        target[0, 0, 15:16, 15:17] = 1  # 2 pixels (should be in 2-10cc)
        target[0, 0, 20:21, 20:29] = 1  # 9 pixels (should be in 2-10cc)
        target[0, 0, 25:27, 25:30] = 1  # 10 pixels (should be in >10cc)

        # Prediction: Overlaps all GT
        pred = torch.zeros(batch_size, num_classes, h, w, device=device)
        pred[0, 0, :, :] = 0.1
        pred[0, 1, :, :] = 0.0
        pred[0, 1, 0:1, 0:1] = 0.9
        pred[0, 1, 5:6, 5:6] = 0.9
        pred[0, 1, 10:11, 10:11] = 0.9
        pred[0, 1, 15:16, 15:17] = 0.9
        pred[0, 1, 20:21, 20:29] = 0.9
        pred[0, 1, 25:27, 25:30] = 0.9

        # Reset and run metric
        simple_metric.reset_instance_scores()
        simple_metric(pred, target)

        # Get FP/TP/FN statistics
        stats = simple_metric.get_fp_tp_fn_statistics()

        # Check binning: 3 instances of 1 pixel (< 2)
        assert stats["0-2cc"]["TP"] == 3

        # Check binning: 2 instances of 2 and 9 pixels (< 10, >= 2)
        assert stats["2-10cc"]["TP"] == 2

        # Check binning: 1 instance of 10 pixels (>= 10)
        assert stats[">10cc"]["TP"] == 1

    def test_partial_overlap_counts_as_tp(self, simple_metric, device):
        """Test that even partial overlap (1 pixel) counts as TP."""
        batch_size = 1
        num_classes = 2
        h, w = 10, 10

        # GT: 3x3 instance
        target = torch.zeros(batch_size, 1, h, w, device=device)
        target[0, 0, 4:7, 4:7] = 1  # 9 pixels

        # Prediction: Only 1 pixel overlaps
        pred = torch.zeros(batch_size, num_classes, h, w, device=device)
        pred[0, 0, :, :] = 0.1
        pred[0, 1, 4, 4] = 0.9  # Only 1 pixel overlap

        # Reset and run metric
        simple_metric.reset_instance_scores()
        simple_metric(pred, target)

        # Get FP/TP/FN statistics
        stats = simple_metric.get_fp_tp_fn_statistics()

        # Should still be TP (any overlap > 0)
        assert stats["all"]["TP"] == 1
        assert stats["all"]["FN"] == 0

    def test_reset_clears_fp_tp_fn_state(self, simple_metric, device):
        """Test that reset_instance_scores clears FP/TP/FN tracking."""
        batch_size = 1
        num_classes = 2
        h, w = 10, 10

        # First run: 1 TP
        target = torch.zeros(batch_size, 1, h, w, device=device)
        target[0, 0, 4:7, 4:7] = 1
        pred = torch.zeros(batch_size, num_classes, h, w, device=device)
        pred[0, 0, :, :] = 0.1
        pred[0, 1, 4:7, 4:7] = 0.9

        simple_metric.reset_instance_scores()
        simple_metric(pred, target)
        stats_1 = simple_metric.get_fp_tp_fn_statistics()
        assert stats_1["all"]["TP"] == 1

        # Reset and run again
        simple_metric.reset_instance_scores()
        simple_metric(pred, target)
        stats_2 = simple_metric.get_fp_tp_fn_statistics()

        # Should still be 1 TP (not accumulated to 2)
        assert stats_2["all"]["TP"] == 1

    def test_no_double_counting(self, simple_metric, device):
        """Test that instances are not double-counted."""
        batch_size = 1
        num_classes = 2
        h, w = 10, 10

        # GT: 2 separate instances
        target = torch.zeros(batch_size, 1, h, w, device=device)
        target[0, 0, 2:4, 2:4] = 1  # Instance 1
        target[0, 0, 6:8, 6:8] = 1  # Instance 2

        # Prediction: Overlaps both
        pred = torch.zeros(batch_size, num_classes, h, w, device=device)
        pred[0, 0, :, :] = 0.1
        pred[0, 1, 2:4, 2:4] = 0.9
        pred[0, 1, 6:8, 6:8] = 0.9

        # Reset and run metric
        simple_metric.reset_instance_scores()
        simple_metric(pred, target)

        # Get FP/TP/FN statistics
        stats = simple_metric.get_fp_tp_fn_statistics()

        # Should be exactly 2 TPs (not 4 or any other number)
        assert stats["all"]["TP"] == 2
        assert stats["all"]["FN"] == 0
        assert stats["all"]["FP"] == 0

        # Check that total across bins equals total in "all"
        total_tp = stats["0-2cc"]["TP"] + stats["2-10cc"]["TP"] + stats[">10cc"]["TP"]
        assert total_tp == stats["all"]["TP"]
