"""
Tests for preprocessing and cropping utilities.

Tests the cropping module functions that remove zero background regions
and prepare datasets for training.
"""


import numpy as np
import pytest

from src.preprocessing.cropping import (
    create_nonzero_mask,
    crop_to_nonzero,
    get_bbox_from_mask,
)


class TestCreateNonzeroMask:
    """Tests for create_nonzero_mask function."""

    def test_3d_single_channel_mask_creation(self) -> None:
        """Test mask creation for 3D single-channel data."""
        # Create 3D single-channel data with foreground in center
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[3:7, 3:7, 3:7] = 1.0  # Foreground region

        mask = create_nonzero_mask(data)

        assert mask.shape == (10, 10, 10)
        assert mask.dtype == bool
        # Check that foreground region is True
        assert np.all(mask[3:7, 3:7, 3:7])
        # Check that background is False
        assert not np.any(mask[0:2, 0:2, 0:2])

    def test_4d_multichannel_mask_creation(self) -> None:
        """Test mask creation for 4D multi-channel 3D data."""
        # Create 4D multi-channel data: (C, H, W, D)
        data = np.zeros((2, 10, 10, 10), dtype=np.float32)
        # Channel 0: foreground in one region
        data[0, 2:5, 2:5, 2:5] = 1.0
        # Channel 1: foreground in different region
        data[1, 5:8, 5:8, 5:8] = 1.0

        mask = create_nonzero_mask(data)

        assert mask.shape == (10, 10, 10)  # Spatial dimensions only
        # Both foreground regions should be in mask (OR operation)
        assert np.any(mask[2:5, 2:5, 2:5])
        assert np.any(mask[5:8, 5:8, 5:8])

    def test_2d_multichannel_mask_creation(self) -> None:
        """Test mask creation for 2D multi-channel data."""
        # Create 2D multi-channel data: (C, H, W)
        data = np.zeros((3, 20, 20), dtype=np.float32)
        data[:, 5:15, 5:15] = 1.0  # Foreground region

        mask = create_nonzero_mask(data)

        assert mask.shape == (20, 20)
        assert mask.dtype == bool

    def test_empty_data_returns_empty_mask(self) -> None:
        """Test that all-zero data produces empty mask."""
        data = np.zeros((10, 10, 10), dtype=np.float32)
        mask = create_nonzero_mask(data)

        assert mask.dtype == bool
        assert not np.any(mask)

    def test_morphological_hole_filling(self) -> None:
        """Test that morphological hole-filling closes small holes."""
        # Create data with a small hole in the middle
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:8, 2:8, 2:8] = 1.0
        data[4:5, 4:5, 4:5] = 0.0  # Small hole in center

        mask = create_nonzero_mask(data)

        # After hole-filling, the small hole should be filled
        assert mask[4, 4, 4], "Small hole should be filled by morphological operation"

    def test_negative_values_treated_as_nonzero(self) -> None:
        """Test that negative values are correctly identified as non-zero."""
        data = np.zeros((5, 5, 5), dtype=np.float32)
        data[1:4, 1:4, 1:4] = -1.0  # Negative values

        mask = create_nonzero_mask(data)

        assert np.all(mask[1:4, 1:4, 1:4]), "Negative values should be non-zero"


class TestGetBboxFromMask:
    """Tests for get_bbox_from_mask function."""

    def test_bbox_from_simple_mask(self) -> None:
        """Test bounding box calculation from simple mask."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:7, 3:8, 1:6] = True

        bbox = get_bbox_from_mask(mask)

        assert len(bbox) == 3
        assert bbox[0] == [2, 7]  # z-axis
        assert bbox[1] == [3, 8]  # x-axis
        assert bbox[2] == [1, 6]  # y-axis

    def test_bbox_full_data(self) -> None:
        """Test bounding box when foreground occupies full data."""
        mask = np.ones((10, 10, 10), dtype=bool)

        bbox = get_bbox_from_mask(mask)

        assert bbox == [[0, 10], [0, 10], [0, 10]]

    def test_bbox_single_voxel(self) -> None:
        """Test bounding box for single voxel foreground."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True

        bbox = get_bbox_from_mask(mask)

        assert bbox == [[5, 6], [5, 6], [5, 6]]

    def test_bbox_empty_mask(self) -> None:
        """Test bounding box for empty mask."""
        mask = np.zeros((10, 10, 10), dtype=bool)

        bbox = get_bbox_from_mask(mask)

        # Empty mask returns minimal bbox
        assert bbox == [[0, 0], [0, 0], [0, 0]]

    def test_bbox_2d_mask(self) -> None:
        """Test bounding box for 2D mask."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:7, 3:9] = True

        bbox = get_bbox_from_mask(mask)

        assert len(bbox) == 2
        assert bbox[0] == [2, 7]
        assert bbox[1] == [3, 9]


class TestCropToNonzero:
    """Tests for crop_to_nonzero function."""

    def test_crop_3d_single_channel_image_only(self) -> None:
        """Test cropping 3D single-channel image without segmentation."""
        # Create image with foreground in center
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:8, 3:9, 1:7] = 1.0

        cropped_data, cropped_seg, bbox = crop_to_nonzero(data)

        assert cropped_data.shape == (6, 6, 6)
        assert cropped_seg is None
        assert bbox == [[2, 8], [3, 9], [1, 7]]

    def test_crop_3d_multichannel_with_segmentation(self) -> None:
        """Test cropping 3D multi-channel image with segmentation."""
        # Create multi-channel image
        image = np.zeros((2, 10, 10, 10), dtype=np.float32)
        image[:, 2:8, 3:9, 1:7] = 1.0

        # Create corresponding segmentation
        seg = np.zeros((2, 10, 10, 10), dtype=np.uint8)
        seg[:, 2:8, 3:9, 1:7] = 1

        cropped_image, cropped_seg, _bbox = crop_to_nonzero(image, seg)

        assert cropped_image.shape == (2, 6, 6, 6)
        assert cropped_seg is not None
        assert cropped_seg.shape == (2, 6, 6, 6)
        assert np.all(cropped_seg == 1)

    def test_crop_with_precomputed_mask(self) -> None:
        """Test cropping with pre-computed mask."""
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[3:7, 3:7, 3:7] = 1.0

        # Pre-compute mask
        mask = create_nonzero_mask(data)

        cropped_data, _, _ = crop_to_nonzero(data, mask=mask)

        assert cropped_data.shape == (4, 4, 4)

    def test_crop_preserves_data_values(self) -> None:
        """Test that cropping preserves actual data values."""
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:8, 2:8, 2:8] = 3.14

        cropped_data, _, _ = crop_to_nonzero(data)

        assert np.allclose(cropped_data, 3.14)

    def test_crop_shape_mismatch_raises_error(self) -> None:
        """Test that mismatched image and segmentation shapes raise error."""
        image = np.zeros((10, 10, 10), dtype=np.float32)
        seg = np.zeros((10, 10, 8), dtype=np.uint8)  # Wrong shape

        with pytest.raises(ValueError, match="data and seg must have same spatial shape"):
            crop_to_nonzero(image, seg)

    def test_crop_2d_data(self) -> None:
        """Test cropping 2D multi-channel data."""
        # Create 2D multi-channel data: (C, H, W)
        image = np.zeros((3, 20, 20), dtype=np.float32)
        image[:, 5:15, 7:17] = 1.0

        cropped_image, _, bbox = crop_to_nonzero(image)

        assert cropped_image.shape == (3, 10, 10)
        assert len(bbox) == 2

    def test_crop_maintains_channel_dimension(self) -> None:
        """Test that channel dimension is preserved during cropping."""
        # Create 4D multi-channel data
        image = np.zeros((4, 20, 20, 20), dtype=np.float32)
        image[:, 5:15, 5:15, 5:15] = 1.0

        cropped_image, _, _ = crop_to_nonzero(image)

        # Channel dimension should be preserved
        assert cropped_image.shape[0] == 4
        assert cropped_image.shape[1:] == (10, 10, 10)

    def test_crop_empty_image(self) -> None:
        """Test cropping all-zero image."""
        data = np.zeros((10, 10, 10), dtype=np.float32)

        _cropped_data, _, bbox = crop_to_nonzero(data)

        # Should return minimal bbox [0, 0]
        assert bbox == [[0, 0], [0, 0], [0, 0]]


class TestCroppingIntegration:
    """Integration tests for complete cropping workflow."""

    def test_realistic_medical_image_cropping(self) -> None:
        """Test cropping realistic medical image with multiple regions."""
        # Simulate 3D medical image with anatomical structure
        image = np.zeros((1, 50, 50, 50), dtype=np.float32)

        # Simulate multiple anatomical structures
        # Structure 1: centered cube
        image[:, 10:30, 10:30, 10:30] = 100.0
        # Structure 2: in one corner
        image[:, 5:15, 5:15, 5:15] = 50.0

        seg = np.zeros((1, 50, 50, 50), dtype=np.uint8)
        seg[:, 10:30, 10:30, 10:30] = 1  # Segment structure 1
        seg[:, 5:15, 5:15, 5:15] = 2  # Segment structure 2

        cropped_image, cropped_seg, _bbox = crop_to_nonzero(image, seg)

        # Should crop to bounds of both structures
        assert cropped_image.shape[0] == 1  # Channel preserved
        assert cropped_seg is not None
        assert cropped_image.shape[1:] == cropped_seg.shape[1:]

    def test_cropping_reduces_size(self) -> None:
        """Test that cropping reduces data size."""
        # Large volume with small foreground
        image = np.zeros((1, 100, 100, 100), dtype=np.float32)
        image[:, 25:75, 25:75, 25:75] = 1.0

        original_size = image.nbytes
        cropped_image, _, _ = crop_to_nonzero(image)
        cropped_size = cropped_image.nbytes

        assert cropped_size < original_size
        # Foreground is 50×50×50, should crop to approximately that
        assert cropped_image.shape[1:] == (50, 50, 50)

    def test_roundtrip_crop_consistency(self) -> None:
        """Test that cropping is consistent across multiple calls."""
        data = np.random.rand(1, 30, 30, 30).astype(np.float32)
        data[:, :10, :, :] = 0  # Add some zeros
        data[:, 20:, :, :] = 0  # Add some zeros

        # Crop twice and verify same result
        cropped1, _, bbox1 = crop_to_nonzero(data)
        cropped2, _, bbox2 = crop_to_nonzero(data)

        assert np.array_equal(cropped1, cropped2)
        assert bbox1 == bbox2
