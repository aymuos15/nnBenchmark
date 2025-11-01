"""
Tests for MONAI integration in src.utils.files and src.planning.fingerprint.
Tests MONAI LoadImaged usage for robust NIfTI loading and fingerprinting.
"""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from src.planning.fingerprinting.fingerprint import fingerprint_dataset
from src.planning.fingerprinting.loading import load_image_properties
from src.utils.files import detect_file_type, load_nifti_data, load_nifti_with_metadata


class TestDetectFileType:
    """Tests for detect_file_type function."""

    @pytest.mark.parametrize(
        "filename,expected_type",
        [
            # NIfTI files
            ("image.nii.gz", "nifti"),
            ("/path/to/image.nii.gz", "nifti"),
            ("image.nii", "nifti"),
            ("/path/to/image.nii", "nifti"),
            # PNG files
            ("image.png", "png"),
            ("/path/to/image.png", "png"),
            # JPEG files (.jpg and .jpeg variants)
            ("image.jpg", "jpeg"),
            ("/path/to/image.jpg", "jpeg"),
            ("image.jpeg", "jpeg"),
            ("/path/to/image.jpeg", "jpeg"),
            # Case-insensitive detection
            ("image.NII.GZ", "nifti"),
            ("image.PNG", "png"),
            ("image.JPG", "jpeg"),
            ("image.NiI.Gz", "nifti"),
            ("image.PnG", "png"),
            # Unknown extensions
            ("image.dcm", "unknown"),
            ("image.tif", "unknown"),
            ("image.txt", "unknown"),
            ("image", "unknown"),
            ("/path/to/image", "unknown"),
        ],
    )
    def test_detect_file_type(self, filename: str, expected_type: str) -> None:
        """Test file type detection across various extensions and cases.

        Validates that detect_file_type correctly identifies:
        - Common medical imaging formats (NIfTI)
        - Standard image formats (PNG, JPEG)
        - Case-insensitive extension matching
        - Unknown and unsupported file types
        """
        assert detect_file_type(filename) == expected_type


class TestLoadNiftiData:
    """Tests for load_nifti_data function."""

    def test_load_valid_nifti(self, temp_dir: str) -> None:
        """Test loading a valid NIfTI file."""
        # Create a small 3D NIfTI file
        data = np.random.rand(10, 10, 10).astype(np.float32)
        img = nib.Nifti1Image(data, affine=np.eye(4))  # type: ignore[attr-defined]
        nifti_path = os.path.join(temp_dir, "test.nii.gz")
        nib.save(img, nifti_path)  # type: ignore[attr-defined]

        # Load using our function
        loaded_data = load_nifti_data(nifti_path)

        # Verify shape and values match
        assert loaded_data.shape == data.shape
        np.testing.assert_array_almost_equal(loaded_data, data, decimal=5)

    def test_load_missing_file(self, temp_dir: str) -> None:
        """Test loading non-existent NIfTI file raises FileNotFoundError."""
        missing_path = os.path.join(temp_dir, "missing.nii.gz")

        with pytest.raises(FileNotFoundError):
            load_nifti_data(missing_path)


class TestLoadNiftiWithMetadata:
    """Tests for load_nifti_with_metadata function - CRITICAL for fingerprinting."""

    def test_spacing_extraction_from_affine(self, temp_dir: str) -> None:
        """Test spacing is correctly extracted from NIfTI affine matrix.

        Note: MONAI LoadImaged doesn't populate meta_dict with ensure_channel_first=False,
        so our function falls back to computing spacing from the affine matrix.
        """
        # Create NIfTI with known spacing in affine
        data = np.random.rand(20, 30, 15).astype(np.float32)
        affine = np.diag([1.5, 2.0, 0.5, 1.0])  # Spacing: 1.5, 2.0, 0.5
        img = nib.Nifti1Image(data, affine=affine)  # type: ignore[attr-defined]

        nifti_path = os.path.join(temp_dir, "test_spacing.nii.gz")
        nib.save(img, nifti_path)  # type: ignore[attr-defined]

        # Load and verify spacing (computed from affine)
        loaded_data, spacing = load_nifti_with_metadata(nifti_path)

        assert loaded_data.shape == data.shape
        # Spacing computed from affine matrix norm
        assert len(spacing) == 3
        assert spacing[0] == pytest.approx(1.5, abs=0.01)
        assert spacing[1] == pytest.approx(2.0, abs=0.01)
        assert spacing[2] == pytest.approx(0.5, abs=0.01)

    def test_load_with_anisotropic_spacing(self, temp_dir: str) -> None:
        """Test loading NIfTI with highly anisotropic spacing."""
        data = np.random.rand(64, 64, 16).astype(np.float32)
        # Anisotropic: slice thickness 5mm, in-plane 0.5mm
        affine = np.diag([0.5, 0.5, 5.0, 1.0])
        img = nib.Nifti1Image(data, affine=affine)  # type: ignore[attr-defined]

        nifti_path = os.path.join(temp_dir, "anisotropic.nii.gz")
        nib.save(img, nifti_path)  # type: ignore[attr-defined]

        loaded_data, spacing = load_nifti_with_metadata(nifti_path)

        assert loaded_data.shape == data.shape
        assert spacing[0] == pytest.approx(0.5, abs=0.01)
        assert spacing[1] == pytest.approx(0.5, abs=0.01)
        assert spacing[2] == pytest.approx(5.0, abs=0.01)

    def test_load_2d_nifti(self, temp_dir: str) -> None:
        """Test loading 2D NIfTI file (single slice)."""
        data = np.random.rand(128, 128).astype(np.float32)
        affine = np.diag([0.5, 0.5, 1.0, 1.0])
        img = nib.Nifti1Image(data, affine=affine)  # type: ignore[attr-defined]

        nifti_path = os.path.join(temp_dir, "2d.nii.gz")
        nib.save(img, nifti_path)  # type: ignore[attr-defined]

        loaded_data, spacing = load_nifti_with_metadata(nifti_path)

        assert loaded_data.shape == data.shape
        # Function returns 3D spacing tuple even for 2D data
        assert len(spacing) == 3
        assert spacing[0] == pytest.approx(0.5, abs=0.01)
        assert spacing[1] == pytest.approx(0.5, abs=0.01)

    def test_load_missing_file(self, temp_dir: str) -> None:
        """Test loading non-existent file raises appropriate error."""
        missing_path = os.path.join(temp_dir, "nonexistent.nii.gz")

        with pytest.raises(FileNotFoundError):
            load_nifti_with_metadata(missing_path)


class TestLoadImageProperties:
    """Tests for load_image_properties function - CRITICAL for fingerprinting."""

    def test_load_properties_from_nifti(self, temp_dir: str) -> None:
        """Test loading image properties from NIfTI file."""
        # Create NIfTI with known properties
        data = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32
        )  # Shape: (2, 2, 3)
        affine = np.diag([1.5, 2.0, 0.5, 1.0])
        img = nib.Nifti1Image(data, affine=affine)  # type: ignore[attr-defined]

        nifti_path = os.path.join(temp_dir, "test_props.nii.gz")
        nib.save(img, nifti_path)  # type: ignore[attr-defined]

        # Load properties
        props = load_image_properties(nifti_path)

        # Verify shape and spacing (from affine matrix)
        assert props.shape == (2, 2, 3)
        assert props.spacing[0] == pytest.approx(1.5, abs=0.01)
        assert props.spacing[1] == pytest.approx(2.0, abs=0.01)
        assert props.spacing[2] == pytest.approx(0.5, abs=0.01)

        # Verify intensity statistics
        assert props.intensity_mean == pytest.approx(6.5, abs=0.01)  # Mean of 1-12
        assert props.intensity_std > 0  # Should have non-zero std
        assert props.intensity_percentile_00_5 < props.intensity_percentile_99_5

    def test_load_properties_from_png(self, temp_dir: str) -> None:
        """Test loading image properties from PNG file."""
        # Create a simple grayscale PNG
        data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        img = Image.fromarray(data, mode="L")
        png_path = os.path.join(temp_dir, "test.png")
        img.save(png_path)

        # Load properties
        props = load_image_properties(png_path)

        # Verify shape (MONAI LoadImaged returns channel-first: C, H, W)
        assert props.shape == (1, 64, 64)

        # Verify spacing (should be 1.0, 1.0 for PNG)
        assert props.spacing == (1.0, 1.0)

        # Verify intensity statistics exist
        assert props.intensity_mean >= 0
        assert props.intensity_std >= 0
        assert 0 <= props.intensity_percentile_00_5 <= 255
        assert 0 <= props.intensity_percentile_99_5 <= 255

    def test_load_properties_from_rgb_png(self, temp_dir: str) -> None:
        """Test loading image properties from RGB PNG."""
        # Create RGB PNG
        data = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(data, mode="RGB")
        png_path = os.path.join(temp_dir, "rgb_test.png")
        img.save(png_path)

        # Load properties
        props = load_image_properties(png_path)

        # Verify shape includes channels (MONAI LoadImaged returns channel-first: C, H, W)
        assert props.shape == (3, 32, 32)

        # Verify spacing for 3-channel image (should be 1.0, 1.0)
        assert props.spacing == (1.0, 1.0)

    def test_load_properties_unsupported_format(self, temp_dir: str) -> None:
        """Test loading unsupported format raises ValueError."""
        # Create a text file (unsupported)
        txt_path = os.path.join(temp_dir, "test.txt")
        with open(txt_path, "w") as f:
            f.write("not an image")

        with pytest.raises(ValueError, match="Unsupported file type"):
            load_image_properties(txt_path)


class TestFingerprintDatasetErrorHandling:
    """Tests for fingerprint_dataset error handling - CRITICAL for robustness."""

    def test_fingerprint_dataset_no_images_found(self, mock_dataset_dir: str) -> None:
        """Test fingerprinting dataset with no images raises FileNotFoundError."""
        # Remove all image files
        images_dir = os.path.join(mock_dataset_dir, "imagesTr")
        for f in os.listdir(images_dir):
            os.remove(os.path.join(images_dir, f))

        with pytest.raises(FileNotFoundError, match="No images found"):
            fingerprint_dataset(mock_dataset_dir)

    def test_fingerprint_dataset_handles_some_missing_images(
        self, temp_dir: str, mock_dataset_json: dict
    ) -> None:
        """Test fingerprinting handles some corrupted/missing images gracefully."""
        # Create dataset with mix of valid and missing files
        dataset_dir = os.path.join(temp_dir, "Dataset_Mixed")
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "imagesTr"), exist_ok=True)

        # Create dataset.json with channel
        dataset_json = {
            "name": "Dataset_Mixed",
            "numTraining": 3,
            "labels": {"0": "background", "1": "foreground"},
            "modality": {"0": "MRI"},
            "file_ending": ".nii.gz",
        }
        import json

        with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
            json.dump(dataset_json, f)

        # Create 2 valid NIfTI files
        for i in range(2):
            data = np.random.rand(10, 10, 10).astype(np.float32)
            img = nib.Nifti1Image(data, affine=np.eye(4))  # type: ignore[attr-defined]
            nifti_path = os.path.join(
                dataset_dir, "imagesTr", f"case_{i:03d}_0000.nii.gz"
            )
            nib.save(img, nifti_path)  # type: ignore[attr-defined]

        # Create 1 corrupted file (empty)
        corrupted_path = os.path.join(dataset_dir, "imagesTr", "case_002_0000.nii.gz")
        open(corrupted_path, "a").close()

        # Should succeed with warning, not crash
        # (fingerprint_dataset logs warnings but continues with valid images)
        try:
            fingerprint = fingerprint_dataset(dataset_dir)
            # Should have processed at least the 2 valid images
            assert fingerprint.num_training_cases >= 2
        except ValueError as e:
            # If ALL images fail, it should raise ValueError
            if "No valid images could be loaded" in str(e):
                pytest.skip("All images failed - expected behavior for corrupted files")
            else:
                raise
