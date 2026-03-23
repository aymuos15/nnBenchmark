"""
Image loading and property extraction for dataset fingerprinting.
Handles NIfTI, PNG, and JPEG formats with automatic format detection.
"""


from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from src.planning.constants import PLANNING_CONSTANTS
from src.utils.files import detect_file_type, load_nifti_with_metadata


@dataclass
class ImageProperties:
    """Properties extracted from a single image."""

    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    foreground_intensities: (
        np.ndarray
    )  # Sampled foreground voxel intensities for pooling
    intensity_mean: float  # Mean of foreground voxel intensities
    intensity_std: float  # Standard deviation of foreground voxel intensities
    intensity_percentile_00_5: float  # 0.5th percentile of foreground intensities
    intensity_percentile_99_5: float  # 99.5th percentile of foreground intensities


def load_image_properties(
    image_path: str, label_path: str | None = None
) -> ImageProperties:
    """
    Load image and extract properties.

    Supports NIfTI (.nii.gz) and PNG/JPEG with automatic format detection.
    Uses MONAI's LoadImaged for robust NIfTI image loading.

    Args:
        image_path: Path to image file
        label_path: Path to corresponding label/segmentation file (optional)
    """
    file_type = detect_file_type(image_path)

    if file_type == "nifti":
        # Load NIfTI with MONAI for robust, consistent handling
        data, spacing = load_nifti_with_metadata(image_path)

    elif file_type in ["png", "jpeg"]:
        # Load image with PIL, assume spacing [1.0, 1.0]
        img = Image.open(image_path)
        data = np.array(img)

        # Handle grayscale vs RGB - ensure channel-first format
        if len(data.shape) == 2:
            # Grayscale image: (H, W) -> (1, H, W)
            data = data[np.newaxis, :]
            spacing = (1.0, 1.0)
        elif len(data.shape) == 3:
            # RGB image: (H, W, C) -> (C, H, W)
            data = np.transpose(data, (2, 0, 1))
            spacing = (1.0, 1.0)
        else:
            raise ValueError(f"Unexpected PNG/JPEG shape: {data.shape}")

    else:
        raise ValueError(f"Unsupported file type: {file_type} for {image_path}")

    # Collect sampled foreground voxel intensities (nnUNet v2.4.1 style)
    if label_path is not None and Path(label_path).exists():
        # Load label/segmentation
        label_file_type = detect_file_type(label_path)
        if label_file_type == "nifti":
            label_data, _ = load_nifti_with_metadata(label_path)
        else:
            label_data = np.array(Image.open(label_path))
            # Ensure channel-first format for labels if 2D
            if len(label_data.shape) == 2:
                # (H, W) -> (1, H, W)
                label_data = label_data[np.newaxis, :]
            elif len(label_data.shape) == 3 and label_data.shape[2] <= 4:
                # If last dimension is small (channels), transpose to channel-first
                # (H, W, C) -> (C, H, W)
                label_data = np.transpose(label_data, (2, 0, 1))

        # Create foreground mask (all non-zero labels)
        foreground_mask = label_data > 0

        # Extract foreground voxels only
        if np.any(foreground_mask):
            data_foreground = data[foreground_mask]
        else:
            # If no foreground, fall back to all voxels
            data_foreground = data.flatten()
    else:
        # No label available, use all voxels
        data_foreground = data.flatten()

    # Sample foreground intensities (nnUNet uses 10,000 samples per case)
    num_samples = min(
        PLANNING_CONSTANTS.FOREGROUND_SAMPLES_PER_CASE, len(data_foreground)
    )
    if len(data_foreground) > num_samples:
        # Randomly sample to limit memory usage
        rng = np.random.RandomState(PLANNING_CONSTANTS.RANDOM_SEED)
        sampled_indices = rng.choice(
            len(data_foreground), size=num_samples, replace=False
        )
        sampled_intensities = data_foreground[sampled_indices]
    else:
        sampled_intensities = data_foreground

    # Compute intensity statistics for this image
    intensity_mean = float(np.mean(sampled_intensities))
    intensity_std = float(np.std(sampled_intensities))
    intensity_percentile_00_5 = float(np.percentile(sampled_intensities, 0.5))
    intensity_percentile_99_5 = float(np.percentile(sampled_intensities, 99.5))

    return ImageProperties(
        shape=data.shape,
        spacing=spacing,
        foreground_intensities=sampled_intensities,
        intensity_mean=intensity_mean,
        intensity_std=intensity_std,
        intensity_percentile_00_5=intensity_percentile_00_5,
        intensity_percentile_99_5=intensity_percentile_99_5,
    )


def load_image_properties_safe(args: tuple[str, str | None]) -> ImageProperties | None:
    """
    Wrapper for load_image_properties that handles exceptions.

    Used in parallel processing to gracefully handle individual image failures.
    Returns None if loading fails, otherwise returns ImageProperties.

    Args:
        args: Tuple of (image_path, label_path)
    """
    image_path, label_path = args
    try:
        return load_image_properties(image_path, label_path)
    except Exception as e:
        logger.warning(f"Failed to load {image_path}: {e}")
        return None
