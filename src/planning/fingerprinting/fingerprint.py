"""
Dataset fingerprinting for automatic configuration generation.
Analyzes training images to extract shape, spacing, and intensity statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.utils.files import detect_file_type, load_json, load_nifti_with_metadata


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


@dataclass
class DatasetFingerprint:
    """Aggregated dataset properties for experiment planning."""

    # Dataset identification
    dataset_name: str
    num_training_cases: int
    num_classes: int
    channel: str

    # Dimensionality
    is_2d: bool  # True if dataset is 2D, False if 3D

    # Shape statistics
    median_shape: tuple[int, ...]
    percentile_10_shape: tuple[int, ...]
    percentile_90_shape: tuple[int, ...]

    # Spacing statistics
    median_spacing: tuple[float, ...]
    percentile_10_spacing: tuple[float, ...]
    percentile_90_spacing: tuple[float, ...]

    # Anisotropy flags
    is_anisotropic: bool  # True if worst axis > 4x better axes
    anisotropy_axis: int | None  # Which axis is anisotropic (None if isotropic)

    # Intensity statistics (from foreground voxels if labels available)
    intensity_mean: float
    intensity_std: float
    intensity_percentile_00_5: float
    intensity_percentile_99_5: float

    # Normalization scheme determined from channel
    normalization_scheme: str  # 'CTNormalization', 'ZScoreNormalization', etc.


def _load_image_properties(
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
        from PIL import Image

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
            from PIL import Image

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
    num_samples = min(10000, len(data_foreground))
    if len(data_foreground) > num_samples:
        # Randomly sample to limit memory usage
        rng = np.random.RandomState(12345)
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


def _load_image_properties_safe(args: tuple[str, str | None]) -> ImageProperties | None:
    """
    Wrapper for _load_image_properties that handles exceptions.

    Used in parallel processing to gracefully handle individual image failures.
    Returns None if loading fails, otherwise returns ImageProperties.

    Args:
        args: Tuple of (image_path, label_path)
    """
    image_path, label_path = args
    try:
        return _load_image_properties(image_path, label_path)
    except Exception as e:
        logger.warning(f"Failed to load {image_path}: {e}")
        return None


def _determine_normalization_scheme(channel: str) -> str:
    """
    Determine normalization scheme based on channel.

    Following nnU-Net conventions:
    - CT → CTNormalization (percentile clipping + dataset-wide z-score)
    - Other channels → ZScoreNormalization (per-case z-score)
    """
    channel_lower = channel.lower()

    if "ct" in channel_lower:
        return "CTNormalization"
    else:
        return "ZScoreNormalization"


def _detect_anisotropy(
    median_spacing: tuple[float, ...], median_shape: tuple[int, ...]
) -> tuple[bool, int | None]:
    """
    Detect if dataset is anisotropic following nnU-Net heuristics.

    Returns:
        (is_anisotropic, anisotropy_axis)

    nnU-Net considers dataset anisotropic if:
    - Worst spacing axis > 3× better axes spacing AND
    - Worst axis has < 25% voxels of better axes in median shape
    """
    if len(median_spacing) < 2:
        return False, None

    # Find worst (largest) spacing axis
    worst_axis = int(np.argmax(median_spacing))
    worst_spacing = median_spacing[worst_axis]

    # Get better axes
    other_axes = [i for i in range(len(median_spacing)) if i != worst_axis]
    better_spacings = [median_spacing[i] for i in other_axes]
    better_shapes = [median_shape[i] for i in other_axes]

    # Check if worst axis is 3x worse than better axes
    spacing_ratio = worst_spacing / np.median(better_spacings)

    # Check if worst axis has < 25% voxels
    voxel_ratio = median_shape[worst_axis] / np.median(better_shapes)

    aniso_threshold = 3.0  # nnUNet v2 ANISO_THRESHOLD
    is_anisotropic = bool(spacing_ratio > aniso_threshold and voxel_ratio < 0.25)

    return is_anisotropic, worst_axis if is_anisotropic else None


def _scan_unique_label_values(label_paths: list[str], num_samples: int = 50) -> int:
    """
    Scan label files to determine the actual number of unique classes.

    Samples a subset of label files to find unique label values.
    Returns the maximum label value + 1 (assuming 0-indexed classes).

    Args:
        label_paths: List of paths to label files
        num_samples: Number of label files to sample (default: 50)

    Returns:
        Number of classes (max_label_value + 1)
    """
    if not label_paths:
        return 0

    # Sample up to num_samples files
    import random
    sample_paths = random.sample(label_paths, min(num_samples, len(label_paths)))

    unique_values = set()
    for label_path in sample_paths:
        try:
            file_type = detect_file_type(label_path)
            if file_type == "nifti":
                label_data, _ = load_nifti_with_metadata(label_path)
            else:
                from PIL import Image
                label_data = np.array(Image.open(label_path))

            # Get unique values from this label file
            unique_values.update(np.unique(label_data).tolist())
        except Exception as e:
            logger.warning(f"Failed to read label file {label_path}: {e}")
            continue

    if not unique_values:
        logger.warning("No unique label values found in sampled files")
        return 0

    # Number of classes = max label value + 1 (assuming 0-indexed)
    max_label = int(max(unique_values))
    num_classes = max_label + 1

    logger.info(f"Scanned {len(sample_paths)} label files, found {len(unique_values)} unique label values: {sorted(unique_values)}")
    logger.info(f"Determined num_classes = {num_classes} (max label value: {max_label})")

    return num_classes


def fingerprint_dataset(
    dataset_dir: str, num_workers: int | None = None
) -> DatasetFingerprint:
    """
    Analyze dataset and create fingerprint for experiment planning.

    Args:
        dataset_dir: Path to dataset directory containing dataset.json and imagesTr/
        num_workers: Number of parallel workers for processing images.
                     If None, defaults to min(cpu_count(), 8).
                     Set to 1 to disable parallel processing.

    Returns:
        DatasetFingerprint with aggregated statistics

    """
    logger.info(f"Fingerprinting dataset: {dataset_dir}")

    # Determine number of workers
    if num_workers is None:
        # Auto-detect optimal workers using resource detection
        # Using conservative strategy to avoid issues with parallel processing
        try:
            from src.planning.fingerprinting.resources import (
                calculate_optimal_workers,
                detect_cpu_cores,
            )

            _, logical_cores = detect_cpu_cores()
            num_workers = calculate_optimal_workers(
                logical_cores, strategy="conservative"
            )
            logger.info(
                f"Auto-detected {num_workers} worker(s) from {logical_cores} CPU cores"
            )
        except Exception as e:
            logger.debug(f"Failed to auto-detect workers: {e}. Defaulting to 1.")
            num_workers = 1

    logger.info(f"Using {num_workers} worker(s) for parallel processing")

    # Load dataset.json
    dataset_json_path = str(Path(dataset_dir) / "dataset.json")
    dataset_info: dict[str, Any] = load_json(dataset_json_path, "dataset.json")

    dataset_name = dataset_info.get("name", "Unknown")
    # num_classes will be determined by scanning actual label files

    # Get channel (first channel if multiple)
    channel_dict = dataset_info.get("modality", {"0": "Unknown"})
    channel = list(channel_dict.values())[0]

    # Find all training images
    # NOTE: Fingerprinting is done on preprocessed (cropped) images in nnUNet_preprocessed
    # This function should be called with the preprocessed directory path
    images_dir = Path(dataset_dir) / "imagesTr"
    labels_dir = Path(dataset_dir) / "labelsTr"

    # Support multiple formats
    image_paths = []
    for pattern in ["*.nii.gz", "*.nii", "*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(
            str(p)
            for p in images_dir.glob(pattern)
            if not p.name.startswith("._")  # Skip macOS metadata files
        )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {images_dir}. "
            f"Supported formats: .nii.gz, .nii, .png, .jpg"
        )

    logger.info(f"Found {len(image_paths)} training images")

    # Create pairs of (image_path, label_path)
    image_label_pairs = []
    for img_path in image_paths:
        img_name = Path(img_path).name
        # Strip channel suffix (e.g., "_0000") to match label naming convention
        # Image: 000_0000.png -> Label: 000.png
        label_name = img_name
        if "_" in label_name:
            # Remove the channel suffix (_0000, _0001, etc.)
            parts = label_name.split("_")
            if (
                len(parts) > 1
                and parts[-1].replace(".png", "").replace(".nii.gz", "").isdigit()
            ):
                # Reconstruct without channel suffix
                # Handle .nii.gz extension properly (Path.suffix only returns .gz)
                if label_name.endswith(".nii.gz"):
                    label_ext = ".nii.gz"
                else:
                    label_ext = Path(label_name).suffix
                label_name = "_".join(parts[:-1]) + label_ext

        label_path = labels_dir / label_name

        if label_path.exists():
            image_label_pairs.append((img_path, str(label_path)))
        else:
            # No label found, use None
            image_label_pairs.append((img_path, None))
            logger.debug(
                f"No label found for {img_name}, using all voxels for intensity stats"
            )

    # Determine num_classes by scanning actual label files
    label_paths_for_scanning = [lp for _, lp in image_label_pairs if lp is not None]
    if label_paths_for_scanning:
        num_classes = _scan_unique_label_values(label_paths_for_scanning)
    else:
        # Fallback to dataset.json if no labels found
        num_classes = len(dataset_info.get("labels", {}))
        logger.warning(f"No label files found, using num_classes from dataset.json: {num_classes}")

    # Extract properties from all images
    properties_list: list[ImageProperties] = []

    if num_workers > 1:
        # Parallel processing
        logger.info("Processing images in parallel...")

        with Pool(processes=num_workers) as pool:
            # Use imap to get results as they complete for progress tracking
            results = []
            for i, result in enumerate(
                pool.imap(_load_image_properties_safe, image_label_pairs)
            ):
                if i % 50 == 0 and i > 0:
                    logger.info(f"Processed {i}/{len(image_label_pairs)} images...")
                results.append(result)

            # Filter out None values (failed loads)
            properties_list = [r for r in results if r is not None]
    else:
        # Sequential processing (num_workers == 1)
        logger.info("Processing images sequentially...")

        for i, (img_path, label_path) in enumerate(image_label_pairs):
            if i % 50 == 0:
                logger.info(f"Processing image {i + 1}/{len(image_label_pairs)}...")

            try:
                props = _load_image_properties(img_path, label_path)
                properties_list.append(props)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                continue

    if not properties_list:
        raise ValueError(f"Failed to load any images from {images_dir}")

    logger.info(f"Successfully loaded {len(properties_list)} images")

    # Aggregate shape statistics
    shapes = np.array([p.shape for p in properties_list])
    median_shape = tuple(int(x) for x in np.median(shapes, axis=0))
    percentile_10_shape = tuple(int(x) for x in np.percentile(shapes, 10, axis=0))
    percentile_90_shape = tuple(int(x) for x in np.percentile(shapes, 90, axis=0))

    # Aggregate spacing statistics
    spacings = np.array([p.spacing for p in properties_list])
    median_spacing = tuple(float(x) for x in np.median(spacings, axis=0))
    percentile_10_spacing = tuple(float(x) for x in np.percentile(spacings, 10, axis=0))
    percentile_90_spacing = tuple(float(x) for x in np.percentile(spacings, 90, axis=0))

    # Detect 2D vs 3D
    # Following nnU-Net v2: Only spatial dimensions are considered (skip channel dimension)
    # Images are loaded in channel-first format: [C, D, H, W] or [C, H, W]
    # Extract spatial dimensions by skipping the first (channel) dimension
    spatial_shape = median_shape[1:] if len(median_shape) > 2 else median_shape

    # nnUNet logic: Dataset is 2D only if it has 2 spatial dimensions (no depth)
    # Datasets with shape [C, 1, H, W] are 3D (anisotropic), not 2D
    is_2d = len(spatial_shape) == 2

    # Detect anisotropy
    is_anisotropic, anisotropy_axis = _detect_anisotropy(median_spacing, median_shape)

    # Aggregate intensity statistics (nnUNet v2.4.1 style: pool all foreground voxels)
    # Concatenate all sampled intensities from all cases
    all_foreground_intensities = np.concatenate(
        [p.foreground_intensities for p in properties_list]
    )

    # Compute statistics on pooled data
    intensity_mean = float(np.mean(all_foreground_intensities))
    intensity_std = float(np.std(all_foreground_intensities))
    intensity_percentile_00_5 = float(np.percentile(all_foreground_intensities, 0.5))
    intensity_percentile_99_5 = float(np.percentile(all_foreground_intensities, 99.5))

    # Determine normalization scheme
    normalization_scheme = _determine_normalization_scheme(channel)

    fingerprint = DatasetFingerprint(
        dataset_name=dataset_name,
        num_training_cases=len(properties_list),
        num_classes=num_classes,
        channel=channel,
        is_2d=is_2d,
        median_shape=median_shape,
        percentile_10_shape=percentile_10_shape,
        percentile_90_shape=percentile_90_shape,
        median_spacing=median_spacing,
        percentile_10_spacing=percentile_10_spacing,
        percentile_90_spacing=percentile_90_spacing,
        is_anisotropic=is_anisotropic,
        anisotropy_axis=anisotropy_axis,
        intensity_mean=intensity_mean,
        intensity_std=intensity_std,
        intensity_percentile_00_5=intensity_percentile_00_5,
        intensity_percentile_99_5=intensity_percentile_99_5,
        normalization_scheme=normalization_scheme,
    )

    logger.info("Dataset fingerprinting complete!")
    logger.info(f"  Channel: {fingerprint.channel}")
    logger.info(f"  2D/3D: {'2D' if fingerprint.is_2d else '3D'}")
    logger.info(f"  Median shape: {fingerprint.median_shape}")
    logger.info(f"  Median spacing: {fingerprint.median_spacing}")
    logger.info(f"  Anisotropic: {fingerprint.is_anisotropic}")
    logger.info(f"  Normalization: {fingerprint.normalization_scheme}")

    return fingerprint
