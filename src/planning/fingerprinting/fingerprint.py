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

from src.planning.fingerprinting.loading import (
    ImageProperties,
    load_image_properties,
    load_image_properties_safe,
)
from src.planning.fingerprinting.metadata import (
    determine_normalization_scheme,
    scan_unique_label_values,
)
from src.planning.fingerprinting.spacing import detect_anisotropy
from src.utils.files import load_json


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


# DOC: DATASET_FINGERPRINTING | Category: Adaptive | Documentation: docs/planning.md Step 1
# Description: Parallel analysis of preprocessed images to extract statistical properties
# Function: fingerprint_dataset | Documentation: docs/planning.md Step 1
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
        num_classes = scan_unique_label_values(label_paths_for_scanning)
    else:
        # Fallback to dataset.json if no labels found
        num_classes = len(dataset_info.get("labels", {}))
        logger.warning(
            f"No label files found, using num_classes from dataset.json: {num_classes}"
        )

    # Extract properties from all images
    properties_list: list[ImageProperties] = []

    if num_workers > 1:
        # Parallel processing
        logger.info("Processing images in parallel...")

        with Pool(processes=num_workers) as pool:
            # Use imap to get results as they complete for progress tracking
            results = []
            for i, result in enumerate(
                pool.imap(load_image_properties_safe, image_label_pairs)
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
                props = load_image_properties(img_path, label_path)
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
    is_anisotropic, anisotropy_axis = detect_anisotropy(median_spacing, median_shape)

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
    normalization_scheme = determine_normalization_scheme(channel)

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
