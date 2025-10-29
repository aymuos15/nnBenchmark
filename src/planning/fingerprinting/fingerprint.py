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
    intensity_mean: float
    intensity_std: float
    intensity_percentile_00_5: float
    intensity_percentile_99_5: float


@dataclass
class DatasetFingerprint:
    """Aggregated dataset properties for experiment planning."""

    # Dataset identification
    dataset_name: str
    num_training_cases: int
    num_classes: int
    modality: str

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

    # Normalization scheme determined from modality
    normalization_scheme: str  # 'CTNormalization', 'ZScoreNormalization', etc.


def _load_image_properties(image_path: str) -> ImageProperties:
    """
    Load image and extract properties.

    Supports NIfTI (.nii.gz) and PNG/JPEG with automatic format detection.
    Uses MONAI's LoadImaged for robust NIfTI image loading.
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

        # Handle grayscale vs RGB
        if len(data.shape) == 2:
            spacing = (1.0, 1.0)
        elif len(data.shape) == 3:
            # RGB image - consider as multi-channel 2D
            spacing = (1.0, 1.0)
        else:
            raise ValueError(f"Unexpected PNG/JPEG shape: {data.shape}")

    else:
        raise ValueError(f"Unsupported file type: {file_type} for {image_path}")

    # Calculate intensity statistics
    # Flatten data for statistics
    data_flat = data.flatten()

    return ImageProperties(
        shape=data.shape,
        spacing=spacing,
        intensity_mean=float(np.mean(data_flat)),
        intensity_std=float(np.std(data_flat)),
        intensity_percentile_00_5=float(np.percentile(data_flat, 0.5)),
        intensity_percentile_99_5=float(np.percentile(data_flat, 99.5)),
    )


def _load_image_properties_safe(image_path: str) -> ImageProperties | None:
    """
    Wrapper for _load_image_properties that handles exceptions.

    Used in parallel processing to gracefully handle individual image failures.
    Returns None if loading fails, otherwise returns ImageProperties.
    """
    try:
        return _load_image_properties(image_path)
    except Exception as e:
        logger.warning(f"Failed to load {image_path}: {e}")
        return None


def _determine_normalization_scheme(modality: str) -> str:
    """
    Determine normalization scheme based on modality.

    Following nnU-Net conventions:
    - CT → CTNormalization (percentile clipping + dataset-wide z-score)
    - Other modalities → ZScoreNormalization (per-case z-score)
    """
    modality_lower = modality.lower()

    if "ct" in modality_lower:
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
    num_classes = len(dataset_info.get("labels", {}))

    # Get modality (first modality if multiple)
    modality_dict = dataset_info.get("modality", {"0": "Unknown"})
    modality = list(modality_dict.values())[0]

    # Find all training images
    images_dir = Path(dataset_dir) / "imagesTr"

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

    # Extract properties from all images
    properties_list: list[ImageProperties] = []

    if num_workers > 1:
        # Parallel processing
        logger.info("Processing images in parallel...")

        with Pool(processes=num_workers) as pool:
            # Use imap to get results as they complete for progress tracking
            results = []
            for i, result in enumerate(
                pool.imap(_load_image_properties_safe, image_paths)
            ):
                if i % 50 == 0 and i > 0:
                    logger.info(f"Processed {i}/{len(image_paths)} images...")
                results.append(result)

            # Filter out None values (failed loads)
            properties_list = [r for r in results if r is not None]
    else:
        # Sequential processing (num_workers == 1)
        logger.info("Processing images sequentially...")

        for i, img_path in enumerate(image_paths):
            if i % 50 == 0:
                logger.info(f"Processing image {i + 1}/{len(image_paths)}...")

            try:
                props = _load_image_properties(img_path)
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
    # Following nnU-Net: if median shape has one axis = 1 (or very small), it's 2D
    is_2d = any(s <= 1 for s in median_shape) or (len(median_shape) == 2)

    # Detect anisotropy
    is_anisotropic, anisotropy_axis = _detect_anisotropy(median_spacing, median_shape)

    # Aggregate intensity statistics
    intensity_mean = float(np.mean([p.intensity_mean for p in properties_list]))
    intensity_std = float(np.mean([p.intensity_std for p in properties_list]))
    intensity_percentile_00_5 = float(
        np.percentile([p.intensity_percentile_00_5 for p in properties_list], 0.5)
    )
    intensity_percentile_99_5 = float(
        np.percentile([p.intensity_percentile_99_5 for p in properties_list], 99.5)
    )

    # Determine normalization scheme
    normalization_scheme = _determine_normalization_scheme(modality)

    fingerprint = DatasetFingerprint(
        dataset_name=dataset_name,
        num_training_cases=len(properties_list),
        num_classes=num_classes,
        modality=modality,
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
    logger.info(f"  Modality: {fingerprint.modality}")
    logger.info(f"  2D/3D: {'2D' if fingerprint.is_2d else '3D'}")
    logger.info(f"  Median shape: {fingerprint.median_shape}")
    logger.info(f"  Median spacing: {fingerprint.median_spacing}")
    logger.info(f"  Anisotropic: {fingerprint.is_anisotropic}")
    logger.info(f"  Normalization: {fingerprint.normalization_scheme}")

    return fingerprint
