"""
Metadata extraction for dataset fingerprinting.
Determines normalization schemes and scans label values.
"""

from __future__ import annotations

import random

import numpy as np
from loguru import logger
from PIL import Image

from src.utils.files import detect_file_type, load_nifti_with_metadata


# DOC: NORMALIZATION_SCHEME | Category: Constant | Documentation: docs/planning.md
# Description: CT → CTNormalization, Other → ZScoreNormalization
# Function: determine_normalization_scheme | Documentation: docs/planning.md Step 1
def determine_normalization_scheme(channel: str) -> str:
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


def scan_unique_label_values(label_paths: list[str], num_samples: int = 50) -> int:
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
    sample_paths = random.sample(label_paths, min(num_samples, len(label_paths)))

    unique_values = set()
    for label_path in sample_paths:
        try:
            file_type = detect_file_type(label_path)
            if file_type == "nifti":
                label_data, _ = load_nifti_with_metadata(label_path)
            else:
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

    logger.debug(
        f"Scanned {len(sample_paths)} label files, found {len(unique_values)} unique label values: {sorted(unique_values)}"
    )
    logger.debug(
        f"Determined num_classes = {num_classes} (max label value: {max_label})"
    )

    return num_classes
