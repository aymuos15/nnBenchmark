"""
Spacing analysis for dataset fingerprinting.
Detects anisotropy following nnU-Net heuristics.
"""


import numpy as np

from src.planning.constants import PLANNING_CONSTANTS


def detect_anisotropy(
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

    aniso_threshold = PLANNING_CONSTANTS.ANISOTROPY_THRESHOLD
    voxel_ratio_threshold = PLANNING_CONSTANTS.ANISOTROPY_VOXEL_RATIO
    is_anisotropic = bool(
        spacing_ratio > aniso_threshold and voxel_ratio < voxel_ratio_threshold
    )

    return is_anisotropic, worst_axis if is_anisotropic else None
