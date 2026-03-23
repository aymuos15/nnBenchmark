"""
Patch size and batch size calculation following nnU-Net heuristics.
"""


import numpy as np
from loguru import logger

from src.planning.constants import PLANNING_CONSTANTS


def calculate_initial_patch_size(
    spacing: tuple[float, ...], median_shape: tuple[int, ...], is_2d: bool
) -> tuple[int, ...]:
    """
    Calculate initial patch size following nnU-Net exact heuristics.

    This matches the nnU-Net implementation exactly:
    - 3D: tmp = 1/spacing * (256^3 / prod(1/spacing))^(1/3)
    - 2D: tmp = 1/spacing * (2048^2 / prod(1/spacing))^(1/2)
    - Clip to median shape
    """
    tmp = 1 / np.array(spacing)

    if len(spacing) == 3:
        initial_patch_size = [
            round(i)
            for i in tmp
            * (PLANNING_CONSTANTS.PATCH_NORM_3D**3 / np.prod(tmp)) ** (1 / 3)
        ]
    elif len(spacing) == 2:
        initial_patch_size = [
            round(i)
            for i in tmp
            * (PLANNING_CONSTANTS.PATCH_NORM_2D**2 / np.prod(tmp)) ** (1 / 2)
        ]
    else:
        raise RuntimeError(f"Unsupported dimensionality: {len(spacing)}")

    initial_patch_size = np.array(
        [min(i, j) for i, j in zip(initial_patch_size, median_shape[: len(spacing)])]
    )

    return tuple(initial_patch_size)


def calculate_batch_size(
    patch_size: tuple[int, ...],
    num_pool_per_axis: list[int],
    num_input_channels: int,
    num_classes: int,
    is_2d: bool,
    approximate_n_voxels_dataset: float,
    gpu_memory_target_gb: float = 8.0,
) -> int:
    """
    Calculate batch size following nnU-Net exact formula.

    nnU-Net uses reference VRAM values and scales based on patch size complexity.

    Constants (from nnU-Net):
    - UNet_reference_val_3d = 560000000 (reference complexity for 3D)
    - UNet_reference_val_2d = 85000000 (reference complexity for 2D)
    - UNet_reference_val_corresp_GB = 8 (GB for reference)
    - UNet_reference_val_corresp_bs_3d = 2 (reference batch size 3D)
    - UNet_reference_val_corresp_bs_2d = 12 (reference batch size 2D)
    - max_dataset_covered = 0.05 (max 5% dataset per batch)
    - UNet_min_batch_size = 2

    Formula:
    1. estimate = static_estimate_VRAM_usage(...)
    2. reference = UNet_reference_val * (target_GB / corresp_GB)
    3. batch_size = round((reference / estimate) * ref_bs)
    4. batch_size = max(min(batch_size, bs_5_percent), min_batch_size)
    """
    # Get reference values based on dimensionality
    if is_2d:
        unet_reference_val = PLANNING_CONSTANTS.UNET_REFERENCE_VAL_2D
        ref_bs = PLANNING_CONSTANTS.UNET_REFERENCE_CORRESP_BS_2D
    else:
        unet_reference_val = PLANNING_CONSTANTS.UNET_REFERENCE_VAL_3D
        ref_bs = PLANNING_CONSTANTS.UNET_REFERENCE_CORRESP_BS_3D

    # Calculate reference scaled to target GPU memory
    reference = unet_reference_val * (
        gpu_memory_target_gb / PLANNING_CONSTANTS.UNET_REFERENCE_CORRESP_GB
    )

    # Estimate VRAM complexity (simplified - nnU-Net instantiates model)
    # This is a rough approximation of model complexity
    patch_voxels = np.prod(patch_size)
    if patch_voxels == 0:
        raise ValueError(
            f"Invalid patch size {patch_size}: all dimensions must be > 0. "
            f"This may indicate improper calculation during planning."
        )

    num_pool_total = sum(num_pool_per_axis)
    # Rough estimate based on patch size and pooling depth
    estimate = patch_voxels * (2**num_pool_total) * num_input_channels * num_classes

    if estimate <= 0:
        raise ValueError(
            f"Invalid complexity estimate: {estimate}. "
            f"Check that patch_size={patch_size}, num_pool_per_axis={num_pool_per_axis}, "
            f"num_input_channels={num_input_channels}, num_classes={num_classes} are all positive."
        )

    # Calculate batch size from reference
    batch_size = round((reference / estimate) * ref_bs)

    # Cap to 5% of dataset coverage
    bs_corresponding_to_5_percent = round(
        approximate_n_voxels_dataset
        * PLANNING_CONSTANTS.MAX_DATASET_COVERED
        / np.prod(patch_size, dtype=np.float64)
    )

    # Apply constraints
    batch_size = max(
        min(batch_size, bs_corresponding_to_5_percent),
        PLANNING_CONSTANTS.UNET_MIN_BATCH_SIZE,
    )

    logger.debug(
        f"Batch size calculation: estimate={estimate:.0f}, reference={reference:.0f}, "
        f"initial_bs={round((reference / estimate) * ref_bs)}, "
        f"5%_cap={bs_corresponding_to_5_percent}, final_bs={batch_size}"
    )

    return batch_size
