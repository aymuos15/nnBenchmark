"""
Patch size and batch size calculation following nnU-Net heuristics.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


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
            round(i) for i in tmp * (256**3 / np.prod(tmp)) ** (1 / 3)
        ]
    elif len(spacing) == 2:
        initial_patch_size = [
            round(i) for i in tmp * (2048**2 / np.prod(tmp)) ** (1 / 2)
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
    # nnU-Net constants
    unet_reference_val_3d = 560000000
    unet_reference_val_2d = 85000000
    unet_reference_val_corresp_gb = 8
    unet_reference_val_corresp_bs_3d = 2
    unet_reference_val_corresp_bs_2d = 12
    max_dataset_covered = 0.05
    unet_min_batch_size = 2

    # Get reference values based on dimensionality
    if is_2d:
        unet_reference_val = unet_reference_val_2d
        ref_bs = unet_reference_val_corresp_bs_2d
    else:
        unet_reference_val = unet_reference_val_3d
        ref_bs = unet_reference_val_corresp_bs_3d

    # Calculate reference scaled to target GPU memory
    reference = unet_reference_val * (
        gpu_memory_target_gb / unet_reference_val_corresp_gb
    )

    # Estimate VRAM complexity (simplified - nnU-Net instantiates network)
    # This is a rough approximation of network complexity
    patch_voxels = np.prod(patch_size)
    num_pool_total = sum(num_pool_per_axis)
    # Rough estimate based on patch size and pooling depth
    estimate = patch_voxels * (2**num_pool_total) * num_input_channels * num_classes

    # Calculate batch size from reference
    batch_size = round((reference / estimate) * ref_bs)

    # Cap to 5% of dataset coverage
    bs_corresponding_to_5_percent = round(
        approximate_n_voxels_dataset
        * max_dataset_covered
        / np.prod(patch_size, dtype=np.float64)
    )

    # Apply constraints
    batch_size = max(
        min(batch_size, bs_corresponding_to_5_percent), unet_min_batch_size
    )

    logger.debug(
        f"Batch size calculation: estimate={estimate:.0f}, reference={reference:.0f}, "
        f"initial_bs={round((reference / estimate) * ref_bs)}, "
        f"5%_cap={bs_corresponding_to_5_percent}, final_bs={batch_size}"
    )

    return batch_size
