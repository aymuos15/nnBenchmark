"""GPU-accelerated Connected Components utilities for loss and metric functions.

This module provides GPU-accelerated connected components labeling and region mapping,
shared between CCLoss and CCMetric to avoid code duplication.

Based on GPU-Connected-Components: https://github.com/aymuos15/GPU-Connected-Components
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    import cupy as cp  # type: ignore[import]
    from cucim.skimage import measure as cucim_measure  # type: ignore[import]
    from cupyx.scipy.ndimage import distance_transform_edt  # type: ignore[import]
else:
    try:
        import cupy as cp  # type: ignore[import]
        from cucim.skimage import measure as cucim_measure  # type: ignore[import]
        from cupyx.scipy.ndimage import distance_transform_edt  # type: ignore[import]
    except ImportError:
        cp = None  # type: ignore
        cucim_measure = None  # type: ignore
        distance_transform_edt = None  # type: ignore


def gpu_connected_components(
    img: torch.Tensor, connectivity: Optional[int] = None
) -> Tuple[torch.Tensor, int]:
    """
    PyTorch wrapper for calculating connected components on a GPU using cupy and cucim.

    From: https://github.com/aymuos15/GPU-Connected-Components

    Args:
        img: Input image tensor
        connectivity: Connectivity defining the neighborhood. Default is None.

    Returns:
        tuple: (labeled_img, num_features)
            - labeled_img: Labeled image with each connected component having a unique label
            - num_features: Number of connected components found

    Raises:
        ImportError: If cupy and cucim are not installed
    """
    if cp is None:
        raise ImportError(
            "cupy and cucim are required for GPU connected components. "
            "Install with: pip install cupy-cuda11x cucim"
        )

    img_cupy = cp.asarray(img)
    labeled_img, num_features = cucim_measure.label(
        img_cupy, connectivity=connectivity, return_num=True
    )
    labeled_img_torch = torch.as_tensor(labeled_img, device=img.device)
    return labeled_img_torch, num_features


def get_gt_regions(
    gt: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Divides the ground truth segmentation space into regions based on proximity to instances.

    This function uses GPU-accelerated distance transforms to create a Voronoi-like partition
    of the image space, where each pixel is assigned to the closest ground truth instance.

    Args:
        gt: Ground truth segmentation for a single class
        device: Device to place tensors on

    Returns:
        tuple: (region_map, labeled_gt, num_features)
            - region_map: Tensor where each pixel is labeled with the nearest region ID
            - labeled_gt: Tensor with connected component labels from original ground truth
            - num_features: Number of distinct regions/connected components

    Raises:
        ImportError: If cupy and cupyx are not installed
    """
    if distance_transform_edt is None:
        raise ImportError(
            "cupyx is required for distance transforms. "
            "Install with: pip install cupy-cuda11x"
        )

    # Identify connected components in the ground truth
    labeled_gt, num_features = gpu_connected_components(gt)

    if num_features == 0:
        return torch.zeros_like(gt, dtype=torch.long), 0

    # Initialize distance map (stores distance to nearest instance)
    distance_map = torch.zeros_like(gt, dtype=torch.float32)
    # Initialize region map (stores region ID for each pixel)
    region_map = torch.zeros_like(gt, dtype=torch.long)

    # Process each connected component (region) in the ground truth
    for region_label in range(1, num_features + 1):
        # Create binary mask for current region
        region_mask = labeled_gt == region_label

        # Convert PyTorch tensor to CuPy array (stays on GPU)
        region_mask_cupy = cp.asarray(region_mask)

        # Calculate Euclidean distance transform using GPU-accelerated cupyx
        distance_cupy = distance_transform_edt(
            ~region_mask_cupy,
            float64_distances=False,  # Use float32 for speed
        )

        # Convert CuPy array back to PyTorch tensor (stays on GPU)
        distance = torch.as_tensor(distance_cupy, device=device)

        # Initialize or update based on closest distance
        if region_label == 1 or distance_map.max() == 0:
            distance_map = distance
            region_map = region_label * torch.ones_like(gt, dtype=torch.long)
        else:
            # Update pixels that are closer to this region
            update_mask = distance < distance_map
            distance_map[update_mask] = distance[update_mask]
            region_map[update_mask] = region_label

    return region_map, labeled_gt, num_features


__all__ = ["gpu_connected_components", "get_gt_regions"]
