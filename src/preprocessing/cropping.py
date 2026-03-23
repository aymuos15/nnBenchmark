"""
Cropping utilities for nnBenchmark.

Implements crop-to-nonzero preprocessing to match nnU-Net's pipeline.
Removes zero background regions to reduce computational burden and memory usage.

Based on nnU-Net v2.4.1 implementation from:
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/cropping/cropping.py
"""


from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes


def create_nonzero_mask(data: NDArray) -> NDArray:
    """
    Create binary mask identifying non-zero regions across all channels.

    Combines channels using logical OR to identify voxels where any channel
    has non-zero values. Applies morphological hole-filling to create a
    continuous mask.

    Args:
        data: Input image array, shape can be:
            - (H, W, D): Single channel 3D image
            - (C, H, W, D): Multi-channel 3D image
            - Other spatial dimensions supported

    Returns:
        Binary mask array, same shape as input spatial dimensions.
        True/1 where any channel is non-zero, False/0 elsewhere.
    """
    # Determine if input has channel dimension
    # Format: (C, H, W) for 2D spatial or (C, H, W, D) for 3D spatial
    if data.ndim == 4:
        # Clearly (C, H, W, D) format - 3D spatial with multiple channels
        channels_first = True
        n_channels = data.shape[0]
    elif data.ndim == 3:
        # Could be (C, H, W) for 2D spatial, or (H, W, D) for 3D spatial single-channel
        # Check if first dimension looks like channels (small value, typically 1-4)
        if data.shape[0] <= 4:
            # Likely (C, H, W) format - 2D spatial with channels
            channels_first = True
            n_channels = data.shape[0]
        else:
            # Likely (H, W, D) format - 3D spatial single channel
            channels_first = False
            n_channels = 1
    else:
        raise ValueError(f"Expected 3D or 4D array, got {data.ndim}D array")

    # Create mask by checking if any channel is non-zero
    if channels_first:
        mask = np.zeros(data.shape[1:], dtype=bool)
        for c in range(n_channels):
            mask |= data[c] != 0
    else:
        mask = data != 0

    # Apply morphological hole-filling to create continuous mask
    # This fills small holes in the foreground region
    mask = binary_fill_holes(mask)

    return np.asarray(mask)


def get_bbox_from_mask(mask: NDArray) -> list[list[int]]:
    """
    Find bounding box coordinates from a binary mask.

    Identifies the minimal rectangular region containing all True values
    in the mask. Returns coordinates suitable for slicing arrays.

    Args:
        mask: Binary mask array (True where foreground, False for background)

    Returns:
        List of [min, max+1] coordinates for each dimension, e.g.:
        [[minz, maxz], [minx, maxx], [miny, maxy]]
        Coordinates are in numpy slicing format (max is exclusive).
    """
    # Find indices where mask is True
    nonzero_indices = np.where(mask)

    if len(nonzero_indices[0]) == 0:
        # Empty mask - return minimal bbox
        return [[0, 0], [0, 0], [0, 0]]

    # Find min and max for each dimension
    bbox = []
    for dim in range(mask.ndim):
        min_idx = int(np.min(nonzero_indices[dim]))
        max_idx = int(np.max(nonzero_indices[dim])) + 1  # +1 for exclusive upper bound
        bbox.append([min_idx, max_idx])

    return bbox


def crop_to_nonzero(
    data: NDArray,
    seg: Optional[NDArray] = None,
    mask: Optional[NDArray] = None,
) -> tuple[NDArray, Optional[NDArray], list[list[int]]]:
    """
    Crop image and segmentation to their nonzero bounding box.

    This is the main cropping function matching nnU-Net's behavior.
    Creates a mask, computes its bounding box, and crops both image and
    segmentation to that box.

    Args:
        data: Input image array, can be:
            - (H, W, D): Single channel 3D
            - (C, H, W, D): Multi-channel 3D
        seg: Optional segmentation array, same shape as data. If provided,
            will be cropped to the same bounding box.
        mask: Optional pre-computed binary mask. If None, will be created
            from data. Useful if mask is computed once for efficiency.

    Returns:
        Tuple of (cropped_data, cropped_seg, bbox):
        - cropped_data: Image cropped to nonzero bounding box
        - cropped_seg: Segmentation cropped to same bbox (None if seg was None)
        - bbox: Bounding box coordinates [[minz, maxz], [minx, maxx], [miny, maxy]]

    Raises:
        ValueError: If data and seg shapes don't match (when seg is provided)
    """
    # Validate inputs
    if seg is not None and data.shape != seg.shape:
        raise ValueError(
            f"data and seg must have same shape. Got data: {data.shape}, seg: {seg.shape}"
        )

    # Create mask if not provided
    if mask is None:
        mask = create_nonzero_mask(data)

    # Get bounding box from mask
    bbox = get_bbox_from_mask(mask)
    n_spatial_dims = len(bbox)  # Number of spatial dimensions (2 for 2D, 3 for 3D)

    # Determine data format
    # 4D: (C, H, W, D) - multi-channel 3D spatial data
    # 3D: (C, H, W) - multi-channel 2D spatial data, or (H, W, D) - single-channel 3D spatial
    is_multichannel = data.ndim == (n_spatial_dims + 1)

    # Build slicing indices based on number of spatial dimensions
    cropped_seg: Optional[NDArray] = None
    if is_multichannel:
        # Multi-channel format: skip first dimension (channels)
        slices = [slice(None)]  # Keep all channels
        for i in range(n_spatial_dims):
            slices.append(slice(bbox[i][0], bbox[i][1]))
        cropped_data = data[tuple(slices)]

        if seg is not None:
            cropped_seg = seg[tuple(slices)]
    else:
        # Single-channel format
        slices = tuple(slice(bbox[i][0], bbox[i][1]) for i in range(n_spatial_dims))
        cropped_data = data[slices]

        if seg is not None:
            cropped_seg = seg[slices]

    return cropped_data, cropped_seg, bbox
