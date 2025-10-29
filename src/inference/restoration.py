"""
Inference restoration utilities for nnBenchmark.

Implements coordinate restoration pipeline to reverse preprocessing steps applied during training.
Matches nnU-Net's inference-time transformations for accurate prediction restoration.

Based on nnU-Net v2.4.1 implementation from:
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def pad_nd_image(
    data: NDArray,
    new_shape: tuple[int, ...],
    mode: str = "constant",
    kwargs: Optional[dict] = None,
    return_slicer: bool = False,
) -> NDArray | tuple[NDArray, tuple[slice, ...]]:
    """
    Pad n-dimensional image to specified shape.

    Used to ensure image dimensions are divisible by 2^N for sliding window inference
    (where N is the number of pooling operations in the network).

    Args:
        data: Input array to pad
        new_shape: Target shape for each dimension
        mode: Padding mode ('constant', 'edge', 'reflect', etc.)
        kwargs: Additional arguments for np.pad (e.g., {'constant_values': 0})
        return_slicer: If True, return tuple of (padded_data, slicer_revert_padding)
            slicer_revert_padding is coordinates to slice out original data from padded result

    Returns:
        Padded array, or (padded_array, slicer_revert_padding) if return_slicer=True

    Raises:
        ValueError: If new_shape is smaller than data shape in any dimension
    """
    if kwargs is None:
        kwargs = {}

    if len(data.shape) != len(new_shape):
        raise ValueError(
            f"data.ndim ({len(data.shape)}) != new_shape.ndim ({len(new_shape)})"
        )

    # Check if new_shape is smaller in any dimension
    for i, (old_size, new_size) in enumerate(zip(data.shape, new_shape)):
        if new_size < old_size:
            raise ValueError(
                f"new_shape[{i}] ({new_size}) < data.shape[{i}] ({old_size}). "
                "Can only pad, not crop."
            )

    # Calculate padding for each dimension
    # Pad symmetrically (split extra padding between before and after)
    pad_width = []
    slicers = []

    for old_size, new_size in zip(data.shape, new_shape):
        total_pad = new_size - old_size
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before

        pad_width.append((pad_before, pad_after))

        # Create slicer to extract original data
        slicer = slice(pad_before, pad_before + old_size)
        slicers.append(slicer)

    # Apply padding
    padded = np.pad(data, pad_width, mode=mode, **kwargs)  # type: ignore[arg-type]

    if return_slicer:
        return padded, tuple(slicers)
    else:
        return padded


def get_padding_for_divisibility(
    shape: tuple[int, ...],
    divisor: int,
) -> tuple[int, ...]:
    """
    Calculate target shape to make dimensions divisible by a number.

    Used to determine padding needed for sliding window inference where
    divisor = 2^N (N = number of pooling operations).

    Args:
        shape: Current image shape
        divisor: Number that dimensions must be divisible by (typically 2^N)

    Returns:
        New shape with each dimension padded up to nearest multiple of divisor
    """
    new_shape = []
    for size in shape:
        # Round up to nearest multiple of divisor
        new_size = int(np.ceil(size / divisor) * divisor)
        new_shape.append(new_size)

    return tuple(new_shape)


def uncrop_predictions(
    predictions: NDArray,
    crop_bbox: list[list[int]],
    original_shape: tuple[int, ...] | list[int],
) -> NDArray:
    """
    Uncrop predictions back to original image bounding box.

    Reverses the crop-to-nonzero operation applied during preprocessing.
    Places cropped predictions back into original image space and fills
    background regions with zeros.

    Args:
        predictions: Cropped prediction array, shape can be:
            - (H, W, D): Single output channel
            - (C, H, W, D): Multi-channel predictions
        crop_bbox: Bounding box used to crop, format:
            [[minz, maxz], [minx, maxx], [miny, maxy]]
        original_shape: Original image shape before cropping (spatial dims only)
            - Can be (H, W, D) for single channel
            - Can be (C, H, W, D) for multi-channel

    Returns:
        Predictions restored to original image shape with background filled with zeros.

    Raises:
        ValueError: If crop_bbox format is invalid or predictions don't fit
    """
    original_shape = tuple(original_shape)

    # Determine if multi-channel
    is_multichannel = predictions.ndim == 4

    # Determine spatial dimensions (without channel)
    if is_multichannel:
        spatial_shape = original_shape[1:]
        n_channels = original_shape[0]
        pred_channels = predictions.shape[0]

        if pred_channels != n_channels:
            raise ValueError(
                f"Predictions channels ({pred_channels}) != original channels ({n_channels})"
            )
    else:
        spatial_shape = original_shape
        n_channels = 1

    # Create output array filled with zeros
    if is_multichannel:
        output = np.zeros(original_shape, dtype=predictions.dtype)
    else:
        output = np.zeros(spatial_shape, dtype=predictions.dtype)

    # Validate crop_bbox
    if len(crop_bbox) != 3:
        raise ValueError(
            f"crop_bbox must have 3 elements (z, x, y), got {len(crop_bbox)}"
        )

    for bbox_elem in crop_bbox:
        if len(bbox_elem) != 2:
            raise ValueError(f"Each bbox element must have [min, max], got {bbox_elem}")

    # Extract bbox coordinates
    z_min, z_max = crop_bbox[0]
    x_min, x_max = crop_bbox[1]
    y_min, y_max = crop_bbox[2]

    # Validate predictions fit in bbox
    expected_shape = (z_max - z_min, x_max - x_min, y_max - y_min)
    pred_spatial_shape = predictions.shape[1:] if is_multichannel else predictions.shape

    if pred_spatial_shape != expected_shape:
        raise ValueError(
            f"Predictions spatial shape {pred_spatial_shape} doesn't match bbox size {expected_shape}"
        )

    # Place predictions back into original space
    if is_multichannel:
        output[:, z_min:z_max, x_min:x_max, y_min:y_max] = predictions
    else:
        output[z_min:z_max, x_min:x_max, y_min:y_max] = predictions

    return output


def revert_padding(
    predictions: NDArray,
    slicer_revert_padding: tuple[slice, ...],
) -> NDArray:
    """
    Remove padding added for sliding window inference.

    Reverses the pad_nd_image operation by slicing out the original data
    from the padded result.

    Args:
        predictions: Padded predictions array
        slicer_revert_padding: Tuple of slices (from pad_nd_image with return_slicer=True)

    Returns:
        Unpadded predictions with original shape
    """
    return predictions[slicer_revert_padding]


def convert_predictions_to_original_space(
    predictions: NDArray,
    data_properties: dict,
) -> NDArray:
    """
    Convert predictions from network output space back to original image space.

    Master function that applies all restoration operations in correct order:
    1. Revert padding (if applied)
    2. Uncrop to original bounding box
    (Note: Resampling to original spacing should be done separately with resampling utilities)

    Args:
        predictions: Raw network predictions (C, H, W, D or H, W, D)
        data_properties: Dictionary containing transformation metadata:
            - 'crop_bbox': [[minz, maxz], [minx, maxx], [miny, maxy]]
            - 'original_shape': Original spatial shape before cropping
            - 'slicer_revert_padding' (optional): Slicer to revert padding

    Returns:
        Predictions restored to original image space
    """
    # Step 1: Revert padding if it was applied
    if "slicer_revert_padding" in data_properties:
        predictions = revert_padding(
            predictions, data_properties["slicer_revert_padding"]
        )

    # Step 2: Uncrop to original bounding box
    if "crop_bbox" in data_properties and "original_shape" in data_properties:
        predictions = uncrop_predictions(
            predictions,
            data_properties["crop_bbox"],
            data_properties["original_shape"],
        )

    return predictions
