"""
nnU-Net heuristics for spacing, feature channels, and deep supervision.
"""

from __future__ import annotations

from src.planning.fingerprinting.fingerprint import DatasetFingerprint


def calculate_target_spacing(fingerprint: DatasetFingerprint) -> tuple[float, ...]:
    """
    Calculate target spacing following nnU-Net heuristics.

    - Default: Use median spacing (50th percentile)
    - Anisotropic adjustment: For anisotropic axis, use 10th percentile if:
      - Worst axis > 4× better axes AND
      - Worst axis has < 25% voxels
    """
    if fingerprint.is_anisotropic and fingerprint.anisotropy_axis is not None:
        # Use 10th percentile for anisotropic axis
        target = list(fingerprint.median_spacing)
        target[fingerprint.anisotropy_axis] = fingerprint.percentile_10_spacing[
            fingerprint.anisotropy_axis
        ]
        return tuple(target)
    else:
        # Use median spacing
        return fingerprint.median_spacing


def calculate_feature_channels(num_stages: int, is_2d: bool) -> list[int]:
    """
    Calculate feature channels per level following nnU-Net exact conventions.

    For a UNet with num_stages pooling operations, there are num_stages+1 decoder levels:
    - Input level (32 channels)
    - After each pooling: double channels (64, 128, 256, ...)
    - Channels double each level, starting at 32 (UNet_base_num_features)
    - 2D: Cap at 512 (UNet_max_features_2d)
    - 3D: Cap at 320 (UNet_max_features_3d)

    Args:
        num_stages: Number of pooling operations (encoder depth)
        is_2d: Whether this is 2D data

    Returns:
        List of feature channels for each decoder level (num_stages + 1 values)
    """
    base_features = 32  # UNet_base_num_features
    max_features = 512 if is_2d else 320  # UNet_max_features_2d/3d

    channels = []
    # num_stages pooling ops means num_stages+1 decoder levels
    for i in range(num_stages + 1):
        features = min(max_features, base_features * (2**i))
        channels.append(features)

    return channels


def calculate_deep_supervision_weights(num_stages: int) -> list[float]:
    """
    Calculate deep supervision weights following nnU-Net style (decreasing).

    Uses exponential decay: weights[i] = 2^(-i)
    This favors the final output while still providing gradients to earlier layers.

    Args:
        num_stages: Number of decoder stages

    Returns:
        List of weights, one per decoder output level

    """
    weights = [2.0 ** (-i) for i in range(num_stages)]
    return weights
