"""
nnU-Net heuristics for spacing, feature channels, and deep supervision.
"""

from __future__ import annotations

from src.planning.constants import PLANNING_CONSTANTS
from src.planning.fingerprinting.fingerprint import DatasetFingerprint


# DOC: TARGET_SPACING_CALCULATION | Category: Constant+Adaptive | Documentation: docs/planning.md
# Description: Median spacing by default, 10th percentile for anisotropic axis
# Function: calculate_target_spacing | Constants: aniso_threshold=3.0 | Documentation: docs/planning.md Step 2
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


# DOC: FEATURE_CHANNELS_CALCULATION | Category: Constant+Adaptive | Documentation: docs/planning.md
# Description: base=32, doubles per level, capped at 512(2D)/320(3D)
# Function: calculate_feature_channels | Constants: base=32, max_2d=512, max_3d=320 | Documentation: docs/planning.md Step 2
def calculate_feature_channels(num_stages: int, is_2d: bool) -> list[int]:
    """
    Calculate feature channels per level following nnU-Net exact conventions.

    For DynUNet with num_stages encoder stages, we need num_stages filter values:
    - Each stage has its own filter count
    - Channels double each level, starting at 32 (UNet_base_num_features)
    - 2D: Cap at 512 (UNet_max_features_2d)
    - 3D: Cap at 320 (UNet_max_features_3d)

    Args:
        num_stages: Number of encoder stages (equals number of strides)
        is_2d: Whether this is 2D data

    Returns:
        List of feature channels for each encoder stage (num_stages values)
    """
    base_features = PLANNING_CONSTANTS.BASE_FEATURES  # UNet_base_num_features
    max_features = (
        PLANNING_CONSTANTS.MAX_FEATURES_2D
        if is_2d
        else PLANNING_CONSTANTS.MAX_FEATURES_3D
    )  # UNet_max_features_2d/3d

    channels = []
    # DynUNet: num_stages encoder stages means num_stages filter values
    for i in range(num_stages):
        features = min(max_features, base_features * (2**i))
        channels.append(features)

    return channels


# DOC: DEEP_SUPERVISION_WEIGHTS | Category: Constant | Documentation: docs/planning.md
# Description: Exponential decay formula 2^(-i), deep_supr_num=1
# Function: calculate_deep_supervision_weights | Constants: deep_supr_num=1 | Documentation: docs/planning.md Step 2
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
