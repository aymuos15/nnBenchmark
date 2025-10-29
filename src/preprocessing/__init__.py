"""
Preprocessing module for nnBenchmark.

Provides utilities for image preprocessing including cropping, padding, and normalization.
Designed to match nnU-Net v2.4.1 preprocessing pipeline exactly.
"""

from src.preprocessing.cropping import (
    create_nonzero_mask,
    crop_to_nonzero,
    get_bbox_from_mask,
)

__all__ = [
    "create_nonzero_mask",
    "get_bbox_from_mask",
    "crop_to_nonzero",
]
