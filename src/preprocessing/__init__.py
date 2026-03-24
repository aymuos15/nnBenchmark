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
from src.preprocessing.tensor_cache import preprocess_to_tensors

__all__ = [
    "create_nonzero_mask",
    "get_bbox_from_mask",
    "crop_to_nonzero",
    "preprocess_to_tensors",
]
