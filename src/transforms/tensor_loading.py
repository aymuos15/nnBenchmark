"""Transform for loading preprocessed .pt tensor files.

Replaces the common transforms pipeline (LoadImaged, NormalizeIntensityd,
SpatialPadd, ToTensord) with a single .pt file load using memory-mapped
access for efficient large-dataset support.
"""

from __future__ import annotations

from pathlib import Path

import torch
from monai.transforms import MapTransform


class LoadPreprocessedTensord(MapTransform):
    """Load preprocessed .pt tensor from the tensor_cache path in data dict.

    Uses torch.load(mmap=True) for memory-mapped access so large datasets
    don't need to fit entirely in RAM.

    Expects data dict to contain a "tensor_cache" key pointing to a .pt file
    with {"image": Tensor, "label": Tensor}.
    """

    def __init__(self, keys: tuple[str, ...] = ("image", "label")):
        super().__init__(keys)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        tensor_path = d["tensor_cache"]
        cached = torch.load(tensor_path, weights_only=False, mmap=True)
        for key in self.keys:
            d[key] = cached[key]
        return d
