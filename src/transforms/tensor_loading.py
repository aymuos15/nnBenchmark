"""Transform for loading preprocessed .pt tensor files.

Replaces the common transforms pipeline (LoadImaged, NormalizeIntensityd,
SpatialPadd, ToTensord) with a single .pt file load using memory-mapped
access for efficient large-dataset support.

Falls back to common transforms when tensor_cache key is missing (e.g.,
during inference with -i on raw NIfTI files).
"""

from __future__ import annotations

from typing import Any

import torch
from monai.transforms import MapTransform


class LoadPreprocessedTensord(MapTransform):
    """Load preprocessed .pt tensor from the tensor_cache path in data dict.

    Uses torch.load(mmap=True) for memory-mapped access so large datasets
    don't need to fit entirely in RAM.

    If "tensor_cache" key is missing from data dict (e.g., raw inference
    input), passes through unchanged so downstream common transforms can
    handle NIfTI loading.
    """

    def __init__(self, keys: tuple[str, ...] = ("image", "label")):
        super().__init__(keys)
        self._common_transforms: Any = None

    def set_fallback(self, common_transforms: Any) -> None:
        """Set common transforms to use when tensor_cache is not available."""
        self._common_transforms = common_transforms

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        tensor_path = d.get("tensor_cache")

        if tensor_path:
            cached = torch.load(tensor_path, weights_only=False, mmap=True)
            for key in self.keys:
                d[key] = cached[key]
            return d

        # No tensor cache — apply common transforms (NIfTI loading path)
        if self._common_transforms is not None:
            return self._common_transforms(d)

        raise KeyError(
            "Data dict has no 'tensor_cache' key and no fallback transforms set. "
            "Run 'nnBench.plan' to create tensor cache, or set fallback transforms."
        )
