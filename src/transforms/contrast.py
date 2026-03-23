"""
Mean-centered multiplicative contrast transform.

Matches nnU-Net's ContrastTransform from batchgeneratorsv2:
    img = (img - mean) * factor + mean

This is NOT gamma correction. It scales deviations from the mean,
preserving the mean intensity while adjusting contrast.

Reference:
    batchgeneratorsv2/transforms/intensity/contrast.py
"""


from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform
from monai.utils import convert_to_tensor
from monai.utils.misc import ensure_tuple


class ContrastTransform:
    """
    Apply mean-centered multiplicative contrast adjustment.

    Formula: output = (input - mean) * factor + mean

    Args:
        factor: Multiplicative contrast factor.
            >1 increases contrast, <1 decreases contrast.
        preserve_range: If True, clamp output to input min/max range.
    """

    def __init__(self, factor: float = 1.0, preserve_range: bool = True) -> None:
        self.factor = factor
        self.preserve_range = preserve_range

    def __call__(self, img: torch.Tensor, factor: float | None = None) -> torch.Tensor:
        img = convert_to_tensor(img)
        factor = factor if factor is not None else self.factor
        mean = img.mean()

        if self.preserve_range:
            minm = img.min()
            maxm = img.max()

        img = (img - mean) * factor + mean

        if self.preserve_range:
            img = torch.clamp(img, minm, maxm)

        return img


class RandContrastd(RandomizableTransform, MapTransform):
    """
    Dictionary-based random mean-centered multiplicative contrast.

    Matches nnU-Net's ContrastTransform: (x - mean) * factor + mean

    Args:
        keys: Keys of the data dictionary to transform.
        contrast_range: Range (min, max) to sample factor from.
        preserve_range: If True, clamp output to input min/max range.
        prob: Probability of applying the transform.
        allow_missing_keys: If True, don't raise error for missing keys.
    """

    def __init__(
        self,
        keys: KeysCollection,
        contrast_range: tuple[float, float] = (0.75, 1.25),
        preserve_range: bool = True,
        prob: float = 0.15,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.contrast_range = (min(contrast_range), max(contrast_range))
        self.adjuster = ContrastTransform(factor=1.0, preserve_range=preserve_range)
        self.factor_value: float = 1.0

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> "RandContrastd":
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return
        self.factor_value = self.R.uniform(
            low=self.contrast_range[0], high=self.contrast_range[1]
        )

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d
        for key in self.key_iterator(d):
            d[key] = self.adjuster(d[key], factor=self.factor_value)
        return d
