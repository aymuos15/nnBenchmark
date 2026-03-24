"""Combined rotation+scaling transform with single interpolation pass.

MONAI's RandRotated + RandZoomd apply two separate grid_sample calls,
degrading image quality through double interpolation. nnU-Net's
SpatialTransform combines rotation and scaling into a single affine
matrix and applies one grid_sample. This transform does the same.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.transforms import MapTransform, RandomizableTransform
from monai.utils import ensure_tuple


class RandAffined(RandomizableTransform, MapTransform):
    """Combined random rotation + scaling in a single interpolation pass.

    Matches nnU-Net's SpatialTransform behavior: builds a combined affine
    matrix for rotation and scaling, then applies a single grid_sample.

    Args:
        keys: Keys of data dict to transform.
        prob_rotate: Probability of rotation.
        prob_scale: Probability of scaling.
        rotate_range: Rotation range in radians for each axis (x, y, z).
        scale_range: (min_scale, max_scale) tuple.
        mode: Interpolation mode per key ('bilinear' or 'nearest').
        padding_mode: Padding mode for grid_sample ('zeros', 'border', 'reflection').
    """

    def __init__(
        self,
        keys: tuple[str, ...],
        prob_rotate: float = 0.2,
        prob_scale: float = 0.2,
        rotate_range: tuple[float, float, float] = (0.5236, 0.5236, 0.5236),
        scale_range: tuple[float, float] = (0.7, 1.4),
        mode: tuple[str, ...] = ("bilinear", "nearest"),
        padding_mode: str = "border",
    ):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=1.0)
        self.prob_rotate = prob_rotate
        self.prob_scale = prob_scale
        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.mode = ensure_tuple(mode)
        self.padding_mode = padding_mode
        self._do_rotate = False
        self._do_scale = False
        self._angles: list[float] = [0.0, 0.0, 0.0]
        self._scales: list[float] = [1.0, 1.0, 1.0]

    def randomize(self, data=None):
        self._do_rotate = self.R.random() < self.prob_rotate
        self._do_scale = self.R.random() < self.prob_scale

        if self._do_rotate:
            self._angles = [
                self.R.uniform(-r, r) for r in self.rotate_range
            ]
        else:
            self._angles = [0.0, 0.0, 0.0]

        if self._do_scale:
            # nnU-Net synchronizes scaling across axes by default
            s = self.R.uniform(self.scale_range[0], self.scale_range[1])
            self._scales = [s, s, s]
        else:
            self._scales = [1.0, 1.0, 1.0]

    def __call__(self, data):
        d = dict(data)
        self.randomize()

        if not self._do_rotate and not self._do_scale:
            return d

        affine = _create_affine_3d(self._angles, self._scales)

        for idx, key in enumerate(self.keys):
            mode = self.mode[idx] if idx < len(self.mode) else self.mode[-1]
            d[key] = _apply_affine(d[key], affine, mode, self.padding_mode)

        return d


def _create_affine_3d(angles: list[float], scales: list[float]) -> torch.Tensor:
    """Create a 3x3 affine matrix combining rotation (x,y,z) and scaling.

    Matches nnU-Net's create_affine_matrix_3d: Rz @ Ry @ Rx @ S
    """
    ax, ay, az = angles
    sx, sy, sz = scales

    # Rotation matrices
    cos_x, sin_x = np.cos(ax), np.sin(ax)
    cos_y, sin_y = np.cos(ay), np.sin(ay)
    cos_z, sin_z = np.cos(az), np.sin(az)

    Rx = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x],
    ])
    Ry = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y],
    ])
    Rz = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1],
    ])

    S = np.diag([1.0 / sx, 1.0 / sy, 1.0 / sz])

    # nnU-Net order: Rz @ Ry @ Rx @ S
    affine = Rz @ Ry @ Rx @ S
    return torch.from_numpy(affine).float()


def _apply_affine(
    img: torch.Tensor,
    affine: torch.Tensor,
    mode: str,
    padding_mode: str,
) -> torch.Tensor:
    """Apply affine transform via grid_sample with a single interpolation."""
    spatial_shape = img.shape[1:]  # (C, D, H, W)
    ndim = len(spatial_shape)

    # Create identity grid centered at origin
    ranges = [torch.linspace(-1, 1, s) for s in spatial_shape]
    grid = torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1)  # (D, H, W, 3)

    # Apply affine: rotate grid coordinates
    grid = torch.matmul(grid, affine.T)

    # grid_sample expects grid in (N, D, H, W, 3) with values in [-1, 1]
    # and axis order (x, y, z) = (W, H, D), so we need to flip
    grid = grid.flip(-1)

    result = torch.nn.functional.grid_sample(
        img.unsqueeze(0).float(),
        grid.unsqueeze(0),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
    return result[0].to(img.dtype)
