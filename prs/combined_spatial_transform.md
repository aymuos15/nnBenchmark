# Combined Spatial Transform to Avoid Double Interpolation

## Problem

nnBenchmark uses separate `RandRotated` + `RandZoomd` MONAI transforms, which apply two sequential `grid_sample` interpolation passes. nnU-Net's `SpatialTransform` combines rotation and scaling into a single affine matrix and applies one `grid_sample` call.

Double interpolation degrades image quality — each resampling introduces interpolation error and blurring. This is especially impactful for small structures like Hippocampus subregions.

## nnU-Net's approach

```python
# SpatialTransform combines rotation + scaling:
affine = create_affine_matrix_3d(angles, scales)  # single 3x3 matrix
grid = torch.matmul(identity_grid, affine)          # one grid transform
result = grid_sample(img, grid, mode='bilinear')     # ONE interpolation
```

Reference: `batchgeneratorsv2/transforms/spatial/spatial.py`

## Current fix

Added `RandAffined` transform in `src/transforms/spatial.py` that:
- Builds a combined 3x3 affine matrix (Rz @ Ry @ Rx @ S) matching nnU-Net's order
- Applies rotation and scaling in a single `grid_sample` call
- Supports per-key interpolation modes (bilinear for images, nearest for labels)

## Proposed change

The nnBenchmark planner should generate configs using `RandAffined` instead of separate `RandRotated` + `RandZoomd` by default.
