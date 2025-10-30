# Deep Supervision Output Dimensions

## Problem

DynUNet with `deep_supervision=True` outputs different tensor shapes depending on:
1. **Training vs. Eval mode**: DS outputs only appear in training mode
2. **Spatial dimensions**: 2D vs. 3D models have different base dimensions

## Tensor Shapes

### Without Deep Supervision
```
2D: [B, C, H, W]       → 4D
3D: [B, C, D, H, W]    → 5D
```

### With Deep Supervision (Training Mode Only)
```
2D: [B, num_outputs, C, H, W]       → 5D
3D: [B, num_outputs, C, D, H, W]    → 6D
```

## Detection Formula

To detect if a tensor is in deep supervision format:

```python
expected_ds_ndim = 3 + spatial_dims

# 2D: 3 + 2 = 5D
# 3D: 3 + 3 = 6D
```

## Extraction

When deep supervision is detected, extract the final output:

```python
if outputs.ndim == expected_ds_ndim:
    final_output = outputs[:, 0, ...]  # First output = final prediction
```

This gives you `[B, C, spatial...]` for metric computation.

## Critical Note

**Never use `len(outputs.shape) in (5, 6)` for detection!**

This incorrectly matches 3D outputs without deep supervision (5D), causing:
- Wrong dimension extraction
- argmax on spatial dims instead of class dim
- CUDA assert errors from invalid class indices
