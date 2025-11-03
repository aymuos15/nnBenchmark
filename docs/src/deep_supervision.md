# Deep Supervision

When enabled, models output extra intermediate predictions during training. Output tensor shape depends on training mode and spatial dimensions.

## Output Shapes

| Configuration | Shape | Dims |
|---------------|-------|------|
| 2D, no DS | `[B, C, H, W]` | 4D |
| 3D, no DS | `[B, C, D, H, W]` | 5D |
| 2D, with DS | `[B, num_outputs, C, H, W]` | 5D |
| 3D, with DS | `[B, num_outputs, C, D, H, W]` | 6D |

**Note**: Deep supervision outputs only appear in training mode.

## Detecting Deep Supervision

```python
expected_ds_ndim = 3 + spatial_dims
is_deep_supervision = outputs.ndim == expected_ds_ndim
```

For 2D: `3 + 2 = 5D` | For 3D: `3 + 3 = 6D`

## Extracting Final Output

```python
if outputs.ndim == expected_ds_ndim:
    final_output = outputs[:, 0, ...]  # First output = final prediction
```

Result: `[B, C, spatial...]` ready for metric computation.