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

## Automatic Parameter Filtering (v0.2.2+)

nnBenchmark automatically filters out deep supervision parameters for models that don't support them. This prevents initialization errors when you specify these parameters for incompatible models.

**Supported models** (support deep supervision):
- `DynUNet`
- `BasicUNetPlusPlus`

**Unsupported models** (deep supervision params automatically removed):
- `UNet` and all other MONAI models

**Parameters filtered**:
- `deep_supervision` - Whether to enable deep supervision
- `deep_supr_num` - Number of deep supervision outputs
- `ds_weights` - Loss weights for each supervision level (used by DeepSupervisionLossWrapper)

**Example:**

```yaml
# Configuration specifying deep supervision
model:
  type: UNet  # Does NOT support deep supervision
  in_channels: 1
  out_channels: 3
  deep_supervision: true  # Will be automatically removed
  deep_supr_num: 2        # Will be automatically removed
  ds_weights: [1.0, 0.5]  # Will be automatically removed
```

When this config is loaded, the filtering logic in `src/engines/common.py:157-159` removes the unsupported parameters before model instantiation, allowing the model to initialize without error.

**When used with supported models:**

```yaml
model:
  type: DynUNet  # Supports deep supervision
  in_channels: 1
  out_channels: 3
  deep_supervision: true  # Kept and used
  deep_supr_num: 2        # Kept and used
  ds_weights: [1.0, 0.5]  # Kept and used by loss wrapper
```

The parameters are preserved and passed to the model, enabling deep supervision training.
