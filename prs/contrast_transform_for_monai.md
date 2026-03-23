# ContrastTransform for MONAI

## Problem

MONAI lacks a mean-centered multiplicative contrast transform matching nnU-Net's `ContrastTransform`.

MONAI's `RandAdjustContrastd` does gamma correction (`x^gamma`), which is a different operation.

## nnU-Net's ContrastTransform

```
output = (input - mean) * factor + mean
if preserve_range: clamp(output, input_min, input_max)
```

- Factor range: (0.75, 1.25)
- `factor > 1` increases contrast, `factor < 1` decreases contrast
- Preserves mean intensity
- Reference: `batchgeneratorsv2/transforms/intensity/contrast.py`

## Current workaround

Custom `RandContrastd` in `src/transforms/contrast.py` following MONAI's `RandomizableTransform + MapTransform` pattern.

## Proposed MONAI contribution

Add `ContrastTransform` / `RandContrastd` to `monai.transforms.intensity` as a complement to `AdjustContrast` (gamma).
