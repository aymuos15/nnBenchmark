# Deep Supervision Loss Wrapper for MONAI

## Problem

MONAI's DynUNet with `deep_supervision=True` upsamples all decoder outputs to the final resolution, so the standard loss function works as-is. But with native-resolution deep supervision (see `native_ds_for_monai.md`), decoder outputs have different spatial sizes. There is no MONAI loss wrapper that downsamples labels to match each decoder level and computes a weighted sum.

## nnU-Net's approach

nnU-Net computes loss at each decoder's native resolution by downsampling labels with nearest interpolation, then sums with per-level weights (e.g., `[1.0, 0.5, 0.25, 0.125]`).

```
for output, weight in zip(decoder_outputs, ds_weights):
    labels_down = F.interpolate(labels, size=output.shape[2:], mode="nearest")
    total_loss += weight * loss_fn(output, labels_down)
```

## Current workaround

`DeepSupervisionLossWrapper` in `src/engines/ignite_utils/trainer.py` wraps any loss function, handles both tensor (stacked) and list DS output formats, and downsamples labels per level.

## Proposed MONAI contribution

Add a `DeepSupervisionLoss` wrapper to `monai.losses` that:
- Takes any base loss function + per-level weights
- Downsamples labels to match each decoder output's spatial size
- Works with both list and stacked-tensor DS output formats

This is the natural companion to adding `native_ds` support in DynUNet.
