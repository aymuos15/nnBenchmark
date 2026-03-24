# Foreground Oversampling Ratio Mismatch with nnU-Net

## Problem

nnBenchmark's planner generates `RandCropByPosNegLabeld` with `pos=1, neg=2`, giving a 33% foreground sampling probability. However, nnU-Net's effective foreground ratio depends on batch size due to its deterministic oversampling strategy, and is often higher than 33%.

## nnU-Net's approach

nnU-Net uses `oversample_foreground_percent=0.33` with a **deterministic per-batch** strategy:

```python
def _oversample_last_XX_percent(self, sample_idx):
    return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
```

With `batch_size=2`: `round(2 * 0.67) = 1`, so sample index 0 is random, index 1 is **always** foreground → **50% effective foreground rate**.

With `batch_size=9`: `round(9 * 0.67) = 6`, so 3 out of 9 are foreground → **33% effective foreground rate**.

The effective rate converges to `oversample_foreground_percent` only at large batch sizes. At small batch sizes, rounding pushes it significantly higher.

Reference: `nnunetv2/training/dataloading/data_loader.py`, `_oversample_last_XX_percent()`.

## Current workaround

`equalize.py` in the compare repo adjusts `pos` and `neg` in `RandCropByPosNegLabeld` to match nnU-Net's effective ratio for the given batch size.

## Proposed fix

The nnBenchmark planner should compute the effective foreground ratio using the same rounding logic as nnU-Net, based on the planned batch size, and set `pos`/`neg` accordingly.
