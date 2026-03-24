# DS Weight Normalization to Match nnU-Net

## Problem

nnBenchmark uses raw deep supervision weights `[1.0, 0.5, 0.25]` (sum=1.75) while nnU-Net normalizes them to sum to 1 and drops the lowest resolution output (weight=0).

This causes two issues:
1. Effective loss is 1.75x larger, making the optimizer overshoot relative to the LR schedule
2. Gradient signal is wasted on the lowest-resolution DS output that nnU-Net ignores

## nnU-Net's approach

```python
weights = np.array([1 / (2 ** i) for i in range(num_ds_levels)])  # [1, 0.5, 0.25]
weights[-1] = 0  # drop lowest resolution
weights = weights / weights.sum()  # normalize to sum=1 → [0.667, 0.333, 0]
```

Reference: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`, `_build_loss()` method.

## Current workaround

`equalize.py` in the compare repo post-processes the generated config to set `ds_weights[-1] = 0` and normalize.

## Proposed fix

The nnBenchmark planner should generate normalized DS weights matching nnU-Net's scheme by default: drop the lowest resolution output and normalize remaining weights to sum to 1.
