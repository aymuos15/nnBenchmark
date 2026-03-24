# Dice Loss Formulation Mismatch with nnU-Net

## Problem

MONAI's `DiceCELoss` produces ~2x larger loss values than nnU-Net's `DC_and_CE_loss` on the same input, causing nnBenchmark to train with effectively double the gradient magnitude.

Key differences:
1. **Dice formula**: nnU-Net's `MemoryEfficientSoftDiceLoss` computes `-2 * (intersection + smooth) / (sum_pred + sum_label + smooth)`, returning negative values. MONAI's `DiceLoss` computes `1 - dice_score`, always positive (0-1 range).
2. **Background handling**: nnU-Net uses `do_bg=False` to exclude background from dice. MONAI's `DiceCELoss` includes background by default.
3. **Net effect**: MONAI total loss is ~1.5-2x higher → proportionally larger gradients → equivalent to training with a higher effective learning rate.

## nnU-Net's approach

```python
# DC_and_CE_loss with MemoryEfficientSoftDiceLoss
loss = DC_and_CE_loss(
    {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False},
    {}, weight_ce=1, weight_dice=1,
    dice_class=MemoryEfficientSoftDiceLoss
)
```

Reference: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`, `_build_loss()` method.

## Proposed fix

Replace MONAI's `DiceCELoss` with nnU-Net's `DC_and_CE_loss` in the nnBenchmark training engine for exact gradient parity. The loss function should be configurable via the YAML config with nnU-Net's formulation as the default.
