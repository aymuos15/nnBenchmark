# Native-Resolution Deep Supervision for MONAI DynUNet

## Problem

MONAI's DynUNet with `deep_supervision=True` upsamples all decoder outputs to the final output resolution before returning them. nnU-Net computes deep supervision loss at each decoder's native resolution by downsampling labels instead.

Upsampling outputs introduces interpolation artifacts and weaker gradient signal to deeper layers. nnU-Net's native-resolution approach provides ~4% better Dice in controlled comparisons.

## Current workaround

`NativeDSDynUNet` subclass in `src/models/dynunet.py` overrides `forward()` to return DS heads at native resolution.

## Proposed MONAI contribution

Add a `native_ds` flag to `DynUNet` that skips the internal `interpolate()` call and returns heads at native resolution. This is a one-line change in `DynUNet.forward()`.
