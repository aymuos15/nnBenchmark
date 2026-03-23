# Weight Init Option for MONAI DynUNet

## Problem

DynUNet uses `kaiming_normal_(a=0.01)` for weight init. nnU-Net's PlainConvUNet uses PyTorch default `kaiming_uniform_(a=sqrt(5))`. This init difference causes variance in early training dynamics.

## Current workaround

`NativeDSDynUNet` re-applies `kaiming_uniform_(a=sqrt(5))` after `DynUNet.__init__`.

## Proposed MONAI contribution

Add an `init_mode` parameter to DynUNet that selects between `kaiming_normal` (current default) and `kaiming_uniform` (PyTorch default / nnU-Net compatible).
