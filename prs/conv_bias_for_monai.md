# Conv Bias Option for MONAI DynUNet

## Problem

MONAI's `Convolution` block disables conv bias when using InstanceNorm (`bias=False`). nnU-Net's PlainConvUNet uses `conv_bias=True` regardless of normalization. This creates a parameter count mismatch (2,214 params for Hippocampus config) and affects training dynamics.

## nnU-Net's approach

PlainConvUNet always uses `conv_bias=True` for all convolutions (encoder, decoder, transposed convs, and seg head convs). The bias terms participate in training even with InstanceNorm.

## Current workaround

`NativeDSDynUNet._add_bias()` walks all conv layers after `DynUNet.__init__` and adds `nn.Parameter(torch.zeros(out_channels))` to any conv that has `bias=None`. This covers encoder, decoder, transposed conv, and deep supervision head convs.

## Proposed MONAI contribution

Add a `conv_bias` parameter to DynUNet that propagates to all internal `Convolution` blocks, overriding the default behavior of disabling bias with normalization. This matches nnU-Net's behavior and eliminates the parameter count difference.
