"""DynUNet with native-resolution deep supervision outputs and nnU-Net weight init.

MONAI's DynUNet upsamples all DS heads to the final output resolution.
nnU-Net computes DS loss at each decoder's native resolution.
This subclass returns DS heads at native resolution to match nnU-Net.

MONAI's DynUNet uses kaiming_normal_(a=0.01).
nnU-Net's PlainConvUNet uses PyTorch default kaiming_uniform_(a=sqrt(5)).
This subclass re-initializes weights to match nnU-Net.

MONAI disables conv bias when using InstanceNorm; nnU-Net keeps it.
This subclass adds bias to all conv layers to match nnU-Net's param count.
"""

import math

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet


class NativeDSDynUNet(DynUNet):
    """DynUNet with native-resolution DS outputs and nnU-Net-style weight init."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MONAI disables conv bias when using InstanceNorm; nnU-Net keeps it.
        # Add bias to all conv layers that lack it, then re-init everything.
        self.apply(self._add_bias)
        self.apply(self._nnunet_init)

    @staticmethod
    def _add_bias(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            if module.bias is None:
                module.bias = nn.Parameter(torch.zeros(module.out_channels))

    @staticmethod
    def _nnunet_init(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x):
        out = self.skip_layers(x)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            # self.heads are populated by DynUNetSkipLayer via super_head
            # (already projected to out_channels). Return at native resolution
            # instead of upsampling to match nnU-Net's DS loss computation.
            return [out, *self.heads]
        return out
