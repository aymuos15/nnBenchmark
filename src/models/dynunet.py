"""DynUNet with native-resolution deep supervision outputs and nnU-Net weight init.

MONAI's DynUNet upsamples all DS heads to the final output resolution.
nnU-Net computes DS loss at each decoder's native resolution.
This subclass returns DS heads at native resolution to match nnU-Net.

MONAI's DynUNet uses kaiming_normal_(a=0.01).
nnU-Net's PlainConvUNet uses PyTorch default kaiming_uniform_(a=sqrt(5)).
This subclass re-initializes weights to match nnU-Net.
"""

import math

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet


class NativeDSDynUNet(DynUNet):
    """DynUNet with native-resolution DS outputs and nnU-Net-style weight init."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Re-initialize weights to match PyTorch default (same as nnU-Net's PlainConvUNet)
        self.apply(self._nnunet_init)

    @staticmethod
    def _nnunet_init(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x):
        out = self.skip_layers(x)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            return [out, *self.heads]
        return out
