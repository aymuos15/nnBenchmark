"""DynUNet with native-resolution deep supervision outputs.

MONAI's DynUNet upsamples all DS heads to the final output resolution.
nnU-Net computes DS loss at each decoder's native resolution.
This subclass returns DS heads at native resolution to match nnU-Net.
"""

import torch
from monai.networks.nets import DynUNet


class NativeDSDynUNet(DynUNet):
    """DynUNet that returns deep supervision outputs at native resolution."""

    def forward(self, x):
        out = self.skip_layers(x)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            # Return heads at native resolution (no upsampling)
            return [out, *self.heads]
        return out
