"""DiceCE loss matching nnU-Net's DC_and_CE_loss formulation.

Wraps nnU-Net's DC_and_CE_loss with MemoryEfficientSoftDiceLoss to provide
exact gradient parity. Compatible with _target_ instantiation.
"""

from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class NnUNetDiceCELoss(nn.Module):
    """DiceCE loss using nnU-Net's exact formulation.

    Args:
        batch_dice: Whether to compute dice over the entire batch (True) or
            per-sample then average (False). Should match nnU-Net's plan.
        smooth: Smoothing term for dice computation.
        do_bg: Whether to include background in dice computation.
        softmax: Ignored (kept for config compatibility with MONAI DiceCELoss).
        to_onehot_y: Ignored (nnU-Net handles label encoding internally).
    """

    def __init__(
        self,
        batch_dice: bool = False,
        smooth: float = 1e-5,
        do_bg: bool = False,
        softmax: bool = True,
        to_onehot_y: bool = True,
    ):
        super().__init__()
        self.loss = DC_and_CE_loss(
            soft_dice_kwargs={
                "batch_dice": batch_dice,
                "smooth": smooth,
                "do_bg": do_bg,
                "ddp": False,
            },
            ce_kwargs={},
            weight_ce=1,
            weight_dice=1,
            ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

    def forward(self, input, target):
        return self.loss(input, target)
