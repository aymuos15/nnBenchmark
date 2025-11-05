"""KiU-Net: Kite-Net U-Net architecture for medical image segmentation.

This module implements KiU-Net architecture following the original paper's design.
KiU-Net uses dual encoder-decoder branches (U-Net and Ki-Net) with inline
Cross-Resolution Fusion Blocks (CRFB) for multi-scale feature interaction.

Reference:
    Valanarasu et al. "KiU-Net: Overcomplete Convolutional Architectures for
    Biomedical Image and Volumetric Segmentation." IEEE TMI, 2021.
    https://arxiv.org/abs/2010.01663

Key features:
    - Dual-branch architecture (U-Net: under-complete, Ki-Net: over-complete)
    - Inline CRFB fusion at encoder and decoder levels
    - Configurable normalization and activation (defaults: InstanceNorm + LeakyReLU)
    - Deep supervision support for improved gradient flow
    - 2D and 3D support with unified interface
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution

__all__ = ["KiUNet", "KiUNet2D", "KiUNet3D"]


class KiUNet(nn.Module):
    """KiU-Net: Dual-branch segmentation network with cross-resolution fusion.

    Implements both over-complete (Ki-Net) and under-complete (U-Net) branches
    with inline cross-resolution fusion at each encoder/decoder level.

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3)
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        features: Sequence of feature channel counts at each encoder level
        norm_name: Type of normalization ('batch', 'instance', 'group')
        act_name: Type of activation ('relu', 'leakyrelu', 'prelu')
        deep_supervision: If True, return auxiliary outputs for deep supervision
        deep_supr_num: Number of deep supervision outputs (from coarsest levels)

    Example:
        >>> # 2D model with 3 encoder levels
        >>> model = KiUNet(
        ...     spatial_dims=2,
        ...     in_channels=1,
        ...     out_channels=2,
        ...     features=[16, 32, 64],
        ...     norm_name="instance",
        ...     act_name="leakyrelu",
        ...     deep_supervision=True,
        ... )
        >>> x = torch.randn(2, 1, 256, 256)
        >>> outputs = model(x)  # Returns (main, [aux1, aux2, ...])
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        features: Sequence[int] = (16, 32, 64),
        norm_name: str = "instance",
        act_name: str = "leakyrelu",
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
    ) -> None:
        super().__init__()

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = list(features)
        self.num_levels = len(features)
        self.norm_name = norm_name
        self.act_name = act_name
        self.deep_supervision = deep_supervision
        self.deep_supr_num = deep_supr_num

        # Select pooling type based on spatial dimensions
        self.pool_type = nn.MaxPool2d if spatial_dims == 2 else nn.MaxPool3d
        conv_type = nn.Conv2d if spatial_dims == 2 else nn.Conv3d

        # ============================================================
        # U-Net Branch (under-complete): encoder with max pooling
        # ============================================================
        self.unet_encoders = nn.ModuleList()

        for i, feat in enumerate(features):
            in_ch = in_channels if i == 0 else features[i - 1]
            self.unet_encoders.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=feat,
                    kernel_size=3,
                    norm=norm_name,
                    act=act_name,
                )
            )

        # ============================================================
        # Ki-Net Branch (over-complete): encoder with upsampling
        # ============================================================
        self.kinet_encoders = nn.ModuleList()

        for i, feat in enumerate(features):
            in_ch = in_channels if i == 0 else features[i - 1]
            self.kinet_encoders.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=feat,
                    kernel_size=3,
                    norm=norm_name,
                    act=act_name,
                )
            )

        # ============================================================
        # CRFB Convolutions for Encoder (3 levels)
        # ============================================================
        # intere_i_1: Ki-Net → U-Net
        # intere_i_2: U-Net → Ki-Net

        self.intere1_1 = Convolution(spatial_dims, features[0], features[0], 3, norm=norm_name, act=None)
        self.intere1_2 = Convolution(spatial_dims, features[0], features[0], 3, norm=norm_name, act=None)

        self.intere2_1 = Convolution(spatial_dims, features[1], features[1], 3, norm=norm_name, act=None)
        self.intere2_2 = Convolution(spatial_dims, features[1], features[1], 3, norm=norm_name, act=None)

        # Level 3: Channel adaptation (Ki-Net and U-Net have different channels)
        # If features = [16, 32, 64], then level 3 has U-Net=64, Ki-Net=32
        # But in 3-level case, they're the same. In 4-level+ case, they differ.
        # Following original paper: intere3_1 does channel expansion, intere3_2 does reduction
        if self.num_levels >= 3:
            kinet_ch_level3 = features[-2] if self.num_levels > 3 else features[-1]
            unet_ch_level3 = features[-1]
            self.intere3_1 = Convolution(spatial_dims, kinet_ch_level3, unet_ch_level3, 3, norm=norm_name, act=None)
            self.intere3_2 = Convolution(spatial_dims, unet_ch_level3, kinet_ch_level3, 3, norm=norm_name, act=None)

        # ============================================================
        # Decoders (both branches) - decode independently
        # ============================================================
        self.unet_decoders = nn.ModuleList()
        self.kinet_decoders = nn.ModuleList()

        # Reverse features for decoder (coarse to fine)
        decoder_features = features[::-1]

        for i in range(len(decoder_features) - 1):
            in_feat = decoder_features[i]
            out_feat = decoder_features[i + 1]

            # U-Net decoder: takes concatenated input (decoder_out + skip)
            self.unet_decoders.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_feat + out_feat,
                    out_channels=out_feat,
                    kernel_size=3,
                    norm=norm_name,
                    act=act_name,
                )
            )

            # Ki-Net decoder: takes concatenated input (decoder_out + skip)
            self.kinet_decoders.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_feat + out_feat,
                    out_channels=out_feat,
                    kernel_size=3,
                    norm=norm_name,
                    act=act_name,
                )
            )

        # ============================================================
        # CRFB Convolutions for Decoder (only first 2 levels have CRFB)
        # ============================================================
        # Following original paper: decoder levels 1 and 2 have CRFB, level 3 doesn't

        if self.num_levels >= 2:
            # Decoder level 1 CRFB (after first decoder stage)
            dec_ch_1 = decoder_features[1] if len(decoder_features) > 1 else decoder_features[0]
            self.interd1_1 = Convolution(spatial_dims, dec_ch_1, dec_ch_1, 3, norm=norm_name, act=None)
            self.interd1_2 = Convolution(spatial_dims, dec_ch_1, dec_ch_1, 3, norm=norm_name, act=None)

        if self.num_levels >= 3:
            # Decoder level 2 CRFB (after second decoder stage)
            dec_ch_2 = decoder_features[2] if len(decoder_features) > 2 else decoder_features[1]
            self.interd2_1 = Convolution(spatial_dims, dec_ch_2, dec_ch_2, 3, norm=norm_name, act=None)
            self.interd2_2 = Convolution(spatial_dims, dec_ch_2, dec_ch_2, 3, norm=norm_name, act=None)

        # ============================================================
        # Final layers to match output resolution
        # ============================================================
        final_feat = features[0]

        # Final convolutions before fusion
        self.unet_final = Convolution(
            spatial_dims=spatial_dims,
            in_channels=final_feat,
            out_channels=final_feat // 2,
            kernel_size=3,
            norm=norm_name,
            act=act_name,
        )

        self.kinet_final = Convolution(
            spatial_dims=spatial_dims,
            in_channels=final_feat,
            out_channels=final_feat // 2,
            kernel_size=3,
            norm=norm_name,
            act=act_name,
        )

        # Segmentation head (1x1 conv)
        self.seg_head = conv_type(
            in_channels=final_feat // 2,
            out_channels=out_channels,
            kernel_size=1,
        )

        # ============================================================
        # Deep supervision heads
        # ============================================================
        if deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            # Create heads for the coarsest deep_supr_num decoder levels
            for i in range(min(deep_supr_num, len(decoder_features) - 1)):
                ds_feat = decoder_features[i + 1]
                self.deep_supervision_heads.append(
                    conv_type(
                        in_channels=ds_feat,
                        out_channels=out_channels,
                        kernel_size=1,
                    )
                )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through KiU-Net with inline CRFB fusion.

        Args:
            x: Input tensor of shape (B, C, H, W) for 2D or (B, C, H, W, D) for 3D

        Returns:
            If deep_supervision=False:
                Output tensor of shape (B, out_channels, H, W) or (B, out_channels, H, W, D)
            If deep_supervision=True:
                Tuple of (main_output, [aux_output_1, aux_output_2, ...])
        """
        input_size = x.shape[2:]
        mode = "bilinear" if self.spatial_dims == 2 else "trilinear"

        # Scale factors for CRFB (from original paper)
        # Encoder: Ki-Net grows 2x at each level, U-Net shrinks 2x
        # Level 1: U-Net at H/2, Ki-Net at H*2 → ratio 1:4
        # Level 2: U-Net at H/4, Ki-Net at H*4 → ratio 1:16
        # Level 3: U-Net at H/8, Ki-Net at H*8 → ratio 1:64
        encoder_scale_down = [0.25, 0.0625, 0.015625]
        encoder_scale_up = [4, 16, 64]

        # Decoder: Reverse scaling
        decoder_scale_down = [0.0625, 0.25]  # Only 2 levels have CRFB
        decoder_scale_up = [16, 4]

        # ============================================================
        # Encoder: U-Net branch (pooling) + Ki-Net branch (upsampling)
        # with inline CRFB fusion
        # ============================================================
        unet_features = []
        kinet_features = []

        unet_x = x
        kinet_x = x

        for i in range(self.num_levels):
            # Apply convolutions (Convolution block already includes activation)
            unet_x = self.unet_encoders[i](unet_x)
            kinet_x = self.kinet_encoders[i](kinet_x)

            # Apply pooling/upsampling (NO extra activation - Convolution already did it)
            unet_x = self.pool_type(kernel_size=2, stride=2)(unet_x)
            kinet_x = F.interpolate(kinet_x, scale_factor=2, mode=mode, align_corners=True)

            # CRFB fusion (inline) - use F.relu on intere convs (they have act=None)
            if i == 0:
                tmp = unet_x
                unet_x = unet_x + F.interpolate(
                    F.relu(self.intere1_1(kinet_x)),
                    size=unet_x.shape[2:],
                    mode=mode,
                    align_corners=True
                )
                kinet_x = kinet_x + F.interpolate(
                    F.relu(self.intere1_2(tmp)),
                    size=kinet_x.shape[2:],
                    mode=mode,
                    align_corners=True
                )
            elif i == 1:
                tmp = unet_x
                unet_x = unet_x + F.interpolate(
                    F.relu(self.intere2_1(kinet_x)),
                    size=unet_x.shape[2:],
                    mode=mode,
                    align_corners=True
                )
                kinet_x = kinet_x + F.interpolate(
                    F.relu(self.intere2_2(tmp)),
                    size=kinet_x.shape[2:],
                    mode=mode,
                    align_corners=True
                )
            elif i == 2 and self.num_levels >= 3:
                tmp = unet_x
                unet_x = unet_x + F.interpolate(
                    F.relu(self.intere3_1(kinet_x)),
                    size=unet_x.shape[2:],
                    mode=mode,
                    align_corners=True
                )
                kinet_x = kinet_x + F.interpolate(
                    F.relu(self.intere3_2(tmp)),
                    size=kinet_x.shape[2:],
                    mode=mode,
                    align_corners=True
                )

            # Save features for skip connections
            unet_features.append(unet_x)
            kinet_features.append(kinet_x)

        # ============================================================
        # Decoder: Reverse order, apply skip connections and CRFB
        # ============================================================
        unet_x = unet_features[-1]
        kinet_x = kinet_features[-1]

        deep_supervision_outputs = []

        for i in range(len(self.unet_decoders)):
            # Get skip connection features (from encoder)
            skip_idx = self.num_levels - 2 - i
            unet_skip = unet_features[skip_idx]
            kinet_skip = kinet_features[skip_idx]

            # U-Net decoder: upsample then concatenate skip connection
            unet_x = F.interpolate(
                unet_x,
                size=unet_skip.shape[2:],
                mode=mode,
                align_corners=True,
            )
            unet_x = torch.cat([unet_x, unet_skip], dim=1)
            unet_x = self.unet_decoders[i](unet_x)

            # Ki-Net decoder: downsample (pool) then concatenate skip connection
            kinet_x = self.pool_type(kernel_size=2, stride=2)(kinet_x)
            kinet_x = torch.cat([kinet_x, kinet_skip], dim=1)
            kinet_x = self.kinet_decoders[i](kinet_x)

            # CRFB fusion (only at first 2 decoder levels)
            if i == 0 and self.num_levels >= 2:
                tmp = unet_x
                unet_x = unet_x + F.interpolate(
                    F.relu(self.interd1_1(kinet_x)),
                    size=unet_x.shape[2:],
                    mode=mode,
                    align_corners=True,
                )
                kinet_x = kinet_x + F.interpolate(
                    F.relu(self.interd1_2(tmp)),
                    size=kinet_x.shape[2:],
                    mode=mode,
                    align_corners=True,
                )
            elif i == 1 and self.num_levels >= 3:
                tmp = unet_x
                unet_x = unet_x + F.interpolate(
                    F.relu(self.interd2_1(kinet_x)),
                    size=unet_x.shape[2:],
                    mode=mode,
                    align_corners=True,
                )
                kinet_x = kinet_x + F.interpolate(
                    F.relu(self.interd2_2(tmp)),
                    size=kinet_x.shape[2:],
                    mode=mode,
                    align_corners=True,
                )

            # Deep supervision: capture outputs at intermediate decoder levels
            if self.deep_supervision and i < self.deep_supr_num:
                # Use U-Net branch for deep supervision (it has correct resolution)
                ds_out = self.deep_supervision_heads[i](unet_x)
                # Upsample to input resolution
                ds_out = F.interpolate(
                    ds_out,
                    size=input_size,
                    mode=mode,
                    align_corners=True,
                )
                deep_supervision_outputs.append(ds_out)

        # ============================================================
        # Final layers and fusion
        # ============================================================
        unet_x = self.unet_final(unet_x)
        kinet_x = self.kinet_final(kinet_x)

        # Ensure both branches are at input resolution before fusion
        unet_x = F.interpolate(
            unet_x, size=input_size, mode=mode, align_corners=True
        )
        kinet_x = F.interpolate(
            kinet_x, size=input_size, mode=mode, align_corners=True
        )

        # Fuse branches via addition (like original paper)
        fused = unet_x + kinet_x

        # Final segmentation
        output = self.seg_head(fused)

        if self.deep_supervision and len(deep_supervision_outputs) > 0:
            return output, deep_supervision_outputs
        else:
            return output


def KiUNet2D(
    in_channels: int,
    out_channels: int,
    features: Sequence[int] = (16, 32, 64),
    norm_name: Literal["batch", "instance", "group"] = "instance",
    act_name: Literal["relu", "leakyrelu", "prelu"] = "leakyrelu",
    deep_supervision: bool = False,
    deep_supr_num: int = 1,
) -> KiUNet:
    """Create a 2D KiU-Net model.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        features: Sequence of feature channel counts at each encoder level.
                 Default: (16, 32, 64) for 3-level encoder
        norm_name: Type of normalization ('batch', 'instance', 'group')
        act_name: Type of activation ('relu', 'leakyrelu', 'prelu')
        deep_supervision: If True, return auxiliary outputs for deep supervision
        deep_supr_num: Number of deep supervision outputs

    Returns:
        KiUNet model configured for 2D inputs

    Example:
        >>> model = KiUNet2D(in_channels=1, out_channels=2, features=[32, 64, 128])
        >>> x = torch.randn(2, 1, 256, 256)
        >>> y = model(x)
        >>> print(y.shape)  # (2, 2, 256, 256)
    """
    return KiUNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        norm_name=norm_name,
        act_name=act_name,
        deep_supervision=deep_supervision,
        deep_supr_num=deep_supr_num,
    )


def KiUNet3D(
    in_channels: int,
    out_channels: int,
    features: Sequence[int] = (16, 32, 64),
    norm_name: Literal["batch", "instance", "group"] = "instance",
    act_name: Literal["relu", "leakyrelu", "prelu"] = "leakyrelu",
    deep_supervision: bool = False,
    deep_supr_num: int = 1,
) -> KiUNet:
    """Create a 3D KiU-Net model.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
        features: Sequence of feature channel counts at each encoder level.
                 Default: (16, 32, 64) for 3-level encoder
        norm_name: Type of normalization ('batch', 'instance', 'group')
        act_name: Type of activation ('relu', 'leakyrelu', 'prelu')
        deep_supervision: If True, return auxiliary outputs for deep supervision
        deep_supr_num: Number of deep supervision outputs

    Returns:
        KiUNet model configured for 3D inputs

    Example:
        >>> model = KiUNet3D(in_channels=1, out_channels=2, features=[32, 64, 128])
        >>> x = torch.randn(2, 1, 64, 64, 64)
        >>> y = model(x)
        >>> print(y.shape)  # (2, 2, 64, 64, 64)
    """
    return KiUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        norm_name=norm_name,
        act_name=act_name,
        deep_supervision=deep_supervision,
        deep_supr_num=deep_supr_num,
    )
