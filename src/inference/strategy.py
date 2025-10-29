"""
Inference strategy classes for model prediction on test data.

Provides abstraction layer for different inference approaches:
- FullVolumeInferer: Direct full-volume inference (original approach)
- SlidingWindowInferer: Memory-efficient sliding window inference (for large volumes)

This strategy pattern allows easy addition of new inference types (e.g., TTA, ensemble)
in the future without modifying the evaluation loop.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from monai.inferers.inferer import SlidingWindowInferer as MONAISlidingWindowInferer
from torch.amp.autocast_mode import autocast


class InferenceStrategy(ABC):
    """
    Abstract base class for inference strategies.

    Defines the interface that all inference strategies must implement.
    """

    @abstractmethod
    def infer(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        device: torch.device,
        use_amp: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on input data.

        Args:
            model: PyTorch model to use for inference
            inputs: Input tensor to process
            device: Device to run inference on (cuda or cpu)
            use_amp: Whether to use automatic mixed precision (FP16)

        Returns:
            Output tensor from model
        """
        raise NotImplementedError


class FullVolumeInferer(InferenceStrategy):
    """
    Direct full-volume inference strategy.

    Processes entire volume in a single forward pass without any patching.
    This is the original approach in nnBenchmark.

    Pros:
    - Simple and straightforward
    - No boundary artifacts
    - Fast for volumes that fit in GPU memory

    Cons:
    - Requires entire volume to fit in GPU memory
    - Not suitable for large 3D medical imaging volumes
    """

    def infer(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        device: torch.device,
        use_amp: bool = False,
    ) -> torch.Tensor:
        """
        Perform direct full-volume inference.

        Args:
            model: PyTorch model to use for inference
            inputs: Input tensor of shape [B, C, H, W(, D)]
            device: Device to run inference on
            use_amp: Whether to use automatic mixed precision

        Returns:
            Output tensor from model with same spatial shape as input
        """
        with torch.no_grad():
            if use_amp:
                with autocast(device.type):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

        return outputs


class SlidingWindowInferer(InferenceStrategy):
    """
    Memory-efficient sliding window inference strategy.

    Processes large volumes using overlapping patches with blending.
    Implements MONAI's SlidingWindowInferer which is the industry standard
    for medical imaging inference (used by nnU-Net, etc.).

    Pros:
    - Memory efficient: processes volume in manageable chunks
    - Reduces boundary artifacts through overlapping and blending
    - Production-ready standard approach
    - Configurable overlap and blending modes

    Cons:
    - Slower than full-volume inference for small volumes
    - Requires proper parameter tuning for best results

    References:
        https://docs.monai.io/en/stable/inferers.html#slidingwindowinferrer
        https://github.com/NVIDIA/MONAI/blob/dev/monai/inferers/inferer.py
    """

    def __init__(
        self,
        roi_size: tuple[int, ...] | list[int],
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        mode: str = "gaussian",
        padding_mode: str = "constant",
    ):
        """
        Initialize sliding window inferer.

        Args:
            roi_size: Region of interest (patch) size as tuple or list.
                     Must match training patch size.
            sw_batch_size: Batch size for processing patches internally.
                          Higher values use more memory but may be faster.
                          Default: 4 (adjust based on GPU memory)
            overlap: Overlap ratio between patches (0.0 to 0.99).
                    Higher overlap produces better quality but is slower.
                    Default: 0.5 (50% overlap, standard in medical imaging)
            mode: Blending mode for overlapping regions.
                 Options: "gaussian" (recommended), "constant"
                 Default: "gaussian" (smooth blending, reduces artifacts)
            padding_mode: Padding mode for volume edges.
                         Options: "constant", "edge", "reflect", "wrap"
                         Default: "constant" (zero-padding, standard)

        """
        # Ensure roi_size is a tuple
        if isinstance(roi_size, list):
            roi_size = tuple(roi_size)

        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.padding_mode = padding_mode

        # Create MONAI inferer
        self.inferer = MONAISlidingWindowInferer(
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            overlap=self.overlap,
            mode=self.mode,
            padding_mode=self.padding_mode,
        )

    def infer(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        device: torch.device,
        use_amp: bool = False,
    ) -> torch.Tensor:
        """
        Perform sliding window inference on input.

        Processes input volume using overlapping patches. Output shape matches input shape.

        Args:
            model: PyTorch model to use for inference
            inputs: Input tensor of shape [B, C, H, W(, D)]
            device: Device to run inference on (cuda or cpu)
            use_amp: Whether to use automatic mixed precision (FP16)

        Returns:
            Output tensor from model with same spatial shape as input [B, num_classes, H, W(, D)]

        Note:
            The MONAI SlidingWindowInferer handles:
            - Generating sliding window positions with specified overlap
            - Processing patches in batches (sw_batch_size)
            - Blending overlapping regions using specified mode
            - Padding edges if needed
            - Returning full-resolution output matching input size
        """
        with torch.no_grad():
            if use_amp:
                # Create wrapper to use AMP during inference
                def amp_forward(x: torch.Tensor) -> torch.Tensor:
                    with autocast(device.type):
                        return model(x)

                outputs = self.inferer(inputs, amp_forward)
            else:
                outputs = self.inferer(inputs, model)

        return outputs  # type: ignore


def create_inferer(config: dict[str, Any]) -> InferenceStrategy:
    """
    Factory function to create appropriate inferer based on config.

    Args:
        config: Configuration dictionary with optional 'testing.sliding_window' section

    Returns:
        InferenceStrategy instance (FullVolumeInferer or SlidingWindowInferer)

    Raises:
        ValueError: If sliding_window config is invalid

    """
    testing_cfg = config.get("testing", {})
    sliding_window_cfg = testing_cfg.get("sliding_window", {})

    # Check if sliding window is enabled
    enabled = sliding_window_cfg.get("enabled", False)

    if not enabled:
        # Use default full-volume inference
        return FullVolumeInferer()

    # Extract sliding window parameters with defaults
    roi_size = sliding_window_cfg.get("roi_size", None)
    sw_batch_size = sliding_window_cfg.get("sw_batch_size", 4)
    overlap = sliding_window_cfg.get("overlap", 0.5)
    mode = sliding_window_cfg.get("mode", "gaussian")
    padding_mode = sliding_window_cfg.get("padding_mode", "constant")

    # If roi_size is not specified, use training patch size
    if roi_size is None:
        dataset_cfg = config.get("dataset", {})
        roi_size = dataset_cfg.get("spatial_size", None)

        if roi_size is None:
            raise ValueError(
                "sliding_window.roi_size is not specified and could not infer from "
                "dataset.spatial_size. Please specify roi_size in testing.sliding_window config "
                "or ensure dataset.spatial_size is defined."
            )

    # Validate roi_size
    if not isinstance(roi_size, (list, tuple)):
        raise ValueError(
            f"sliding_window.roi_size must be a list or tuple, got {type(roi_size).__name__}"
        )

    # Create and return sliding window inferer
    return SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode=mode,
        padding_mode=padding_mode,
    )
