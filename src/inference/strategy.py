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
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """
        Perform inference on input data.

        Args:
            model: PyTorch model to use for inference
            inputs: Input tensor to process
            device: Device to run inference on (cuda or cpu)
            use_amp: Whether to use automatic mixed precision (FP16)

        Returns:
            Output from model. Can be:
            - torch.Tensor: Single output tensor
            - tuple[torch.Tensor, ...]: Multiple output tensors
            - dict[str, torch.Tensor]: Named output tensors
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
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """
        Perform direct full-volume inference.

        Args:
            model: PyTorch model to use for inference
            inputs: Input tensor of shape [B, C, H, W(, D)]
            device: Device to run inference on
            use_amp: Whether to use automatic mixed precision

        Returns:
            Output from model with same spatial shape as input
        """
        with torch.no_grad():
            if use_amp:
                with autocast(device.type):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

        return outputs  # type: ignore[return-value]


class SlidingWindowInferer(InferenceStrategy):
    """
    Sliding window inference strategy using MONAI's SlidingWindowInferer.

    Processes large volumes in overlapping patches and blends results
    to produce seamless output matching input shape.
    """

    roi_size: tuple[int, ...] | list[int]
    sw_batch_size: int
    overlap: float
    mode: str
    padding_mode: str
    inferer: MONAISlidingWindowInferer

    def __init__(
        self,
        roi_size: tuple[int, ...] | list[int],
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        mode: str = "gaussian",
        padding_mode: str = "constant",
    ) -> None:
        """
        Initialize SlidingWindowInferer.

        Args:
            roi_size: Region of interest size for each sliding window patch
            sw_batch_size: Number of sliding window patches to process in each batch
            overlap: Overlap between patches as fraction (0-1)
            mode: Blending mode for overlapping regions ("gaussian" or "constant")
            padding_mode: Padding mode for edges ("constant", "edge", etc.)
        """
        # Convert list to tuple if needed
        self.roi_size = tuple(roi_size) if isinstance(roi_size, list) else roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.padding_mode = padding_mode

        # Initialize MONAI's SlidingWindowInferer
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
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """
        Perform sliding window inference on input.

        Processes input volume using overlapping patches. Output shape matches input shape.

        Args:
            model: PyTorch model to use for inference
            inputs: Input tensor of shape [B, C, H, W(, D)]
            device: Device to run inference on (cuda or cpu)
            use_amp: Whether to use automatic mixed precision (FP16)

        Returns:
            Output from model with same spatial shape as input. Can be:
            - torch.Tensor: [B, num_classes, H, W(, D)]
            - tuple[torch.Tensor, ...]: Multiple outputs from model
            - dict[str, torch.Tensor]: Named outputs from model

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

        return outputs


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
