"""Learning Rate Scheduler for polynomial and linear decay."""

from torch.optim.lr_scheduler import LRScheduler


class PolyLRScheduler(LRScheduler):
    """
    Learning rate scheduler with polynomial or linear decay.

    Supports two decay modes:
    - Polynomial: lr = initial_lr * (1 - epoch / max_epochs) ^ exponent
    - Linear: lr = initial_lr - (epoch * decay_rate)

    This implementation matches nnU-Net v2.4.1's PolyLRScheduler with linear mode support.
    Reference: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/lr_scheduler/polylr.py

    Args:
        optimizer: PyTorch optimizer
        initial_lr: Initial learning rate
        max_epochs: Maximum number of epochs for training
        exponent: Polynomial exponent (default: 0.9). Ignored if mode='linear'.
        mode: Decay mode - 'polynomial' or 'linear' (default: 'linear')
        decay_rate: Learning rate reduction per epoch in linear mode (default: 0.00001)
    """

    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_epochs: int,
        exponent: float = 0.9,
        mode: str = "linear",
        decay_rate: float = 0.00001,
    ):
        """
        Initialize PolyLRScheduler.

        Args:
            optimizer: PyTorch optimizer
            initial_lr: Initial learning rate
            max_epochs: Total number of epochs
            exponent: Polynomial exponent (default: 0.9). Only used in polynomial mode.
            mode: 'polynomial' or 'linear' (default: 'linear')
            decay_rate: Learning rate reduction per epoch in linear mode (default: 0.00001)
        """
        if mode not in ("polynomial", "linear"):
            raise ValueError(f"mode must be 'polynomial' or 'linear', got {mode}")

        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.exponent = exponent
        self.mode = mode
        self.decay_rate = decay_rate
        self.current_epoch = 0

        super().__init__(optimizer, last_epoch=-1)

    def step(self, epoch: int | None = None) -> None:
        """
        Step the learning rate scheduler.

        Args:
            epoch: Current epoch number (optional, will auto-increment if not provided)
        """
        if epoch is None:
            epoch = self.current_epoch
            self.current_epoch += 1

        # Calculate new learning rate based on mode
        if self.mode == "polynomial":
            # Polynomial decay: lr = initial_lr * (1 - epoch / max_epochs) ^ exponent
            new_lr = self.initial_lr * max(0, 1 - epoch / self.max_epochs) ** self.exponent
        else:
            # Linear decay: lr = initial_lr - (epoch * decay_rate)
            new_lr = max(0.0, self.initial_lr - (epoch * self.decay_rate))

        # Update learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        # Store for get_last_lr()
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self) -> list[float]:
        """Return last computed learning rates."""
        return self._last_lr
