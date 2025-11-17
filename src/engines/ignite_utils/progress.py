"""
Console progress handler for training visualization.
Provides real-time feedback similar to PyTorch Lightning's progress bar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ignite.engine import Engine, Events
from tqdm import tqdm

if TYPE_CHECKING:
    from loguru._logger import Logger


class ConsoleProgressHandler:
    """
    Displays training progress with tqdm progress bars.
    Shows epoch progress, loss, and learning rate.
    """

    def __init__(self, logger: Logger, max_epochs: int):
        """
        Args:
            logger: Logger instance
            max_epochs: Total number of epochs
        """
        self.logger = logger
        self.max_epochs = max_epochs
        self.epoch_pbar = None
        self.current_epoch = 0

    def attach(self, engine: Engine) -> None:
        """Attach handler to engine events."""
        engine.add_event_handler(Events.STARTED, self._on_started)
        engine.add_event_handler(Events.EPOCH_STARTED, self._on_epoch_started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._on_iteration)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._on_epoch_completed)
        engine.add_event_handler(Events.COMPLETED, self._on_completed)

    def _on_started(self, engine: Engine) -> None:
        """Initialize progress tracking."""
        print(f"\nStarting training for {self.max_epochs} epochs...")

    def _on_epoch_started(self, engine: Engine) -> None:
        """Start progress bar for new epoch."""
        self.current_epoch = engine.state.epoch
        total_iterations = len(engine.state.dataloader)  # type: ignore[arg-type]

        # Create progress bar for this epoch
        self.epoch_pbar = tqdm(
            total=total_iterations,
            desc=f"Epoch {self.current_epoch}/{self.max_epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

    def _on_iteration(self, engine: Engine) -> None:
        """Update progress bar after each iteration."""
        if self.epoch_pbar is not None:
            # Get current loss
            loss = engine.state.output.get("loss", 0.0)  # type: ignore[union-attr]

            # Build postfix with loss and learning rate
            postfix = {"loss": f"{loss:.4f}"}

            # Try to get current learning rate from optimizer
            if hasattr(engine, "optimizer") and engine.optimizer is not None:  # type: ignore[union-attr]
                lr = engine.optimizer.param_groups[0]["lr"]  # type: ignore[union-attr]
                postfix["lr"] = f"{lr:.6f}"

            # Update progress bar with loss and learning rate
            self.epoch_pbar.set_postfix(postfix)
            self.epoch_pbar.update(1)

    def _on_epoch_completed(self, engine: Engine) -> None:
        """Close progress bar after epoch."""
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
            self.epoch_pbar = None

    def _on_completed(self, engine: Engine) -> None:
        """Training completed."""
        print("\nTraining completed!")
