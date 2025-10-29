"""
Training plot generation.
Contains functions for plotting training loss curves.
"""

import matplotlib.pyplot as plt

# Configure matplotlib styles on import
from src.plotting.styles import mpl  # noqa: F401


def plot_training_loss(
    epochs: list[int], train_loss: list[float], save_path: str
) -> None:
    """
    Plot training loss vs epoch using matplotlib.

    Args:
        epochs: List of epoch numbers
        train_loss: List of training loss values
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, "-o", label="Training Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True, edgecolor="black")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
