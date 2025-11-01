"""
Validation plot generation and visualizations.
Contains functions for plotting validation metrics and saving validation visualizations.
"""

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Configure matplotlib styles on import
# Configure matplotlib styles (side-effect import - must run before plotting)
from src.plotting.styles import mpl  # noqa: F401
from src.utils.files import ensure_directory


def plot_validation_metric(
    epochs: list[int],
    metric_values: list[float],
    metric_name: str,
    save_path: str,
    per_class_values: dict[str, list[float]] | None = None,
) -> None:
    """
    Plot validation metric vs epoch with optional per-class curves using matplotlib.

    Args:
        epochs: List of epoch numbers where validation was run
        metric_values: List of overall mean metric values
        metric_name: Name of the metric (e.g., "Dice")
        save_path: Path to save the plot
        per_class_values: Optional dict mapping class names to their metric values over epochs
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot per-class curves if available
    if per_class_values:
        for class_name, class_vals in per_class_values.items():
            ax.plot(
                epochs,
                class_vals,
                "--o",
                markersize=3,
                alpha=0.6,
                label=f"{class_name}",
            )

    # Plot overall mean with thicker line
    ax.plot(
        epochs,
        metric_values,
        "-o",
        linewidth=2,
        markersize=5,
        color="black",
        label="Mean",
        zorder=10,
    )

    # Mark the best overall score
    if metric_values:
        best_idx = np.argmax(metric_values)
        best_epoch = epochs[best_idx]
        best_value = metric_values[best_idx]
        ax.scatter(
            [best_epoch],
            [best_value],
            color="red",
            s=100,
            zorder=15,
            marker="*",
            edgecolors="black",
            linewidths=1,
            label=f"Best: {best_value:.4f}",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{metric_name} Score")
    ax.legend(loc="best", frameon=True, edgecolor="black")
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def save_validation_visualizations(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    save_dir: str,
    epoch: int,
    spatial_dims: int,
) -> None:
    """
    Save visualization of first batch: image, reference label, and prediction in one row.

    Args:
        images: Input images tensor (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
        labels: Reference label tensor
        predictions: Model predictions tensor
        save_dir: Directory to save visualizations
        epoch: Current epoch number
        spatial_dims: 2 for 2D data, 3 for 3D data
    """
    # Create visualizations subdirectory
    vis_dir = ensure_directory(str(Path(save_dir) / "visualizations"))

    # Get batch size (save entire first batch)
    batch_size = images.shape[0]

    # Move to CPU and convert to numpy
    images_np: np.ndarray = images.detach().cpu().numpy()
    labels_np: np.ndarray = labels.detach().cpu().numpy()
    predictions_np: np.ndarray = predictions.detach().cpu().numpy()

    if spatial_dims == 3:
        # For 3D data, take the middle slice along depth dimension
        # Shape: (B, C, D, H, W) -> take middle slice along D
        depth_idx = images_np.shape[2] // 2
        images_np = images_np[:, :, depth_idx, :, :]
        labels_np = labels_np[:, :, depth_idx, :, :]
        predictions_np = predictions_np[:, :, depth_idx, :, :]

    # Create figure with subplots: 3 columns (image, Ref, pred) x batch_size rows
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))

    # Handle single case
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for idx in range(batch_size):
        # Get single case
        image = images_np[idx]  # (C, H, W)
        label = labels_np[idx]  # (1, H, W) or (C, H, W)
        pred = predictions_np[idx]  # (1, H, W) or (C, H, W)

        # Handle multi-channel images (e.g., RGB)
        image_display: np.ndarray
        if image.shape[0] == 3:
            # RGB image - transpose to (H, W, C)
            image_display = np.transpose(image, (1, 2, 0))
        else:
            # Grayscale - take first channel
            image_display = image[0]

        # Handle labels and predictions (take first channel or argmax)
        if label.shape[0] == 1:
            label_display = label[0]
            pred_display = pred[0]
        else:
            # Multi-class: show class indices
            label_display = label[0]  # Assuming already class indices
            pred_display = pred[0]

        # Plot image
        axes[idx, 0].imshow(image_display, cmap="gray" if image.shape[0] == 1 else None)
        axes[idx, 0].set_title(f"Sample {idx + 1}: Input Image")
        axes[idx, 0].axis("off")

        # Plot reference label
        axes[idx, 1].imshow(label_display, cmap="jet", interpolation="nearest")
        axes[idx, 1].set_title(f"Sample {idx + 1}: Reference Label")
        axes[idx, 1].axis("off")

        # Plot prediction
        axes[idx, 2].imshow(pred_display, cmap="jet", interpolation="nearest")
        axes[idx, 2].set_title(f"Sample {idx + 1}: Prediction")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    save_path = str(Path(vis_dir) / f"validation_epoch_{epoch:03d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
