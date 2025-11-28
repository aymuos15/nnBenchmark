#!/usr/bin/env python3
"""Visualize predictions for a specific sample.

Usage:
    python scripts/viz.py --dataset Dataset001_Cellpose --config fold_0 --sample 007_0000.png
    python scripts/viz.py --dataset Data --config fold_0 --sample 015_0000.png
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from monai import networks
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
)

from src.config import get_datasets_root, get_results_root


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize predictions for a specific sample",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/viz.py --dataset Dataset001_Cellpose --config fold_0 --sample 007_0000.png
  python scripts/viz.py --dataset Data --config fold_0 --sample 015_0000.png
  python scripts/viz.py --dataset Data --config fold_0 --sample 015_0000.png --output my_viz.png
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., Dataset001_Cellpose, Data)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config/fold name (e.g., fold_0)",
    )
    parser.add_argument(
        "--sample",
        type=str,
        required=True,
        help="Sample filename (e.g., 007_0000.png)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (optional, default: display to screen)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base data directory (default: nnBench_raw environment variable)",
    )

    return parser.parse_args()


def find_config_yaml(config_name: str, dataset_name: str) -> Path:
    """Find the config YAML file for a dataset and config."""
    # Look in docs/datasets/{dataset_name}/
    config_dir = Path.cwd() / "docs" / "datasets" / dataset_name

    if not config_dir.exists():
        raise FileNotFoundError(f"Could not find config directory: {config_dir}")

    # Try exact match first
    config_path = config_dir / f"{config_name}.yaml"
    if config_path.exists():
        return config_path

    # Try with fold_ prefix if not present
    if not config_name.startswith("fold_"):
        config_path = config_dir / f"fold_{config_name}.yaml"
        if config_path.exists():
            return config_path

    # List available configs
    available = list(config_dir.glob("*.yaml"))
    raise FileNotFoundError(
        f"Could not find config {config_name}.yaml in {config_dir}\n"
        f"Available configs: {[p.stem for p in available]}"
    )


def find_checkpoint(config_name: str, dataset_name: str) -> Path:
    """Find the best checkpoint for a config/dataset combination."""
    if config_name.startswith("fold_"):
        fold = config_name.split("_")[1]
    else:
        fold = config_name

    results_root = get_results_root()

    # Check multiple possible paths
    possible_paths = [
        results_root / dataset_name / config_name / "checkpoints",
        results_root / dataset_name / f"fold_{fold}" / "checkpoints",
    ]

    for checkpoint_dir in possible_paths:
        if checkpoint_dir.exists():
            # Find best checkpoint
            pt_files = list(checkpoint_dir.glob("best_*.pt"))
            if pt_files:
                return pt_files[0]

    raise FileNotFoundError(
        f"Could not find checkpoint for dataset={dataset_name}, config={config_name}"
    )


def find_sample_path(
    dataset_dir: Path, sample_name: str, data_type: str = "imagesTs"
) -> Path:
    """Find the path to a sample image."""
    sample_path = dataset_dir / data_type / sample_name

    if not sample_path.exists():
        # Try without extension
        base_name = sample_name.split(".")[0]

        # For labels, try removing _0000 suffix (common pattern: images have _0000, labels don't)
        patterns = [f"{base_name}.*", f"{base_name}_0000.*"]
        if base_name.endswith("_0000"):
            base_without_suffix = base_name[:-5]  # Remove "_0000"
            patterns.insert(0, f"{base_without_suffix}.*")

        for pattern in patterns:
            matches = list(dataset_dir.glob(f"{data_type}/{pattern}"))
            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"Could not find sample: {sample_name} in {dataset_dir / data_type}"
        )

    return sample_path


def load_sample(
    image_path: Path,
    label_path: Path = None,
    spatial_dims: int = 2,
    patch_size: list = None,
):
    """Load image and label with proper preprocessing."""
    from monai.transforms import SpatialPadd

    # Load transforms
    keys = ["image", "label"] if label_path else ["image"]
    transform_list = [
        LoadImaged(keys=keys, ensure_channel_first=True),
        NormalizeIntensityd(keys=["image"]),
    ]

    # Add padding if patch size is specified
    if patch_size:
        transform_list.append(
            SpatialPadd(keys=keys, spatial_size=patch_size, mode="constant")
        )

    transforms = Compose(transform_list)

    data = {"image": str(image_path)}
    if label_path:
        data["label"] = str(label_path)

    result = transforms(data)
    return result


def predict_sample(
    model,
    image: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Make prediction for a single sample."""
    model.eval()
    with torch.no_grad():
        # Add batch dimension if needed
        if image.dim() == 3:  # 2D: (C, H, W)
            image = image.unsqueeze(0)
        elif image.dim() == 4:  # 3D: (C, D, H, W)
            image = image.unsqueeze(0)

        image = image.to(device)
        output = model(image)

        # Apply softmax
        if output.shape[1] > 1:
            output = torch.softmax(output, dim=1)

        return output.squeeze(0).cpu()


def visualize_prediction(
    image: torch.Tensor,
    label: torch.Tensor,
    prediction: torch.Tensor,
    sample_name: str,
    output_path: str = None,
    spatial_dims: int = 2,
):
    """Visualize image, label, and prediction."""
    # Prepare data for visualization
    images = image.unsqueeze(0) if image.dim() == 3 else image
    labels = label.unsqueeze(0) if label.dim() == 2 else label
    predictions = prediction.unsqueeze(0) if prediction.dim() == 3 else prediction

    # Use existing visualization function
    try:
        from src.plotting.validation import save_validation_visualizations

        output_dir = Path(output_path).parent if output_path else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        save_validation_visualizations(
            images=images,
            labels=labels,
            predictions=predictions,
            save_dir=str(output_dir),
            epoch=0,
            spatial_dims=spatial_dims,
            filename_prefix=sample_name.split(".")[0],
        )

        print(f"✓ Visualization saved to {output_dir}")

    except Exception as e:
        print(f"Warning: Could not use standard visualization: {e}")
        print("Falling back to basic matplotlib visualization...")
        visualize_with_matplotlib(image, label, prediction, sample_name, output_path)


def visualize_with_matplotlib(
    image: torch.Tensor,
    label: torch.Tensor,
    prediction: torch.Tensor,
    sample_name: str,
    output_path: str = None,
):
    """Fallback visualization using matplotlib with connected components analysis."""
    try:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import ndimage

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Image (take first channel if multi-channel)
        img_data = image[0].numpy() if image.shape[0] > 1 else image.squeeze().numpy()
        axes[0, 0].imshow(img_data, cmap="gray")
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis("off")

        # Label
        label_data = label.squeeze().numpy() if label.dim() > 2 else label.numpy()
        axes[0, 1].imshow(label_data, cmap="tab20")
        axes[0, 1].set_title("Ground Truth")
        axes[0, 1].axis("off")

        # Prediction (argmax if multi-class)
        if prediction.shape[0] > 1:
            pred_data = torch.argmax(prediction, dim=0).numpy()
        else:
            pred_data = prediction.squeeze().numpy()
        axes[0, 2].imshow(pred_data, cmap="tab20")
        axes[0, 2].set_title("Prediction")
        axes[0, 2].axis("off")

        # Ensure label and prediction have the same size
        if label_data.shape != pred_data.shape:
            print(
                f"⚠️  Warning: Label shape {label_data.shape} != Prediction shape {pred_data.shape}"
            )
            # Crop to the smaller size
            min_h = min(label_data.shape[0], pred_data.shape[0])
            min_w = min(label_data.shape[1], pred_data.shape[1])
            label_data = label_data[:min_h, :min_w]
            pred_data = pred_data[:min_h, :min_w]
            print(f"   Cropped to common size: {label_data.shape}")

        # Connected components analysis for ground truth
        binary_label = (label_data > 0).astype(int)
        labeled_label, num_label_features = ndimage.label(binary_label)
        print(
            f"🔍 GT: binary_label unique = {np.unique(binary_label)}, labeled_label max = {labeled_label.max()}"
        )

        # Create colormap for ground truth components
        if num_label_features > 0:
            try:
                tab20 = plt.colormaps()["tab20"]
            except TypeError:
                tab20 = plt.cm.get_cmap("tab20")
            num_colors = max(20, num_label_features)
            colors = [tab20(i % 20) for i in range(num_colors)]
            cmap_label = mcolors.ListedColormap(colors)
            cmap_label.set_under("black")
        else:
            cmap_label = "gray"

        # Hide the first subplot (bottom left)
        axes[1, 0].axis("off")

        axes[1, 1].imshow(labeled_label, cmap=cmap_label, vmin=0)
        axes[1, 1].set_title(f"GT Components ({num_label_features} objects)")

        # Add component numbers for ground truth (BEFORE turning off axis)
        if num_label_features > 0:
            center_of_mass_label = ndimage.center_of_mass(
                binary_label, labeled_label, range(1, num_label_features + 1)
            )
            for i, (y, x) in enumerate(center_of_mass_label, 1):
                axes[1, 1].text(
                    x,
                    y,
                    str(i),
                    color="white",
                    fontsize=12,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    zorder=100,
                )
            print(f"📊 GT Components numbered 1-{num_label_features}")

        axes[1, 1].axis("off")

        # Connected components analysis for prediction
        print(
            f"🔍 PRED before: pred_data shape={pred_data.shape}, unique={sorted(np.unique(pred_data))}"
        )
        binary_pred = (pred_data > 0).astype(int)
        print(f"🔍 PRED binary: unique={np.unique(binary_pred)}")
        labeled_pred, num_pred_features = ndimage.label(binary_pred)
        print(
            f"🔍 PRED: labeled_pred max = {labeled_pred.max()}, num_pred_features = {num_pred_features}"
        )

        # Create colormap for prediction components
        if num_pred_features > 0:
            try:
                tab20 = plt.colormaps()["tab20"]
            except TypeError:
                tab20 = plt.cm.get_cmap("tab20")
            num_colors = max(20, num_pred_features)
            colors = [tab20(i % 20) for i in range(num_colors)]
            cmap_pred = mcolors.ListedColormap(colors)
            cmap_pred.set_under("black")
        else:
            cmap_pred = "gray"

        axes[1, 2].imshow(labeled_pred, cmap=cmap_pred, vmin=0)
        axes[1, 2].set_title(f"Pred Components ({num_pred_features} objects)")

        # Add component numbers for prediction (BEFORE turning off axis)
        if num_pred_features > 0:
            center_of_mass_pred = ndimage.center_of_mass(
                binary_pred, labeled_pred, range(1, num_pred_features + 1)
            )
            for i, (y, x) in enumerate(center_of_mass_pred, 1):
                axes[1, 2].text(
                    x,
                    y,
                    str(i),
                    color="white",
                    fontsize=12,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    zorder=100,
                )
            print(f"📊 Pred Components numbered 1-{num_pred_features}")

        axes[1, 2].axis("off")

        plt.suptitle(f"Sample: {sample_name}")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Visualization saved to {output_path}")
        else:
            plt.show()

        plt.close()

    except ImportError as e:
        print(f"Error: Missing required library for visualization: {e}")


def main():
    """Main entry point."""
    args = parse_args()

    # Set up paths
    if args.data_dir:
        base_data_dir = Path(args.data_dir)
    else:
        base_data_dir = get_datasets_root()

    dataset_dir = base_data_dir / args.dataset

    if not dataset_dir.exists():
        print(f"❌ Error: Dataset not found: {dataset_dir}")
        sys.exit(1)

    print(f"📦 Dataset: {args.dataset}")
    print(f"⚙️  Config: {args.config}")
    print(f"📄 Sample: {args.sample}")
    print()

    try:
        # Load config from YAML
        print("📋 Loading config...")
        config_path = find_config_yaml(args.config, args.dataset)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"✓ Config: {config_path}")
        print()

        # Find checkpoint
        print("🔍 Finding checkpoint...")
        checkpoint_path = find_checkpoint(args.config, args.dataset)
        print(f"✓ Checkpoint: {checkpoint_path}")
        print()

        # Find sample
        print("🔍 Finding sample...")
        image_path = find_sample_path(dataset_dir, args.sample, "imagesTs")
        label_path = find_sample_path(dataset_dir, args.sample, "labelsTs")
        print(f"✓ Image: {image_path}")
        print(f"✓ Label: {label_path}")
        print()

        # Load checkpoint
        print("⚡ Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("✓ Checkpoint loaded")
        print()

        # Extract config parameters
        dataset_config = config.get("dataset", {})
        model_config = config.get("model", {})

        spatial_dims = dataset_config.get("spatial_dims", 2)
        num_classes = dataset_config.get("num_classes", 2)
        in_channels = model_config.get("in_channels", 1)
        model_type = model_config.get("type", "DynUNet")

        print("📊 Model Configuration:")
        print(f"  Model type: {model_type}")
        print(f"  Spatial dims: {spatial_dims}D")
        print(f"  In channels: {in_channels}")
        print(f"  Num classes: {num_classes}")
        print()

        # Build and load model
        print("🏗️  Building model...")
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Build complete model config from YAML
        complete_model_config = {
            "type": model_type,
            "spatial_dims": spatial_dims,
            "in_channels": in_channels,
            "out_channels": num_classes,
        }

        # Add model-specific parameters
        if model_type == "DynUNet" and "DynUNet" in model_config:
            complete_model_config.update(model_config["DynUNet"])
        elif model_type == "UNet" and "UNet" in model_config:
            complete_model_config.update(model_config["UNet"])

        # Build model via getattr (supports any MONAI model)
        model_type_name = complete_model_config.pop("type")
        model_class = getattr(networks.nets, model_type_name)
        model = model_class(**complete_model_config).to(device)

        # Load state dict with strict=False to handle architecture mismatches
        # (e.g., deep supervision heads that may not be in the built model)
        missing, unexpected = model.load_state_dict(
            checkpoint.get("model"), strict=False
        )
        if missing or unexpected:
            print(
                f"⚠️  Note: {len(missing) if missing else 0} missing, {len(unexpected) if unexpected else 0} unexpected keys"
            )
        print(f"✓ Model built and loaded on {device}")
        print()

        # Load sample
        print("📥 Loading sample...")
        patch_size = dataset_config.get("spatial_size")
        sample_data = load_sample(image_path, label_path, spatial_dims, patch_size)
        image = sample_data["image"]
        label = sample_data.get("label")
        print(f"✓ Image shape: {image.shape}")
        if label is not None:
            print(f"✓ Label shape: {label.shape}")
        print()

        # Make prediction
        print("🔮 Making prediction...")
        prediction = predict_sample(model, image, device)
        print(f"✓ Prediction shape: {prediction.shape}")
        print()

        # Visualize
        print("🎨 Visualizing...")
        visualize_prediction(
            image,
            label if label is not None else torch.zeros_like(image[0:1]),
            prediction,
            args.sample,
            args.output,
            spatial_dims,
        )

        print()
        print("✨ Done!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
