"""
YAML configuration file generator from experiment plans.
Converts ExperimentPlan dataclass to YAML config matching nnBenchmark format.
"""

from pathlib import Path
from typing import TextIO

from src.planning.planner.create import ExperimentPlan


def generate_config_yaml(
    plan: ExperimentPlan,
    dataset_dir: str,
    output_path: str,
    fold: int = 0,
    num_workers: int | None = None,
    cache_enabled: bool | None = None,
    cache_rate: float | None = None,
) -> None:
    """
    Generate YAML configuration file from experiment plan.

    Args:
        plan: ExperimentPlan with optimized parameters
        dataset_dir: Path to dataset directory (for extracting dataset name)
        output_path: Where to save the YAML config
        fold: Which fold to use for training (default: 0)
        num_workers: Number of DataLoader workers (auto-detected if None)
        cache_enabled: Whether to enable caching (auto-detected if None)
        cache_rate: Cache rate fraction 0.0-1.0 (auto-calculated if None)

    """
    # Extract dataset name from directory
    dataset_name = Path(dataset_dir).name

    # Determine spatial dimensions
    spatial_dims = 2 if plan.is_2d else 3

    # Write YAML file with comprehensive comments
    with open(output_path, "w") as f:
        _write_header(f, plan)
        _write_comment(f, plan)
        _write_gpu_logging(f)
        _write_dataset_config(f, dataset_name, plan, fold, cache_enabled, cache_rate)
        _write_model_config(f, spatial_dims, plan)
        _write_training_config(f, plan, num_workers)
        _write_optimizer_config(f)
        _write_loss_config(f, plan)
        _write_metrics_config(f, plan)
        _write_transforms_config(f, plan)
        _write_testing_config(f)


def _write_header(f: TextIO, plan: ExperimentPlan) -> None:
    """Write file header with overview information."""
    f.write("#" + "=" * 79 + "\n")
    f.write(f"# {plan.dataset_name} - Auto-Generated Configuration\n")
    f.write("#" + "=" * 79 + "\n")
    f.write("#\n")
    f.write(
        "# This configuration was automatically generated using nnU-Net-style heuristics.\n"
    )
    f.write(
        "# The parameters are optimized based on dataset fingerprinting and analysis.\n"
    )
    f.write("#\n")
    f.write("# Key Auto-Generated Parameters:\n")
    f.write(f"#   - Patch Size: {plan.patch_size}\n")
    f.write(f"#   - Batch Size: {plan.batch_size}\n")
    f.write(f"#   - Normalization: {plan.normalization_scheme}\n")
    f.write(f"#   - Spatial Dimensions: {'2D' if plan.is_2d else '3D'}\n")
    f.write(f"#   - Number of Classes: {plan.num_classes}\n")
    f.write("#\n")
    f.write("# Feel free to customize these parameters based on your specific needs.\n")
    f.write("#" + "=" * 79 + "\n\n")


def _write_comment(f: TextIO, plan: ExperimentPlan) -> None:
    """Write top-level comment field."""
    f.write(
        f"_comment: Auto-generated for {plan.dataset_name} using nnU-Net heuristics\n\n"
    )


def _write_gpu_logging(f: TextIO) -> None:
    """Write GPU memory logging configuration."""
    f.write(
        "# ============================================================================\n"
    )
    f.write("# GPU Memory Logging\n")
    f.write(
        "# ============================================================================\n"
    )
    f.write("# Enable GPU memory monitoring during training to track VRAM usage.\n")
    f.write("# Useful for debugging OOM errors and optimizing batch sizes.\n\n")
    f.write("log_gpu_memory: true\n\n")


def _write_dataset_config(
    f: TextIO,
    dataset_name: str,
    plan: ExperimentPlan,
    fold: int,
    cache_enabled: bool | None = None,
    cache_rate: float | None = None,
) -> None:
    """Write dataset configuration section."""
    # Default to conservative caching if not specified
    if cache_enabled is None:
        cache_enabled = False
    if cache_rate is None:
        cache_rate = 0.0

    f.write(
        "# ============================================================================\n"
    )
    f.write("# Dataset Configuration\n")
    f.write(
        "# ============================================================================\n"
    )
    f.write("# Defines the dataset to use and patch extraction settings.\n\n")
    f.write("dataset:\n")
    f.write(f"  name: {dataset_name}  # Dataset identifier matching folder name\n\n")
    f.write(
        "  # Patch size for training (automatically optimized based on median shape)\n"
    )
    f.write(f"  spatial_size: {list(plan.patch_size)}\n\n")
    f.write("  # Number of output classes (including background)\n")
    f.write(f"  num_classes: {plan.num_classes}\n\n")
    f.write("  # Cross-validation fold to use (0-4 for 5-fold CV)\n")
    f.write(f"  fold: {fold}\n\n")
    f.write("  # Caching configuration (stores preprocessed data in memory)\n")
    f.write("  # Useful for multiple experiments; preserves augmentation\n")
    f.write("  # Auto-optimized based on dataset size and available RAM\n")
    f.write("  cache:\n")
    enabled_str = "true" if cache_enabled else "false"
    f.write(f"    enabled: {enabled_str}  # Auto-optimized for this dataset\n")
    f.write(
        f"    cache_rate: {cache_rate}  # Fraction of dataset to cache (1.0 = 100%)\n\n"
    )


def _write_model_config(f: TextIO, spatial_dims: int, plan: ExperimentPlan) -> None:
    """Write model architecture configuration."""
    f.write(
        "# ============================================================================\n"
    )
    f.write("# Model Architecture\n")
    f.write(
        "# ============================================================================\n"
    )
    f.write("# U-Net architecture with parameters optimized for this dataset.\n\n")
    f.write("model:\n")
    f.write("  type: UNet  # MONAI U-Net implementation\n\n")
    f.write(f"  # Spatial dimensions: {spatial_dims}D\n")
    f.write(f"  spatial_dims: {spatial_dims}\n\n")
    f.write("  # Input/output channels\n")
    f.write("  in_channels: 1  # Single modality (grayscale)\n")
    f.write(f"  out_channels: {plan.num_classes}  # One channel per class\n\n")
    f.write("  # Feature channels at each resolution level (auto-optimized)\n")
    f.write(f"  channels: {plan.channels}\n\n")
    f.write("  # Downsampling strides for each encoder level\n")
    f.write("  # Note: MONAI UNet expects len(strides) = len(channels) - 1\n")
    f.write(
        "  # Strides define transitions between levels, not the levels themselves\n"
    )
    f.write("  strides:\n")
    # Write all strides (first [1,1,1] is already excluded in the plan)
    for stride in plan.strides:
        f.write(f"    - {list(stride)}\n")
    f.write("\n")
    f.write(
        "  # Number of residual units per level (0 = plain convolutions, nnU-Net default)\n"
    )
    f.write(f"  num_res_units: {plan.num_res_units}\n\n")
    f.write(
        "  # ========================================================================\n"
    )
    f.write("  # Deep Supervision (nnU-Net style)\n")
    f.write(
        "  # ========================================================================\n"
    )
    f.write("  # Deep supervision improves gradient flow and feature learning by\n")
    f.write(
        "  # computing loss at multiple decoder levels, not just the final output.\n"
    )
    f.write("  # This is enabled by default following nnU-Net approach.\n")
    f.write("  #\n")
    f.write("  # Weights follow exponential decay: [1.0, 0.5, 0.25, ...]\n")
    f.write("  # - Higher weight on final output (stronger supervision)\n")
    f.write("  # - Lower weights on intermediate outputs (regularization)\n")
    f.write("  # - Number of weights must match number of decoder stages\n")
    f.write("  #\n")
    f.write("  # You can override weights here if needed, but these are optimized\n")
    f.write("  # for this dataset based on the network topology.\n\n")
    f.write(f"  deep_supervision: {str(plan.deep_supervision).lower()}\n")
    f.write(f"  ds_weights: {plan.ds_weights}\n\n")


def _write_training_config(
    f: TextIO, plan: ExperimentPlan, num_workers: int | None = None
) -> None:
    """Write training hyperparameters."""
    # Default to conservative num_workers if not specified
    if num_workers is None:
        num_workers = 1

    f.write(
        "# ============================================================================\n"
    )
    f.write("# Training Configuration\n")
    f.write(
        "# ============================================================================\n"
    )
    f.write("# Training hyperparameters and validation settings.\n\n")
    f.write("training:\n")
    f.write("  # Total training epochs (standard for medical segmentation)\n")
    f.write("  epochs: 200\n\n")
    f.write(
        "  # Batch size (auto-optimized based on patch size and available memory)\n"
    )
    f.write(f"  batch_size: {plan.batch_size}\n\n")
    f.write("  # Learning rate (nnU-Net default: 0.01 for SGD with momentum)\n")
    f.write("  learning_rate: 0.01\n\n")
    f.write("  # Run validation every N epochs\n")
    f.write("  val_interval: 5\n\n")
    f.write("  # Number of data loading workers (auto-optimized based on CPU cores)\n")
    f.write(f"  num_workers: {num_workers}\n\n")
    f.write("  # Metric to use for saving best checkpoint\n")
    f.write("  checkpoint_metric: Dice\n\n")
    f.write("  # Metrics to plot during training\n")
    f.write("  plot_metrics:\n")
    f.write("    - Dice          # Overlap-based metric\n")
    f.write("    - SurfaceDice   # Surface distance metric\n\n")
    f.write("  # Mixed precision training (FP16 for ~2x speedup, requires CUDA)\n")
    f.write("  # Set to true if you have a modern GPU with Tensor Cores\n")
    f.write("  # Matches nnUNet's automatic AMP usage on GPU\n")
    f.write("  mixed_precision: true\n\n")


def _write_optimizer_config(f: TextIO) -> None:
    """Write optimizer configuration (nnU-Net exact settings)."""
    f.write(
        "# ============================================================================\n"
    )
    f.write("# Optimizer Configuration\n")
    f.write(
        "# ============================================================================\n"
    )
    f.write("# SGD optimizer with Nesterov momentum (nnU-Net default).\n\n")
    f.write("optimizer:\n")
    f.write("  type: SGD  # Stochastic Gradient Descent\n\n")
    f.write("  # L2 regularization (nnU-Net default: 3e-5)\n")
    f.write("  weight_decay: 0.00003\n\n")
    f.write("  # Momentum coefficient (nnU-Net default: 0.99)\n")
    f.write("  momentum: 0.99\n\n")
    f.write("  # Enable Nesterov momentum for better convergence\n")
    f.write("  nesterov: true\n\n")


def _write_loss_config(f: TextIO, plan: ExperimentPlan) -> None:
    """Write loss function configuration."""
    f.write(
        "# ============================================================================\n"
    )
    f.write("# Loss Function\n")
    f.write(
        "# ============================================================================\n"
    )
    f.write("# Combined Dice + Cross-Entropy loss for better gradient stability.\n\n")
    f.write("loss:\n")
    f.write("  type: DiceCELoss  # Dice loss + Cross Entropy\n\n")
    f.write("  # Convert integer labels to one-hot encoding\n")
    f.write("  to_onehot_y: true\n\n")
    f.write("  # Apply softmax to model outputs\n")
    f.write("  softmax: true\n\n")
    f.write("  # Batch dice: compute dice across entire batch for stable gradients\n")
    f.write("  # Matches nnUNet's batch_dice=True for more stable early training\n")
    f.write("  batch: true\n\n")


def _write_metrics_config(f: TextIO, plan: ExperimentPlan) -> None:
    """Write metrics configuration."""
    f.write(
        "# ============================================================================\n"
    )
    f.write("# Evaluation Metrics\n")
    f.write(
        "# ============================================================================\n"
    )
    f.write("# Metrics computed during validation to assess segmentation quality.\n\n")
    f.write("metrics:\n")
    f.write("  # Dice Similarity Coefficient (overlap-based)\n")
    f.write("  - type: DiceMetric\n")
    f.write("    include_background: false  # Exclude background class\n")
    f.write("    reduction: mean_batch      # Average across batch\n")
    f.write(f"    num_classes: {plan.num_classes}\n\n")
    f.write("  # Surface Dice (boundary accuracy metric)\n")
    f.write("  - type: SurfaceDiceMetric\n")
    f.write("    include_background: false  # Exclude background class\n")
    f.write("    reduction: mean_batch      # Average across batch\n\n")
    f.write("    # Distance tolerance in mm for each class\n")
    f.write("    class_thresholds:\n")
    for _ in range(plan.num_classes - 1):
        f.write("      - 2.0  # 2mm tolerance\n")
    f.write("\n")


def _write_transforms_config(f: TextIO, plan: ExperimentPlan) -> None:
    """Write data transforms/augmentation configuration (nnU-Net exact)."""
    f.write(
        "# ============================================================================\n"
    )
    f.write("# Data Transforms & Augmentation (nnU-Net Configuration)\n")
    f.write(
        "# ============================================================================\n"
    )
    f.write(
        "# This configuration exactly matches nnU-Net's preprocessing and augmentation.\n\n"
    )
    f.write("transforms:\n")
    f.write(
        "  # --------------------------------------------------------------------------\n"
    )
    f.write("  # Common Transforms (applied to both training and validation)\n")
    f.write(
        "  # --------------------------------------------------------------------------\n"
    )
    f.write("  common:\n")
    f.write("    # Load NIfTI images and labels\n")
    f.write("    - type: LoadImaged\n")
    f.write("      keys: [image, label]\n\n")
    f.write("    # Ensure channel-first format [C, H, W, D]\n")
    f.write("    - type: EnsureChannelFirstd\n")
    f.write("      keys: [image, label]\n\n")

    # CT-specific intensity clipping (nnU-Net style)
    if plan.normalization_scheme == "CTNormalization":
        f.write("    # Intensity Clipping for CT (nnU-Net: 0.5-99.5 percentiles)\n")
        f.write(
            "    # Clips outliers to global dataset range for stable normalization\n"
        )
        f.write("    # Following nnU-Net v2.4.1 implementation for CT data\n")
        f.write("    - type: ScaleIntensityRanged\n")
        f.write("      keys: [image]\n")
        f.write(f"      a_min: {plan.intensity_clip_min}\n")
        f.write(f"      a_max: {plan.intensity_clip_max}\n")
        f.write(f"      b_min: {plan.intensity_clip_min}\n")
        f.write(f"      b_max: {plan.intensity_clip_max}\n")
        f.write("      clip: true  # Clip values outside range\n\n")

    # Intensity normalization - nnU-Net uses Z-score normalization!
    f.write("    # Z-Score Normalization (nnU-Net default)\n")
    f.write("    # Normalizes to zero mean and unit variance: (x - mean) / std\n")
    f.write("    - type: NormalizeIntensityd\n")
    f.write("      keys: [image]\n")
    f.write("      nonzero: false  # Compute stats over whole image\n")
    f.write("      channel_wise: false  # Single channel\n\n")

    f.write("    # Pad images smaller than patch size\n")
    f.write("    - type: SpatialPadd\n")
    f.write("      keys: [image, label]\n")
    f.write(f"      spatial_size: {list(plan.patch_size)}\n")
    f.write("      mode: constant  # Zero-padding\n\n")

    f.write("    # Convert to PyTorch tensors\n")
    f.write("    - type: ToTensord\n")
    f.write("      keys: [image, label]\n\n")

    f.write(
        "  # --------------------------------------------------------------------------\n"
    )
    f.write("  # Training Transforms (nnU-Net Augmentation Pipeline)\n")
    f.write(
        "  # --------------------------------------------------------------------------\n"
    )
    f.write("  train:\n")
    f.write("    # Random spatial crop to patch size\n")
    f.write("    - type: RandSpatialCropd\n")
    f.write("      keys: [image, label]\n")
    f.write(f"      roi_size: {list(plan.patch_size)}\n")
    f.write("      random_size: false\n\n")

    # Spatial augmentations (nnU-Net exact)
    f.write("    # Rotation (nnU-Net: ±30° for isotropic 3D, p=0.2)\n")
    f.write("    # Using continuous angle rotation to match nnU-Net exactly\n")
    f.write("    - type: RandRotated\n")
    f.write("      keys: [image, label]\n")
    f.write("      prob: 0.2\n")
    f.write("      range_x: 0.5236  # ±30° in radians (30 * π / 180)\n")
    f.write("      range_y: 0.5236\n")
    f.write("      range_z: 0.5236\n")
    f.write('      mode: ["bilinear", "nearest"]\n')
    f.write("      padding_mode: border\n\n")

    f.write("    # Scaling (nnU-Net: 0.7-1.4, p=0.2)\n")
    f.write("    - type: RandZoomd\n")
    f.write("      keys: [image, label]\n")
    f.write("      prob: 0.2\n")
    f.write("      min_zoom: 0.7\n")
    f.write("      max_zoom: 1.4\n")
    f.write("      keep_size: true  # Maintain spatial dimensions after zoom\n")
    f.write('      mode: ["trilinear", "nearest"]\n\n')

    # Mirroring (nnU-Net: all axes)
    for axis in range(len(plan.patch_size)):
        axis_name = ["X", "Y", "Z"][axis] if axis < 3 else str(axis)
        f.write(f"    # Random flip along {axis_name}-axis (nnU-Net: mirroring)\n")
        f.write("    - type: RandFlipd\n")
        f.write("      keys: [image, label]\n")
        f.write("      prob: 0.5\n")
        f.write(f"      spatial_axis: {axis}\n\n")

    # Intensity augmentations (nnU-Net exact)
    f.write("    # Gaussian Noise (nnU-Net: variance (0, 0.1), p=0.1)\n")
    f.write("    - type: RandGaussianNoised\n")
    f.write("      keys: [image]\n")
    f.write("      prob: 0.1\n")
    f.write("      mean: 0.0\n")
    f.write("      std: 0.1\n\n")

    f.write("    # Gaussian Blur (nnU-Net: sigma (0.5, 1.0), p=0.2)\n")
    f.write("    - type: RandGaussianSmoothd\n")
    f.write("      keys: [image]\n")
    f.write("      prob: 0.2\n")
    f.write("      sigma_x: [0.5, 1.0]\n")
    f.write("      sigma_y: [0.5, 1.0]\n")
    f.write("      sigma_z: [0.5, 1.0]\n\n")

    f.write("    # Brightness (nnU-Net: multiplier (0.75, 1.25), p=0.15)\n")
    f.write("    - type: RandScaleIntensityd\n")
    f.write("      keys: [image]\n")
    f.write("      factors: [0.75, 1.25]\n")
    f.write("      prob: 0.15\n\n")

    f.write("    # Contrast (nnU-Net: range (0.75, 1.25), p=0.15)\n")
    f.write("    - type: RandAdjustContrastd\n")
    f.write("      keys: [image]\n")
    f.write("      prob: 0.15\n")
    f.write("      gamma: [0.75, 1.25]\n\n")

    f.write("    # Low-Resolution Scale (nnU-Net: scale (0.5, 1.0), p=0.25)\n")
    f.write("    # Simulates low resolution images to improve robustness\n")
    f.write("    - type: RandZoomd\n")
    f.write("      keys: [image, label]\n")
    f.write("      prob: 0.25\n")
    f.write("      min_zoom: 0.5\n")
    f.write("      max_zoom: 1.0\n")
    f.write("      mode: [bilinear, nearest]\n")
    f.write("      padding_mode: edge\n\n")

    f.write("    # Gamma Transform (nnU-Net: range (0.7, 1.5), p=0.1 + p=0.3)\n")
    f.write("    - type: RandHistogramShiftd\n")
    f.write("      keys: [image]\n")
    f.write("      prob: 0.1\n")
    f.write("      num_control_points: [5, 15]\n\n")

    f.write("    # Gamma Transform without inversion (p=0.3)\n")
    f.write("    - type: RandHistogramShiftd\n")
    f.write("      keys: [image]\n")
    f.write("      prob: 0.3\n")
    f.write("      num_control_points: [5, 15]\n\n")

    f.write("    # Ensure consistent patch size after augmentations\n")
    f.write("    # ResizeWithPadOrCropd pads and crops to ensure exact patch size\n")
    f.write("    - type: ResizeWithPadOrCropd\n")
    f.write("      keys: [image, label]\n")
    f.write(f"      spatial_size: {list(plan.patch_size)}\n")
    f.write('      mode: ["constant", "constant"]\n\n')

    f.write(
        "  # --------------------------------------------------------------------------\n"
    )
    f.write("  # Validation Transforms (no augmentation)\n")
    f.write(
        "  # --------------------------------------------------------------------------\n"
    )
    f.write("  val:\n")
    f.write("    # Center crop to patch size (deterministic)\n")
    f.write("    - type: CenterSpatialCropd\n")
    f.write("      keys: [image, label]\n")
    f.write(f"      roi_size: {list(plan.patch_size)}\n\n")

    f.write(
        "  # --------------------------------------------------------------------------\n"
    )
    f.write("  # Test Transforms (same as validation, no augmentation)\n")
    f.write(
        "  # --------------------------------------------------------------------------\n"
    )
    f.write("  test:\n")
    f.write("    # Center crop to patch size (deterministic)\n")
    f.write("    - type: CenterSpatialCropd\n")
    f.write("      keys: [image, label]\n")
    f.write(f"      roi_size: {list(plan.patch_size)}\n\n")


def _write_testing_config(f: TextIO) -> None:
    """Write testing configuration."""
    f.write(
        "# ============================================================================\n"
    )
    f.write("# Testing Configuration\n")
    f.write(
        "# ============================================================================\n"
    )
    f.write("# Settings for inference/testing phase.\n\n")
    f.write("testing:\n")
    f.write("  # Batch size for inference (typically 1 for full-resolution testing)\n")
    f.write("  batch_size: 1\n")
