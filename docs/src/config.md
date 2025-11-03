# Configuration Reference

Complete reference of all configuration options supported by nnBenchmark.

## Structure

```yaml
# Top-level options
seed: 12345                 # Random seed for reproducibility (default: 12345)

# Main sections
dataset:     # Dataset and data loading configuration
model:       # Model architecture configuration
training:    # Training hyperparameters
optimizer:   # Optimizer configuration
loss:        # Loss function configuration
metrics:     # Evaluation metrics
transforms:  # Data augmentation and preprocessing
inference:   # Inference configuration
```

## Top-Level Options

```yaml
# Random seed for reproducibility
seed: 12345                 # Integer seed for random, numpy, torch (default: 12345)
                           # Can also be specified in training.seed or inference.seed
```

## Dataset Configuration

```yaml
dataset:
  name: Dataset001_Example  # Dataset identifier (must match folder name)

  # Dataset fingerprint properties (from preprocessing/planning)
  median_shape: [512, 512, 128]      # Median image shape in voxels
  median_spacing: [1.0, 1.0, 3.0]    # Median voxel spacing in mm
  foreground_intensity_mean: 100.5   # Mean intensity of foreground region

  # Patch extraction for training
  spatial_size: [128, 128, 128]      # 3D patch size or [256, 256] for 2D

  # Number of output classes (including background)
  num_classes: 3            # Background + foreground classes

  # Cross-validation fold (-1 = train on all data, no validation)
  fold: 0                   # 0-4 for 5-fold CV, -1 for all data

  # Data caching (store preprocessed data in RAM)
  cache:
    enabled: false          # true = enable caching, false = disable
    cache_rate: 0.0         # Fraction to cache: 0.0-1.0 (1.0 = 100%)
```

## Model Configuration

Supports both **flat** (backward compatible) and **nested** (multi-model) formats.

### Nested Format (Multi-Model Support)

```yaml
model:
  # Model selection
  type: DynUNet             # DynUNet or UNet

  # Shared parameters (common to all models)
  spatial_dims: 3           # 2 for 2D, 3 for 3D
  in_channels: 1            # Number of input channels
  out_channels: 3           # Number of output classes

  # Deep supervision (nnU-Net style, exposed for DynUNet)
  deep_supervision: true    # Enable deep supervision
  deep_supr_num: 1          # Number of deep supervision outputs
  ds_weights: [1.0, 0.5]    # Loss weights for each supervision level

  # DynUNet-specific parameters
  DynUNet:
    filters: [32, 64, 128, 256, 320]  # Feature channels per encoder stage

    kernel_size:            # Kernel sizes per stage (typically all 3x3x3)
      - [3, 3, 3]
      - [3, 3, 3]
      - [3, 3, 3]
      - [3, 3, 3]
      - [3, 3, 3]

    strides:                # Strides per stage ([1,1,1] = no downsampling)
      - [1, 1, 1]
      - [2, 2, 2]
      - [2, 2, 2]
      - [2, 2, 2]
      - [2, 2, 2]

    upsample_kernel_size:   # Upsampling kernel sizes for decoder
      - [2, 2, 2]
      - [2, 2, 2]
      - [2, 2, 2]
      - [2, 2, 2]

    # Normalization layer: INSTANCE or BATCH
    norm_name: [INSTANCE, {affine: true}]

    # Activation function: leakyrelu, relu, prelu, etc.
    act_name: [leakyrelu, {inplace: true, negative_slope: 0.01}]

    # Residual connections
    res_block: false        # true = residual blocks, false = plain convolutions

    # Transpose convolution bias (nnU-Net default: true)
    trans_bias: true        # true = bias in upsampling, false = no bias

  # UNet-specific parameters (auto-derived from DynUNet)
  UNet:
    channels: [32, 64, 128, 256, 320]  # Feature channels per stage
    strides: [2, 2, 2, 2]              # Downsampling strides (simplified)
    num_res_units: 2                   # Number of residual units per stage
```

### Flat Format (Backward Compatible)

```yaml
model:
  type: DynUNet             # Model type
  spatial_dims: 3           # Dimensionality
  in_channels: 1            # Input channels
  out_channels: 3           # Output classes

  # Deep supervision
  deep_supervision: true
  deep_supr_num: 1
  ds_weights: [1.0, 0.5]

  # DynUNet parameters (all at top level)
  filters: [32, 64, 128, 256, 320]
  kernel_size: [[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]]
  strides: [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]
  upsample_kernel_size: [[2,2,2], [2,2,2], [2,2,2], [2,2,2]]
  norm_name: [INSTANCE, {affine: true}]
  act_name: [leakyrelu, {inplace: true, negative_slope: 0.01}]
  res_block: false
```

## Training Configuration

```yaml
training:
  epochs: 200               # Total training epochs
  batch_size: 2             # Batch size (auto-optimized by planner)
  learning_rate: 0.01       # Initial learning rate (nnU-Net default)
  val_interval: 5           # Run validation every N epochs
  num_workers: 4            # DataLoader workers (auto-detected by planner)

  # Metric for checkpoint selection
  checkpoint_metric: DiceMetric  # Metric name from metrics section

  # Metrics to plot during training
  plot_metrics:
    - DiceMetric
    - SurfaceDiceMetric

  # Mixed precision training (FP16 for speedup on modern GPUs)
  mixed_precision: true     # true = FP16, false = FP32 (default: true)
```

## Optimizer Configuration

```yaml
optimizer:
  type: SGD                 # SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, Adamax, NAdam, RAdam

  # SGD-specific parameters
  weight_decay: 0.00003     # L2 regularization (nnU-Net default)
  momentum: 0.99            # Momentum coefficient (nnU-Net default)
  nesterov: true            # Enable Nesterov momentum

  # Adam/AdamW parameters (if using Adam-based optimizers)
  # betas: [0.9, 0.999]     # Beta coefficients
  # eps: 1.0e-8             # Epsilon for numerical stability
  # amsgrad: false          # Use AMSGrad variant
```

## Loss Configuration

```yaml
loss:
  # Loss function type
  # Available: DiceCELoss, DiceLoss, DiceFocalLoss, FocalLoss,
  #            GeneralizedDiceLoss, TverskyLoss, GeneralizedWassersteinDiceLoss, MaskedDiceLoss
  type: DiceCELoss

  # DiceCELoss parameters
  to_onehot_y: true         # Convert integer labels to one-hot
  softmax: true             # Apply softmax to predictions
  batch: true               # Batch dice (compute across entire batch)

  # Additional parameters (loss-specific)
  # include_background: false  # Exclude background in loss calculation
  # squared_pred: false        # Square predictions in Dice calculation
  # jaccard: false             # Use Jaccard index instead of Dice
  # smooth_nr: 1.0e-5          # Numerator smoothing
  # smooth_dr: 1.0e-5          # Denominator smoothing
  # lambda_dice: 1.0           # Dice loss weight (DiceCELoss)
  # lambda_ce: 1.0             # CE loss weight (DiceCELoss)
  # gamma: 2.0                 # Focal loss gamma (FocalLoss, DiceFocalLoss)
  # alpha: 0.5                 # Tversky alpha (TverskyLoss)
  # beta: 0.5                  # Tversky beta (TverskyLoss)
```

## Metrics Configuration

```yaml
metrics:
  # Dice Similarity Coefficient
  - type: DiceMetric
    include_background: false  # Exclude background class (class 0)
    reduction: mean_batch      # mean_batch, mean, sum, none
    num_classes: 3             # Total number of classes
    # get_not_nans: false      # Return only non-NaN values
    # ignore_empty: true       # Ignore empty predictions/labels

  # Surface Dice (boundary accuracy)
  - type: SurfaceDiceMetric
    include_background: false
    reduction: mean_batch
    class_thresholds:          # Distance tolerance (mm) per class
      - 2.0                    # Class 1 tolerance
      - 2.0                    # Class 2 tolerance
    # use_subvoxels: true      # Use subvoxel accuracy

  # Hausdorff Distance
  # - type: HausdorffDistanceMetric
  #   include_background: false
  #   percentile: 95           # 95th percentile (robust HD95)
  #   directed: false          # Symmetric distance

  # Surface Distance
  # - type: SurfaceDistanceMetric
  #   include_background: false
  #   symmetric: true          # Symmetric surface distance

  # Mean Intersection over Union
  # - type: MeanIoU
  #   include_background: false

  # Confusion Matrix
  # - type: ConfusionMatrixMetric
  #   include_background: false
  #   metric_name: [sensitivity, specificity, precision, accuracy]
```

## Transforms Configuration

Transforms are split into `common` (applied to all modes) and mode-specific (`train`, `val`, `test`).

### Common Transforms

```yaml
transforms:
  common:
    # Load images and labels (NIfTI for 3D, PNG/JPEG for 2D)
    - type: LoadImaged
      keys: [image, label]
      ensure_channel_first: true  # For 2D only; omit for 3D NIfTI

    # Intensity clipping (CT-specific, optional)
    # - type: ScaleIntensityRanged
    #   keys: [image]
    #   a_min: -79.0
    #   a_max: 304.0
    #   b_min: -79.0
    #   b_max: 304.0
    #   clip: true

    # Z-score normalization (nnU-Net default)
    - type: NormalizeIntensityd
      keys: [image]
      nonzero: false           # Compute stats over whole image
      channel_wise: false      # Single channel

    # Pad images smaller than patch size
    - type: SpatialPadd
      keys: [image, label]
      spatial_size: [128, 128, 128]
      mode: constant           # Zero-padding

    # Convert to PyTorch tensors
    - type: ToTensord
      keys: [image, label]
```

### Training Transforms (Augmentation)

```yaml
  train:
    # Random crop to patch size
    - type: RandSpatialCropd
      keys: [image, label]
      roi_size: [128, 128, 128]
      random_size: false

    # Rotation (nnU-Net: ±30°, p=0.2)
    - type: RandRotated
      keys: [image, label]
      prob: 0.2
      range_x: 0.5236          # ±30° in radians
      range_y: 0.5236
      range_z: 0.5236
      mode: [bilinear, nearest]
      padding_mode: border

    # Scaling (nnU-Net: 0.7-1.4, p=0.2)
    - type: RandZoomd
      keys: [image, label]
      prob: 0.2
      min_zoom: 0.7
      max_zoom: 1.4
      keep_size: true
      mode: [trilinear, nearest]

    # Random flips (nnU-Net: p=0.5 per axis)
    - type: RandFlipd
      keys: [image, label]
      prob: 0.5
      spatial_axis: 0          # X-axis

    - type: RandFlipd
      keys: [image, label]
      prob: 0.5
      spatial_axis: 1          # Y-axis

    - type: RandFlipd
      keys: [image, label]
      prob: 0.5
      spatial_axis: 2          # Z-axis

    # Gaussian noise (nnU-Net: std=0.1, p=0.1)
    - type: RandGaussianNoised
      keys: [image]
      prob: 0.1
      mean: 0.0
      std: 0.1

    # Gaussian blur (nnU-Net: sigma=0.5-1.0, p=0.2)
    - type: RandGaussianSmoothd
      keys: [image]
      prob: 0.2
      sigma_x: [0.5, 1.0]
      sigma_y: [0.5, 1.0]
      sigma_z: [0.5, 1.0]

    # Brightness (nnU-Net: 0.75-1.25, p=0.15)
    - type: RandScaleIntensityd
      keys: [image]
      factors: [0.75, 1.25]
      prob: 0.15

    # Contrast (nnU-Net: gamma=0.75-1.25, p=0.15)
    - type: RandAdjustContrastd
      keys: [image]
      prob: 0.15
      gamma: [0.75, 1.25]

    # Low-resolution simulation (nnU-Net: 0.5-1.0, p=0.25)
    - type: RandZoomd
      keys: [image, label]
      prob: 0.25
      min_zoom: 0.5
      max_zoom: 1.0
      mode: [bilinear, nearest]
      padding_mode: edge

    # Gamma transform (nnU-Net: p=0.1 + p=0.3)
    - type: RandHistogramShiftd
      keys: [image]
      prob: 0.1
      num_control_points: [5, 15]

    - type: RandHistogramShiftd
      keys: [image]
      prob: 0.3
      num_control_points: [5, 15]

    # Ensure consistent patch size after augmentation
    - type: ResizeWithPadOrCropd
      keys: [image, label]
      spatial_size: [128, 128, 128]
      mode: [constant, constant]
```

### Validation/Test Transforms

```yaml
  val:
    # Center crop (deterministic, no augmentation)
    - type: CenterSpatialCropd
      keys: [image, label]
      roi_size: [128, 128, 128]

  test:
    # Same as validation
    - type: CenterSpatialCropd
      keys: [image, label]
      roi_size: [128, 128, 128]
```

## Inference Configuration

```yaml
inference:
  # Batch size for inference
  batch_size: 1             # Typically 1 for full-resolution inference

  # Sliding window inference (optional, for large volumes)
  sliding_window:
    enabled: false          # true = use sliding window, false = full volume
    roi_size: [128, 128, 128]  # Window size (defaults to dataset.spatial_size)
    overlap: 0.5            # Window overlap: 0.0-0.99
    sw_batch_size: 4        # Number of windows per batch
    mode: gaussian          # gaussian or constant (blending mode)
    padding_mode: constant  # constant, edge, reflect, wrap
```

## Available Components

### Models
- `DynUNet` - MONAI DynUNet (nnU-Net architecture)
- `UNet` - MONAI UNet (simpler, faster alternative)

### Losses
- `DiceCELoss` - Combined Dice + Cross-Entropy (nnU-Net default)
- `DiceLoss` - Dice loss only
- `DiceFocalLoss` - Combined Dice + Focal loss
- `FocalLoss` - Focal loss for class imbalance
- `GeneralizedDiceLoss` - Generalized Dice with class weighting
- `TverskyLoss` - Tversky loss (weighted Dice variant)
- `GeneralizedWassersteinDiceLoss` - Wasserstein Dice
- `MaskedDiceLoss` - Dice with masking support

### Optimizers
- `SGD` - Stochastic Gradient Descent (nnU-Net default)
- `Adam` - Adaptive Moment Estimation
- `AdamW` - Adam with decoupled weight decay
- `RMSprop` - Root Mean Square Propagation
- `Adagrad` - Adaptive Gradient
- `Adadelta` - Adaptive Learning Rate
- `Adamax` - Adam with infinity norm
- `NAdam` - Adam with Nesterov momentum
- `RAdam` - Rectified Adam

### Metrics
- `DiceMetric` - Dice Similarity Coefficient
- `SurfaceDiceMetric` - Surface Dice (boundary accuracy)
- `HausdorffDistanceMetric` - Hausdorff Distance
- `SurfaceDistanceMetric` - Average Surface Distance
- `MeanIoU` - Mean Intersection over Union
- `ConfusionMatrixMetric` - Confusion matrix metrics

### Transforms
All MONAI transforms are supported. Common ones:
- **Loading**: `LoadImaged`, `EnsureChannelFirstd`
- **Spatial**: `RandSpatialCropd`, `CenterSpatialCropd`, `SpatialPadd`, `ResizeWithPadOrCropd`
- **Intensity**: `NormalizeIntensityd`, `ScaleIntensityRanged`, `RandScaleIntensityd`, `RandAdjustContrastd`
- **Augmentation**: `RandRotated`, `RandZoomd`, `RandFlipd`, `RandGaussianNoised`, `RandGaussianSmoothd`, `RandHistogramShiftd`
- **Utility**: `ToTensord`, `EnsureTyped`, `Spacingd`

See [MONAI transforms documentation](https://docs.monai.io/en/stable/transforms.html) for complete list.

**Implementation**: `src/config/`, `src/factory/`
