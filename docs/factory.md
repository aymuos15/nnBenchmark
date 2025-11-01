# Factory Pattern Architecture

This document describes the factory pattern implementation in nnBenchmark, which provides a flexible and extensible way to create models, losses, and optimizers from configuration files.

## Overview

The factory pattern allows nnBenchmark to support multiple model architectures, loss functions, and optimizers without hardcoding specific implementations. Each component type (models, losses, optimizers) has its own registry that manages available implementations.

**Key Benefits:**
- **Extensibility**: Easy to add new models, losses, or optimizers
- **Configuration-driven**: All components instantiated from YAML config
- **Type safety**: Registry validates component types at build time
- **Native parameters**: Each component uses its original MONAI/PyTorch parameter names
- **Discoverability**: List all available components programmatically

## Architecture

### Directory Structure

```
src/factory/
├── __init__.py              # Public API exports
├── models/
│   ├── __init__.py
│   └── registry.py         # ModelRegistry class
├── losses/
│   ├── __init__.py
│   └── registry.py         # LossRegistry class
└── optimizers/
    ├── __init__.py
    └── registry.py         # OptimizerRegistry class
```

### Component Flow

```
Configuration File (YAML)
        ↓
   load_config()
        ↓
   builders.py (build_model, build_loss, build_optimizer)
        ↓
   Factory Registries (model_registry, loss_registry, optimizer_registry)
        ↓
   Component Instantiation (MONAI/PyTorch)
```

## Registries

### ModelRegistry

**Purpose**: Creates segmentation models from configuration with proper weight initialization.

**Registered Models**:
- `DynUNet` - Dynamic UNet matching nnU-Net PlainConvUNet architecture (default)
- `UNet` - MONAI UNet with simpler architecture for faster training

**Key Features**:
- Kaiming Normal weight initialization (nnU-Net style)
- Automatic device placement
- DynUNet-specific parameter handling (tuple conversion)
- Excludes non-model parameters (e.g., `ds_weights`)

**Usage**:
```python
from src.factory import model_registry

# List available models
available = model_registry.list_available()
print(available)  # ['AttentionUnet', 'DynUNet', 'UNet', ...]

# Build model from config
config = {
    "type": "UNet",
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 3,
    "channels": [16, 32, 64],
    "strides": [2, 2],
    "num_res_units": 0
}
model = model_registry.build(config, device)
```

### LossRegistry

**Purpose**: Creates loss functions from configuration using native MONAI parameters.

**Registered Losses**:
- `DiceCELoss` - Combined Dice + Cross-Entropy (default)
- `DiceLoss` - Sørensen-Dice coefficient
- `DiceFocalLoss` - Dice + Focal loss combination
- `FocalLoss` - Focal loss for class imbalance
- `GeneralizedDiceLoss` - Generalized Dice for multi-class
- `TverskyLoss` - Tversky index-based loss
- `GeneralizedWassersteinDiceLoss` - Wasserstein distance-based
- `MaskedDiceLoss` - Dice with masking support

**Usage**:
```python
from src.factory import loss_registry

# Build loss from config
config = {
    "type": "DiceCELoss",
    "to_onehot_y": True,
    "softmax": True,
    "batch": True
}
loss_fn = loss_registry.build(config)
```

### OptimizerRegistry

**Purpose**: Creates PyTorch optimizers from configuration.

**Registered Optimizers**:
- `SGD` - Stochastic Gradient Descent (nnU-Net default)
- `Adam` - Adaptive Moment Estimation
- `AdamW` - Adam with weight decay fix
- `RMSprop` - RMSProp optimizer
- `Adagrad`, `Adadelta`, `Adamax` - Adaptive learning rate variants
- `NAdam`, `RAdam` - Adam improvements

**Usage**:
```python
from src.factory import optimizer_registry

# Build optimizer from config
config = {
    "type": "SGD",
    "weight_decay": 0.00003,
    "momentum": 0.99,
    "nesterov": True
}
optimizer = optimizer_registry.build(config, model.parameters(), learning_rate=0.01)
```

### MetricRegistry

**Purpose**: Creates evaluation metrics from configuration and returns them as a dictionary.

**Registered Metrics**:
- `DiceMetric` - Dice Similarity Coefficient (overlap-based)
- `SurfaceDiceMetric` - Surface Dice (boundary accuracy)
- `HausdorffDistanceMetric` - Hausdorff distance
- `SurfaceDistanceMetric` - Surface distance metrics
- `MeanIoU` - Mean Intersection over Union
- `ConfusionMatrixMetric` - Confusion matrix computation

**Key Feature**: Returns dictionary mapping full metric type names to instances.

**Usage**:
```python
from src.factory import metric_registry

# Build metrics from config
config = {
    "metrics": [
        {
            "type": "DiceMetric",
            "include_background": False,
            "reduction": "mean_batch",
            "num_classes": 3
        },
        {
            "type": "SurfaceDiceMetric",
            "include_background": False,
            "reduction": "mean_batch",
            "class_thresholds": [2.0, 2.0]
        }
    ]
}
metrics_dict = metric_registry.build(config)
# Returns: {"DiceMetric": DiceMetric(...), "SurfaceDiceMetric": SurfaceDiceMetric(...)}
```

**Important**: Metric names use **full type names** (e.g., `"DiceMetric"` not `"Dice"`). This is a breaking change from previous versions.

### TransformRegistry

**Purpose**: Creates MONAI transform pipelines from configuration with mode-specific composition.

**Transform Composition**:
- **Common transforms** (applied to all modes)
- **Mode-specific transforms** (train/val/test) appended after common
- Returns `monai.transforms.Compose` pipeline

**Simplified Logic**: Mode-specific transforms are appended after common transforms (no special insertion logic).

**Usage**:
```python
from src.factory import transform_registry

# Build transform pipeline
config = {
    "transforms": {
        "common": [
            {"type": "LoadImaged", "keys": ["image", "label"]},
            {"type": "NormalizeIntensityd", "keys": ["image"]},
            {"type": "ToTensord", "keys": ["image", "label"]}
        ],
        "train": [
            {"type": "RandSpatialCropd", "keys": ["image", "label"], "roi_size": [64, 64, 64]},
            {"type": "RandFlipd", "keys": ["image", "label"], "prob": 0.5, "spatial_axis": 0}
        ],
        "val": [
            {"type": "CenterSpatialCropd", "keys": ["image", "label"], "roi_size": [64, 64, 64]}
        ]
    }
}

# Build mode-specific pipelines
train_pipeline = transform_registry.build(config, mode="train")
val_pipeline = transform_registry.build(config, mode="val")
```

**Breaking Change**: Mode-specific transforms are now simply appended after common transforms. Previous version had special ToTensord insertion logic.

## Configuration Format

All factories use **native MONAI/PyTorch parameter names** - no translation layer. This means you use the exact parameter names from the original library documentation.

### Nested Config Structure (Multi-Model Support)

The planning pipeline now generates configs with **nested model-specific sections**, allowing configs to include parameters for multiple model architectures simultaneously. This makes it easy to switch between models by changing a single field.

**Structure:**
```yaml
model:
  type: DynUNet  # Change to 'UNet' to switch models

  # Shared parameters (common to all models)
  spatial_dims: 3
  in_channels: 1
  out_channels: 3
  deep_supervision: true
  ds_weights: [1.0, 0.5, 0.25]

  # DynUNet-specific parameters
  DynUNet:
    filters: [32, 64, 128, 256]
    kernel_size: [[3,3,3], [3,3,3], [3,3,3], [3,3,3]]
    strides: [[1,1,1], [2,2,2], [2,2,2], [2,2,2]]
    upsample_kernel_size: [[2,2,2], [2,2,2], [2,2,2]]
    norm_name: [INSTANCE, {affine: true}]
    act_name: [leakyrelu, {inplace: true, negative_slope: 0.01}]
    res_block: false

  # UNet-specific parameters (auto-derived from DynUNet)
  UNet:
    channels: [32, 64, 128, 256]  # Matches DynUNet filters
    strides: [2, 2, 2]  # Simplified from DynUNet
    num_res_units: 2
```

**Benefits:**
- **Easy model switching**: Change one field (`type`) to try different architectures
- **Architecture equivalence**: UNet params derived from DynUNet for fair comparison
- **Single planning**: One `nnBench.plan` command generates configs for both models
- **Backward compatible**: Flat configs (all params at top level) still work

### Model Configuration Examples

#### UNet (Lighter Architecture)

```yaml
model:
  type: UNet
  spatial_dims: 3
  in_channels: 1
  out_channels: 3
  channels: [16, 32, 64]      # Feature channels per level
  strides: [2, 2]              # Downsampling between levels
  num_res_units: 0             # No residual connections
```

**Use Cases**:
- Quick prototyping and experimentation
- Resource-constrained environments
- Baseline comparisons
- Smaller datasets where DynUNet might overfit

#### DynUNet (nnU-Net Architecture)

```yaml
model:
  type: DynUNet
  spatial_dims: 3
  in_channels: 1
  out_channels: 3
  filters: [32, 64, 128, 256]
  kernel_size:
    - [3, 3, 3]
    - [3, 3, 3]
    - [3, 3, 3]
    - [3, 3, 3]
  strides:
    - [1, 1, 1]  # Full resolution at first level
    - [2, 2, 2]
    - [2, 2, 2]
    - [2, 2, 2]
  upsample_kernel_size:
    - [2, 2, 2]
    - [2, 2, 2]
    - [2, 2, 2]
  norm_name: [INSTANCE, {affine: true}]
  act_name: [leakyrelu, {inplace: true, negative_slope: 0.01}]
  res_block: false
  deep_supervision: true
  deep_supr_num: 1
  ds_weights: [1.0, 0.5, 0.25, 0.125]
```

**Use Cases**:
- State-of-the-art segmentation performance
- Medical imaging challenges
- Exact nnU-Net replication
- Deep supervision training

### Loss Configuration Examples

#### DiceCELoss (Default)

```yaml
loss:
  type: DiceCELoss
  to_onehot_y: true    # Convert integer labels to one-hot
  softmax: true        # Apply softmax to predictions
  batch: true          # Batch Dice for stable gradients
```

#### DiceLoss (Pure Dice)

```yaml
loss:
  type: DiceLoss
  to_onehot_y: true
  softmax: true
  squared_pred: false
```

#### FocalLoss (Class Imbalance)

```yaml
loss:
  type: FocalLoss
  to_onehot_y: true
  gamma: 2.0           # Focusing parameter
  alpha: 0.25          # Class weight
```

### Optimizer Configuration Examples

#### SGD (nnU-Net Default)

```yaml
optimizer:
  type: SGD
  weight_decay: 0.00003
  momentum: 0.99
  nesterov: true

training:
  learning_rate: 0.01
```

#### Adam

```yaml
optimizer:
  type: Adam
  weight_decay: 0.0001
  betas: [0.9, 0.999]
  eps: 1e-8

training:
  learning_rate: 0.001
```

#### AdamW

```yaml
optimizer:
  type: AdamW
  weight_decay: 0.01
  betas: [0.9, 0.999]

training:
  learning_rate: 0.001
```

### Metrics Configuration Examples

#### DiceMetric + SurfaceDiceMetric

```yaml
metrics:
  - type: DiceMetric
    include_background: false
    reduction: mean_batch
    num_classes: 3

  - type: SurfaceDiceMetric
    include_background: false
    reduction: mean_batch
    class_thresholds:
      - 2.0  # Class 1: 2mm tolerance
      - 2.0  # Class 2: 2mm tolerance

training:
  checkpoint_metric: DiceMetric  # Use full name
  plot_metrics:
    - DiceMetric
    - SurfaceDiceMetric
```

#### Multiple Evaluation Metrics

```yaml
metrics:
  - type: DiceMetric
    include_background: false
    reduction: mean_batch
    num_classes: 3

  - type: HausdorffDistanceMetric
    include_background: false
    reduction: mean_batch
    percentile: 95

  - type: MeanIoU
    include_background: false
    reduction: mean_batch

training:
  checkpoint_metric: DiceMetric
  plot_metrics:
    - DiceMetric
    - MeanIoU
```

### Transform Configuration Examples

#### 3D Medical Imaging

```yaml
transforms:
  common:
    - type: LoadImaged
      keys: [image, label]

    - type: ScaleIntensityRanged
      keys: [image]
      a_min: -79.0
      a_max: 304.0
      b_min: -79.0
      b_max: 304.0
      clip: true

    - type: NormalizeIntensityd
      keys: [image]
      nonzero: false
      channel_wise: false

    - type: SpatialPadd
      keys: [image, label]
      spatial_size: [64, 128, 128]
      mode: constant

    - type: ToTensord
      keys: [image, label]

  train:
    - type: RandSpatialCropd
      keys: [image, label]
      roi_size: [64, 128, 128]
      random_size: false

    - type: RandFlipd
      keys: [image, label]
      prob: 0.5
      spatial_axis: 0

    - type: RandRotated
      keys: [image, label]
      prob: 0.2
      range_x: 0.5236  # ±30° in radians
      mode: [bilinear, nearest]

  val:
    - type: CenterSpatialCropd
      keys: [image, label]
      roi_size: [64, 128, 128]

  test:
    - type: CenterSpatialCropd
      keys: [image, label]
      roi_size: [64, 128, 128]
```

## Model Comparison

### UNet vs DynUNet

| Feature | UNet | DynUNet |
|---------|------|---------|
| **Parameters** | Fewer | More |
| **Speed** | Faster | Slower |
| **Memory** | Lower | Higher |
| **Configuration** | Simple | Detailed |
| **Deep Supervision** | Internal (not exposed) | ✅ Exposed |
| **nnU-Net Alignment** | ❌ | ✅ |
| **Recommended For** | Quick experiments | State-of-the-art results |
| **Typical Use** | Prototyping, baselines | Final models, competitions |

### When to Use Each Model

**Use UNet when:**
- Prototyping new ideas quickly
- Limited GPU memory (<8GB VRAM)
- Training on small datasets (<50 cases)
- Need fast iteration cycles
- Establishing baseline performance

**Use DynUNet when:**
- Aiming for state-of-the-art results
- Participating in challenges
- Need exact nnU-Net replication
- Have sufficient compute resources
- Dataset size >100 cases

**Note**: Only DynUNet and UNet are currently registered. Other MONAI models can be added using `model_registry.register()`.

## Adding Custom Components

### Adding a Custom Model

```python
from src.factory import model_registry
from monai.networks.nets import CustomUNet  # Your custom model

# Register the model
model_registry.register("CustomUNet", CustomUNet)

# Now use it in config
config = {
    "type": "CustomUNet",
    # ... custom parameters
}
model = model_registry.build(config, device)
```

### Adding a Custom Loss

```python
from src.factory import loss_registry
from my_losses import CustomLoss

# Register the loss
loss_registry.register("CustomLoss", CustomLoss)

# Use in config
config = {
    "type": "CustomLoss",
    "param1": value1,
    "param2": value2
}
loss_fn = loss_registry.build(config)
```

### Adding a Custom Optimizer

```python
from src.factory import optimizer_registry
from custom_optimizers import CustomOptim

# Register the optimizer
optimizer_registry.register("CustomOptim", CustomOptim)

# Use in config
config = {
    "type": "CustomOptim",
    "param1": value1
}
optimizer = optimizer_registry.build(config, model.parameters(), lr=0.01)
```

## Implementation Details

### Weight Initialization

All models use **Kaiming Normal initialization** optimized for LeakyReLU activation:

```python
def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
```

This matches nnU-Net v2.4.1 initialization strategy for deep networks.

### DynUNet Special Handling

DynUNet requires some parameters as tuples (e.g., `norm_name`, `act_name`). The factory automatically converts these from lists:

```python
# In config (YAML - uses lists)
norm_name: [INSTANCE, {affine: true}]

# Converted to (internally - uses tuples)
norm_name: ("INSTANCE", {"affine": True})
```

The factory also adds nnU-Net compatibility defaults:
- `trans_bias: True` - Enable bias in transpose convolutions

### Deep Supervision Support

Deep supervision is configured at the model level but weights are stored separately:

```yaml
model:
  type: DynUNet
  deep_supervision: true
  deep_supr_num: 1
  ds_weights: [1.0, 0.5, 0.25, 0.125]  # Not passed to model constructor
```

The `ds_weights` parameter is excluded from model instantiation and used later in loss computation.

## Backward Compatibility

The factory pattern maintains **full backward compatibility** with existing code:

- Existing `builders.py` functions (`build_model`, `build_loss`, `build_optimizer`) continue to work
- Old DynUNet configs work without modification
- The builders now delegate to factory registries internally
- All existing tests pass without changes

## Migration Guide

### Updating Existing Configs

**Before (hardcoded DynUNet only):**
```yaml
model:
  type: DynUNet  # Only DynUNet was supported
  # ... DynUNet parameters
```

**After (any registered model):**
```yaml
model:
  type: UNet  # or DynUNet
  # ... model-specific parameters
```

### Using Multiple Models

With nested configs, you can switch models by changing a single field:

```yaml
# Try DynUNet (default, state-of-the-art)
model:
  type: DynUNet
  # ... rest of config

# Try UNet (faster alternative)
model:
  type: UNet
  # ... rest of config
```

Alternatively, maintain separate config files:
```
configs/
├── dataset001_hippo_dynunet.yaml   # State-of-the-art
└── dataset001_hippo_unet.yaml      # Baseline / faster training
```

## Testing

All factory functionality is tested through:

1. **Registry Tests**: Verify models/losses/optimizers are registered
2. **Building Tests**: Test instantiation from config
3. **Forward Pass Tests**: Verify models produce correct output shapes
4. **Integration Tests**: Test builder functions work with registries

Run tests with:
```bash
python3 -m pytest tests/test_config.py
python3 -m pytest tests/test_nnunet_exact_match.py
```

## Troubleshooting

### Error: "Model type 'X' is not registered"

**Solution**: Check available models:
```python
from src.factory import model_registry
print(model_registry.list_available())
```

### Error: "TypeError: __init__() got an unexpected keyword argument 'Y'"

**Solution**: Check that you're using native MONAI parameter names from the documentation. The factory does not translate parameter names.

### Error: "KeyError: 'ds_weights'"

**Solution**: If using deep supervision, ensure `ds_weights` is specified in the model config:
```yaml
model:
  deep_supervision: true
  ds_weights: [1.0, 0.5, 0.25]  # Required when deep_supervision=true
```

## References

- [MONAI Models Documentation](https://docs.monai.io/en/stable/networks.html)
- [nnU-Net Paper](https://www.nature.com/articles/s41592-020-01008-z)
- [nnU-Net v2 Documentation](https://github.com/MIC-DKFZ/nnUNet)
- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)

## See Also

- [hyperparameters.md](hyperparameters.md) - nnU-Net alignment details
- [deep_supervision_dims.md](deep_supervision_dims.md) - Deep supervision explanation
- [configs/examples/](../configs/examples/) - Example configurations
