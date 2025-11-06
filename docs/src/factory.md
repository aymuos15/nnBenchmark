# Factory Pattern

The factory pattern provides flexible component creation from configuration files.

## What is It?

A registry system that instantiates models, losses, optimizers, metrics, and transforms from YAML configuration without hardcoding implementations.

**Components**:
- **Models**: DynUNet, UNet, KiUNet2D, KiUNet3D, and custom architectures
- **Losses**: DiceCELoss, FocalLoss, TverskyLoss, and variants
- **Optimizers**: SGD, Adam, AdamW, and others
- **Metrics**: Dice, Surface Dice, Hausdorff Distance, IoU, Connected Components Metric (CCMetric)
- **Transforms**: MONAI augmentation pipelines

## Why Use It?

1. **Extensibility** - Add new models/losses/optimizers without code changes
2. **Configuration-driven** - All components instantiated from YAML
3. **Type-safe** - Registry validates component types at build time
4. **Native parameters** - Uses original MONAI/PyTorch parameter names (no translation)
5. **Multi-model support** - Switch models by changing one config field

## Configuration Examples

### DynUNet (nnU-Net Architecture)

```yaml
model:
  type: DynUNet
  spatial_dims: 3
  in_channels: 1
  out_channels: 3
  filters: [32, 64, 128, 256]
  kernel_size: [[3,3,3], [3,3,3], ...]
  strides: [[1,1,1], [2,2,2], ...]
  # ... more DynUNet-specific parameters
```

### KiU-Net (Dual-Branch Architecture)

```yaml
model:
  type: KiUNet2D  # or KiUNet3D for 3D data
  spatial_dims: 2
  in_channels: 1
  out_channels: 2
  deep_supervision: true
  KiUNet2D:
    features: [32, 64, 128, 256]  # Channels at each encoder level
    norm_name: instance  # 'instance', 'batch', or 'group'
    act_name: relu  # 'relu', 'leakyrelu', or 'prelu'
```

### Loss and Optimizer

```yaml
loss:
  type: DiceCELoss
  to_onehot_y: true
  softmax: true

optimizer:
  type: SGD
  weight_decay: 0.00003
  momentum: 0.99
```

## Adding Custom Components

```python
from src.factory import model_registry, loss_registry

# Register custom component
model_registry.register("CustomModel", CustomModelClass)
loss_registry.register("CustomLoss", CustomLossClass)

# Now use in config
config = {"type": "CustomModel", "param1": value1}
```

## Model Architectures

### DynUNet
Exact replication of nnU-Net's PlainConvUNet architecture using MONAI's implementation. Features:
- Configurable encoder/decoder depth
- InstanceNorm + LeakyReLU (nnU-Net defaults)
- Deep supervision support
- Residual blocks (optional)

### UNet
MONAI's standard U-Net implementation. Simpler and faster than DynUNet but less configurable.

### KiU-Net (Dual-Branch Architecture)
Novel dual-branch architecture that combines over-complete and under-complete paths:

**Architecture**:
- **U-Net Branch**: Standard encoder-decoder with max pooling (under-complete)
- **Ki-Net Branch**: Encoder-decoder with upsampling (over-complete)
- **Feature Fusion**: Branches fused at output resolution for final prediction
- **Deep Supervision**: Optional auxiliary outputs from intermediate decoder levels

**Key Features**:
- Captures both coarse (U-Net) and fine (Ki-Net) scale features
- Configurable feature channels, normalization, and activation
- Both 2D (KiUNet2D) and 3D (KiUNet3D) variants
- MONAI-style configuration and integration

**Reference**: Valanarasu et al. "KiU-Net: Overcomplete Convolutional Architectures for Biomedical Image and Volumetric Segmentation." IEEE TMI, 2021.

**Example Configs**: See `docs/datasets/Dataset001_Cellpose/KiUNet2D_fold_0.yaml` and `KiUNet3D_example.yaml`

## Metrics

### Available Metrics

- `DiceMetric` - Dice Similarity Coefficient
- `SurfaceDiceMetric` - Surface Dice (boundary accuracy)
- `HausdorffDistanceMetric` - Hausdorff Distance
- `SurfaceDistanceMetric` - Average Surface Distance
- `MeanIoU` - Mean Intersection over Union
- `ConfusionMatrixMetric` - Confusion matrix metrics
- `CCMetric` - Connected Components Metric for multi-instance segmentation
  - Evaluates predictions at region level using connected components
  - Supports dice, surface dice, or combined metrics per region
  - Parameters: `metric_type`, `class_thresholds`, `distance_metric`
  - Ideal for cell/nuclei segmentation and other multi-instance tasks

See [Configuration Reference - Metrics](config.md#metrics-configuration) for complete parameter documentation and examples.

## Implementation

- **Registries**: `src/factory/models/registry.py`, `src/factory/losses/registry.py`, etc.
- **KiU-Net Implementation**: `src/factory/models/kiunet.py`
- **Builders**: `src/factory/builders.py` - High-level build functions
- **Weight Initialization**: Kaiming Normal (nnU-Net style)

All components use native MONAI/PyTorch parameter names from their original documentation.
