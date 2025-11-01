# Factory Pattern

The factory pattern provides flexible component creation from configuration files.

## What is It?

A registry system that instantiates models, losses, optimizers, metrics, and transforms from YAML configuration without hardcoding implementations.

**Components**:
- **Models**: DynUNet, UNet, and custom architectures
- **Losses**: DiceCELoss, FocalLoss, TverskyLoss, and variants
- **Optimizers**: SGD, Adam, AdamW, and others
- **Metrics**: Dice, Surface Dice, Hausdorff Distance, IoU
- **Transforms**: MONAI augmentation pipelines

## Why Use It?

1. **Extensibility** - Add new models/losses/optimizers without code changes
2. **Configuration-driven** - All components instantiated from YAML
3. **Type-safe** - Registry validates component types at build time
4. **Native parameters** - Uses original MONAI/PyTorch parameter names (no translation)
5. **Multi-model support** - Switch models by changing one config field

## Configuration Example

```yaml
model:
  type: DynUNet
  spatial_dims: 3
  in_channels: 1
  out_channels: 3
  filters: [32, 64, 128, 256]
  # ... model-specific parameters

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

## Implementation

- **Registries**: `src/factory/models/registry.py`, `src/factory/losses/registry.py`, etc.
- **Builders**: `src/factory/builders.py` - High-level build functions
- **Weight Initialization**: Kaiming Normal (nnU-Net style)

All components use native MONAI/PyTorch parameter names from their original documentation.
