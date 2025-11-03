# Adding a Custom Loss Function

This guide shows how to add a new loss function to nnBenchmark and use it in your training configuration.

## Overview

nnBenchmark uses a factory pattern for losses. Any loss function can be registered and used via configuration files without code changes.

## Step 1: Create Your Loss Class

Create a PyTorch loss module in `src/factory/losses/`:

```python
# src/factory/losses/my_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomLoss(nn.Module):
    """Your custom loss implementation."""

    # Add type hints for class attributes
    param1: bool
    param2: float

    def __init__(
        self,
        param1: bool = False,
        param2: float = 1.0,
    ) -> None:
        """Initialize the loss."""
        super().__init__()
        self.param1 = param1  # type: ignore
        self.param2 = param2  # type: ignore

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            pred: Predictions of shape (B, C, H, W) or (B, C, H, W, D)
            target: Target labels of shape (B, H, W) or (B, H, W, D)

        Returns:
            Scalar loss tensor
        """
        # Your loss computation here
        loss = F.mse_loss(pred, target)
        return loss
```

**Important:**
- Inherit from `torch.nn.Module`
- Add type hints to class attributes
- Implement `forward(pred, target)` returning a scalar tensor
- Use `# type: ignore` on attribute assignments (PyTorch quirk)

## Step 2: Register Your Loss

Add your loss to `src/factory/losses/registry.py`:

```python
from src.factory.losses.my_loss import MyCustomLoss

class LossRegistry(BaseRegistry):
    def _register_default_losses(self) -> None:
        # ... existing registrations ...
        self.register("MyCustomLoss", MyCustomLoss)
```

## Step 3: Export Your Loss

Update `src/factory/losses/__init__.py`:

```python
from src.factory.losses.my_loss import MyCustomLoss

__all__ = ["LossRegistry", "loss_registry", "MyCustomLoss"]
```

## Step 4: Use in Configuration

Update your YAML config file (e.g., `fold_0.yaml`):

```yaml
loss:
  type: MyCustomLoss
  param1: true
  param2: 0.5
```

## Real Example: Connected Components Dice Loss

Here's how CCLoss was added:

**File:** `src/factory/losses/cc.py`

CCLoss is a region-aware Dice loss that evaluates predictions at the connected component level rather than globally. This is particularly useful for multi-instance segmentation tasks like cell segmentation.

```python
class CCLoss(nn.Module):
    """Connected Components Dice Loss for multi-instance segmentation."""

    def __init__(
        self,
        to_onehot_y: bool = False,
        softmax: bool = False,
        sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.to_onehot_y = to_onehot_y  # type: ignore
        self.softmax = softmax  # type: ignore
        self.sigmoid = sigmoid  # type: ignore

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Connected Components Dice Loss.

        Args:
            pred: Predictions shape (B, C, H, W) or (B, C, H, W, D)
            target: Target labels shape (B, H, W) or (B, H, W, D)
                   - class indices if to_onehot_y=False
                   - one-hot encoded if to_onehot_y=True

        Returns:
            Scalar loss tensor
        """
        # 1. Normalize target shape for spurious channel dimensions
        #    (When LoadImaged with ensure_channel_first=true creates (B, 1, H, W) targets)
        if target.dim() == pred.dim() and target.shape[1] == 1:
            target = target.squeeze(1)

        # 2. Convert to one-hot if needed
        if self.to_onehot_y and target.dim() != pred.dim():
            target = self._to_onehot(target, pred.shape[1])

        # 3. For each sample in batch:
        #    - Find connected components in ground truth
        #    - Compute Dice score per component
        #    - Average across components

        # Returns: 1 - mean(dice_scores)
```

**Key Features:**
- **Region-aware evaluation**: Computes Dice per connected component, not globally
- **Multi-instance support**: Ideal for cell/nuclei segmentation where each instance matters
- **Robust shape handling**: Automatically normalizes spurious channel dimensions from transforms
- **Configurable activations**: Supports softmax, sigmoid, or no activation

**Registered in:** `src/factory/losses/registry.py`
```python
self.register("CCLoss", CCLoss)
```

**Used in config:**
```yaml
loss:
  type: CCLoss
  sigmoid: true
  to_onehot_y: true
```

## Configuration Parameters

All parameters passed in the YAML config are forwarded to your `__init__` method:

```yaml
loss:
  type: MyCustomLoss
  param1: true      # Becomes: MyCustomLoss(param1=True)
  param2: 0.5       # Becomes: MyCustomLoss(param1=True, param2=0.5)
```

## Best Practices

1. **Always inherit from `nn.Module`** - Required for PyTorch compatibility
2. **Use proper type hints** - Helps with IDE autocomplete and type checking
3. **Add `# type: ignore` on attribute assignments** - Works around PyTorch's `__setattr__` quirks
4. **Return scalar tensors** - Loss should be a single number, not a tensor of losses
5. **Handle edge cases** - Empty batches, mismatched shapes, etc.
6. **Document parameters** - Add docstrings explaining what each parameter does
7. **Normalize target shapes** - Like MONAI losses, handle spurious channel dimensions from transforms:
   ```python
   # Handle case where LoadImaged with ensure_channel_first=true adds unwanted channels
   if target.dim() == pred.dim() and target.shape[1] == 1:
       target = target.squeeze(1)
   ```
   This makes your loss robust to different data pipeline configurations.

## Testing Your Loss

Before using in training:

```python
from src.factory.losses import loss_registry

# Build from config
config = {
    "type": "MyCustomLoss",
    "param1": True,
    "param2": 0.5,
}
loss_fn = loss_registry.build(config)

# Test forward pass
pred = torch.randn(2, 3, 64, 64, requires_grad=True)
target = torch.randint(0, 3, (2, 64, 64))

loss = loss_fn(pred, target)
loss.backward()  # Should work without errors

print(f"Loss: {loss.item():.4f}")
print(f"Gradients: {pred.grad is not None}")
```

## Switching Between Losses

**Before:**
```yaml
loss:
  type: DiceCELoss
  to_onehot_y: true
  softmax: true
```

**After:**
```yaml
loss:
  type: CCLoss
  to_onehot_y: true
  sigmoid: true
```

No code changes needed - just update the YAML config!