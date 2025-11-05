# Checkpointing

Automatic model checkpointing for training resumption and inference with comprehensive state preservation.

## Checkpoint Files

Checkpoints are automatically saved during training with full training state:

- `checkpoint_final_checkpoint.pt` - Latest checkpoint (updated every epoch)
- `best_model_model_key_metric=<value>.pt` - Best model based on validation metric (e.g., key_metric=0.8523)

**Location**: `results/<dataset_name>/fold_<N>/`

**Format**: PyTorch checkpoint dictionary (`.pt` files) containing:
- `model` - Model weights (state_dict)
- `optimizer` - Optimizer state (momentum, learning rate history)
- `lr_scheduler` - Learning rate scheduler state
- `scaler` - GradScaler state (for mixed precision training)
- `epoch` - Current epoch number
- `config_metadata` - Dataset name, fold, model type, total epochs

## Checkpoint Selection

**Validation mode** (fold 0-4):
- Saves best model based on `training.checkpoint_metric` (e.g., DiceMetric)
- Keeps top-1 checkpoint based on validation metric
- Filename includes metric value: `best_model_model_key_metric=0.8523.pt`
- Also saves latest checkpoint every epoch: `checkpoint_final_checkpoint.pt`

**All-data mode** (fold -1):
- No validation performed, saves latest checkpoint every epoch
- Saved as `checkpoint_final_checkpoint.pt`

## Resuming Training

### Automatic Resumption (Default Behavior)

Training **automatically resumes** if a checkpoint is detected - no flags needed!

```bash
# Automatically resumes if checkpoint exists, otherwise starts fresh
nnBench.train --config fold_0.yaml
```

**What happens:**
1. ✅ Automatically detects latest checkpoint (`checkpoint_final_checkpoint.pt` or best model)
2. ✅ Validates checkpoint matches current config (dataset, fold, model type)
3. ✅ Checks if training is already complete → exits gracefully if done
4. ✅ Loads full training state and continues from last epoch
5. ✅ If no checkpoint found → starts fresh training

**Fully restored state:**
- Model weights
- Optimizer state (momentum, learning rate history)
- Learning rate scheduler state (continues schedule, no reset)
- GradScaler state (maintains AMP stability)
- Epoch number (resumes from correct epoch)

### Training Completion Detection

If training is already complete, the system exits gracefully:

```bash
$ nnBench.train --config fold_0.yaml

Checkpoint detected: results/Dataset001/fold_0/checkpoint_final_checkpoint.pt
Training already complete! Checkpoint at epoch 100/100
No further training needed. Exiting.
```

### Force Fresh Start

To delete checkpoints and start from scratch:

```bash
# Using --fresh flag
nnBench.train --config fold_0.yaml --fresh

# Or --no-resume
nnBench.train --config fold_0.yaml --no-resume
```

### Legacy --continue Flag

The `--continue` flag is now **deprecated** (resumption is automatic):

```bash
# Still works but unnecessary (deprecated)
nnBench.train --config fold_0.yaml --continue
```

## Loading for Inference

Checkpoints contain full training state, but for inference you only need the model weights:

```python
import torch
from src.config import load_config
from src.factory import model_registry

# Load config
cfg = load_config("fold_0.yaml")
model = model_registry.build(cfg)

# Load checkpoint (new format with comprehensive state)
checkpoint_path = "results/Dataset001/fold_0/best_model_model_key_metric=0.8523.pt"
checkpoint = torch.load(checkpoint_path, map_location='cuda:0')

# Extract model weights from checkpoint dictionary
model.load_state_dict(checkpoint["model"])
model.eval()

# Use model for inference
with torch.no_grad():
    predictions = model(images)
```

**Note**: Checkpoint format changed from bare `state_dict` to dictionary with `"model"` key. Old checkpoints (before automatic resumption) can still be loaded directly.

**CLI**: `nnBench.inference` automatically uses the best checkpoint by default.

## Configuration

```yaml
training:
  checkpoint_metric: DiceMetric  # Metric for best model selection (validation mode)
  epochs: 100                    # Total epochs (used for completion detection)
```

## Checkpoint Validation Warnings

When resuming, the system validates checkpoint compatibility:

```bash
$ nnBench.train --config fold_1.yaml

Checkpoint detected: results/Dataset001/fold_0/checkpoint_final_checkpoint.pt
⚠️  WARNING: Checkpoint config validation issues detected:
  - Fold mismatch: checkpoint=0, current=1
Continuing with checkpoint load (proceed with caution)...
```

**Validation checks:**
- Dataset name match
- Fold number match
- Model type match

Warnings are shown but training continues (useful for debugging or intentional config changes).

## Implementation Details

**Custom Handler**: `ComprehensiveCheckpointHandler` in `src/engines/train/handlers.py`
- Extends MONAI/Ignite event system
- Saves all training state (not just model weights)
- Automatic checkpoint detection and validation
- Training completion detection

**Checkpoint Detection**: Priority order in `find_latest_checkpoint()`:
1. `checkpoint_final_checkpoint.pt` (most recent)
2. `best_model_model_key_metric=*.pt` (best validation)
3. `best_model_model_final_iteration=*.pt` (legacy format)

**Entry Point**: `src/engines/train/run.py` with automatic resumption logic
