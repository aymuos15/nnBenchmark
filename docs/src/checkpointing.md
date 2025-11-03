# Checkpointing

Automatic model checkpointing for training resumption and inference using MONAI SupervisedTrainer with Ignite.

## Checkpoint Files

Checkpoints are automatically saved during training using MONAI's `CheckpointSaver`:

- `best_model_model_key_metric=<metric_name>.pt` - Best model based on validation metric (e.g., key_metric=dice_score)
- `best_model_model_final_iteration=<epoch>.pt` - Final epoch checkpoint (fallback if no validation)

**Location**: `results/<dataset_name>/fold_<N>/`

**Format**: PyTorch model state dict (`.pt` files), compatible with MONAI

## Checkpoint Selection

**Validation mode** (fold 0-4):
- Saves best model based on `training.checkpoint_metric` (e.g., DiceMetric)
- MONAI CheckpointSaver keeps top-1 checkpoint based on validation metric
- Filename includes metric name: `best_model_model_key_metric=dice_score.pt`

**All-data mode** (fold -1):
- No validation performed, uses final epoch checkpoint
- Saved as `best_model_model_final_iteration=<epoch>.pt`

## Resuming Training

```bash
# Resume from last checkpoint
nnBench.train --config fold_0.yaml --continue

# Or with -c flag
nnBench.train --config fold_0.yaml -c
```

Resumes from the latest checkpoint with preserved:
- Model weights
- Optimizer state
- Training epoch
- Learning rate schedule

## Loading for Inference

```python
import torch
from src.config import load_config
from src.factory import model_registry

# Load config
cfg = load_config("fold_0.yaml")
model = model_registry.build(cfg)

# Load best checkpoint
checkpoint_path = "results/Dataset001/fold_0/best_model_model_key_metric=dice_score.pt"
checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
model.load_state_dict(checkpoint)
model.eval()

# Use model for inference
with torch.no_grad():
    predictions = model(images)
```

**CLI**: `nnBench.inference` automatically uses the best checkpoint by default.

## Configuration

```yaml
training:
  checkpoint_metric: DiceMetric  # Metric for best model selection
```

## MONAI Checkpoint Naming Convention

MONAI's CheckpointSaver uses descriptive filenames:
- `best_model_model_key_metric=<metric>` - Best model on specified metric
- `best_model_model_final_iteration=<N>` - Final iteration/epoch
- Metric names come from `training.checkpoint_metric` config value

**Implementation**: MONAI `CheckpointSaver` in `src/engines/train/run.py` with Ignite event handlers
