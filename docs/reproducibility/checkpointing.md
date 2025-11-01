# Checkpointing

Automatic model checkpointing for training resumption and inference.

## Checkpoint Files

Two checkpoint files are automatically created during training:

- `best_model.ckpt` - Best model based on validation metric (or last epoch if training on all data)
- `last.ckpt` - Most recent checkpoint for resuming interrupted training

**Location**: `results/<dataset_name>/fold_<N>/`

## Checkpoint Selection

**Validation mode** (fold 0-4):
- Saves best model based on `training.checkpoint_metric` (e.g., DiceMetric)
- Monitors validation metric and keeps top-1 checkpoint

**All-data mode** (fold -1):
- Saves checkpoint every epoch (no validation)
- `best_model.ckpt` contains the final epoch

## Resuming Training

```bash
# Resume from last checkpoint
nnBench.train --config fold_0.yaml --continue

# Or with -c flag
nnBench.train --config fold_0.yaml -c
```

Resumes from `last.ckpt` with preserved:
- Model weights
- Optimizer state
- Training epoch
- Learning rate schedule

## Loading for Inference

```python
from src.lightning import SegmentationModule

# Load best model checkpoint
lit_module = SegmentationModule.load_from_checkpoint(
    "results/Dataset001/fold_0/best_model.ckpt",
    cfg=cfg,
    device=device,
    map_location=device
)
model = lit_module.model
```

**CLI**: `nnBench.inference` automatically uses `best_model.ckpt` by default.

## Configuration

```yaml
training:
  checkpoint_metric: DiceMetric  # Metric for best model selection
```

**Implementation**: PyTorch Lightning `ModelCheckpoint` callback in `src/train/run.py:169-191`
