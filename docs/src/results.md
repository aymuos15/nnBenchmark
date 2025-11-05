# Results Storage

Results are saved in `nnBench_results/` (or `nnUNet_results/` for backward compatibility).

## Directory Structure

```
{dataset_name}/
├── configs/
│   ├── fold_0.yaml                    # Training configurations
│   └── custom_exp.yaml
└── {config_name}/                     # Results directory (created during training)
    ├── training_history.json          # Training/validation metrics per epoch
    ├── test_history.json              # Test/inference results (if nnBench.inference run)
    ├── best_model_model_key_metric=*.pt   # Best checkpoint (selected by validation metric)
    ├── best_model_model_final_iteration=*.pt  # Final epoch checkpoint (fallback)
    ├── train.log                      # Training logs
    ├── test.log                       # Inference logs
    ├── visualizations/                # Validation slice visualizations
    └── plots/                         # Generated plots
        ├── training_loss.png
        ├── val_{MetricName}.png
        └── test_cls_wise_{MetricName}_scores.png
```

## Key Files

### training_history.json
```json
{
  "epochs": [1, 2, 3, ...],
  "train_loss": [0.45, 0.38, ...],
  "val_epochs": [5, 10, 15, ...],
  "val_DiceMetric": [0.65, 0.72, ...],
  "val_DiceMetric_Class1": [0.60, 0.70, ...]
}
```
- Updated after every epoch (training loss) and validation (metrics)
- Omitted for `fold: -1` (all-data training): only `epochs` and `train_loss` present

### test_history.json
```json
{
  "config_name": "fold_0.yaml",
  "dataset_name": "Dataset001_Example",
  "fold": 0,
  "metrics": ["DiceMetric", "SurfaceDiceMetric"],
  "summary": {
    "DiceMetric": {
      "mean": 0.87,
      "std": 0.05,
      "per_class": { "Class1": {"mean": 0.85, ...}, ... }
    }
  },
  "per_sample_scores": { ... },
  "sample_names": [...]
}
```
- Created after `nnBench.inference`

### Checkpoints (MONAI Format)

**Best model checkpoints** (saved automatically during training):
- `best_model_model_key_metric=<metric_name>.pt`: Best model based on validation metric (e.g., `dice_score`)
- `best_model_model_final_iteration=<epoch_num>.pt`: Final epoch checkpoint (fallback if no validation)

Files are in PyTorch state dict format (`.pt`), loadable with `torch.load()`.

See [Checkpointing Guide](./reproducibility/checkpointing.md) for loading instructions.

## Generate Plots

```bash
nnBench.plot --config fold_0.yaml --dataset Dataset001_Example
```

Creates visualization plots in `plots/` directory from training and test results.

## Resume Training

```bash
nnBench.train --config fold_0.yaml --dataset Dataset001_Example --continue
```

Automatically loads best checkpoint, optimizer state, and appends to training history.
