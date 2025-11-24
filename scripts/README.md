# nnBenchmark Scripts

Utility scripts for analyzing and visualizing nnBenchmark results.

## Available Scripts

### `viz.py` - Visualize Sample Predictions

Visualize predictions for a specific sample with input image, ground truth label, and model prediction side-by-side.

#### Usage

```bash
# Basic usage
python scripts/viz.py --dataset Dataset001_Cellpose --config fold_0 --sample 007_0000.png

# Save to file instead of displaying
python scripts/viz.py --dataset Dataset001_Cellpose --config fold_0 --sample 007_0000.png --output viz_007.png

# Use CPU instead of GPU
python scripts/viz.py --dataset Dataset001_Cellpose --config fold_0 --sample 007_0000.png --device cpu

# Specify custom data directory
python scripts/viz.py --dataset Data --config fold_0 --sample 015_0000.png --data-dir /custom/data/path
```

#### Arguments

- `--dataset` (required): Dataset name (e.g., `Dataset001_Cellpose`, `Data`)
- `--config` (required): Config or fold name (e.g., `fold_0`)
- `--sample` (required): Sample filename (e.g., `007_0000.png`)
- `--output` (optional): Save visualization to file instead of displaying
- `--device` (optional): Device to use (`cuda` or `cpu`, default: `cuda`)
- `--data-dir` (optional): Base data directory (default: `~/CAI4Soumya/SegData`)

#### Features

- Automatically finds the best checkpoint for the config
- Loads image and label from the dataset
- Makes prediction using the trained model
- Displays input, ground truth, and prediction side-by-side
- Supports both 2D and 3D data
- Can save visualization to file or display interactively
- Shows progress and diagnostics

#### Example Output

```
📦 Dataset: Dataset001_Cellpose
⚙️  Config: fold_0
📄 Sample: 007_0000.png

🔍 Finding checkpoint...
✓ Checkpoint: /home/user/CAI4Soumya/SegData/nnUNet_results/Dataset001_Cellpose/fold_0/checkpoints/best_loss=0.5253.pt

🔍 Finding sample...
✓ Image: /home/user/CAI4Soumya/SegData/Dataset001_Cellpose/imagesTs/007_0000.png
✓ Label: /home/user/CAI4Soumya/SegData/Dataset001_Cellpose/labelsTs/007_0000.png

⚡ Loading model...
  Spatial dims: 2D
  Num classes: 2

🏗️  Building model...
✓ Model loaded on cuda

📥 Loading sample...
✓ Image shape: torch.Size([1, 256, 256])
✓ Label shape: torch.Size([256, 256])

🔮 Making prediction...
✓ Prediction shape: torch.Size([2, 256, 256])

🎨 Visualizing...
✓ Visualization saved to current_directory

✨ Done!
```

---

### `table.py` - Display Metrics in Formatted Tables

Display validation or test results in a beautifully formatted rich table.

#### Usage

```bash
# Display test results
python scripts/table.py results/history/test.json

# Display validation results
python scripts/table.py results/history/validation_epoch_010.json

# From the nnBenchmark root directory
python scripts/table.py ~/CAI4Soumya/SegData/nnUNet_results/Dataset001_Cellpose/fold_0/history/test.json
```

#### Output

The script displays:

1. **Configuration** - Dataset name, model path, config name
2. **Metrics Summary** - Overall statistics (mean, std, min, max) for all metrics
3. **Binned Metrics** - CC-based metrics broken down by instance size:
   - `all`: All instances
   - `0-2cc`: Small instances (0-2 pixels/voxels)
   - `2-10cc`: Medium instances (2-10 pixels/voxels)
   - `>10cc`: Large instances (>10 pixels/voxels)
4. **Per-Sample Summary** - Instance size breakdowns for the first 10 samples

#### Requirements

- `rich` library: `pip install rich`

#### Example Output

```
⚙️  Configuration
┌──────────────────────┬────────────────────────────────┐
│ Config Name          │ fold_0                         │
│ Dataset Name         │ Dataset001_Cellpose            │
└──────────────────────┴────────────────────────────────┘

📊 Metrics Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ Metric                    ┃   Mean ┃    Std ┃    Min ┃    Max ┃ Cases ┃
├───────────────────────────┼────────┼────────┼────────┼────────┼───────┤
│ DiceMetric                │ 0.7664 │ 0.1559 │ 0.2084 │ 0.9467 │    68 │
│ CCMetric_dice             │ 0.6737 │ 0.1896 │ 0.2244 │ 0.9467 │    68 │
└───────────────────────────┴────────┴────────┴────────┴────────┴───────┘

📦 Binned Metrics by Instance Size
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━┳━━━━┳━━━━┳━━━━━┳━━━━┓
┃ Metric              ┃ Bin     ┃Mean ┃Std ┃Min ┃Max  ┃Cnt │
├─────────────────────┼─────────┼─────┼────┼────┼─────┼────┤
│ CCMetric_dice       │ all     │0.67 │0.19│0.22│0.95 │1859│
│                     │ 0-2cc   │0.00 │0.00│0.00│0.00 │  4 │
│                     │ >10cc   │0.63 │0.25│0.00│0.96 │1845│
└─────────────────────┴─────────┴─────┴────┴────┴─────┴────┘
```

### Command Line Arguments

- `<path/to/json>`: Path to the JSON results file (required)

### Color Coding

- **Cyan**: Headers and "all" bins
- **Red**: 0-2cc (small instances)
- **Blue**: 2-10cc (medium instances)
- **Yellow**: >10cc (large instances)
- **Green**: Mean values
- **Magenta**: Sample names and bin labels

## Quick Reference

| Script | Purpose | Command |
|--------|---------|---------|
| `viz.py` | Visualize sample predictions | `python scripts/viz.py --dataset <name> --config <fold> --sample <file>` |
| `table.py` | Display metrics in tables | `python scripts/table.py <path/to/test.json>` |

## Future Scripts

This folder is intended to grow with additional analysis tools:
- Comparison across multiple runs
- Statistical significance testing
- Per-class metric breakdowns
- Training history visualization
- Batch prediction visualization
- Metrics comparison across folds
