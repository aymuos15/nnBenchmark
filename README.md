[![Actions status](https://github.com/aymuos15/nnBenchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/aymuos15/nnBenchmark/actions)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?style=flat-square&logo=python&logoColor=white)

# nnBenchmark

Config-driven 3D medical image segmentation using MONAI with a focus on reproducibility and benchmarking.

## Quick Start

### Installation

```
pip install -e .
```

## Usage

After setting up your data according to [nnUNet convention](https://github.com/MIC-DKFZ/nnUNet): see [docs/dataset_preparation.md](docs/dataset_preparation.md)

### Complete Workflow

**1. Auto-generate config (recommended):**
```bash
nnBench.plan --dataset datasets/Dataset001_Hippo
```
This analyzes your dataset and creates an optimal configuration automatically using nnU-Net heuristics.

For verbose logging or parallel fingerprinting:
```bash
nnBench.plan --dataset datasets/Dataset001_Hippo --num-workers 8 --verbose
```

Or create a config manually (see [docs/config_reference.md](docs/config_reference.md)).

**2. Train a model:**
```bash
nnBench.train --config configs/dataset001_hippo.yaml
```

**2b. Resume training (if interrupted):**
```bash
nnBench.train --config configs/dataset001_hippo.yaml --continue
```

**3. Test the model:**
```bash
nnBench.test --config configs/dataset001_hippo.yaml --use-val-split
```

**4. Generate plots:**
```bash
nnBench.plot --config configs/dataset001_hippo.yaml
```
Generates all plots in `results/<config_name>/plots/`

### Training on All Data (No Validation)

For final production models or small datasets, train on 100% of data without validation:

```yaml
# In your config YAML:
dataset:
  fold: -1  # Use all data for training, no validation
```

## Results

All results saved to `results/[config_name]/`:

```
results/[config_name]/
├── best_model.ckpt                   # Best model checkpoint (Lightning format)
├── last.ckpt                         # Latest checkpoint (for resumption)
├── training_history.json             # Training metrics and loss history
├── test_history.json                 # Test results for all metrics
├── train.log                         # Training logs
├── test.log                          # Test logs
├── plots/                            # All generated plots
│   ├── training_loss.png
│   ├── val_Dice.png
│   ├── val_SurfaceDice.png
│   ├── test_cls_wise_Dice_scores.png
│   └── test_cls_wise_SurfaceDice_scores.png
└── visualizations/                   # Validation case visualizations
    └── validation_epoch_*.png
```

### Metrics Tracked

By default, both **Dice** and **Surface Dice (NSD)** metrics are computed and:

- Overall mean across all classes
- Per-class scores for detailed analysis
- Complete statistics (mean, std, min, max) for each test case

## Documentation

In pipeline

## Development

### Setup

```bash
git clone git@github.com:aymuos15/nnBenchmark.git

cd nnBenchmark

pip install uv # https://github.com/astral-sh/uv

# Install all dev dependencies.
uv pip install -e ".[dev]"

# Install pre-commit hooks (recommended)
pre-commit install
```

### Testing

Run the test suite:

```bash
# Quick tests with uv
uv run pytest
```

### Pre-commit Checks

```bash
tox -e all                # Full validation (lint + type + tests)
```

## Todos

- [] Stress test in a Multi-GPU set up. 
