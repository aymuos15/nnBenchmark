[![Actions status](https://github.com/aymuos15/nnBenchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/aymuos15/nnBenchmark/actions)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?style=flat-square&logo=python&logoColor=white)

# nnBenchmark

Config-driven medical image segmentation with a focus on reproducibility and benchmarking.

## Quick Start

### Installation

> **Note**: Currently in alpha testing on TestPyPI. Full PyPI release coming soon.

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nnbenchmark
```

For users without TestPyPI access, install from source:
```bash
git clone https://github.com/aymuos15/nnBenchmark.git
cd nnBenchmark
pip install -e .
```

### Setup Environment Variables

> **Note**: We currently support both `nnBench_*` and `nnUNet_*` environment variables. If both are set, `nnBench_*` takes precedence.

nnBenchmark uses three environment variables to organize data:

```bash
export nnBench_raw="/path/to/nnBench_raw"
export nnBench_preprocessed="/path/to/nnBench_preprocessed"
export nnBench_results="/path/to/nnBench_results"
```

Add these to your `~/.bashrc` or `~/.zshrc` for persistence.

### Prepare Your Dataset

Organize your dataset in the following way:

```
nnBench_raw/Dataset001_YourDataset/
├── dataset.json          # Dataset metadata
├── imagesTr/             # Training images: case_001_0000.nii.gz, case_002_0000.nii.gz, ...
└── labelsTr/             # Training labels: case_001.nii.gz, case_002.nii.gz, ...
```

See [docs/src/dataset_format.md](docs/src/dataset_format.md) for details.

### Basic Workflow

nnBenchmark uses a 4-step workflow:

```bash
# 1. Plan - Generate optimal configuration
nnBench.plan --dataset Dataset001_YourDataset --verbose
# Output: nnBench_results/Dataset001_YourDataset/fold_0/fold_0.yaml
#         nnBench_preprocessed/Dataset001_YourDataset/splits.json

# 2. Train - Train the model
nnBench.train --config fold_0.yaml --dataset Dataset001_YourDataset
# To resume: add --continue flag
# Output: nnBench_results/Dataset001_YourDataset/fold_0/checkpoints/
#         nnBench_results/Dataset001_YourDataset/fold_0/training_history.json

# 3. Inference - Run predictions
nnBench.inference --config fold_0.yaml --dataset Dataset001_YourDataset
# To use validation set: add --use-val-split flag
# Output: nnBench_results/Dataset001_YourDataset/fold_0/predictions/
#         nnBench_results/Dataset001_YourDataset/fold_0/metrics.json

# 4. Plot - Visualize results
nnBench.plot --config fold_0.yaml --dataset Dataset001_YourDataset
# Output: nnBench_results/Dataset001_YourDataset/fold_0/plots/
```

For an extended workflow, refer to: **[Workflow](docs/src/workflow.md)**

## Important Documentation

- **[Configuration Reference](docs/src/config.md)** - Complete guide to all config options
  - Includes [Connected Components Metric (CCMetric)](docs/src/config.md#connected-components-metric-ccmetric) for multi-instance segmentation
- **[Terminology](docs/src/terminology.md)** - Key terms and concepts
- **[Contributing](CONTRIBUTING.md)** - Development setup and contribution guidelines

## Acknowledgement

This project builds upon and is inspired by:

- **[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)** - Self-configuring method for deep learning-based biomedical image segmentation
- **[MONAI](https://github.com/Project-MONAI/MONAI)** - PyTorch-based framework for deep learning in healthcare imaging
