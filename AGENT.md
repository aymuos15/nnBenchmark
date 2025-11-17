This file provides guidance to coding agents when working with code in this repository.

## About nnBenchmark

Config-driven 3D medical image segmentation framework using MONAI SupervisedTrainer. Focuses on reproducibility and benchmarking with automatic configuration generation from dataset properties.

## Installation

### For Users (TestPyPI - Alpha Testing)

Currently in alpha testing phase. Install from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nnbenchmark
```

Or install from source:
```bash
git clone https://github.com/aymuos15/nnBenchmark.git
cd nnBenchmark
pip install -e .
```

**Note**: Full PyPI release planned after alpha testing phase concludes.

### For Development

```bash
# Install uv (fast Python package manager)
pip install uv

# Clone and install with dev dependencies
git clone https://github.com/aymuos15/nnBenchmark.git
cd nnBenchmark
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Environment Variables

Set these for all installation methods (add to `~/.bashrc` or `~/.zshrc`):

```bash
export nnBench_raw="/path/to/nnBench_raw"
export nnBench_preprocessed="/path/to/nnBench_preprocessed"
export nnBench_results="/path/to/nnBench_results"

# Note: Old nnUNet_* variables still work for backward compatibility (deprecated)
```

### Environment Variable Deprecation

When running tests or commands with old `nnUNet_*` environment variables, you'll see DeprecationWarnings. The code automatically falls back to old variables if new ones aren't set, but warnings indicate the need for migration. Update to `nnBench_*` variables to eliminate these warnings.

## Core Commands

### Testing
```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/test_planning.py

# Run specific test function
uv run pytest tests/test_planning.py::test_function_name

# Fast mode
tox -e fast

# Coverage report
tox -e coverage
```

### Code Quality
```bash
# Run all checks (lint + type + tests)
tox -e all

# Individual checks
tox -e lint        # Ruff linting
tox -e type        # Type checking with ty
tox -e pyright     # Type checking with pyright
```

### CLI Tools
```bash
# Automatic config generation from dataset
nnBench.plan --dataset Dataset002_HippocampusMedDecathalon --verbose

# Training
nnBench.train --config fold_0.yaml --dataset Dataset001_Cellpose
nnBench.train --config fold_0.yaml --dataset Dataset001_Cellpose --continue  # Resume from checkpoint (optional, automatic)

# Validation (post-training)
nnBench.validate --config fold_0.yaml --dataset Dataset001_Cellpose
nnBench.validate --config fold_0.yaml --dataset Dataset001_Cellpose --checkpoint path/to/checkpoint.pt  # Single checkpoint

# Inference
nnBench.inference --config fold_0.yaml --dataset Dataset001_Cellpose
nnBench.inference --config fold_0.yaml --dataset Dataset001_Cellpose --use-val-split

# Plotting
nnBench.plot --config fold_0.yaml --dataset Dataset001_Cellpose
```

## Architecture

### Core Philosophy
**Every experiment requires a config file. No CLI overrides allowed.** This enforces reproducibility and traceability. Different fold = new config, different parameter = new config.

### Environment Variables

The codebase uses three environment variables to locate data:
- `nnBench_raw` - Original datasets (imagesTr/, labelsTr/, dataset.json)
- `nnBench_preprocessed` - Preprocessed/cropped data
- `nnBench_results` - Training results, configs, checkpoints

**Backward compatibility**: Old `nnUNet_*` variables still work but will show deprecation warnings. Set the new `nnBench_*` variables to avoid warnings.

These are accessed via `src/config/paths.py`:
- `get_datasets_root()` → nnBench_raw
- `get_preprocessed_root()` → nnBench_preprocessed
- `get_results_root()` → nnBench_results

### Project Structure

**src/planning/** - Automatic configuration generation
- 5-step workflow: Preprocessing → Fingerprinting → Experiment Planning → YAML Generation → Cross-Validation Splits
- Analyzes dataset properties to determine optimal model architecture, batch size, patch size
- Deterministic: same dataset produces same config
- Key modules:
  - `fingerprinting/` - Parallel dataset analysis (shape, spacing, intensity statistics)
  - `planner/` - Architecture calculation heuristics (topology, sizing, feature channels)
  - `yaml_generator.py` - Complete config generation

**src/factory/** - Registry-based component instantiation
- Models, losses, optimizers, metrics, transforms registered in type-safe registries
- Configuration-driven: all components built from YAML
- Uses native MONAI/PyTorch parameter names (no translation layer)
- Enables multi-model support by changing single config field

**src/engines/train/** - MONAI SupervisedTrainer integration with Ignite
- `run.py` - Training execution with automatic checkpoint resumption
- `handlers.py` - Custom handlers for training history, logging, GPU memory, checkpointing
- Deep supervision support via loss wrapper
- Event-driven training with Ignite events
- Checkpoints saved as numbered files per epoch: checkpoint_epoch_XXX.pt
- Validation removed from training (now separate post-training step)

**src/engines/validate/** - Independent post-training validation module
- `cli.py` - CLI entry point for `nnBench.validate` command
- `run.py` - Validation orchestration with checkpoint discovery
- `engine.py` - ValidationEngine for event-driven validation
- `handlers.py` - Custom handlers for validation metrics, progress, results, visualization
- Auto-discovers and validates all epoch checkpoints sequentially
- Results saved as validation_history_epoch_XXX.json per checkpoint
- Can run independently without requiring training to complete

**src/config/** - Configuration loading and validation
- `load.py` - YAML loading with environment variable expansion
- `validation.py` - Schema validation for configs
- `resolution.py` - Nested config format resolution (multi-model support)

**src/engines/inference/** - Model inference with Ignite
- Supports sliding window inference for large volumes
- Loads MONAI checkpoints (.pt format) directly
- Handles full-resolution predictions
- Saves predictions in original dataset format
- Event-driven inference using Ignite engine

**src/preprocessing/** - Dataset preprocessing
- Crops images to nonzero regions using binary masks
- Reduces dataset size by 25-50% for brain MRI
- Preserves metadata (spacing, affine transforms)

**src/utils/** - Utility functions
- `lr_scheduler.py` - PolyLRScheduler (nnU-Net style learning rate decay)
- `data.py` - Data loading utilities
- `seeding.py` - Reproducibility utilities
- `files.py` - File I/O helpers

**src/plotting/** - Results visualization
- Training curves, validation metrics
- Uses results from training history

### Dataset Format
Follows nnU-Net conventions:
- Structure: `datasets/Dataset{XXX}_{Name}/imagesTr/`, `labelsTr/`, `imagesTs/`, `labelsTs/`
- Naming: Images `{case_id}_{XXXX}.{ext}` (XXXX = channel), Labels `{case_id}.{ext}`
- Formats: 3D (`.nii.gz`), 2D (`.png`, `.jpg`)
- Metadata: `dataset.json` defines channels and classes
- Splits: `splits.json` generated by `nnBench.plan` (5-fold CV, seed=12345)

### Configuration System
All parameters in YAML configs (see `docs/config.md` for full reference).

Key sections:
- `dataset`: patch_size, num_classes, fold, caching
- `model`: type (DynUNet/UNet), architecture params, deep supervision
- `training`: epochs, batch_size, learning_rate, mixed_precision
- `optimizer`: type, weight_decay, momentum
- `loss`: type (DiceCELoss, etc.), parameters
- `metrics`: list of metrics (DiceMetric, SurfaceDiceMetric, CCMetric with metric_type parameter, etc.)
- `transforms`: common/train/val/test pipelines

Supports nested format for multi-model configs (DynUNet + UNet in single file).

## Key Terminology

Use these consistently throughout code and docs (see `docs/terminology.md`):

- **Channel** - Single imaging input (e.g., T1 MRI, T2 MRI, CT)
- **Case** - Single patient's complete dataset with all channels and labels
- **Reference Label** - Expert annotations (NOT "ground truth" - acknowledges subjectivity)
- **Model** - Deep learning architecture being trained
- **Inference** - Using trained model to make predictions (NOT "testing")
- **Prediction** - Model output during inference

## Planning Module Guidelines

When modifying `src/planning/`:

1. **Add DOC comments** referencing:
   - Category (Fingerprinting, Patch Sizing, Model Topology, etc.)
   - Source constant from `src/planning/constants.py`
   - Corresponding section in `docs/planning.md`

2. **Update documentation**:
   - `docs/planning.md` - Update workflow step and factor analysis
   - `src/planning/constants.py` - Add/update constants
   - `docs/terminology.md` - Add new terms if concepts introduced

3. **Mark factors as [Constant] or [Adaptive]**:
   - [Constant] - Fixed hyperparameters (e.g., base_features=32)
   - [Adaptive] - Derived from dataset (e.g., median_spacing)

## Testing Requirements

- Fixtures in `tests/conftest.py` provide mock datasets
- Environment variables auto-configured by `_setup_nnunet_env_vars` fixture
- All tests must pass CI checks (lint, type check, pytest on Python 3.10-3.13)
- Use `tox -e all` to replicate CI locally

## CI Pipeline

GitHub Actions (`.github/workflows/ci.yml`) runs on all PRs:
1. Lint (Ruff)
2. Type check (ty + pyright)
3. Tests (pytest with coverage on Python 3.10, 3.11, 3.12, 3.13)
4. Coverage upload (Codecov, Python 3.11 only)

## Important Notes

- Python 3.10+ required
- Uses `uv` for fast package management
- Pre-commit hooks enforce code quality
- Never use `ground truth` - always use `reference label`
- Planning constants in `src/planning/constants.py` are source of truth
- All experiments reproducible via config files + seed=12345
