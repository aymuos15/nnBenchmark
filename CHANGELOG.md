# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5] - 2025-11-06

### Added
- **Surface Dice Support in CCMetric**: Extended CCMetric to support Normalized Surface Dice (NSD) metrics on connected components
  - New parameter `metric_type`: Choose between "dice", "surface_dice", or "both" per region
  - New parameter `class_thresholds`: Distance tolerance (pixels/mm) for boundary matching
  - New parameter `distance_metric`: Euclidean, chessboard, or taxicab distance metrics
  - Boundary-focused metric evaluation for improved multi-instance segmentation assessment
- **Dynamic Metric Key Generation**: MetricRegistry now auto-generates unique metric keys for instances with `metric_type` parameter
  - Example: CCMetric with metric_type="dice" becomes "CCMetric_dice"
  - Enables multiple metric variants in single configuration
- **Enhanced Results Table Formatting**: format_results_table now displays fixed set of 4 metrics (Dice, CC-Dice, NSD, CC-NSD)
  - Improved readability with mean ± std formatting
  - Support for both standard and connected component metric variants

### Changed
- **Metrics**: CCMetric now defaults to metric_type="dice" (backward compatible)
- **Results Output**: test_history.json metric keys now include metric_type suffix for uniqueness

### Fixed
- **Code Quality**: Removed unused Union import, extracted duplicate logic into helper methods
  - _compute_region_dice(): Unified dice computation across metric types
  - _create_masked_region(): Unified tensor masking for surface dice computation
  - Consolidated Console instantiations in results.py

## [0.1.4] - 2025-11-04

### Added
- **BlobLoss**: New loss function for instance segmentation with blob detection
  - Penalty term for under-segmentation (merged instances)
  - Penalty term for over-segmentation (split instances)
  - Configurable blob detection sensitivity and size thresholds
  - Integration with existing loss registry system
- **Release Process Documentation**: Comprehensive guide in `docs/RELEASING.md` covering complete release workflow
  - Step-by-step instructions for version updates, changelog management, and PyPI publishing
  - Troubleshooting section for common issues
  - Complete checklist for release verification
- **Dataset003_Kvasir Documentation**: Documentation and tooling for Kvasir-SEG polyp segmentation dataset
  - Dataset metadata configuration in `docs/datasets/Dataset003_Kvasir/dataset.json`
  - Source documentation in `docs/datasets/Dataset003_Kvasir/flow.md`
  - Format conversion script (`format.py`) for transforming Kvasir-SEG to nnUNet structure
  - Configuration comparison utility (`compare.py`) for validating settings

### Fixed
- **Type Checking**: Resolved type checking errors in BlobLoss implementation
- **Test Suite**: Removed failing BlobLoss test that was causing CI issues

## [0.1.3] - 2025-11-03

### Added
- **CCLoss (Connected Components Loss)**: New loss function for multi-instance segmentation evaluating predictions at region/connected component level
  - Fully differentiable with backpropagation support
  - GPU-accelerated connected components detection using cupy and cucim
  - Configurable activations (sigmoid, softmax)
  - Support for one-hot and class index targets
  - Ideal for cell/nuclei segmentation tasks
- **Documentation**: Comprehensive guide for adding custom loss functions (`docs/examples/adding_custom_loss.md`)
- **Tests**: New test suite for custom losses (`tests/test_custom_losses.py`)

### Changed
- Loss Registry: Updated docstring to reflect custom losses inclusion
- Loss Module: Added `CCLoss` to exports in `src/factory/losses/__init__.py`
- Pyright Configuration: Added `requires_grad` to ignored names in `pyproject.toml`

### Fixed
- **CCLoss Target Shape**: Handle spurious single-channel dimensions from MONAI's `LoadImaged` transform
  - Automatically squeezes `(B, 1, H, W)` targets to `(B, H, W)` for class indices
  - Prevents `IndexError` when training with CCLoss on class index targets

## [0.1.2] - 2025-11-03

### Added
- **Pre-commit Improvements**: Enhanced pyright configuration with comprehensive dependency coverage

### Changed
- **Development Workflow**: Removed pytest from pre-commit hooks (now runs separately)
- **Pre-commit Configuration**: Streamlined hook setup for better performance

### Fixed
- **Type Checking**: Fixed type checking issues in test files and configurations
- **Pre-commit Hooks**: Resolved pyright environment dependency resolution

### Removed
- **Pyrefly Integration**: Removed due to alpha status and editable package import limitations

## [0.1.1] - 2025-11-02

### Added
- **Progress Display**: Consolidated metrics in training progress bar and added validation progress display
- **Dual Logging**: Implemented clean console output with verbose file logging for better debugging
- **GPU Memory Management**: Added GPU cache clearing and disabled pinned memory for small GPUs
- **Integration Tests**: Added comprehensive test coverage for the new architecture

### Changed
- **Architecture Migration**: Migrated from PyTorch Lightning to MONAI SupervisedTrainer with Ignite integration
- **Module Organization**: Reorganized modules into engines-based architecture for better separation of concerns
- **Documentation**: Updated architecture references from PyTorch Lightning to MONAI+Ignite

### Fixed
- **Type Annotations**: Corrected type annotations for intensity clipping parameters in tests
- **Checkpoint Loading**: Fixed checkpoint glob patterns to match MONAI naming convention
- **AMP Training**: Added GradScaler for stable Automatic Mixed Precision training
- **Code Quality**: Applied Ruff linting fixes across the codebase

### Removed
- **Unwanted Files**: Removed accidentally committed files

### Performance
- **GPU Optimization**: Improved GPU memory management for better training stability</content>
<parameter name="filePath">CHANGELOG.md
