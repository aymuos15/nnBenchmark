# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
