# Logging

Centralized logging using [loguru](https://github.com/Delgan/loguru) with dual logging support.

## Dual Logging System

nnBenchmark implements a dual logging philosophy:

- **Console Output**: INFO level only, minimal format for clean user-facing output
- **File Output**: DEBUG level, detailed format with timestamps, module names, and line numbers

This separation keeps the console clean while preserving all debugging information in files.

## Setup Functions

### For Training/Inference (File-only Logging)
- `setup_train_logger(results_dir, resume)` - Logs to `train.log` file only (DEBUG level)
- `setup_test_logger(results_dir)` - Logs to `test.log` file only (DEBUG level)

### For CLI Tools (Dual Logging)
- `setup_dual_logging(log_file)` - Setup dual logging (console INFO + file DEBUG)
  - Example: Planning, main CLI entry points
  - Console: User-facing progress and important messages
  - File: Exhaustive debug information

### Other Functions
- `setup_verbose_logger(level, format_string)` - Custom console output with specified level
- `setup_logger(results_dir, log_name, resume)` - Generic logger setup

## Helper Functions

- `log_and_print(message, level)` - Log to file AND print to console (both formats)
- `log_only(message, level)` - Log to file only (no console output)
- `log_header(message)` - Log formatted section headers with separators
- `log_separator()` - Log visual separators
- `log_system_info()` - Log system/GPU information

## Usage Examples

### Training (File-only logging)
```python
from loguru import logger
from src.logging import setup_train_logger

# Setup logger (removes console output, writes to train.log)
setup_train_logger(results_dir="results/Dataset001/fold_0", resume=False)

# All logs go to file only
logger.info("Starting training...")
logger.debug("Model loaded with parameters...")
```

### Planning (Dual logging)
```python
from src.logging import setup_dual_logging

# Setup dual logging
log_file = setup_dual_logging("planning.log")

# Console: Only INFO level (clean for users)
# File: DEBUG level (detailed for debugging)
logger.info("Fingerprinting dataset...")  # Shows on console
logger.debug("Processing image 001...")   # File only
```

## File Locations

- **Training logs**: `{results_dir}/train.log`
- **Inference logs**: `{results_dir}/test.log`
- **Planning logs**: User-specified path (e.g., `planning.log`)

**Implementation**: `src/logging/` (`setup.py`, `helpers.py`, `system.py`)

## Philosophy
1. **Console**: Minimal, INFO-level only - clean progress output for users
2. **Files**: Exhaustive DEBUG-level logging - complete traceability for troubleshooting
3. **Consistency**: All modules use the same logger instance from loguru