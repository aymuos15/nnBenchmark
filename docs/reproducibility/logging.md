# Logging

Centralized logging using [loguru](https://github.com/Delgan/loguru).

## Setup Functions

- `setup_train_logger(results_dir, resume)` - Logs to `train.log` file only
- `setup_test_logger(results_dir)` - Logs to `test.log` file only
- `setup_verbose_logger(level, format_string)` - Console output for scripts with `--verbose`
- `setup_logger(results_dir, log_name, resume)` - Generic logger setup

## Helper Functions

- `log_and_print(message)` - Log to file and print to console
- `log_only(message)` - Log to file only
- `log_header(message)` - Log formatted section headers
- `log_separator()` - Log visual separators
- `log_system_info()` - Log system/GPU information

## Usage

```python
from loguru import logger
from src.logging import setup_train_logger

# Setup logger (removes console output)
setup_train_logger(results_dir="results/exp1")

# Use logger
logger.info("Starting training...")
```

**Implementation**: `src/logging/`

## Philosophy
1. Minimal logs on console.
2. Exhuastive logs on disk.