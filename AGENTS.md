# Agent Guidelines for nnBenchmark

## Build/Test Commands
- **Run all tests**: `pytest` or `uv run pytest`
- **Single test**: `pytest tests/test_filename.py::TestClass::test_method -v`
- **Fast tests (no coverage)**: `pytest -q --tb=short --no-cov`
- **Lint**: `ruff check .` (auto-fix: `ruff check . --fix`)
- **Type check**: `python -m ty check` or `pyright`
- **Full validation**: `tox -e all` (lint + type + tests)
- **Format**: `ruff format .`

## Code Style
- **Imports**: Always include `from __future__ import annotations` as first import. Use absolute imports (`from src.module...`). Group: stdlib, third-party, local. Ruff auto-sorts (select I).
- **Types**: Use type hints for all functions (parameters + return). Use modern syntax (e.g., `list[str]`, `dict[str, Any]`). Enable pyright standard mode.
- **Formatting**: Line length 88 chars. Ruff handles formatting (E, F, W rules). Ignore E501 (line length) and N812 (lowercase F for torch.nn.functional).
- **Docstrings**: Module-level docstrings required. Function docstrings for public APIs (Google style). Include Args, Returns, Raises sections.
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants. Descriptive names (avoid single letters except i, j in loops).
- **Error handling**: Raise specific exceptions (ValueError, FileNotFoundError, RuntimeError). Include descriptive messages with context.
- **No print statements**: Use loguru for logging (already configured). Pre-commit hook blocks print() in src/ (except test/viz/plot files).
- **Testing**: Use pytest fixtures from conftest.py. Test classes named `Test*`, methods named `test_*`. Mock external dependencies.

## Project Structure
- **src/**: Main package (config, lightning, train, inference, planning, plotting, utils, logging, preprocessing)
- **tests/**: Unit tests mirror src/ structure
- **datasets/**: nnUNet-style datasets (Dataset001_Name format)
- **CLI**: nnBench.train, nnBench.test, nnBench.plot, nnBench.plan
