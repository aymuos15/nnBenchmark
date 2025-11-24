# AGENTS.md - Coding Agent Guidelines for nnBenchmark

## Build/Test Commands
- **Install deps**: `uv pip install -e ".[dev]"`
- **Run all tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/test_planning.py::test_function_name`
- **Fast tests (no coverage)**: `tox -e fast`
- **Lint**: `tox -e lint` (auto-fix: `tox -e lint-fix`)
- **Type check**: `tox -e type` (ty) or `tox -e pyright`
- **All checks**: `tox -e all` (lint + type + tests + coverage)
- **Pre-commit**: `pre-commit run --all-files`

## Required Environment Variables
```bash
export nnBench_raw="/path/to/nnBench_raw"
export nnBench_preprocessed="/path/to/nnBench_preprocessed"
export nnBench_results="/path/to/nnBench_results"
```

## Code Style
- **Python**: 3.10+ (use modern type hints: `dict[str, Any]`, not `Dict[str, Any]`)
- **Line length**: 88 chars (Black-compatible)
- **Imports**: Sorted by ruff (stdlib → third-party → local), absolute imports preferred
- **Type hints**: Required for all public functions/methods (checked by pyright + ty)
- **Docstrings**: Google style, required for public APIs
- **Error handling**: Raise specific exceptions (ValueError, TypeError, etc.) with descriptive messages
- **Naming**: snake_case (functions/vars), PascalCase (classes), SCREAMING_SNAKE_CASE (constants)
- **No print()**: Use loguru logger in src/ (except CLI/plotting/utils modules)
- **CLI flags**: Use hyphens (--gpu-memory-gb, not --gpu_memory_gb)

## Project-Specific Conventions
- **Registry pattern**: All components (losses, metrics, models) use factory registries in `src/factory/`
- **Planning DOC comments**: Reference `src/planning/constants.py` and `docs/planning.md` sections
- **Terminology**: Use consistent terms (see `docs/terminology.md`): "channel" (imaging input), "case" (patient dataset), "class" (segmentation target)
- **Config inheritance**: YAML configs support `base_config` + `overrides` pattern
- **Checkpointing**: Auto-resume from `checkpoints/final.pt` if exists
- **Tests**: Use fixtures from `tests/conftest.py` (mock datasets, auto env var setup)
