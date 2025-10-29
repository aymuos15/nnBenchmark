# Agent Guide for nnBenchmark

## Build/Test/Lint Commands
- Run tests: `pytest` or `uv run pytest`
- Run single test: `pytest tests/test_file.py::test_name`
- Run tests with coverage: `pytest --cov=src --cov-report=term-missing`
- Lint: `ruff check .` or `tox -e lint`
- Lint fix: `ruff check . --fix` or `tox -e lint-fix`
- Type check: `python -m ty check` or `pyright`
- Full validation: `tox -e all` (runs lint, type, dead code analysis, tests)
- Pre-commit: `pre-commit run --all-files`

## Code Style
- **Python**: 3.10+ with `from __future__ import annotations` at top of every file
- **Formatting**: Ruff (88 chars/line). No manual formatting needed - ruff-format handles it
- **Imports**: Ruff sorts imports automatically (I: isort rules). Standard lib → third-party → local
- **Types**: Use type hints everywhere. `dict[str, Any]` not `Dict`. Use `list`, `tuple`, not `List`, `Tuple`
- **Naming**: Snake_case for functions/vars, PascalCase for classes. Allow `F` for `torch.nn.functional`
- **Docstrings**: Google style with Args/Returns/Raises sections. All public functions require docstrings
- **Error handling**: Use descriptive ValueError/RuntimeError with helpful messages and examples
- **No print()**: Use loguru logger. Print statements forbidden in `src/` (except plotting/viz)
- **Testing**: Fixtures in conftest.py. Use descriptive test names. Maintain coverage
