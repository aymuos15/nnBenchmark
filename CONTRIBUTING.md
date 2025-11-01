# Contributing to nnBenchmark

We welcome contributions to nnBenchmark! This guide will help you get started.

## Installation

Set up your development environment:

```bash
# Clone the repository
git clone git@github.com:aymuos15/nnBenchmark.git
cd nnBenchmark

# Install uv (fast Python package manager)
pip install uv

# Install dev dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks (recommended)
pre-commit install
```

## Documentation

We encourage documentation contributions, especially those that:

- **Clarify terminology** - Help users understand key concepts in medical imaging and segmentation
- **Improve existing docs** - Fix unclear explanations, add examples, or update outdated information
- **Add new guides** - Document new features, workflows, or best practices
- **Update terminology** - Ensure consistent use of terms across the codebase and documentation

### Before Making Documentation Changes

**Please raise an issue first** to discuss your proposed changes. This helps ensure:
- Changes align with project goals and documentation structure
- Effort isn't duplicated if someone else is working on the same area
- We maintain consistency in terminology and style

### Terminology Guidelines

nnBenchmark uses specific terminology to maintain consistency. For example:

- **Channel** - A single imaging input (imaging technique like MRI/CT, or a specific sequence like T1/T2/FLAIR)
- **Case** - A single patient/subject's complete dataset with all channels and labels
- **Class** - A segmentation target (background is always class 0)

For more details, see [docs/terminology.md](docs/terminology.md).

## Code Contributions

### Planning Module Changes

When proposing changes to the planner (`src/planning/`), please ensure:

- **DOC comments are included** - All planning functions should have DOC comments that reference:
  - The relevant category (e.g., "Fingerprinting", "Patch Sizing", "Model Topology")
  - The source constant from `src/planning/constants.py` (e.g., `PLANNING_CONSTANTS.ANISOTROPY_THRESHOLD`)
  - The corresponding section in `docs/planning.md`

- **Documentation is updated** - Any changes to planning logic should include updates to:
  - `docs/planning.md` - Update the relevant workflow step and factor analysis
  - `src/planning/constants.py` - Update or add constants if new parameters are introduced
  - `docs/terminology.md` - Add new terms if concepts are introduced

This ensures that code and documentation stay synchronized and maintainable.

## Testing

Run tests before submitting contributions.

### Quick Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_planning.py

# Run specific test function
uv run pytest tests/test_planning.py::test_function_name

# Fast mode (minimal output)
tox -e fast
```

### Coverage

```bash
# Run tests with coverage report
tox -e coverage

# View HTML coverage report
open htmlcov/index.html
```

### Full Validation

```bash
# Run all checks: lint + type check + tests
tox -e all

# Or run all environments
tox

# Individual checks
tox -e lint        # Ruff linting
tox -e type        # Type checking (ty)
tox -e pyright     # Type checking (pyright)
```

### Test Requirements

- Tests use `pytest` with fixtures in `tests/conftest.py`
- Environment variables are auto-configured by `_setup_nnunet_env_vars` fixture
- Mock datasets are available via fixtures for fast testing

## Continuous Integration

All pull requests automatically run CI checks via GitHub Actions (`.github/workflows/ci.yml`).

### CI Pipeline

The CI runs the following checks on every pull request:

1. **Lint** - Ruff code linting
2. **Type Check (ty)** - Type checking with ty
3. **Type Check (pyright)** - Type checking with pyright
4. **Tests** - pytest on Python 3.10, 3.11, 3.12, 3.13 with coverage
5. **Coverage** - Uploads to Codecov (Python 3.11 only)

All checks must pass before merging. Run `tox -e all` locally to replicate CI checks.

---

*More contribution guidelines coming soon.*
