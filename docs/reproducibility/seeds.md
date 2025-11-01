# Seeding

Reproducible random number generation across Python, NumPy, and PyTorch.

## Functions

- `set_random_seeds(seed)` - Sets seeds for `random`, `numpy`, and `torch` (CPU + CUDA)
- `enable_cuda_determinism(deterministic)` - Enable/disable CUDA determinism (slower but reproducible)
- `get_seed_from_config(cfg)` - Extract seed from config with priority: `seed` → `training.seed` → `inference.seed` → `12345`

## Usage

```python
from src.utils.seeding import set_random_seeds, enable_cuda_determinism

# Set seeds
set_random_seeds(seed=42)

# Enable full determinism (slower)
enable_cuda_determinism(deterministic=True)
```

## Configuration

```yaml
seed: 12345  # Top-level seed for reproducibility
```

**Implementation**: `src/utils/seeding.py`
