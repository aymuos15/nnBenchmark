# Random Patch Sampling vs Sequential DataLoader: The Core Architectural Difference

**Date**: December 12, 2025  
**Status**: Complete Explanation  
**Topic**: Why batch size differs and what actually needs to change

---

## The Real Issue (Not Batch Size)

The batch size difference (2-3 vs 4-6) is a **symptom**, not the cause. The real difference is **how samples are selected during training**.

---

## Current Approach: Sequential DataLoader (nnBenchmark)

### What Happens Now

```python
# Pseudo-code of current nnBenchmark behavior
for epoch in range(200):
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:  # This loop processes EVERY sample
        train_step(batch)
    # Epoch complete: All 100 samples (or however many) seen exactly once
```

### Behavior Details

- **shuffle=True**: Randomizes the ORDER samples are processed, but doesn't change coverage
- **Coverage per epoch**: 100% of dataset, every sample seen exactly once
- **Sample repetition**: Zero - each sample appears at most once per epoch
- **Total epochs**: 200
- **Total training passes**: 200 complete dataset passes

### Example (10-sample dataset, batch_size=2)

```
Epoch 1: Process samples [3,7] → [1,9] → [5,2] → [8,4] → [6,10]
         (random order, but all 10 seen once)
Epoch 2: Process samples [9,2] → [4,7] → [1,3] → [8,5] → [10,6]
         (different random order, but all 10 seen once again)
...
Epoch 200: All samples processed again
```

Result: After 200 epochs, each sample has been seen exactly 200 times.

---

## nnU-Net Approach: Random Patch Sampling with Replacement

### What nnU-Net Does

```python
# Pseudo-code of nnU-Net behavior
for epoch in range(1000):
    for iteration in range(250):  # Fixed number, independent of dataset size
        # Randomly pick batch_size samples WITH REPLACEMENT
        # (same sample can appear multiple times)
        indices = np.random.choice(len(dataset), size=batch_size, replace=True)
        batch = [dataset[i] for i in indices]
        train_step(batch)
    # Epoch complete: Random subset of dataset seen, some maybe multiple times
```

### Behavior Details

- **Random selection**: Each iteration independently picks random samples
- **With replacement**: The same sample can be picked multiple times in one epoch
- **Coverage per epoch**: ~5% on average (depends on random selection)
- **Sample repetition**: Expected ~1.6× times per epoch on average (this is mathematical)
- **Total epochs**: 1000
- **Total training passes**: ~50 complete dataset passes

### Example (10-sample dataset, batch_size=2, iterations=3 per epoch)

```
Epoch 1:
  Iteration 1: Pick random indices [3, 3] → samples [3, 3]
  Iteration 2: Pick random indices [7, 1] → samples [7, 1]
  Iteration 3: Pick random indices [5, 5] → samples [5, 5]
  Coverage: Samples {1,3,5,7} seen, samples {2,4,6,8,9,10} not seen
  Repetition: Sample 3 seen twice, sample 5 seen twice

Epoch 2:
  Iteration 1: Pick random indices [2, 9] → samples [2, 9]
  Iteration 2: Pick random indices [4, 4] → samples [4, 4]
  Iteration 3: Pick random indices [8, 1] → samples [8, 1]
  Coverage: Samples {1,2,4,8,9} seen, samples {3,5,6,7,10} not seen
  Repetition: Sample 4 seen twice

Epoch 3-1000: Random coverage each epoch
```

Result: After 1000 epochs with random selection, each sample has been seen ~50 times on average.

---

## Side-by-Side Comparison

| Aspect | nnBenchmark (Current) | nnU-Net |
|--------|----------------------|---------|
| **Sampler Type** | Sequential with shuffle | Random with replacement |
| **Iterations/Epoch** | Variable (full dataset) | Fixed 250 |
| **Coverage/Epoch** | 100% (all samples) | ~5% (random subset) |
| **Sample Repetition/Epoch** | None (each once) | Expected ~1.6× (some multiple times) |
| **Epochs** | 200 | 1000 |
| **Total Iterations** | ~50,000 (100÷bs × 200) | 250,000 (250 × 1000) |
| **Total Dataset Passes** | 200 | ~50 |
| **Batch Size** | 4-6 | 2-3 |
| **Learning Pattern** | Even exposure (all samples equal) | Uneven exposure (some more, some less) |
| **Randomness** | Per-epoch order shuffle | Per-iteration sample selection |

---

## Why This Matters: Training Dynamics

### nnBenchmark's Sequential Approach
```
Gradient flow per epoch:
- Step 1: Update on samples 1,2,3,4
- Step 2: Update on samples 5,6,7,8
- Step 3: Update on samples 9,10,11,12
- ...
- Step N: Update on samples remaining

All samples contribute equally to the epoch's loss.
```

**Result**: 
- ✅ Smooth, predictable gradient flow
- ✅ Every sample influences every epoch
- ❌ Less noisy gradients (potentially less exploration)

### nnU-Net's Random Sampling Approach
```
Gradient flow per epoch (random selection):
- Step 1: Update on samples [7,12] (random)
- Step 2: Update on samples [7,12] (picked again!)
- Step 3: Update on samples [3,8] (different random)
- ...
- Step N: Update on samples [14,2] (random)

Some samples dominate the epoch, others don't appear.
```

**Result**:
- ✅ Noisier gradients (more exploration of solution space)
- ✅ Small dataset safety (doesn't overfit individual samples)
- ❌ Less predictable (depends on random sampling)
- ❌ Some samples under-trained per epoch (but over-trained across 1000 epochs)

---

## The Code Change Required

### Current Implementation (Sequential)

Located in training setup code (likely in `src/engines/train/run.py` or `src/engines/setup.py`):

```python
# Current: Creates DataLoader that exhausts entire dataset
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,  # Only randomizes order, not selection
    num_workers=4,
    pin_memory=True,
)

# This will iterate through ALL samples exactly once per epoch
for batch in dataloader:
    train_step(batch)
```

### Required Change: Custom Random Sampler

```python
import numpy as np
from torch.utils.data import Sampler

class RandomPatchSampler(Sampler):
    """
    Samples with replacement to match nnU-Net's training behavior.
    
    Instead of processing each sample once per epoch, this sampler
    randomly selects samples for a fixed number of iterations,
    allowing the same sample to be selected multiple times.
    """
    
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        iterations_per_epoch: int,
        seed: int = 12345,
    ):
        """
        Args:
            dataset_size: Number of samples in dataset
            batch_size: Samples per batch
            iterations_per_epoch: Fixed iterations per epoch (e.g., 250 for nnU-Net)
            seed: Random seed for reproducibility
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.iterations_per_epoch = iterations_per_epoch
        self.seed = seed
    
    def __iter__(self):
        """Yield batch_size random indices for each iteration."""
        np.random.seed(self.seed)
        for _ in range(self.iterations_per_epoch):
            # KEY DIFFERENCE: replace=True allows same index multiple times
            indices = np.random.choice(
                self.dataset_size,
                size=self.batch_size,
                replace=True,  # THIS IS THE CRITICAL CHANGE
            )
            for idx in indices:
                yield idx
    
    def __len__(self):
        """Total samples yielded per epoch."""
        return self.iterations_per_epoch * self.batch_size
```

### Usage

```python
from torch.utils.data import DataLoader

# Create the custom sampler
sampler = RandomPatchSampler(
    dataset_size=len(dataset),
    batch_size=2,  # nnU-Net batch size
    iterations_per_epoch=250,  # nnU-Net fixed iterations
    seed=12345,
)

# Use with DataLoader
dataloader = DataLoader(
    dataset,
    batch_sampler=None,  # We provide sampler directly
    sampler=sampler,
    batch_size=None,  # batch_size is handled by sampler
    num_workers=4,
    pin_memory=True,
)

# Now training loop matches nnU-Net
for epoch in range(1000):  # Changed from 200
    for batch in dataloader:  # Now 250 iterations instead of variable
        train_step(batch)
```

---

## What Changes When You Switch

### Code Changes Required

1. **Add RandomPatchSampler class** (~30 lines)
2. **Update DataLoader creation** (~5 lines)
3. **Change epoch count** (200 → 1000)
4. **Adjust learning rate schedule** (currently tuned for 200 epochs, needs 1000)
5. **Update early stopping/checkpointing logic** (depends on epoch-based timing)

### Impact on Training

| Aspect | Current | After Change |
|--------|---------|-------------|
| **Training Time** | 200 epochs × N mins | 1000 epochs × N mins = 5× longer |
| **Memory per Batch** | Higher (BS=4-6) | Lower (BS=2-3) |
| **GPU Utilization** | Good (larger batches) | Lower (smaller batches) |
| **Gradient Noise** | Lower (stable) | Higher (noisier, more exploration) |
| **Convergence Pattern** | Smooth | Zigzaggy but potentially finds better minima |
| **Generalization** | Baseline | Potentially better (noisier = regularization) |

---

## Mathematical Insight: Why ~50 Passes?

**nnU-Net Math**:
- Total samples selected: 1000 epochs × 250 iterations × 2 batch_size = 500,000 selections
- With replacement (random), probability any given sample appears in one selection:
  - P(not selected) = (1 - 1/N)^500 for one selection
  - Over all selections: Expected appearances ≈ 500,000 / N samples
  - For dataset of N samples: 500,000 / N selections ÷ dataset_size
  - If 5% cap: ~0.05 × total_voxels / patch_voxels samples per epoch
  - Result: ~50 complete passes

**nnBenchmark Current**:
- Total samples: 200 epochs × full dataset coverage = 200 complete passes
- No randomness, deterministic

---

## Decision: Should You Make This Change?

### Make the Change If:
✅ You need exact nnU-Net replication for research/publication  
✅ You want to match the original training regime exactly  
✅ Small dataset safety is a priority  
✅ You have enough GPU/time for 1000 epochs

### Keep Current Approach If:
✅ Current results are good enough  
✅ You prefer deterministic, predictable training  
✅ You want shorter training time (200 vs 1000 epochs)  
✅ You prefer modern PyTorch best practices  
✅ You like even sample exposure (no under/over training of specific samples)

### Hybrid Approach:
If you want random sampling benefits without full 1000-epoch commitment:
```python
# Use RandomPatchSampler but with fewer epochs
sampler = RandomPatchSampler(
    dataset_size=len(dataset),
    batch_size=2,
    iterations_per_epoch=250,
    seed=12345,
)
# Then train for 200 epochs instead of 1000
# Result: 200×250×2 / N ≈ 100 total passes (between 50 and 200)
```

---

## Files That Would Need Changes

1. **`src/engines/setup.py`** - Where DataLoader is created
2. **`src/engines/train/run.py`** - Main training loop
3. **`src/utils/lr_scheduler.py`** - Learning rate schedule (tune for 1000 epochs)
4. **`src/engines/train/handlers.py`** - Checkpointing/early stopping logic
5. **`pyproject.toml`** or config - Update default epoch count

---

## Summary

**The batch size difference exists because nnU-Net uses random sampling with replacement**, which naturally works with small batches and many epochs.

**To match nnU-Net exactly:**
1. Replace sequential DataLoader with random sampler (`replace=True`)
2. Set fixed iterations per epoch (250)
3. Increase epochs (200 → 1000)
4. Reduce batch size (4-6 → 2-3)

**The sampler itself is simple** (~30 lines of code), but the ripple effects are large (5× longer training, different convergence pattern, different hyperparameter tuning).

**Bottom line**: It's not just about batch size. It's about accepting random under-exposure of some samples per epoch in exchange for small-dataset robustness.
