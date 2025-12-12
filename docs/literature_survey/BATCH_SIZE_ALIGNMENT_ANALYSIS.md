# Batch Size Alignment Analysis: nnBenchmark vs nnU-Net

**Date**: December 12, 2025  
**Status**: Analysis Complete  
**Topic**: Implications of matching nnU-Net's batch sizes (2-3) in nnBenchmark (currently 4-6)

---

## Quick Summary

**Current State**:
- **nnBenchmark**: Batch sizes of 4-6 per GPU (calculated via reference VRAM formula)
- **nnU-Net**: Batch sizes of 2-3 per GPU (derived from same formula with fixed 250 iterations/epoch)

**Question**: What if we force nnBenchmark batch sizes to 2-3?

**Answer**: 
- ✅ **Technically Possible** - Just change constants in `src/planning/constants.py`
- ⚠️ **Major Trade-offs** - Requires epoch/iteration restructuring to maintain training volume
- 🎯 **Key Insight** - The 5% dataset cap is IDENTICAL in both, but usage is fundamentally different

---

## The Core Issue: What is "Batch Size"?

### nnU-Net's Definition (Fixed Training Loop)
```python
# nnU-Net pseudo-code
for epoch in range(1000):                    # 1000 epochs
    for iteration in range(250):              # Fixed 250 iterations/epoch
        batch = get_random_patch_batch(bs=2) # Batch size = 2-3
        train_step(batch)
```

**Result**: 
- Total batches per epoch: 250
- Total voxels per epoch: 250 × batch_size × patch_voxels = 250 × 2 × patch_voxels
- Total epochs: 1000
- **Total training passes**: ~50 complete dataset passes (because 5% coverage × 1000 epochs)

### nnBenchmark's Definition (Modern DataLoader)
```python
# nnBenchmark pseudo-code
for epoch in range(200):                          # 200 epochs
    for batch in DataLoader(dataset, bs=4-6):    # Variable iterations (depends on dataset size)
        train_step(batch)                         # Batch size = 4-6
```

**Result**:
- Total batches per epoch: variable (depends on dataset)
- Total voxels per epoch: 100% dataset coverage per epoch
- Total epochs: 200
- **Total training passes**: 200 complete dataset passes

---

## Scenario Analysis: What If We Matched Batch Sizes?

### Scenario 1: Force Batch Size = 2, Keep Everything Else Same

```
BEFORE (Current nnBenchmark):
- Batch Size: 4-6
- Epochs: 200
- Iterations/Epoch: Variable (100% coverage)
- Total Training Volume: 200 complete passes

AFTER (Force BS=2):
- Batch Size: 2
- Epochs: 200
- Iterations/Epoch: Variable (100% coverage, but each iter processes smaller batch)
- Total Training Volume: 200 complete passes
- GPU Memory: ~66% of original (roughly, depends on model complexity)
```

**Impact**:
- ✅ **Smaller Batches**: Potentially better generalization (noisier gradients)
- ✅ **Lower VRAM**: More flexibility for larger patch sizes or GPU memory constraints
- ❌ **Slower Training**: Takes 2-3× longer per epoch (more batches to process)
- ❌ **Different Learning Dynamics**: Noisier gradients might affect convergence
- ❌ **Total Training Volume**: STILL 200 passes (not matching nnU-Net's 50 passes)

**Verdict**: ❌ **Not aligned with nnU-Net** - You're still doing 200 passes instead of 50.

---

### Scenario 2: Force Batch Size = 2, Reduce Epochs to ~50

```
BEFORE (Current nnBenchmark):
- Batch Size: 4-6
- Epochs: 200
- Total Training Volume: 200 complete passes

AFTER (BS=2 + Epochs=50):
- Batch Size: 2
- Epochs: 50
- Iterations/Epoch: Variable (100% coverage)
- Total Training Volume: 50 complete passes
- GPU Memory: ~66% of original
```

**Impact**:
- ✅ **Smaller Batches**: Noisier gradients, potentially better generalization
- ✅ **Fewer Epochs**: ~4× faster training
- ✅ **Matched Training Volume**: Both now do ~50 complete passes
- ❌ **Different Iteration Distribution**: 50×(100%coverage) vs 1000×(5%coverage) = fundamentally different
- ❌ **Learning Dynamics Still Different**: Sequential access vs random sampling

**Verdict**: ⚠️ **Partially Aligned** - Same total volume, but COMPLETELY different learning pattern.

---

### Scenario 3: Force Batch Size = 2, Keep 1000 Epochs (Match nnU-Net Exactly)

```
BEFORE (Current nnBenchmark):
- Batch Size: 4-6
- Epochs: 200
- Total Training Volume: 200 complete passes

AFTER (BS=2 + Epochs=1000):
- Batch Size: 2
- Epochs: 1000
- Iterations/Epoch: Variable (100% coverage)
- Total Training Volume: 1000 complete passes
- Training Time: 5× longer than current
```

**Impact**:
- ✅ **Smaller Batches**: Match nnU-Net
- ✅ **Same Epoch Count**: Match nnU-Net structure
- ✅ **Training Time Reasonable**: 5× longer but similar to nnU-Net
- ❌ **Iteration Distribution STILL Different**: 1000×(100%) vs 1000×(5%) 
- ❌ **Over-training Risk**: 5× more data exposure than nnU-Net
- ❌ **Breaking Point**: Learning rate schedule, early stopping, checkpoint logic all need adjustment

**Verdict**: ❌ **Fundamentally Misaligned** - Creates an unintended training regime (1000 epochs with 100% coverage).

---

## Root Cause: Framework Architectural Difference

The batch size difference is **not just a number** - it's a symptom of a deeper architectural difference:

### nnU-Net's Architecture
```
Random Patch Sampler
    ↓
[Per-epoch] Random selection of patches (5% dataset coverage)
    ↓
Fixed 250 iterations/epoch × 1000 epochs
    ↓
Total: 250 × 1000 = 250,000 batch iterations
```

### nnBenchmark's Architecture
```
PyTorch DataLoader
    ↓
[Per-epoch] Sequential exhaustion of dataset (100% coverage)
    ↓
Variable iterations/epoch × 200 epochs
    ↓
Total: ~40,000-50,000 batch iterations (depends on dataset size)
```

**Key Difference**: 
- nnU-Net uses **random sampling with repetition** across many epochs
- nnBenchmark uses **deterministic sequential sampling** across fewer epochs
- These create **fundamentally different gradient patterns** even if batch size matched

---

## Detailed Comparison Table

| Factor | nnU-Net | nnBenchmark | Impact if BS Matched |
|--------|---------|------------|----------------------|
| **Batch Size** | 2-3 | 4-6 | ✅ Easily changed |
| **Epochs** | 1000 | 200 | ❌ Requires epoch restructuring |
| **Iterations/Epoch** | 250 (fixed) | Variable | ❌ Fundamentally different |
| **Dataset Coverage/Epoch** | 5% (random) | 100% (sequential) | ❌ Cannot be changed without new sampler |
| **Gradient Noise** | High | Low | ⚠️ Affects convergence differently |
| **Training Duration/Epoch** | Fast (small data) | Varies (full data) | ⚠️ Changes wall-clock time |
| **Learning Stability** | Good (many small steps) | Good (large steps) | ⚠️ Needs retuning |
| **Reproducibility** | Seed-dependent | Deterministic | ✅ Unchanged |

---

## Implementation Options if Matching Batch Sizes

### Option 1: Change Batch Size Constants (SIMPLE, 30 min)

**What**: Modify the reference batch size constants in `src/planning/constants.py`

```python
# Current
UNET_REFERENCE_CORRESP_BS_3D = 2      # Reference batch size for 3D

# Change to force smaller batches
UNET_REFERENCE_CORRESP_BS_2D = 6      # Adjust reference to lower output
```

**Effort**: 30 minutes  
**Impact**: 
- ✅ Batch sizes become 2-3 (matching nnU-Net)
- ❌ Epochs stay at 200 (NOT matching nnU-Net's 1000)
- ❌ Iteration distribution still different
- ⚠️ Training dynamics change (smaller batches = noisier gradients)

**Use When**: You want smaller batches for memory/generalization reasons, not necessarily for nnU-Net fidelity.

---

### Option 2: Implement nnU-Net-Style Iteration Control (COMPLEX, 4-8 hours)

**What**: Replace DataLoader with custom sampler that mimics nnU-Net's behavior

```python
# Pseudo-code
class nnUNetRandomPatchSampler:
    def __init__(self, dataset, batch_size=2, iterations_per_epoch=250):
        self.dataset = dataset
        self.batch_size = batch_size
        self.iters_per_epoch = iterations_per_epoch
    
    def __iter__(self):
        for _ in range(self.iters_per_epoch):
            # Random sampling WITH REPETITION (allows seeing same image multiple times)
            indices = np.random.choice(len(self.dataset), self.batch_size)
            yield batch_from_indices(indices)

# Usage
trainer.fit(
    model,
    sampler=nnUNetRandomPatchSampler(dataset, batch_size=2, iterations_per_epoch=250),
    epochs=1000,
)
```

**Effort**: 4-8 hours
- 2h: Implement custom sampler
- 2h: Integrate with existing MONAI trainer
- 2h: Adjust learning rate schedule for 1000 epochs
- 2h: Testing and validation

**Impact**:
- ✅ Batch sizes = 2-3 (match nnU-Net)
- ✅ Epochs = 1000 (match nnU-Net)
- ✅ Iterations/epoch = 250 (match nnU-Net)
- ✅ Dataset coverage = 5% per epoch (match nnU-Net)
- ❌ Total training passes = 50 (4× less than current 200!)
- ⚠️ Learning rate schedule needs adjustment for 1000 epochs
- ⚠️ Checkpoint/early stopping logic needs rewrite

**Use When**: You need exact nnU-Net training behavior.

---

### Option 3: Hybrid Approach (MODERATE, 2-3 hours)

**What**: Keep DataLoader but add batch size scaling and epoch adjustment

```python
# Adjust only what's needed for closest approximation
- Force batch size = 2-3 (change constants)
- Keep epoch count = 200 (but make flexible)
- Keep sequential DataLoader (modern framework benefit)
- Accept that total training passes ≠ nnU-Net
```

**Effort**: 2-3 hours
- 30m: Change batch size constants
- 1h: Document the implications
- 1.5h: Validate behavior with a test run

**Impact**:
- ✅ Batch sizes = 2-3 (match nnU-Net)
- ✅ Simpler implementation than Option 2
- ❌ Epochs still 200 (don't match nnU-Net's 1000)
- ❌ Total training volume still 200 passes (not 50)
- ⚠️ This is a compromise - matches SOME aspects of nnU-Net

**Use When**: You want to explore smaller batch sizes but don't need full nnU-Net alignment.

---

## Training Impact: Batch Size 2 vs 4-6

### Gradient Statistics
```
Batch Size = 2:
- Per-sample gradient noise: HIGH (noisier, explores solution space more)
- Batch gradient variance: ~1/2 of BS=4 (more exploration)
- Convergence pattern: More zigzag, potentially finds different minima
- Generalization: Often better (noisier gradients = regularization effect)

Batch Size = 4-6:
- Per-sample gradient noise: MODERATE (more stable)
- Batch gradient variance: ~1/4-6 of BS=2 (more stability)
- Convergence pattern: Smoother, more predictable
- Generalization: Baseline (cleaner optimization trajectory)
```

### Learning Rate Implications
```
Current: Learning Rate = 0.01 (tuned for BS=4-6)
With BS=2: 
- Option A: Keep LR=0.01 (conservative, may undershoot)
- Option B: Scale LR = 0.01 × (2/4) = 0.005 (linear scaling)
- Option C: Keep LR=0.01 (embrace higher noise, faster convergence)

Recommendation: Start with Option B (scaled) or keep current and observe convergence
```

### Epoch Count Implications
```
Current: 200 epochs with batch size 4-6
Switch to BS=2 without epoch change:

Per-epoch gradient steps increase:
- Same dataset, smaller batches = MORE iterations per epoch
- More gradient updates per epoch = potentially better convergence
- But total training volume stays at 200 passes

Example (100 samples, 256³ patch):
- BS=4: ~25 batches/epoch (25 × 100% coverage)
- BS=2: ~50 batches/epoch (50 × 100% coverage)
- Result: 2× more gradient steps per epoch

Impact: Likely FASTER convergence, potentially BETTER final metric
```

---

## Recommendation Summary

### If Goal is "Match nnU-Net Exactly"
→ **Option 2** (Custom Sampler, 4-8 hours) is the only way to truly align.  
**But consider**: Is it worth 4× more training time (1000 vs 200 epochs)?

### If Goal is "Explore Smaller Batch Size Effects"
→ **Option 1 or 3** (Simple constants change or hybrid, 30 min - 3 hours).  
**Expected outcome**: Smaller batches might improve generalization, but training will be different.

### If Goal is "Current Practical Use (No Change)"
→ **Keep status quo** (200 epochs, BS=4-6).  
**Rationale**: Works well, modern framework integration, documented differences.

### If Goal is "Compromise: Better nnU-Net Approximation"
→ **Option 3** (Hybrid, 2-3 hours).  
**Trade-off**: Matches batch size direction, stays reasonably close without major restructuring.

---

## Decision Matrix

| Goal | Effort | Best Option | Time to Implement |
|------|--------|------------|------------------|
| Exact nnU-Net replica | High | Option 2 | 4-8 hours |
| Smaller batches exploration | Low | Option 1 | 30 min |
| Good approximation | Medium | Option 3 | 2-3 hours |
| Current approach | None | Keep as-is | 0 |

---

## Conclusion

**Matching batch size alone is not enough to match nnU-Net's training behavior.**

The batch size difference (2-3 vs 4-6) is a **surface-level manifestation** of a deeper architectural difference:
- **nnU-Net**: Small batches, fixed iterations, random sampling, 1000 epochs = 50 total passes
- **nnBenchmark**: Larger batches, variable iterations, sequential sampling, 200 epochs = 200 total passes

Simply changing batch size without restructuring epochs/iterations/sampling creates a **new, unintended training regime**.

### What Each Option Achieves

| Option | BS Match | Epoch Match | Iteration Match | Sampling Match | Total Effort |
|--------|----------|------------|-----------------|----------------|--------------|
| 1 (Constants) | ✅ | ❌ | ❌ | ❌ | 30m |
| 2 (Custom) | ✅ | ✅ | ✅ | ✅ | 4-8h |
| 3 (Hybrid) | ✅ | ❌ | ❌ | ❌ | 2-3h |
| Status Quo | ❌ | ❌ | ❌ | ❌ | 0 |

**The real question is not "should we change batch size?" but rather "how much of nnU-Net's training behavior do we want to match, and is it worth the effort?"**

---

## References

- **Current Batch Size Calculation**: `src/planning/planner/sizing.py:48-132`
- **Constants Location**: `src/planning/constants.py`
- **Batch Size in YAML**: `src/planning/yaml_generator.py:~250`
- **Training Loop**: `src/engines/train/run.py` (uses MONAI SupervisedTrainer)
- **DataLoader Setup**: Check MONAI SupervisedTrainer configuration

---

**Document Version**: 1.0  
**Status**: Complete Analysis  
**Recommendation**: Review options and decide based on your goals (exact replica vs practical improvement)
